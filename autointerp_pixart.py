"""Autointerp diagnostic: why doesn't per-token (pixel-wise) localization work for PixArt?
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import CocoActivationDataset
from feature_usage import compute_feature_usage, labelmap_to_token_labels
from spatial_align import build_spatial_aligner_from_config
from visualize_feature_activations import (
    discover_stems,
    encode_image_tokens,
    find_latest_checkpoint,
    find_raw_image,
    load_checkpoint,
)


DEFAULT_CONFIG = "/content/algoverse_github/config.yaml"
DEFAULT_CACHE = "/content/combined_cache"
DEFAULT_WEIGHTS_DIR = "/content/algoverse_github/weights"
DEFAULT_OUTPUT_DIR = "/content/autointerp_results"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llava:7b"


# --------------------------------------------------------------------------- #
# Stem / split resolution (mirrors feature_atlas.py's resolve_split_stems,
# without the argparse-namespace coupling so this script owns its own flags).
# --------------------------------------------------------------------------- #

def resolve_stems(cache_root: str, split: str, stems_file: Optional[str]) -> Tuple[List[str], str]:
    if stems_file:
        path = stems_file
    elif split == "all":
        return discover_stems(cache_root), "all"
    else:
        path = os.path.join(cache_root, f"{split}_stems.txt")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Stem list not found: {path}\n"
            "  It's written by coco_dataset_setup.split_train_val() the first time "
            "uni_demo.py runs against this cache. Pass --split all to use every cached "
            "image (not held out), or --stems_file to point at an explicit list."
        )
    with open(path) as f:
        wanted = [line.strip() for line in f if line.strip()]

    cached = set(discover_stems(cache_root))
    stems = [s for s in wanted if s in cached]
    return stems, split



def select_features(
    scores: Dict[str, torch.Tensor],
    sources: Tuple[str, str],
    num_features: int,
) -> Tuple[List[int], int]:
    """Rank features that fire for BOTH sources by combined mean activation.

    scores[source] is (n_images, K) per-image max|z|, from encode_image_tokens.
    "Fires" uses feature_usage.compute_feature_usage's ever_fired criterion, so
    this agrees with dictionary_diagnostic.py's definition of "used".
    """
    a, b = sources
    used_a = compute_feature_usage(scores[a], criterion="ever_fired")
    used_b = compute_feature_usage(scores[b], criterion="ever_fired")
    used_both = used_a & used_b
    n_both = int(used_both.sum().item())
    if n_both == 0:
        raise RuntimeError(
            f"No dictionary feature fires for both {a!r} and {b!r} anywhere in this "
            f"sample. Increase --num_selection_images, or check the checkpoint/cache "
            f"actually cover both sources."
        )

    combined = scores[a].mean(dim=0) + scores[b].mean(dim=0)
    combined = torch.where(used_both, combined, torch.full_like(combined, float("-inf")))
    k = min(num_features, n_both)
    top = torch.topk(combined, k=k).indices.tolist()
    return top, n_both


def top_images_for_feature(scores_source: torch.Tensor, feature_idx: int, m: int) -> List[Tuple[int, float]]:
    col = scores_source[:, feature_idx]
    n_nonzero = int((col > 0).sum().item())
    k = min(m, n_nonzero) if n_nonzero > 0 else 0
    if k == 0:
        return []
    top_vals, top_idx = torch.topk(col, k=k)
    return list(zip(top_idx.tolist(), top_vals.tolist()))


def token_peakiness(z: torch.Tensor, feature_idx: int) -> Tuple[float, int, float]:
    """max/mean over tokens for ONE feature -- same definition as
    pixart_timestep_autopsy.localization()'s 'peak', restricted to a single
    feature so we don't pay for all K columns per crop.

    Returns (peakiness, peak_token_idx, peak_value). peakiness has a floor of
    1.0 (uniform activation); higher = more spatially concentrated.
    """
    absz = z[0, :, feature_idx].abs()
    mean_v = float(absz.mean().item())
    max_v, peak_idx = absz.max(dim=0)
    peak_v = float(max_v.item())
    peakiness = peak_v / (mean_v + 1e-8)
    return peakiness, int(peak_idx.item()), peak_v


def token_bbox(token_idx: int, grid_size: int, render_size: int, context: int) -> Tuple[int, int, int, int]:
    """Pixel box (on a render_size x render_size square canvas) for the token
    block centered on token_idx, expanded by `context` tokens on each side.
    """
    patch = render_size / grid_size
    row, col = divmod(token_idx, grid_size)
    r0, r1 = max(row - context, 0), min(row + context + 1, grid_size)
    c0, c1 = max(col - context, 0), min(col + context + 1, grid_size)
    x0, y0 = int(round(c0 * patch)), int(round(r0 * patch))
    x1, y1 = int(round(c1 * patch)), int(round(r1 * patch))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    return x0, y0, x1, y1


def crop_patch(
    base_image: Image.Image,
    token_idx: int,
    grid_size: int,
    render_size: int,
    context: int,
    crop_out: int,
) -> Image.Image:
    canvas = base_image.resize((render_size, render_size), resample=Image.BICUBIC)
    x0, y0, x1, y1 = token_bbox(token_idx, grid_size, render_size, context)
    crop = canvas.crop((x0, y0, x1, y1))
    return crop.resize((crop_out, crop_out), resample=Image.BICUBIC)


def semantic_label_for_token(
    stem: str,
    token_idx: int,
    grid_size: int,
    semantic_mask_dir: str,
    cache: Dict[str, Optional[Tuple[np.ndarray, Dict[int, str]]]],
) -> Optional[str]:
    if stem not in cache:
        labelmap_path = os.path.join(semantic_mask_dir, f"{stem}_labelmap.png")
        legend_path = os.path.join(semantic_mask_dir, f"{stem}_legend.json")
        if os.path.isfile(labelmap_path) and os.path.isfile(legend_path):
            label_map = np.asarray(Image.open(labelmap_path))
            with open(legend_path) as f:
                legend = json.load(f)
            id_to_name = {int(k): v for k, v in legend.get("id_to_name", {}).items()}
            token_labels = labelmap_to_token_labels(label_map, grid_size)
            cache[stem] = (token_labels, id_to_name)
        else:
            cache[stem] = None

    entry = cache[stem]
    if entry is None:
        return None
    token_labels, id_to_name = entry
    if token_idx >= len(token_labels):
        return None
    cid = int(token_labels[token_idx])
    return id_to_name.get(cid, f"class_{cid}")



def build_comparison_montage(
    feature_idx: int,
    rows: List[Tuple[str, List[Tuple[str, Image.Image]]]],
    cell: int = 224,
    pad: int = 8,
    label_h: int = 20,
    row_label_w: int = 110,
) -> Image.Image:
    """rows: [(row_label, [(caption, crop_image), ...]), ...] -- one row per source."""
    if not rows or all(not items for _, items in rows):
        raise ValueError("No crops to render into a montage.")
    n_cols = max((len(items) for _, items in rows), default=0)
    cell_w, cell_h = cell + pad, cell + label_h + pad
    header_h = 24
    width = row_label_w + n_cols * cell_w + pad
    height = header_h + len(rows) * cell_h + pad
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((pad, 4), f"feature {feature_idx}", fill=(255, 255, 255), font=font)
    y = header_h
    for row_label, items in rows:
        draw.text((pad, y + cell // 2), row_label, fill=(255, 255, 0), font=font)
        for i, (caption, img) in enumerate(items):
            x = row_label_w + i * cell_w
            canvas.paste(img.resize((cell, cell)), (x, y))
            draw.text((x, y + cell + 2), caption, fill=(200, 200, 200), font=font)
        y += cell_h
    return canvas



PROMPT_TEMPLATE = """You are looking at a diagnostic image for a single feature (index {feature_idx}) of a shared sparse-autoencoder dictionary trained on activations from two different vision models.

Row "{row1}" shows {n1} image crops. Each crop is centered on the single token where this feature activates MOST STRONGLY in model "{row1}", taken from {n1} different photos.
Row "{row2}" shows {n2} image crops, same idea, but for the SAME feature index in a different model, "{row2}".

Answer in EXACTLY this format and nothing else, one line per field:
ROW1_COHERENT: yes or no
ROW1_LABEL: <3-6 word description of the common visual concept in row 1, or "none" if the crops look unrelated to each other>
ROW2_COHERENT: yes or no
ROW2_LABEL: <3-6 word description of the common visual concept in row 2, or "none" if the crops look unrelated to each other>
SAME_CONCEPT: yes, no, or unclear -- do row 1 and row 2 depict the same visual concept?
"""


def ollama_available(host: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5):
            return True
    except (urllib.error.URLError, OSError):
        return False


def ollama_has_model(host: str, model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        names = {m.get("name", "") for m in data.get("models", [])}
        base = model.split(":")[0]
        return model in names or any(n.split(":")[0] == base for n in names)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


def ollama_pull(host: str, model: str) -> None:
    print(f"[autointerp] pulling Ollama model {model!r} (first run only, can take a few minutes)...")
    try:
        subprocess.run(["ollama", "pull", model], check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"[autointerp] WARNING: could not auto-pull {model!r}: {e}\n  Run manually: ollama pull {model}")


def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_ollama_vision(
    host: str,
    model: str,
    montage: Image.Image,
    feature_idx: int,
    row1: str,
    row2: str,
    n1: int,
    n2: int,
    timeout: float,
) -> Optional[str]:
    prompt = PROMPT_TEMPLATE.format(feature_idx=feature_idx, row1=row1, row2=row2, n1=n1, n2=n2)
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [image_to_b64(montage)],
        "stream": False,
        "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        return body.get("response", "")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"[autointerp] Ollama call failed for feature {feature_idx}: {e}")
        return None


_FIELD_PATTERNS = {
    "row1_coherent": r"ROW1_COHERENT:\s*(yes|no)",
    "row1_label": r"ROW1_LABEL:\s*(.+)",
    "row2_coherent": r"ROW2_COHERENT:\s*(yes|no)",
    "row2_label": r"ROW2_LABEL:\s*(.+)",
    "same_concept": r"SAME_CONCEPT:\s*(yes|no|unclear)",
}


def parse_llm_response(text: Optional[str]) -> dict:
    if not text:
        return {"parsed": False, "raw": text}
    out: dict = {"parsed": True, "raw": text}
    for key, pattern in _FIELD_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        out[key] = m.group(1).strip().lower() if m else None
    out["parsed"] = any(out[k] is not None for k in _FIELD_PATTERNS)
    return out



def summarize(features_report: List[dict], sources: Tuple[str, str], llm_enabled: bool) -> dict:
    peak_a = [f["baseline"]["mean_peakiness"] for f in features_report if f["baseline"]["mean_peakiness"] is not None]
    peak_b = [f["test"]["mean_peakiness"] for f in features_report if f["test"]["mean_peakiness"] is not None]
    summary: dict = {
        "n_features": len(features_report),
        "baseline_source": sources[0],
        "test_source": sources[1],
        "mean_peakiness_baseline": float(np.mean(peak_a)) if peak_a else None,
        "mean_peakiness_test": float(np.mean(peak_b)) if peak_b else None,
    }
    if llm_enabled:
        parsed = [f["llm"] for f in features_report if f.get("llm") and f["llm"].get("parsed")]
        n_llm = len(parsed)
        summary["n_llm_parsed"] = n_llm
        if n_llm:
            c1 = [p["row1_coherent"] == "yes" for p in parsed]
            c2 = [p["row2_coherent"] == "yes" for p in parsed]
            same = [p["same_concept"] for p in parsed]
            summary["pct_coherent_baseline"] = sum(c1) / n_llm
            summary["pct_coherent_test"] = sum(c2) / n_llm
            summary["pct_same_concept"] = same.count("yes") / n_llm
    summary["verdict"] = build_verdict(summary)
    return summary


def build_verdict(s: dict) -> str:
    pa, pb = s.get("mean_peakiness_baseline"), s.get("mean_peakiness_test")
    if pa is None or pb is None:
        return ("Not enough features had usable crops on both sources in this sample to draw "
                "a conclusion -- widen --num_selection_images or --num_features, or check "
                "--raw_image_dir points at the right COCO split.")

    parts = [
        f"Mean peakiness: {s['baseline_source']}={pa:.2f}, {s['test_source']}={pb:.2f} "
        f"(1.0 = uniform activation across the image, i.e. no localization)."
    ]
    localization_bad = pb < 1.5 and (pa - pb) > 1.0
    if localization_bad:
        parts.append(
            f"{s['test_source']}'s top-activating tokens are close to uniform across the image "
            f"while {s['baseline_source']} localizes clearly on the matched features -- consistent "
            f"with the SAE not extracting spatially-resolved structure from {s['test_source']}, "
            f"not with a rendering/visualization bug."
        )
    else:
        parts.append("Peakiness alone does not show a stark localization gap between the two sources in this sample.")

    if s.get("n_llm_parsed"):
        pcb, pct, psame = s["pct_coherent_baseline"], s["pct_coherent_test"], s["pct_same_concept"]
        parts.append(
            f"LLM judgment: {pcb*100:.0f}% of sampled features showed one nameable visual concept "
            f"in {s['baseline_source']}'s crops vs {pct*100:.0f}% in {s['test_source']}'s; "
            f"{psame*100:.0f}% were judged the SAME concept across both sources."
        )
        if pct < 0.4 and (pcb - pct) > 0.3:
            parts.append(
                "The LLM independently agrees the test-source crops share a common concept far less "
                "often than the baseline's -- two independent signals (peakiness and LLM coherence) "
                "point the same way."
            )
    elif "n_llm_parsed" in s:
        parts.append("LLM was enabled but produced no parseable responses -- check the Ollama model/host.")

    return " ".join(parts)


def write_markdown_report(path: str, features_report: List[dict], summary: dict, sources: Tuple[str, str], ckpt_path: str) -> None:
    a, b = sources
    lines = [
        f"# Autointerp report: {a} vs {b}",
        "",
        f"Checkpoint: `{ckpt_path}`",
        "",
        "## Summary",
        "",
        f"- Features interpreted: {summary['n_features']}",
    ]
    if summary["mean_peakiness_baseline"] is not None:
        lines.append(f"- Mean peakiness -- {a}: {summary['mean_peakiness_baseline']:.2f}, {b}: {summary['mean_peakiness_test']:.2f}")
    if summary.get("n_llm_parsed"):
        lines.append(f"- LLM-judged coherent -- {a}: {summary['pct_coherent_baseline']*100:.0f}%, {b}: {summary['pct_coherent_test']*100:.0f}%")
        lines.append(f"- LLM judged same concept in both: {summary['pct_same_concept']*100:.0f}%")
    lines += ["", f"**Verdict:** {summary['verdict']}", "", "## Per-feature detail", ""]
    lines.append(f"| feature | peak[{a}] | peak[{b}] | LLM row1 ({a}) | LLM row2 ({b}) | same concept? | montage |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in features_report:
        llm = e.get("llm") or {}
        peak_a = e["baseline"]["mean_peakiness"]
        peak_b = e["test"]["mean_peakiness"]
        peak_a_str = f"{peak_a:.2f}" if peak_a is not None else "-"
        peak_b_str = f"{peak_b:.2f}" if peak_b is not None else "-"
        row1 = f"{llm.get('row1_label', '-')} ({llm.get('row1_coherent', '-')})"
        row2 = f"{llm.get('row2_label', '-')} ({llm.get('row2_coherent', '-')})"
        rel = os.path.relpath(e["montage_path"], os.path.dirname(path) or ".")
        lines.append(
            f"| {e['feature_idx']} | {peak_a_str} | {peak_b_str} | {row1} | {row2} | "
            f"{llm.get('same_concept', '-')} | [{os.path.basename(rel)}]({rel}) |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--cache_root", default=DEFAULT_CACHE)
    p.add_argument("--weights_dir", default=DEFAULT_WEIGHTS_DIR)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--raw_image_dir", default=None, required=False,
                   help="Directory with the original COCO images. Required in practice -- "
                        "without raw pixels there is nothing to crop or show an LLM.")
    p.add_argument("--semantic_mask_dir", default=None,
                   help="Optional: *_labelmap.png/*_legend.json from semantic_mask.py, for a "
                        "free non-LLM concept label at each crop's peak token.")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    p.add_argument("--split", default="val", choices=("train", "val", "all"))
    p.add_argument("--stems_file", default=None)

    p.add_argument("--baseline_source", default="DinoV2", help="Known-good localization reference.")
    p.add_argument("--test_source", default="PixArt", help="Source under diagnosis.")
    p.add_argument("--pixart_timestep", type=int, default=None,
                   help="Override the diffusion timestep. Defaults to whatever the checkpoint trained on.")

    p.add_argument("--num_selection_images", type=int, default=150,
                   help="How many images to score (both sources) when picking which features to interpret.")
    p.add_argument("--num_features", type=int, default=12,
                   help="How many shared (used-by-both) features to autointerp.")
    p.add_argument("--crops_per_feature", type=int, default=5,
                   help="Top-activating images to crop per feature, per source.")
    p.add_argument("--patch_context", type=int, default=1,
                   help="Tokens of context around the peak token in each crop (0 = single token).")
    p.add_argument("--render_size", type=int, default=448)
    p.add_argument("--crop_size", type=int, default=224)

    p.add_argument("--ollama_host", default=DEFAULT_OLLAMA_HOST)
    p.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--no_llm", action="store_true", help="Skip Ollama entirely; just produce crops + peakiness.")
    p.add_argument("--no_pull", action="store_true", help="Don't auto-pull the Ollama model if missing.")
    p.add_argument("--llm_timeout", type=float, default=120.0)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    crops_dir = os.path.join(args.output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    if not args.raw_image_dir:
        print("[autointerp] WARNING: --raw_image_dir not set; every feature will be skipped "
              "(no pixels to crop). Pass --raw_image_dir /content/coco_data/val2017 or similar.")

    ckpt_path = args.ckpt
    if not ckpt_path or not os.path.isfile(ckpt_path):
        ckpt_path = find_latest_checkpoint(args.weights_dir)
    print(f"[autointerp] checkpoint: {ckpt_path}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    g = cfg["global"]

    model, model_tokens_native = load_checkpoint(ckpt_path, cfg)
    model.eval().to(args.device)

    if model.cls_pool_mode != "none":
        raise ValueError(
            f"cls_pool_mode={model.cls_pool_mode!r} pools all tokens into one vector per image; "
            "per-patch autointerp requires cls_pool_mode='none'."
        )

    eval_g = getattr(model, "_training_global", g)
    aligner = build_spatial_aligner_from_config(eval_g, model_tokens_native)
    if aligner is not None:
        print(f"[autointerp] spatial alignment ON -> target grid {aligner.target_grid_size}x{aligner.target_grid_size}")

    sources = (args.baseline_source, args.test_source)
    stems, split_tag = resolve_stems(args.cache_root, args.split, args.stems_file)
    if not stems:
        raise RuntimeError(f"No stems found for split={args.split!r} in {args.cache_root!r}")
    stems = stems[: args.num_selection_images]
    print(f"[autointerp] split={split_tag}: scoring {len(stems)} images for feature selection")

    ds = CocoActivationDataset(
        cache_root=args.cache_root,
        sources=list(sources),
        combined_npz=True,
        standardize=bool(eval_g.get("standardize", True)),
        return_metadata=True,
        diffusion_models=[s for s in sources if s in model.diffusion_models],
        use_class_tokens=False,
        standardization_stats=getattr(model, "_standardization_stats", None),
        stats_seed=eval_g.get("stats_seed", 0),
        spatial_aligner=aligner,
    )
    stem_to_idx = {s: i for i, s in enumerate(ds.stems)}
    missing = [s for s in stems if s not in stem_to_idx]
    if missing:
        raise RuntimeError(f"{len(missing)} stems from the split are not in this cache (first: {missing[:3]})")

    # ---- selection pass: score every sampled image for both sources ----
    scores: Dict[str, Optional[torch.Tensor]] = {s: None for s in sources}
    acts_meta_cache: Dict[str, tuple] = {}
    for i, stem in enumerate(tqdm(stems, desc="[autointerp] selection pass")):
        (acts, meta), _ = ds[stem_to_idx[stem]]
        acts_meta_cache[stem] = (acts, meta)
        for source in sources:
            sc, _z, _grid = encode_image_tokens(
                model, acts, meta, source, aligner, args.device,
                config_global=eval_g, pixart_timestep_override=args.pixart_timestep,
            )
            if scores[source] is None:
                scores[source] = torch.zeros(len(stems), sc.numel())
            scores[source][i] = sc

    K = scores[sources[0]].shape[1]
    feature_ids, n_both = select_features(scores, sources, args.num_features)
    print(f"[autointerp] {n_both}/{K} features fire for both {sources[0]} and {sources[1]} in this "
          f"sample; interpreting top {len(feature_ids)} by combined activation strength")

    # ---- LLM setup ----
    llm_enabled = not args.no_llm
    if llm_enabled:
        if not ollama_available(args.ollama_host):
            print(f"[autointerp] WARNING: Ollama not reachable at {args.ollama_host}. In Colab:\n"
                  f"    curl -fsSL https://ollama.com/install.sh | sh\n"
                  f"    (ollama serve &) ; sleep 2\n"
                  f"    ollama pull {args.ollama_model}\n"
                  f"  Continuing without LLM labels (--no_llm behavior).")
            llm_enabled = False
        elif not ollama_has_model(args.ollama_host, args.ollama_model):
            if args.no_pull:
                print(f"[autointerp] WARNING: model {args.ollama_model!r} not found on Ollama and "
                      f"--no_pull set. Continuing without LLM labels.")
                llm_enabled = False
            else:
                ollama_pull(args.ollama_host, args.ollama_model)

    semantic_cache: Dict[str, Optional[Tuple[np.ndarray, Dict[int, str]]]] = {}
    features_report: List[dict] = []

    for feature_idx in feature_ids:
        row_crops: Dict[str, List[Tuple[str, Image.Image]]] = {}
        role_stats: Dict[str, dict] = {}

        for source in sources:
            top = top_images_for_feature(scores[source], feature_idx, args.crops_per_feature)
            crops: List[Tuple[str, Image.Image]] = []
            peakinesses: List[float] = []
            used_stems: List[str] = []

            for img_i, score_val in top:
                stem = stems[img_i]
                raw_path = find_raw_image(stem, args.raw_image_dir) if args.raw_image_dir else None
                if raw_path is None:
                    continue
                acts, meta = acts_meta_cache[stem]
                _sc, z, grid_size = encode_image_tokens(
                    model, acts, meta, source, aligner, args.device,
                    config_global=eval_g, pixart_timestep_override=args.pixart_timestep,
                )
                peak, peak_token, _peak_val = token_peakiness(z, feature_idx)
                base_img = Image.open(raw_path).convert("RGB")
                crop = crop_patch(base_img, peak_token, grid_size, args.render_size, args.patch_context, args.crop_size)

                concept = None
                if args.semantic_mask_dir:
                    concept = semantic_label_for_token(stem, peak_token, grid_size, args.semantic_mask_dir, semantic_cache)

                caption = f"{stem[-8:]} a={score_val:.2f} pk={peak:.2f}"
                if concept:
                    caption += f" [{concept}]"
                crops.append((caption, crop))
                peakinesses.append(peak)
                used_stems.append(stem)

            row_crops[source] = crops
            role_stats[source] = {
                "mean_peakiness": float(np.mean(peakinesses)) if peakinesses else None,
                "n_crops": len(crops),
                "top_images": used_stems,
            }

        if not row_crops[sources[0]] or not row_crops[sources[1]]:
            print(f"[autointerp] feature {feature_idx}: skipped (no raw image found on one side -- "
                  f"check --raw_image_dir)")
            continue

        montage = build_comparison_montage(
            feature_idx,
            [(sources[0], row_crops[sources[0]]), (sources[1], row_crops[sources[1]])],
            cell=args.crop_size,
        )
        montage_path = os.path.join(crops_dir, f"feat{feature_idx:05d}.png")
        montage.save(montage_path)

        llm_result = None
        if llm_enabled:
            raw = call_ollama_vision(
                args.ollama_host, args.ollama_model, montage, feature_idx,
                sources[0], sources[1], len(row_crops[sources[0]]), len(row_crops[sources[1]]),
                timeout=args.llm_timeout,
            )
            llm_result = parse_llm_response(raw)

        entry = {
            "feature_idx": feature_idx,
            "baseline": {"source": sources[0], **role_stats[sources[0]]},
            "test": {"source": sources[1], **role_stats[sources[1]]},
            "montage_path": montage_path,
            "llm": llm_result,
        }
        features_report.append(entry)

        pa, pb = role_stats[sources[0]]["mean_peakiness"], role_stats[sources[1]]["mean_peakiness"]
        msg = f"[autointerp] feature {feature_idx}: peak[{sources[0]}]={pa:.2f} peak[{sources[1]}]={pb:.2f}"
        if llm_result and llm_result.get("parsed"):
            msg += f"  same_concept={llm_result.get('same_concept')}"
        print(msg)

    if not features_report:
        raise RuntimeError(
            "No features produced any crops -- almost always means --raw_image_dir doesn't "
            "contain the cached images. Check it points at the same split the cache was built from."
        )

    summary = summarize(features_report, sources, llm_enabled)

    output = {
        "checkpoint": ckpt_path,
        "baseline_source": sources[0],
        "test_source": sources[1],
        "split": split_tag,
        "n_selection_images": len(stems),
        "n_candidate_features_used_by_both": n_both,
        "llm": {
            "enabled": llm_enabled,
            "model": args.ollama_model if llm_enabled else None,
            "host": args.ollama_host if llm_enabled else None,
        },
        "features": features_report,
        "summary": summary,
    }
    json_path = os.path.join(args.output_dir, "autointerp_report.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    md_path = os.path.join(args.output_dir, "autointerp_report.md")
    write_markdown_report(md_path, features_report, summary, sources, ckpt_path)

    print(f"\n[autointerp] wrote {json_path}")
    print(f"[autointerp] wrote {md_path}")
    print(f"[autointerp] montages -> {crops_dir}/feat*.png")
    print(f"\n[verdict] {summary['verdict']}")


if __name__ == "__main__":
    main()
