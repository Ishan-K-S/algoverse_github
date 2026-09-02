"""Side-by-side feature atlas: for every SAE feature, the top-k images that
activate it in EACH model, rendered as one PDF.

This is the standard SAE interpretability figure, and it's the direct test of
the shared-language claim: if feature 1494 means the same thing in both models,
DinoV2's top-5 and PixArt's top-5 for feature 1494 should be the same kind of
image. If they're unrelated, the feature is shared only in the bookkeeping
sense (both models touch that index) -- which aggregate metrics like
partition score and "used by both" cannot distinguish.

Unlike the per-patch heatmaps, this measures ACROSS-image selectivity ("which
images does this feature prefer?") rather than WITHIN-image localization
("where in this image does it fire?"), so a spatially diffuse feature can still
show up as crisply selective here.

Reuses top_activating_images.py for all scoring -- this script only runs it
once per source, joins the results per feature, and renders them.

Outputs (into --output_dir):
  feature_atlas.pdf                 - one block per feature: a row of top-k
                                      images per model, captioned with scores
  top_activations_<source>.json     - raw per-source rankings (from
                                      top_activating_images.compute_top_activations)
  feature_atlas_overlap.json        - per-feature image/label overlap between
                                      the two models' top-k sets, so the
                                      "do they agree?" question has a number
                                      attached and not just a picture

Example:
    python feature_atlas.py --image_dir /content/coco_data/val2017 \
        --output_dir /content/results --top_k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from top_activating_images import (
    build_coco_label_lookup,
    build_spatial_aligner,
    coco_annotation_path,
    compute_top_activations,
    discover_stems,
    find_latest_checkpoint,
    labels_for_stem,
    load_universal_sae,
)
from visualize_feature_activations import find_raw_image

CACHE_ROOT = "/content/combined_cache"
CONFIG_PATH = "/content/algoverse_github/config.yaml"
WEIGHTS_DIR = "/content/algoverse_github/weights"
IMAGE_DIR = "/content/coco_data/val2017"
OUTPUT_DIR = "/content/results"
COCO_ANNOTATIONS_DIR = "/content/coco_annotations"
COCO_SPLIT = "val2017"

DEAD_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Scoring (delegated to top_activating_images.py)
# --------------------------------------------------------------------------- #

def top_activations_for_source(
    source: str,
    model,
    cfg: dict,
    spatial_aligner,
    args,
    coco_labels: Optional[Dict[str, List[str]]],
    n_images: int,
) -> dict:
    """Run (or reload) top_activating_images.py's ranking for one source."""
    json_path = os.path.join(args.output_dir, f"top_activations_{source}.json")

    if args.reuse_json and os.path.isfile(json_path):
        print(f"[atlas] reusing existing rankings -> {json_path}")
        with open(json_path) as f:
            return json.load(f)

    # compute_top_activations takes a PERCENT, not a count: top_n is derived as
    # round(n_images * top_pct / 100). Invert that so --top_k means exactly k.
    top_pct = 100.0 * args.top_k / max(n_images, 1)

    diffusion_models = set(cfg.get("global", {}).get("diffusion_models", []))

    return compute_top_activations(
        cache_root=args.cache_root,
        sae_model=model,
        source=source,
        diffusion_models=diffusion_models,
        output_path=json_path,
        top_pct=top_pct,
        timestep_idx=args.timestep_idx,
        use_cls=args.use_cls,
        batch_size=args.batch_size,
        device=args.device,
        max_images=args.max_images,
        coco_labels=coco_labels,
        feature_pool=args.feature_pool,
        spatial_aligner=spatial_aligner,
        # Mirror top_activating_images.main(): reuse the checkpoint's own training
        # stats rather than recomputing (and rather than silently standardizing
        # eval data differently than training did).
        standardization_stats=getattr(model, "_standardization_stats", None),
        training_global=getattr(model, "_training_global", None),
    )


# --------------------------------------------------------------------------- #
# Join + overlap metrics
# --------------------------------------------------------------------------- #

def entry_labels(entries: Sequence[dict], coco_labels) -> Set[str]:
    """Union of COCO labels across a model's top-k images for one feature.

    The rankings JSON only attaches coco_labels to the rank-0 entry, so look up
    the rest here via top_activating_images.labels_for_stem.
    """
    out: Set[str] = set()
    if coco_labels is None:
        return out
    for e in entries:
        out.update(labels_for_stem(e["filename"], coco_labels) or [])
    return out


def jaccard(a: Set, b: Set) -> Optional[float]:
    union = a | b
    if not union:
        return None
    return len(a & b) / len(union)


def feature_overlap(
    entries_a: Sequence[dict],
    entries_b: Sequence[dict],
    coco_labels,
) -> dict:
    """How much do the two models' top-k sets agree for one feature?

    image_jaccard: same actual images picked (strictest, and free).
    label_jaccard: same COCO semantic content, even if different images --
        the more meaningful "same concept" signal.
    """
    imgs_a = {e["filename"] for e in entries_a}
    imgs_b = {e["filename"] for e in entries_b}
    labels_a = entry_labels(entries_a, coco_labels)
    labels_b = entry_labels(entries_b, coco_labels)
    return {
        "image_jaccard": jaccard(imgs_a, imgs_b),
        "label_jaccard": jaccard(labels_a, labels_b),
        "labels": {"a": sorted(labels_a), "b": sorted(labels_b)},
    }


def is_dead(entries: Sequence[dict]) -> bool:
    return not entries or abs(entries[0].get("score", 0.0)) <= DEAD_EPS


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

class ThumbCache:
    """Decode each distinct image once. There are only ~2000 distinct stems but
    potentially 100k+ thumbnail slots across all features, so caching is the
    difference between minutes and hours.
    """

    def __init__(self, image_dir: str, thumb_px: int):
        self.image_dir = image_dir
        self.thumb_px = thumb_px
        self._cache: Dict[str, Optional[Image.Image]] = {}
        self.missing: Set[str] = set()

    def get(self, stem: str) -> Optional[Image.Image]:
        if stem in self._cache:
            return self._cache[stem]
        path = find_raw_image(stem, self.image_dir)
        if path is None:
            self.missing.add(stem)
            self._cache[stem] = None
            return None
        img = Image.open(path).convert("RGB").resize(
            (self.thumb_px, self.thumb_px), resample=Image.BICUBIC
        )
        self._cache[stem] = img
        return img


def _font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def compose_feature_block(
    feat_id: int,
    per_source: Dict[str, List[dict]],
    sources: Sequence[str],
    overlap: dict,
    thumbs: ThumbCache,
    thumb_px: int,
    top_k: int,
) -> Image.Image:
    """One feature: a title line, then one captioned row of top-k images per model."""
    pad = 6
    caption_h = 14
    row_label_w = 62
    title_h = 20

    row_h = thumb_px + caption_h + pad
    width = row_label_w + top_k * (thumb_px + pad) + pad
    height = title_h + len(sources) * row_h + pad

    block = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(block)
    font = _font()

    img_j = overlap.get("image_jaccard")
    lab_j = overlap.get("label_jaccard")
    title = f"feature {feat_id}"
    if img_j is not None:
        title += f"   image overlap {img_j:.2f}"
    if lab_j is not None:
        title += f"   label overlap {lab_j:.2f}"
    if font is not None:
        draw.text((pad, 4), title, fill=(255, 255, 255), font=font)

    for r, source in enumerate(sources):
        y = title_h + r * row_h
        if font is not None:
            draw.text((pad, y + thumb_px // 2), source, fill=(200, 200, 200), font=font)
        for c, entry in enumerate(per_source[source][:top_k]):
            x = row_label_w + c * (thumb_px + pad)
            thumb = thumbs.get(entry["filename"])
            if thumb is None:
                draw.rectangle(
                    [x, y, x + thumb_px, y + thumb_px], outline=(90, 90, 90)
                )
                if font is not None:
                    draw.text((x + 4, y + thumb_px // 2), "missing", fill=(140, 140, 140), font=font)
            else:
                block.paste(thumb, (x, y))
            if font is not None:
                draw.text(
                    (x, y + thumb_px + 2),
                    f"{entry['filename'][-6:]} {entry['score']:.2f}",
                    fill=(190, 190, 190),
                    font=font,
                )

    return block


def stack_blocks(blocks: Sequence[Image.Image], pad: int = 10) -> Image.Image:
    width = max(b.width for b in blocks) + 2 * pad
    height = sum(b.height for b in blocks) + pad * (len(blocks) + 1)
    page = Image.new("RGB", (width, height), (18, 18, 18))
    y = pad
    for b in blocks:
        page.paste(b, (pad, y))
        y += b.height + pad
    return page


def render_pdf(pages: Sequence[Image.Image], pdf_path: str) -> None:
    """Stream pages into a multi-page PDF.

    Uses matplotlib's PdfPages because it writes incrementally -- PIL's
    save(append_images=...) needs every page resident in memory at once, which
    is not viable at 12288 features.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pdf:
        for page in pages:
            arr = np.asarray(page)
            h, w = arr.shape[:2]
            dpi = 100
            fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
            ax = fig.add_axes((0, 0, 1, 1))
            ax.imshow(arr)
            ax.axis("off")
            pdf.savefig(fig, facecolor="#121212")
            plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cache_root", default=CACHE_ROOT)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--weights_dir", default=WEIGHTS_DIR)
    p.add_argument("--ckpt", default=None, help="Defaults to newest .pt/.pth under --weights_dir")
    p.add_argument("--sources", nargs=2, default=["DinoV2", "PixArt"],
                   help="Exactly two sources, rendered as two rows per feature.")
    p.add_argument("--image_dir", default=IMAGE_DIR, help="Raw images named <stem>.jpg")
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--pdf_name", default="feature_atlas.pdf")
    p.add_argument("--top_k", type=int, default=5, help="Images per feature per model")

    p.add_argument("--features", nargs="+", type=int, default=None,
                   help="Render only these feature ids. Default: all.")
    p.add_argument("--max_features", type=int, default=None,
                   help="Cap how many features get rendered (after filtering).")
    p.add_argument("--only_shared", action="store_true",
                   help="Skip features that are dead in either model -- i.e. keep only "
                        "features both models actually use, which are the ones the "
                        "shared-language claim is about.")
    p.add_argument("--keep_dead", action="store_true",
                   help="Include features that are dead in BOTH models (default: skipped).")
    p.add_argument("--sort_by", default="label_jaccard",
                   choices=("feature_id", "label_jaccard", "image_jaccard"),
                   help="Ordering in the PDF. Overlap sorts put the strongest "
                        "shared-concept evidence on the first pages.")
    p.add_argument("--features_per_page", type=int, default=2)
    p.add_argument("--thumb_px", type=int, default=150)

    p.add_argument("--reuse_json", action="store_true",
                   help="Reuse existing top_activations_<source>.json in --output_dir "
                        "instead of re-running the (slow) scoring pass.")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--timestep_idx", type=int, default=None,
                   help="Diffusion timestep override. Defaults to the checkpoint's.")
    p.add_argument("--use_cls", action="store_true")
    p.add_argument("--feature_pool", default="max", choices=("max", "max_abs"))
    p.add_argument("--coco_annotations_dir", default=COCO_ANNOTATIONS_DIR)
    p.add_argument("--coco_split", default=COCO_SPLIT, choices=("train2017", "val2017"))
    p.add_argument("--skip_coco_labels", action="store_true",
                   help="Skip COCO labels. Disables the label-overlap metric.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    src_a, src_b = args.sources

    ckpt = args.ckpt
    if ckpt is None or not os.path.isfile(ckpt):
        ckpt = find_latest_checkpoint(args.weights_dir)
    print(f"[atlas] checkpoint : {ckpt}")

    coco_labels = None
    if not args.skip_coco_labels:
        coco_labels = build_coco_label_lookup(
            coco_annotation_path(args.coco_annotations_dir, args.coco_split)
        )

    model, cfg = load_universal_sae(ckpt, args.config, args.device)
    eval_g = getattr(model, "_training_global", cfg.get("global", {}))
    spatial_aligner = build_spatial_aligner(
        {"global": eval_g, "model_zoo": cfg.get("model_zoo", {})}
    )

    stems = discover_stems(args.cache_root)
    if args.max_images is not None:
        stems = stems[: args.max_images]
    n_images = len(stems)
    print(f"[atlas] {n_images} cached images | top_k={args.top_k} per feature per model")

    # ---- Score both sources ----
    results = {
        s: top_activations_for_source(
            s, model, cfg, spatial_aligner, args, coco_labels, n_images
        )
        for s in (src_a, src_b)
    }
    top_a = results[src_a]["top_activations"]
    top_b = results[src_b]["top_activations"]

    # ---- Join per feature ----
    feat_ids = sorted(set(top_a) & set(top_b), key=int)
    if args.features is not None:
        wanted = {str(f) for f in args.features}
        feat_ids = [f for f in feat_ids if f in wanted]

    rows = []
    for fid in feat_ids:
        ea, eb = top_a[fid], top_b[fid]
        dead_a, dead_b = is_dead(ea), is_dead(eb)
        if args.only_shared and (dead_a or dead_b):
            continue
        if not args.keep_dead and dead_a and dead_b:
            continue
        ov = feature_overlap(ea, eb, coco_labels)
        rows.append({
            "feature_id": int(fid),
            "dead_in": [s for s, d in ((src_a, dead_a), (src_b, dead_b)) if d],
            **{k: v for k, v in ov.items() if k != "labels"},
            "labels": {src_a: ov["labels"]["a"], src_b: ov["labels"]["b"]},
        })

    if args.sort_by != "feature_id":
        rows.sort(key=lambda r: (r[args.sort_by] is None, -(r[args.sort_by] or 0.0)))
    if args.max_features is not None:
        rows = rows[: args.max_features]

    overlap_path = os.path.join(args.output_dir, "feature_atlas_overlap.json")
    with open(overlap_path, "w") as f:
        json.dump({
            "checkpoint": ckpt,
            "sources": [src_a, src_b],
            "top_k": args.top_k,
            "n_images": n_images,
            "n_features_rendered": len(rows),
            "features": rows,
        }, f, indent=2)
    print(f"[atlas] overlap metrics -> {overlap_path}")

    scored = [r["label_jaccard"] for r in rows if r["label_jaccard"] is not None]
    if scored:
        print(f"[atlas] label overlap across {len(scored)} features: "
              f"mean={np.mean(scored):.3f} median={np.median(scored):.3f}")

    # ---- Render ----
    n_pages = (len(rows) + args.features_per_page - 1) // max(args.features_per_page, 1)
    print(f"[atlas] rendering {len(rows)} features -> ~{n_pages} pages")
    if len(rows) > 500:
        print(f"[atlas] NOTE: {len(rows)} features is a large PDF and will take a while. "
              f"Use --max_features / --only_shared to trim, or --sort_by label_jaccard "
              f"(default) so the most convincing features land on the first pages.")

    thumbs = ThumbCache(args.image_dir, args.thumb_px)
    pages: List[Image.Image] = []
    block_buf: List[Image.Image] = []

    for i, r in enumerate(rows, 1):
        fid = str(r["feature_id"])
        block = compose_feature_block(
            r["feature_id"],
            {src_a: top_a[fid], src_b: top_b[fid]},
            (src_a, src_b),
            r,
            thumbs,
            args.thumb_px,
            args.top_k,
        )
        block_buf.append(block)
        if len(block_buf) == args.features_per_page:
            pages.append(stack_blocks(block_buf))
            block_buf = []
        if i % 200 == 0:
            print(f"  composed {i}/{len(rows)} features")
    if block_buf:
        pages.append(stack_blocks(block_buf))

    if thumbs.missing:
        print(f"[atlas] WARNING: {len(thumbs.missing)} stems had no image under "
              f"{args.image_dir} (e.g. {sorted(thumbs.missing)[:3]}). Rendered as placeholders.")

    pdf_path = os.path.join(args.output_dir, args.pdf_name)
    render_pdf(pages, pdf_path)
    print(f"[atlas] saved -> {pdf_path}  ({len(pages)} pages)")


if __name__ == "__main__":
    main()
