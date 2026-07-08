"""Dense per-pixel semantic segmentation for images.

This is the counterpart to the activation-masking probe: that script *consumes*
a semantic mask (every pixel already labeled with a class id) and aggregates it
onto the patch grid before ablating activations. This script *produces* that
mask by running a real segmentation model over the raw image, so every pixel
gets a class label -- not just patches inside a box or COCO instance polygon.

Output per image (into --output_dir):
  <stem>_labelmap.png  - single-channel PNG, pixel value = class id. This is
                          exactly what patch_mask_probe.py's --semantic_mask
                          flag expects.
  <stem>_legend.json    - id -> class name, plus per-class pixel coverage, so
                          you know which ids to pass to --mask_labels.
  <stem>_overlay.png     - optional color-coded visualization (--overlay).

Example:
    python semantic_segment.py --image photo.jpg --backend deeplabv3
    python semantic_segment.py --image_dir photos/ --backend segformer \
        --segformer_checkpoint nvidia/segformer-b0-finetuned-ade-512-512 \
        --overlay

Then feed the result into the probe:
    python patch_mask_probe.py --source <source> --stem photo \
        --semantic_mask semantic_masks/photo_labelmap.png --mask_labels 15
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


LabelMap = np.ndarray  # HxW array of integer class ids


def load_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def stem_for(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class SegmentationBackend:
    """Common interface: segment(image) -> (label_map, id_to_name)."""

    def segment(self, image: Image.Image) -> Tuple[LabelMap, Dict[int, str]]:
        raise NotImplementedError


class DeepLabV3Backend(SegmentationBackend):
    """torchvision DeepLabV3, 21 PASCAL VOC classes (incl. background)."""

    def __init__(self, device: str, variant: str = "resnet101"):
        try:
            from torchvision.models.segmentation import (
                DeepLabV3_ResNet50_Weights,
                DeepLabV3_ResNet101_Weights,
                deeplabv3_resnet50,
                deeplabv3_resnet101,
            )
        except ImportError as e:
            raise ImportError(
                "The deeplabv3 backend requires torchvision. "
                "Install it with `pip install torchvision`."
            ) from e

        if variant == "resnet50":
            weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            model = deeplabv3_resnet50(weights=weights)
        elif variant == "resnet101":
            weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
            model = deeplabv3_resnet101(weights=weights)
        else:
            raise ValueError(f"Unknown deeplabv3 variant: {variant!r}")

        self.device = device
        self.model = model.to(device).eval()
        self.preprocess = weights.transforms()
        # Class names come straight from the weights' metadata, not
        # hardcoded, so this stays correct across torchvision versions.
        self.categories = list(weights.meta["categories"])

    @torch.no_grad()
    def segment(self, image: Image.Image) -> Tuple[LabelMap, Dict[int, str]]:
        width, height = image.size
        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        logits = self.model(batch)["out"]  # [1, C, h', w'], h'/w' may differ
        # Interpolate in logit space (not on argmax'd ids) then take argmax,
        # so upsampling doesn't invent nonsense boundary labels.
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        label_map = logits.argmax(dim=1).squeeze(0).to(torch.uint8).cpu().numpy()
        id_to_name = {i: name for i, name in enumerate(self.categories)}
        return label_map, id_to_name


class SegformerBackend(SegmentationBackend):
    """HuggingFace SegFormer, typically fine-tuned on ADE20K (150 classes)."""

    def __init__(self, device: str, checkpoint: str):
        try:
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        except ImportError as e:
            raise ImportError(
                "The segformer backend requires `transformers`. "
                "Install it with `pip install transformers`."
            ) from e

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint).to(device).eval()
        self.id_to_name = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def segment(self, image: Image.Image) -> Tuple[LabelMap, Dict[int, str]]:
        width, height = image.size
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # post_process_semantic_segmentation does the correct nearest-style
        # upsample back to the original resolution internally.
        [seg] = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(height, width)]
        )
        dtype = np.uint8 if len(self.id_to_name) <= 255 else np.uint16
        label_map = seg.cpu().numpy().astype(dtype)
        return label_map, self.id_to_name


def build_backend(args) -> SegmentationBackend:
    if args.backend == "deeplabv3":
        return DeepLabV3Backend(args.device, variant=args.deeplabv3_variant)
    if args.backend == "segformer":
        return SegformerBackend(args.device, checkpoint=args.segformer_checkpoint)
    raise ValueError(f"Unknown backend: {args.backend!r}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def palette_for(num_classes: int) -> np.ndarray:
    """Deterministic but arbitrary RGB palette, one row per class id.

    Not any dataset's "official" palette -- just stable, visually distinct
    colors for eyeballing an overlay. Class id 0 is forced to black.
    """
    rng = np.random.RandomState(0)
    colors = rng.randint(0, 256, size=(max(num_classes, 1), 3)).astype(np.uint8)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)
    return colors


def make_overlay(image: Image.Image, label_map: LabelMap, alpha: float = 0.5) -> Image.Image:
    num_classes = int(label_map.max()) + 1
    colors = palette_for(num_classes)
    color_map = colors[label_map]  # HxWx3
    base = np.asarray(image).astype(np.float32)
    blended = base * (1 - alpha) + color_map.astype(np.float32) * alpha
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def summarize(label_map: LabelMap, id_to_name: Dict[int, str]) -> List[dict]:
    ids, counts = np.unique(label_map, return_counts=True)
    total = label_map.size
    rows = [
        {
            "id": int(i),
            "name": id_to_name.get(int(i), f"class_{i}"),
            "pixels": int(c),
            "fraction": float(c) / total,
        }
        for i, c in zip(ids, counts)
    ]
    rows.sort(key=lambda r: r["fraction"], reverse=True)
    return rows


def save_label_map(label_map: LabelMap, path: str) -> None:
    if label_map.dtype == np.uint8:
        Image.fromarray(label_map, mode="L").save(path)
    else:
        # Only reached with >255 classes (e.g. very large taxonomies).
        Image.fromarray(label_map.astype(np.int32), mode="I").save(path)


def process_one(
    backend: SegmentationBackend,
    image_path: str,
    output_dir: str,
    stem: str,
    save_overlay: bool,
) -> Tuple[str, str]:
    image = load_image(image_path)
    label_map, id_to_name = backend.segment(image)

    os.makedirs(output_dir, exist_ok=True)
    mask_path = os.path.join(output_dir, f"{stem}_labelmap.png")
    save_label_map(label_map, mask_path)

    summary = summarize(label_map, id_to_name)
    legend_path = os.path.join(output_dir, f"{stem}_legend.json")
    with open(legend_path, "w") as f:
        json.dump(
            {
                "image": image_path,
                "width": image.size[0],
                "height": image.size[1],
                "classes_present": summary,
                "id_to_name": {str(k): v for k, v in id_to_name.items()},
            },
            f,
            indent=2,
        )

    overlay_path = None
    if save_overlay:
        overlay_path = os.path.join(output_dir, f"{stem}_overlay.png")
        make_overlay(image, label_map).save(overlay_path)

    print(f"[segment] {image_path}")
    print(f"  labelmap -> {mask_path}")
    print(f"  legend   -> {legend_path}")
    if overlay_path:
        print(f"  overlay  -> {overlay_path}")
    for row in summary[:8]:
        print(f"    id {row['id']:>3}  {row['name']:<20} {row['fraction'] * 100:5.1f}%")

    return mask_path, legend_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", default=None, help="Path to a single image.")
    group.add_argument("--image_dir", default=None, help="Directory of images to segment.")
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.png",
        help="Comma-separated extensions to include when using --image_dir.",
    )
    parser.add_argument("--output_dir", default="semantic_masks")
    parser.add_argument("--backend", default="deeplabv3", choices=("deeplabv3", "segformer"))
    parser.add_argument("--deeplabv3_variant", default="resnet101", choices=("resnet50", "resnet101"))
    parser.add_argument(
        "--segformer_checkpoint",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="HF hub checkpoint id for the segformer backend.",
    )
    parser.add_argument("--overlay", action="store_true", help="Also save a color-coded overlay PNG.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    backend = build_backend(args)

    if args.image:
        process_one(backend, args.image, args.output_dir, stem_for(args.image), args.overlay)
        return

    exts = tuple(e.strip().lower() for e in args.extensions.split(",") if e.strip())
    paths = sorted(
        os.path.join(args.image_dir, name)
        for name in os.listdir(args.image_dir)
        if name.lower().endswith(exts)
    )
    if not paths:
        raise ValueError(f"No images with extensions {exts} found in {args.image_dir!r}.")

    for path in paths:
        process_one(backend, path, args.output_dir, stem_for(path), args.overlay)


if __name__ == "__main__":
    main()
