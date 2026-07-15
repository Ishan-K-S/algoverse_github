# PROJECT_STATUS.md

> Working status + investigation log for the Universal SAE (USAE) project.
> Read this alongside `CLAUDE.md` (which covers architecture/how-to). This file
> is the "where are we and why" that `CLAUDE.md` doesn't capture.
> **Last updated: 2026-07-15.**

## Goal (definition of done)

A shared *language* of features across **one non-diffusion model (DinoV2, preferred)**
and **one diffusion model**, that **extrapolates to further models** (SigLIP, Stable
Diffusion, etc.). "Shared language" means the same dictionary features fire on the
same *content* across models — not merely that features co-fire in aggregate usage.


**Scope decisions:** PixArt is **not** required — it can be swapped for a lighter
diffusion model. We need exactly one diffusion + one non-diffusion to start, then
add models to demonstrate extrapolation.

## The central finding (2026-07-13)

**Aggregate partition metrics look healthy, but they measure the wrong thing.**
From the epoch-29 run (`ex16_bs16_topk512_LR0.0005_alignDinoV2_30ep`):

| Metric | Value | Reading |
|---|---|---|
| `partition/score` | 0.18 | looks great (>1.0 = failing) |
| `partition/frac_shared` | 0.54 | looks fine |
| `usage_cosine_DinoV2_vs_PixArt` | 0.857 | looks fine |
| `loss_PixArt_to_PixArt` | 0.199 | **PixArt self-recon is excellent** |
| `loss_DinoV2_to_DinoV2` | 0.285 | fine |
| `loss_DinoV2_to_PixArt` | 0.533 | mediocre |
| `loss_PixArt_to_DinoV2` | **0.948** | **~1.0 = zero cross signal** (standardized MSE) |
| `used_by_none` | 3389 / 12288 | **~28% of dictionary is dead** despite `resample_dead: true` |

**Qualitative signal that started this:** PixArt per-patch heatmaps are **uniform/flat**,
while DinoV2 heatmaps **localize cleanly on objects**. So PixArt latents reconstruct
PixArt well but do **not** live in the shared subspace that decodes to DinoV2. The
"shared language" is currently superficial (co-usage, not spatial/semantic correspondence).
The team had been implicitly optimizing the aggregate metrics, which gave false comfort.

## The iteration bottleneck (I/O, not compute)

Training is **~9 h/run** (~9.9 s/it for a *linear* SAE, 125 it/epoch, 30 epochs,
batch 16, 2000 images). That is **data-loading bound**, not compute:

- PixArt was cached with `num_inference_steps=15`
  (`cache_coco_diffusion_activations.py`), so each PixArt entry is `(15, 1024, 1152)`
  ≈ **71 MB**; batch 16 ≈ **1.1 GB/step** decompressed. DinoV2 is `(256, 384)` ≈ 0.4 MB (negligible).
- Training uses only **one timestep per step** → we load 15× the data we use.

A faster CPU / new laptop does **not** fix this (bottleneck is data volume; Colab
already provides a CUDA GPU the tiny SAE barely needs). The fix is a data-design change.

## Plan (phased, evidence-gated)

- **Phase 0 — diagnose (no training).** `pixart_timestep_autopsy.py` sweeps all 15
  PixArt timesteps for one image, measures feature *localization* (peakiness =
  max-token/mean-token |z|) vs. a DinoV2 baseline, and classifies the uniform heatmap as
  **wrong-timestep** vs. **feature-selection** vs. **real-encoding-failure**. It also
  picks the single best timestep. **← current step; run this next.**
- **Phase 1 — collapse + slim.** Reduce PixArt to the one best timestep, write a slim
  (optionally pre-standardized) cache, set `use_tide: false`. Expect **9 h → ~40 min**.
  Then measure `PixArt→DinoV2` cross-recon + re-render heatmaps to decide **keep vs.
  swap PixArt** (candidate lighter model: SD-Turbo / SD 1.5 — few-step = smaller cache,
  and in the target family).
- **Phase 2 — measure the right thing.** Add a wandb metric for the real goal:
  cross-model **per-token co-fire** on the same image (and/or heatmap IoU between
  DinoV2 and the diffusion model for shared features).
- **Phase 3 — extrapolation = done.** Add SigLIP (cheap, non-diffusion) + the target
  diffusion model; show shared features transfer.

**Candidate config levers (change ONE at a time, and state it explicitly):**
`latent_align_weight` 0.5 → 1.5–2.0; `use_tide` true → false; collapse to 1 timestep.

## Tooling added this cycle

- `pixart_timestep_autopsy.py` — Phase 0 diagnostic (above).
- `run_inference_on_images.py` — image → top SAE features for a list of stems; has
  `--list_stems`, auto-loads newest checkpoint from `weights/`.
- `visualize_feature_activations.py` — per-patch feature heatmaps overlaid on the image;
  multi-stem, `--quiet`, optional semantic-class breakdown.
- `semantic_mask.py` — **note: this is now a segmentation-mask *generator*** (DeepLabV3/
  SegFormer) that writes `<stem>_labelmap.png` + `<stem>_legend.json`, consumed by
  `visualize_feature_activations.py --semantic_mask_dir`. (The old activation-ablation
  probe with `--stem/--source/--box` is `sae_mask.py`.)

## How to run (Colab paths)

```bash
# Phase 0 — run this next:
python pixart_timestep_autopsy.py --stem 000000562818 \
    --raw_image_dir /content/coco_data/val2017 --output_dir /content/pixart_autopsy
```

Report back the printed per-timestep table + the `[verdict]` block, and eyeball
`{stem}_DinoV2_reference.png` vs. the best PixArt timestep grid. That decides Phase 1.

## Operating context

- All training/GPU work runs in **Google Colab** (`/content/...`: cache
  `/content/combined_cache`, raw images `/content/coco_data/val2017`, weights
  `/content/algoverse_github/weights`, config auto-discovers newest `.pth`).
- Repo: `github.com/Ishan-K-S/algoverse_github`. 2000 COCO images cached.
- Claude works from a code mirror without a GPU/cache: writes scripts + patches;
  the user runs them in Colab and pastes raw output. Tight loop, decisive changes.
