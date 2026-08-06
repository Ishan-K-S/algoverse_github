# PROJECT_STATUS.md

> Working status + investigation log for the Universal SAE (USAE) project.
> Read this alongside `CLAUDE.md` (which covers architecture/how-to). This file
> is the "where are we and why" that `CLAUDE.md` doesn't capture.
> **Last updated: 2026-08-05.**
>
> ⚠️ **Read `REPAIR_PLAN.md` first.** A read-only diagnostic on 2026-08-05 traced
> several premises below to *bugs in the measurement tools*, not to model behavior.
> Where the two documents conflict, `REPAIR_PLAN.md` wins. `RECOVERY_LOG.md` records
> what has been repaired since. Corrections are marked inline below rather than
> deleted — the observations were real, the *attributions* were wrong.

## Goal (definition of done)

A shared *language* of features across **one non-diffusion model (DinoV2, preferred)**
and **one diffusion model**, that **extrapolates to further models** (SigLIP, Stable
Diffusion, etc.). "Shared language" means the same dictionary features fire on the
same *content* across models — not merely that features co-fire in aggregate usage.


**Scope decisions:** PixArt is **not** required — it can be swapped for a lighter
diffusion model. We need exactly one diffusion + one non-diffusion to start, then
add models to demonstrate extrapolation.

## The central finding (2026-07-13) — **substantially retracted 2026-08-05**

**Aggregate partition metrics look healthy, but they measure the wrong thing.**
From the epoch-29 run (`ex16_bs16_topk512_LR0.0005_alignDinoV2_30ep`):

| Metric | Value | Reading (2026-07-13) | Status after `REPAIR_PLAN.md` |
|---|---|---|---|
| `partition/score` | 0.18 | looks great (>1.0 = failing) | ⚠️ discontinuous at `used_by_all == 0`, the exact failure it exists to detect (V11, fixed) |
| `partition/frac_shared` | 0.54 | looks fine | ⚠️ measured on training data (V7) |
| `usage_cosine_DinoV2_vs_PixArt` | 0.857 | looks fine | ❌ **reading unsupported.** Cosine of two non-negative rate vectors saturates high whenever a few near-always-on features are shared; it cannot express anti-correlation (V11). Use the new mean-centered cosine / Jaccard instead |
| `loss_PixArt_to_PixArt` | 0.199 | PixArt self-recon is excellent | ⚠️ denominator is wrong — PixArt was never actually unit-variance (V5, V6) |
| `loss_DinoV2_to_DinoV2` | 0.285 | fine | ⚠️ training data (V7) |
| `loss_DinoV2_to_PixArt` | 0.533 | mediocre | ⚠️ not comparable to the row below (different variance scale, V6) |
| `loss_PixArt_to_DinoV2` | **0.948** | ~1.0 = zero cross signal | ⚠️ **the "≈1.0 = predicting the mean" calibration assumed unit-variance targets, which PixArt's were not** (V5, V6). Directionally still bad; the exact number is not trustworthy |
| `used_by_none` | 3389 / 12288 | ~28% dead *despite* `resample_dead: true` | ❌ **not a puzzle — the resampler caused it.** It seeded revived PixArt features from a random timestep while training only ever saw t=10, so they were dead on arrival every cycle (V2a, fixed) |

**Qualitative signal that started this — ❌ this was a visualization bug.**
The flat PixArt heatmaps came from `visualize_feature_activations.py` hardcoding
`t_idx = x.shape[1] - 1` (timestep **14**) while training pinned `fixed_timestep_idx: 10`.
The script was rendering activations the SAE was never trained on, and `config.yaml:41-43`
independently records that t=14 has only 288 active features — flat by the project's own
measurement. The script also recomputed standardization stats from unseeded random samples,
so two renders of the same image could differ. Both are fixed (V1 / Fix 0.1).

**What survives:** the *concern* is still right — aggregate co-usage is not the goal, and
per-token correspondence was never measured. That gap is now closed by
`val/cofire_jaccard_*` + its chance baseline (Fix 3.2). **What does not survive:** every
specific number above as a statement about model behavior, and the flat-heatmap evidence
entirely. **All of it must be re-derived** after Stage 2's re-cache, on the held-out split
that now exists (Fix 2.3). Until then this project has no valid measurement of whether the
shared dictionary works.

## The iteration bottleneck (I/O, not compute)

Training is **~9 h/run** (~9.9 s/it for a *linear* SAE, 125 it/epoch, 30 epochs,
batch 16, 2000 images). That is **data-loading bound**, not compute:

**This diagnosis is confirmed correct** (`REPAIR_PLAN.md` V13) — one of the few premises
in this document that survived review intact.

- PixArt was cached with `num_inference_steps=15`
  (`cache_coco_diffusion_activations.py`), so each PixArt entry is `(15, 1024, 1152)`
  ≈ **35 MB** (float16 — the 71 MB figure originally quoted here assumed float32; the
  `create_PixArt_extractor` helper uses bfloat16 and *would* give 71 MB, but the path
  actually used constructs the extractor with the default `dtype=torch.float16`).
  DinoV2 is `(256, 384)` ≈ 0.4 MB (negligible).
- Training uses only **one timestep per step** → we load 15× the data we use.
- Three amplifiers found on top of that (V13): `mmap_mode="r"` is **silently ignored** on
  `.npz` archives, so every access is a full zlib inflate; `npz[key].copy()` adds a redundant
  memcpy; and `_compute_standardization_stats` re-reads 1000 whole files at every startup.
  The first two are still unfixed.

A faster CPU / new laptop does **not** fix this (bottleneck is data volume; Colab
already provides a CUDA GPU the tiny SAE barely needs). The fix is a data-design change.

## Plan (phased, evidence-gated) — superseded by `REPAIR_PLAN.md` §6

The phased plan below is kept for continuity, but the ordered repair plan in
`REPAIR_PLAN.md` §6 is the live sequence. Current state (2026-08-05):

- **All Stage 0–3 *code* changes are written and committed** on branch `code-fixes`
  (Fixes 0.1–0.4, 1.1, 1.2, 2.1, 2.2, 2.3, 3.2 — see `RECOVERY_LOG.md`).
- **None of them are verified against real data.** Every fix was tested against
  synthetic caches on CPU. No GPU/Colab run has happened.
- **Nothing has been re-measured.** `loss_PixArt_to_DinoV2 ≈ 0.948` is exactly where it
  was, because the fixes that would change it (Stage 2's re-cache) have not been run.

**Do these next, in order:**
1. **Fix 0.1's real render** — `visualize_feature_activations.py` against the real
   epoch-29 checkpoint, same 3 stems (incl. `000000562818`), at t=10 vs t=14, printing
   `n_active` and peakiness. ~30 min, no training. `REPAIR_PLAN.md` calls this "the single
   highest-information experiment in the plan"; it decides how much of the retracted
   central finding above is recoverable. **Still not done.**
2. **Fix 1.1's real tiny run** — 2 epochs, N=64, batch 8, `curriculum_epochs=0`. Note the
   toy-scale verification of this fix is confounded (see the CORRECTION in
   `RECOVERY_LOG.md` Fix 1.1); this run is what actually validates it. Use a
   `resample_interval` above the ~60-step EMA recovery window, or measure `used_by_none`
   immediately *before* each event, or the same confound recurs.
3. **Read `train/grad_norm_preclip`** on that run before trusting `grad_clip_norm: 1.0` —
   it is the plan's starting value, not a measured one.
4. **Stage 2 re-cache** (16 images first, per `REPAIR_PLAN.md` Fix 2.1's own ordering),
   then the full 2000. This invalidates every existing checkpoint — expected.
5. **Then** Fix 3.1: re-measure everything on the val split and make the keep-vs-swap
   PixArt call on trustworthy numbers.

<details><summary>Original phased plan (2026-07-15)</summary>

- **Phase 0 — diagnose (no training).** `pixart_timestep_autopsy.py` sweeps all 15
  PixArt timesteps for one image, measures feature *localization* (peakiness =
  max-token/mean-token |z|) vs. a DinoV2 baseline. ⚠️ `REPAIR_PLAN.md` H1 shows this
  metric has no spatial content (it is permutation-invariant over tokens), ranks its
  baseline and its winner by *different* statistics, and its `WRONG-TIMESTEP` verdict is
  an unconditional fall-through. Keep t=10 as the working default (it lands near the DIFT
  sweet spot for independent reasons) and do not re-derive it until Stage 2 lands.
- **Phase 1 — collapse + slim.** ✅ Implemented as Fix 2.1 (DIFT-style single-timestep
  extraction, cache `(1, 1024, 1152)`), **not yet run**. Expect 9 h → ~40 min.
- **Phase 2 — measure the right thing.** ✅ Implemented as Fix 3.2
  (`val/cofire_jaccard_*` + chance baseline + lift, on the held-out split).
- **Phase 3 — extrapolation = done.** Unstarted. Note `models.py`'s other five encoders
  (SigLIP, CLIP, ViT, ResNet, ConvNeXt) still carry the anisotropic-squash preprocessing
  bug V3 flagged; only `DinoV2` was fixed. Fix them before adding a second non-diffusion
  model.

</details>

**Candidate config levers (change ONE at a time, and state it explicitly):**
`latent_align_weight` 0.5 → 1.5–2.0; `use_tide` true → false; collapse to 1 timestep.
⚠️ `cross_weight` is **not** a usable lever in the way it reads: the reconstruction term is
a weighted *average*, so only the self:cross ratio matters and raising `cross_weight` just
trades self-recon away (V9). Several fixes also change the loss scale (1.2, 2.2), so runs
across those boundaries are not comparable — encode the fix stage in `run_tag`.

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
# RUN THIS NEXT (REPAIR_PLAN.md Fix 0.1 verification) — ~30 min, no training.
# The script now resolves the timestep from the checkpoint, so this renders at the
# t=10 the model was actually trained on. Then force the old t=14 for comparison.
python visualize_feature_activations.py --stem 000000562818 <two more stems> \
    --raw_image_dir /content/coco_data/val2017
python visualize_feature_activations.py --stem 000000562818 <two more stems> \
    --raw_image_dir /content/coco_data/val2017 --pixart_timestep 14
```

Pass criteria: (a) two t=10 renders in the same session are **byte-identical** (proves the
stats nondeterminism is gone), (b) the t=10 PixArt heatmaps show visibly higher spatial
contrast than the t=14 ones. Report `n_active` and max/mean peakiness per render as numbers,
not an eyeball call. **This decides how much of the retracted central finding is recoverable.**

`pixart_timestep_autopsy.py` is deliberately *not* the next step — see H1 in
`REPAIR_PLAN.md` for why its peakiness metric can't support the conclusion it draws.

## Operating context

- All training/GPU work runs in **Google Colab** (`/content/...`: cache
  `/content/combined_cache`, raw images `/content/coco_data/val2017`, weights
  `/content/algoverse_github/weights`, config auto-discovers newest `.pth`).
- Repo: `github.com/Ishan-K-S/algoverse_github`. 2000 COCO images cached.
- Claude works from a code mirror without a GPU/cache: writes scripts + patches;
  the user runs them in Colab and pastes raw output. Tight loop, decisive changes.
