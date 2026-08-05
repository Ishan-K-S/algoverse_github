# REPAIR_PLAN.md

> Read-only diagnostic of the Universal SAE (USAE) project, 2026-08-05.
> Supersedes the diagnosis in `PROJECT_STATUS.md` where the two conflict — several premises in
> that document are traced here to bugs rather than to model behavior.
> No code was modified to produce this report.

## 0. TL;DR — what is actually wrong

Ranked by how much each blocks a valid research result:

1. **The "PixArt heatmaps are flat" finding is a visualization bug, not a model finding.**
   `visualize_feature_activations.py:136` hardcodes `t_idx = x.shape[1] - 1` (= 14) while training
   pins `fixed_timestep_idx: 10`. The viz encodes activations the SAE was never trained on.
   `config.yaml:41-43` itself records that t=14 has only 288 active features. Every qualitative
   conclusion in `PROJECT_STATUS.md` that starts from flat heatmaps must be re-derived.
2. **Dead-feature resampling is destroying training every 500 steps** and is stuck in a permanent
   churn cycle (28% of the dictionary still dead at epoch 29 *despite* resampling). Four
   compounding defects, detailed in V2. This is what the wandb loss spikes are.
3. **DinoV2 and PixArt see different pixels.** DinoV2 squashes the whole frame to 224×224
   anisotropically; PixArt resizes the short side to 512 and center-crops. `latent_align_mode:
   per_token` and cross-reconstruction both assert that token (i,j) is the same image location in
   both models. It is not. No config lever fixes this.
4. **PixArt activations describe a null-prompt hallucination, not the input image.** Extraction
   noises the latent to ~99% noise and then *generates* forward under an empty prompt.
5. **PixArt is not actually standardized.** Stats are pooled over all 15 timesteps; training uses
   only t=10. This also decalibrates the "≈1.0 = zero signal" reading that the central finding
   rests on.
6. **No train/val split exists anywhere in the repo.** Every number, heatmap, and the t=10 choice
   itself were measured on training data.

Two pieces of good news, both corrections to `PROJECT_STATUS.md`:

- The attn1/attn2 hook concern is **moot**. The PixArt cache has always held the full block-8
  residual stream — the richest available choice (see V8). `pixart_attn1_sanity_check.py`'s stated
  premise is false.
- The 9 h/run I/O bottleneck diagnosis in `PROJECT_STATUS.md:42-53` is **correct**, and the Phase 1
  fix is right. It is just not the reason cross-recon fails.

---

## 1. Project summary

**Goal.** Learn a single shared sparse dictionary ("universal SAE") whose feature indices mean the
same thing across a non-diffusion vision model (DinoV2) and a diffusion model (PixArt), such that
the approach extrapolates to further models (SigLIP, SD). "Shared language" is defined as *the same
dictionary features firing on the same content across models* — not merely aggregate co-usage.

**Definition of done** (`PROJECT_STATUS.md:8-18`): one diffusion + one non-diffusion model sharing
features with genuine spatial/semantic correspondence, then demonstrated extrapolation.

**Success criteria in use today.** `loss_<A>_to_<B>` standardized MSE (≈1.0 means "predicting the
mean" = no signal), `partition/frac_shared`, `partition/score`, `usage_cosine`, `used_by_none`, plus
qualitative per-patch heatmaps. Section 4 shows most of these are miscalibrated or measure the
wrong thing.

**Operating context.** All GPU work runs in Google Colab under `/content/...`. 2000 COCO val2017
images are cached. Repo: `github.com/Ishan-K-S/algoverse_github`.

---

## 2. Architecture overview

```
UniversalSAE (universal_sae.py)
├── saes["DinoV2"]  : PerModelTopKSAE(in_dim=384,  latent_dim=12288)
└── saes["PixArt"]  : PerModelTopKSAE(in_dim=1152, latent_dim=12288)
```

Each `PerModelTopKSAE` is linear: `b_pre` (per-model learned input mean) → `W_enc` → **hard TopK**
→ `W_dec`. The **latent index space (12288) is shared**; the encoders/decoders are per-model. That
shared index space is the only thing making it "universal" — plus the alignment losses that try to
force the two encoders to agree on which indices fire.

Key facts (all verified):
- **TopK is per-token, per-model, signed, `K=128`** (`universal_sae.py:253-270`). No ReLU. Applied
  on `dim=-1` of `(B, N, 12288)`, so every one of the 256 spatial positions independently gets
  exactly 128 nonzeros. The docstring calls it a "straight-through trick"; it is not — `mask` comes
  from `torch.zeros_like` and carries no grad, so gradient reaches only the selected 128 indices.
  This is standard hard TopK, and it is exactly why `pre_topk_align_weight` exists.
- **Decoder columns are unit-norm constrained** (`universal_sae.py:130-136`), renormalized after
  every optimizer step (`train.py:873-874`). Consequence: `weight_decay: 1e-5` on `W_dec` is a
  **no-op** (renormalization is scale-invariant). The standard companion trick — projecting the
  component of `W_dec.grad` parallel to each column out before the step — is **missing**.
- `use_tide: false`, `cls_pool_mode: none`, so the timestep-embedding path and pooling path are
  both inactive. `attention_component_loss` (`train.py:163-178`) is fully dead code.
- `models.py` is **not** part of the SAE — it is the frozen feature-extractor zoo. Nothing in
  `train.py` imports it.

**Entry point:** `uni_demo.py` (not `train.py`). It reads `/content/algoverse_github/config.yaml`,
builds the dataset/model/optimizer, and calls `train.train_universal_sae` once per epoch.

---

## 3. Pipeline overview

```
COCO val2017 (2000 imgs, seed 42)          coco_dataset_setup.py
   ├── DinoV2   → cache_coco_activations.py           → (257,384) f32  [CLS included on disk]
   └── PixArt   → cache_coco_diffusion_activations.py → (15,1024,1152) f16 + sigmas + timesteps
                        ↓
                combine_cached_acts.py → <stem>_combined.npz
                        ↓
        data.CocoActivationDataset  (strip CLS → standardize)   data.py:395-453
                        ↓
        DataLoader(bs=16, shuffle=True, workers=8)              uni_demo.py:220-227
                        ↓
   per step: pick source (even→DinoV2, odd→PixArt)              train.py:224-228
             slice PixArt timestep 10                           train.py:697-704
             spatial_align 32×32 → 16×16 (avg_pool2d 2×2)       spatial_align.py:170-191
             encode → decode into every target                  train.py:721-777
             latent alignment (per_token + pre-TopK)            train.py:784-861
                        ↓
        checkpoints → weights/{run_name}/usae_epoch_{n}.pth     uni_demo.py:373-403
                        ↓
        eval: run_inference_on_images / visualize_feature_activations /
              dictionary_diagnostic / cross_model_overlap / pixart_timestep_autopsy / ...
```

**The loss, fully expanded for the current config** (`train.py:767-869`):

```
loss = (self_weight·MSE_self + cross_weight·MSE_cross) / (self_weight + cross_weight)
     + latent_align_weight · [ (1 − cos(z, z_tgt)) + pre_topk_align_weight·(1 − cos(z_pre, z_pre_tgt)) ]

with self=1.0, cross=2.0, align=0.5, pre_topk=1.0:
loss = (MSE_self + 2·MSE_cross)/3  +  0.5·[(1 − cos(z,z_tgt)) + (1 − cos(z_pre,z_pre_tgt))]
```

There is **no L1/sparsity penalty** (TopK handles it), **no auxiliary dead-feature loss**, and
**no gradient clipping anywhere in the repo**.

Timing landmarks (2000 imgs / bs 16 = **125 steps/epoch**, 30 epochs = 3750 steps):
- curriculum ends, cross-recon + alignment turn on: **global step 625** (epoch 5)
- resample events: **steps 500, 1000, 1500, 2000, 2500, 3000, 3500**

---

## 4. Issues

Severity: **BLOCKER** (invalidates results) / **HIGH** / **MEDIUM** / **LOW**.
Confidence: High / Medium / Low. Every item below marked VERIFIED was traced through the actual
code path; HYPOTHESES are in §5 and are kept strictly separate.

### VERIFIED ISSUES

---

#### V1 — Visualization encodes timestep 14; training uses timestep 10
**Severity: BLOCKER · Confidence: High**

**Files:** `visualize_feature_activations.py:130-138`, `:463-471`, `:458`, `:72`

```python
# visualize_feature_activations.py:136
t_idx = x.shape[1] - 1  # final timestep, matches convention elsewhere in repo
```

**Root cause.** `pixart_timestep.py` exists specifically to centralize this decision, and its
docstring names this exact failure ("that drift is how we scored a run at t=14 while judging a
config that meant something else"). Every other current eval script calls
`resolve_pixart_timestep(...)` — `dictionary_diagnostic.py:246`, `cross_model_overlap.py:224`,
`input_rank_diagnostic.py:111`, `feature_duplication_diagnostic.py:201`,
`run_inference_on_images.py:111`, `top_activating_images.py:357`.
`visualize_feature_activations.py` **does not import `pixart_timestep` at all**. Its comment
"matches convention elsewhere in repo" is stale and now false.

**Why it matters.** This script produced the flat PixArt heatmaps that motivate the entire central
finding in `PROJECT_STATUS.md:36-39` and the whole Phase 0/1 plan. `config.yaml:41-43` independently
records that at t=14 only 288 features are active with peakiness 1.22 — i.e. flat *by the project's
own measurement*. The heatmaps are flat because they are rendering the wrong timestep.

**Two further defects in the same file:**
- `:463-471` constructs `CocoActivationDataset` with neither `standardization_stats=` nor
  `stats_seed=`. `data.py:195` then does `np.random.default_rng(None)` → **fresh OS entropy every
  invocation**. Two runs on the same image with the same checkpoint produce different heatmaps.
  `data.py:251-255` warns about exactly this: it "changes the SAE's input coordinate system and can
  make a few global features dominate every image" — the reported symptom.
- `:72` `model.load_state_dict(raw["state_dict"], strict=False)` with the return value discarded. A
  name/shape mismatch silently loads a **randomly initialized** encoder, which also presents as flat
  heatmaps. This loader is shared by `pixart_timestep_autopsy.py:137` and
  `pixart_attn1_sanity_check.py:118`.

**What is NOT wrong here** (checked, so the next session doesn't re-litigate): the reshape is
correct. With `spatial_align_to: DinoV2`, PixArt's 1024 tokens are pooled to 256, so
`infer_grid_size` returns 16 and `reshape(16,16)` is right — there is no 1024→32×32 bug. Row-major
ordering matches `spatial_align.py:174`. Pooling is applied in the same order as training. The
per-feature `heat/heat.max()` normalization is honest.

**Expected impact:** high. Fixing this may substantially or entirely dissolve the qualitative half
of the central finding.

---

#### V2 — Dead-feature resampling wrecks training every 500 steps and never converges
**Severity: BLOCKER · Confidence: High**

**Files:** `train.py:409-527` (`resample_dead_features`), `train.py:656-677` (call site),
`train.py:383-406` (`_reset_adam_state_slice`), `config.yaml:91-97`

**This is what the supplied wandb charts show.** I predicted the spike locations from the code
before looking, and they match exactly:

| Prediction from code | Chart |
|---|---|
| Resample at steps 500, 1000, 1500, 2000, 2500, 3000, 3500 | — |
| Cross-recon losses do not exist before step 625 (curriculum) | — |
| ⇒ `loss_DinoV2_to_PixArt` spikes at **1000, 1500, 2000, 2500, 3000, 3500 only — never 500** | ✅ exactly six spikes at 1k/1.5k/2k/2.5k/3k/3.5k, none at 500 |
| Every multiple of 500 is an **even** step ⇒ `_pick_source` always returns DinoV2 ⇒ spikes appear only in `DinoV2_to_*` | ✅ `loss_PixArt_to_DinoV2` is a flat ~1.0 line with **no** spikes |
| 3750 steps = 30 epochs | ✅ x-axis ends at ~3.75k |

Five compounding defects:

**(a) `fixed_timestep_idx` is never passed to the resampler.** `train.py:485-487`:
```python
x_slice, t_slice, _, _, _, _ = _extract_source_slice(
    x, is_diffusion=True, timestep_values_bt=ts_bt,
)   # no fixed_timestep_idx= → falls through to random.randrange(15)
```
The parameter is not in the function signature (`train.py:409-421`) and the call site
(`train.py:664-674`) does not pass it. So **PixArt's revived features are seeded from the residual
of a random one of the 15 timesteps, while training only ever sees t=10.** They are dead on arrival,
get re-flagged dead at the next event, and the cycle repeats forever. This explains
`used_by_none = 3389/12288` at epoch 29 *despite* `resample_dead: true` and seven resample events of
up to 4096 features each — a fact `PROJECT_STATUS.md:34` flags as puzzling. It is a permanent
one-third-of-dictionary churn loop.

**(b) Sampling is ∝ ‖err‖⁴, not ‖err‖², with replacement.** `train.py:500-507`:
```python
recon_err = ((x_hat_flat - x_cmp_flat) ** 2).sum(dim=-1)  # already squared L2
probs = recon_err.double() ** 2                            # → ‖err‖⁴
chosen = torch.multinomial(probs, num_samples=n_dead, replacement=True)
```
The docstring at `train.py:443-444` claims "probability proportional to squared self-reconstruction
error" (Bricken et al.). It is the fourth power. A token with 2× the error gets **16×** the
probability instead of 4×. With `replacement=True` and up to 4096 draws from only
`B·N = 16·256 = 4096` tokens, the revived set collapses onto a handful of worst tokens →
**thousands of near-duplicate decoder columns**. (`feature_duplication_diagnostic.py` exists to
detect exactly this symptom; this is a likely source of it.)

**(c) Revived features have cosine 1.0 with their seed token, so they hijack TopK.** `train.py:511-521`
writes `W_enc[k] = 0.2·ref_norm·d` where `d` is the *unit direction of the seed token itself*. Its
pre-activation on that token is `0.2·ref_norm·‖x−b_pre‖·1.0`. Trained features are specialized and
have small cosine with any given token, so the 0.2 factor does **not** compensate. Revived features
fire immediately at legitimate-coefficient magnitude, in a direction never fit against anything, and
simultaneously **evict** real features from the top-128. Combined with (b), on the worst token
*thousands* of near-identical revived features fill its entire top-128 and destroy its
reconstruction. **This is the spike mechanism.**

**(d) The first post-resample Adam step is ~3.2× the nominal LR.** `train.py:392-394` claims:
> "with exp_avg/exp_avg_sq zeroed the first revived update is a normal ~lr-scaled step, not an explosion."

`step` is not reset (it cannot be, per-row), so bias correction ≈ 1. After the next backward,
`m = 0.1g`, `v = 0.001g²`, giving `lr·0.1g/√(0.001g²) ≈ 3.16·lr`. Not an explosion, but 3× and
undocumented.

**(e) Structural source bias.** Because resampling triggers on `global_step % 500 == 0` and
`_pick_source` alternates on `global_step % 2`, **every resample event lands on a DinoV2-source
step**. Nobody intended this. Whatever asymmetry it introduces is systematic across all 30 epochs.

Supporting conditions: `resample_max_per_event: 4096` is **one third of `latent_dim: 12288`** in a
single step; **there is no gradient clipping** anywhere in the repo; and usage EMAs are not reset
for revived features, so they are miscounted as `used_by_none` for ~40 steps after each event.

**Also:** `resample_start_step: 200` is a **no-op** — 200 is not a multiple of `resample_interval:
500`, so the first eligible step is 500 regardless.

**Why it matters.** The model is knocked off its optimum seven times per run and spends the
following ~100 steps recovering. One third of the dictionary is in a churn loop that can never
converge. The `used_by_none` metric that motivated turning resampling on is being *caused* by the
resampler's own bug.

---

#### V3 — DinoV2 and PixArt are looking at different pixels
**Severity: BLOCKER · Confidence: High**

**Files:** `models.py:86-89` vs `DiffusionActivationExtractor.py:642-647`

```python
# DinoV2 — models.py:86-89
transforms.Resize((224, 224), ...)      # TUPLE → anisotropic squash, whole frame, no crop

# PixArt — DiffusionActivationExtractor.py:643-644
transforms.Resize(512, ...)             # INT → short side to 512, aspect preserved
transforms.CenterCrop(512)              # sides cropped away
```

**Root cause.** Two independently written preprocessing pipelines that were never unified.

**Why it matters.** For a typical 640×480 COCO image, PixArt's grid covers only the center 480×480
(25% of the width is discarded) while DinoV2's grid covers the full 640×480, non-uniformly stretched.
`spatial_align` then pools both to 16×16 and `latent_align_mode: per_token` (`config.yaml:79`) plus
cross-reconstruction **assert that token (i,j) is the same image location in both models**. It is
not, and the offset varies per image with the aspect ratio.

This is a direct, sufficient explanation for `loss_PixArt_to_DinoV2 ≈ 0.948`: the per-token cross
map the model is asked to learn genuinely does not exist. Unlike the timestep issue, **no config
lever fixes this** — the preprocessing must be unified and PixArt re-cached.

---

#### V4 — PixArt activations describe a null-prompt hallucination, not the input image
**Severity: BLOCKER · Confidence: High**

**Files:** `DiffusionActivationExtractor.py:844-885`, `:680-743`, `:755-762`

```python
# :845-848
noise = torch.randn_like(clean_latents)
initial_t = int(timesteps[0])
alpha = self.scheduler.alphas_cumprod[initial_t].item()
noisy_latents = (alpha ** 0.5) * clean_latents + ((1 - alpha) ** 0.5) * noise
```

I computed `sqrt(ᾱ)` at the first scheduled timestep for both plausible beta schedules:

| cache idx | t | signal fraction (linear) | (scaled_linear) |
|---|---|---|---|
| **0** | 999 | **0.006** | **0.027** |
| 5 | 664 | 0.106 | 0.309 |
| **10** | 329 | **0.572** | **0.826** |
| 14 | 61 | 0.978 | 0.994 |

The starting latent is **0.6–2.7% real image**. The loop then *generates forward* from there under
an empty prompt (`_encode_null_prompt`, `:680-743`), with no CFG and no conditional pass. By cache
index 10 the latent is a null-prompt DDIM sample seeded by ~1–3% of the true image — only weakly
anchored to the input.

This is very different from the DIFT-style protocol the timestep choice is implicitly imitating
(re-noise the *clean* latent directly to a moderate t, take a **single** forward pass). The
DIFT protocol is also ~15× cheaper.

**Timestep semantics, confirmed** (this is easy to get backwards, so it is stated explicitly):
`DDIMScheduler.timesteps` is **descending** and activations are appended in loop order
(`:877-885`), so **cache index 0 = the noisiest step; index 14 = the cleanest.** Index 10 ≈ t=264-329
is low-noise, which lands near the DIFT sweet spot — so the *direction* of the t=10 pick is
reasonable even though the evidence for the exact value is weak (see H1).

**Three further extraction defects in the same file:**
- **`:755-762`** — micro-conditioning is fed **latent** dims, not pixel dims:
  ```python
  B, C, H, W = latents.shape              # H = W = 64 (latent!)
  "resolution": torch.tensor([H, W], ...) # → [64, 64]
  ```
  `PixArt-XL-2-1024-MS`'s `AdaLayerNormSingle` expects pixel resolution (1024/512). The model is
  told the image is 64×64 — out of distribution for every AdaLN modulation in every block.
- **`:685-691`, `:725`, `:771-776`** — no `encoder_attention_mask`. T5 is padded to `max_length=256`
  and cross-attention in blocks 0–7 attends over **255 pad embeddings**.
- **`:845`** — `torch.randn_like` is **unseeded**. Each image's trajectory is driven by a different
  noise draw (an image-independent nuisance variable baked into every cached activation), and
  re-caching will never reproduce.

Also: 512×512 input into a **1024-trained** multi-scale checkpoint (`:633` vs `:642-647`).

---

#### V5 — PixArt is not actually standardized; the "1.0 = zero signal" calibration is wrong
**Severity: HIGH · Confidence: High**

**Files:** `data.py:46-56`, `data.py:182-245`, `data.py:424-430`

```python
# data.py:50-55
if act.dim() == 3:
    t, n, d = act.shape
    return act.reshape(t * n, d)     # (15, 1024, 1152) → (15360, 1152)
```

Stats are per-channel `(D,)` mean/std pooled over **all 15 noise levels** — from index 0 (≈pure
noise) to index 14 (≈clean). Training then consumes **only** `x[:, 10]`, which contributes 1/15 of
the mass. **After "standardization", the t=10 slice is neither zero-mean nor unit-variance.**

**Why it matters.** Two independent consequences:
1. The headline metric is decalibrated. `loss_PixArt_to_DinoV2 = 0.948` is read as "≈1.0 =
   predicting the mean = zero signal" (`PROJECT_STATUS.md:33`). That reading assumes unit-variance
   targets. The DinoV2 target *is* unit-variance; the PixArt target is not. The number the whole
   central finding rests on is measured against the wrong denominator.
2. The SAE's TopK threshold and dead-feature EMA operate on a shifted, mis-scaled input
   distribution for PixArt only — which biases every partition metric and the resampler's dead set.

Related: `data.py:242` clamps `std` to ≥1e-5 and then `data.py:430` adds `1e-5` **again**. Anything
inverting standardization as `x·std + mean` will be slightly off, and
`_validate_standardization_stats` (`data.py:275`) uses a third convention (`clamp_min` without the
additive term).

---

#### V6 — Spatial pooling is applied after standardization, so the two MSE terms are on different scales
**Severity: HIGH · Confidence: High**

**Files:** `data.py:427-430` (standardize) then `train.py:705-706, 749-750, 804-805` (align)

Averaging four unit-variance tokens yields variance `(1+3ρ)/4`. For a realistic DiT neighbour
correlation ρ≈0.6–0.8 that is ~0.7–0.85; for ρ=0 it is 0.25. Either way, **post-alignment PixArt is
no longer unit-variance while DinoV2 still is**, and `mse_flat` (`train.py:21-30`, a plain
per-element mean) sums the two targets with no per-model rescaling.

The learned `b_pre` absorbs the mean shift; **nothing absorbs the variance shift**. So
`loss_DinoV2_to_PixArt` has a systematically lower floor than `loss_PixArt_to_DinoV2`, the two are
not comparable to each other, and neither is comparable to 1.0. This compounds V5.

Secondary: the model never sees or reconstructs native-resolution PixArt — `x_hat` for
`target="PixArt"` is `(B, 256, 1152)`, a decoder of 2×2-averaged features. Any downstream tool
assuming a 32×32 PixArt grid is looking at something else. (`sae_mask.py` does exactly this — V16.)

**What is NOT wrong:** the pooling arithmetic itself is correct. `spatial_align.py:173-191` reshapes
row-major, permutes to NCHW, `avg_pool2d(kernel=2, stride=2)`, and inverts. Both DinoV2's ViT
patches and PixArt's `PatchEmbed` emit row-major `(h·W + w)` order. **No row-major/column-major bug.**

---

#### V7 — No train/val split exists anywhere in the repo
**Severity: HIGH · Confidence: High**

**Files:** `uni_demo.py:203-227`, all diagnostics

There is no `random_split`, no `Subset`, no `SubsetRandomSampler`, and no evaluation loop in
`train.py`. One `CocoActivationDataset` over all 2000 stems, one shuffled `DataLoader`. Every
diagnostic points at the same `/content/combined_cache`. The `viz: { set: val }` block
(`config.yaml:67-70`) is **dead config** — nothing reads it.

So all 2000 images are used for training *and* for every reported metric, every heatmap, every
top-activating-image, and **the `fixed_timestep_idx: 10` selection itself** (that decision was made
by evaluating a trained checkpoint on its own training data).

Note `coco_dataset_setup.py` samples from `val2017` while `cache_coco_activations.py:150-153`
references `train2017` — a held-out set is trivially available and simply unused.

**Why it matters.** Nothing currently produced by this repo is a generalization measurement. For an
interpretability result this is less fatal than for a predictive one, but the *timestep choice* and
any claim of "shared language" need held-out evidence.

---

#### V8 — The attn1/attn2 hook change never took effect for PixArt (corrects `PROJECT_STATUS.md`)
**Severity: HIGH (as a correction) · Confidence: High**

**Files:** `DiffusionActivationExtractor.py:794` and `:870-871` vs `:191`; commits `f73c170`, `7bba701`

Commit `f73c170` ("Hooked Pixart on attn2 instead of attn1") introduced `HOOK_ATTN_NAME` and changed
the **base class** (`:191`) to `getattr(self._get_last_block(), self.HOOK_ATTN_NAME)`. But
`PixArtActivationExtractor.extract_activations` is a **full override**, and its hook line is:

```python
# DiffusionActivationExtractor.py:870-871
last_block = self._get_last_block()          # the BLOCK, not an attention submodule
hook_handle = last_block.register_forward_hook(hook_fn)
```

I checked this line across the full history (`git show <c>:DiffusionActivationExtractor.py`):

| commit | base class (line ~185-191) | **PixArt override (line ~847-870)** |
|---|---|---|
| `7bba701` | `._get_last_block().attn2` | `self._get_last_block()` |
| `f73c170` | `getattr(..., HOOK_ATTN_NAME)` | `self._get_last_block()` |
| `HEAD` | `getattr(..., HOOK_ATTN_NAME)` | `self._get_last_block()` |

**The PixArt path has hooked the whole block since the initial commit. It was never `.attn2`.**

**Consequences:**
1. `HOOK_ATTN_NAME = "attn1"` (`:794`) and its 5-line justifying comment are **dead code for PixArt**.
2. The cache holds the **full post-block residual stream of block 8** (post-attn1 + post-attn2 +
   post-FF, residuals included) — which is *better* than either attention output.
3. **Good news:** the "null-prompt attn2 output is content-free by construction" concern does **not**
   apply to this cache. That rules out the most alarming possible explanation for the cross-recon
   failure.
4. **Bad news:** `pixart_attn1_sanity_check.py:3-9` states as its premise "the existing cache was
   built with the old attn2 hook". **That premise is false**, so any conclusion drawn from that
   script is void. (It also cannot run at all — see V16.)

Related, lower severity: the hook depth is `HOOK_DEPTH_FRAC = 8/27` → **block 8 of 28 (~32% depth)**,
and the method is still named `_get_last_block`. Aligning an early-layer diffusion representation
against DinoV2's *final*-layer patch tokens is an asymmetry worth naming, though not obviously wrong.

---

#### V9 — `cross_weight: 2.0` does not increase cross signal; it cuts self-recon gradient to 1/3
**Severity: HIGH · Confidence: High**

**Files:** `train.py:863-866`

```python
sae_loss = reconstruction_loss / reconstruction_weight_total
```

The reconstruction term is a **weighted average**, not a weighted sum, so `self_weight` and
`cross_weight` only ever matter as a *ratio* — their absolute magnitudes are normalized away.
Raising `cross_weight` 1.0 → 2.0 does not add cross gradient; it moves the split from (½, ½) to
(⅓, ⅔) and **shrinks the self-recon gradient by 33%**.

**Why it matters.** `config.yaml:75-76` reasons about this lever as "back to the old 2:1 cross/self
ratio … cross-recon is the whole point". The ratio reasoning is right; the implied magnitude is not.
Anyone tuning `cross_weight` expecting more cross pressure is actually only trading away self-recon.
This is listed in `PROJECT_STATUS.md:73-74` as a candidate lever, so it will be reached for.

---

#### V10 — The curriculum boundary is a simultaneous three-way discontinuity
**Severity: MEDIUM · Confidence: High**

**Files:** `train.py:644`, `:722-723`, `:786`

`in_curriculum = epoch < curriculum_epochs` is computed once per epoch. At **epoch 5 / global step
625**, three things change in a single step:
1. self-recon gradient is instantly scaled by **1/3** (V9),
2. `MSE_cross` enters at 2/3 weight starting from ≈1.0 (untrained cross-decode) → adds ~0.67,
3. both cosine terms start near 1.0 → add ~0.5·(1+1) = ~1.0.

`train/total_loss` should jump from ≈`MSE_self` (~0.2) to ≈1.73 — an ~8× step — and *self*-recon
quality typically regresses afterward. This is by construction, not a bug, but there is no LR
re-warmup and `model._train_loss_ema` (`train.py:643`) is **not reset** at the boundary, so
`total_loss_ema` renders it as a smooth ramp rather than a step and hides it.

Structural point worth flagging: during epochs 0–4, `_pick_source` still alternates, so every other
step trains *only* DinoV2 and every other *only* PixArt. **There is zero coupling between the two
dictionaries during curriculum.** At epoch 5 the alignment loss is suddenly asked to reconcile two
independently-converged dictionaries whose index assignments are arbitrary relative to each other.
The `config.yaml:11-15` rationale for adding the curriculum (prevent early collapse onto a shared
near-constant clique) is sound, but this is its cost and it is not currently acknowledged.

Also asymmetric gating: cross-recon is gated on `in_curriculum AND curriculum_self_only`; alignment
only on `not in_curriculum`. Setting `curriculum_self_only: false` would enable cross-recon during
curriculum but still not alignment.

---

#### V11 — wandb metrics are largely uninformative or miscalibrated
**Severity: MEDIUM · Confidence: High**

**Files:** `train.py:887-966`

- **`train/latent_sparsity`** (`:917`) is `(z == 0).float().mean()`. TopK guarantees exactly 128
  nonzeros per token always, so this is the **constant 0.98958** for the entire run. Zero information.
- **`partition/score`** (`:950-955`) is `max_excl / max(used_by_all, 1)`. When `used_by_all == 0` —
  total partitioning, **the exact failure the metric exists to detect** — the denominator becomes 1
  and the metric silently changes units from a ratio to a raw count (jumping to e.g. 6000 rather than
  reporting undefined). It is discontinuous precisely at the point of interest. It is also `max` over
  models rather than symmetric, so it describes only the worse-partitioned model.
- **`partition/usage_cosine`** (`:960-966`) is a cosine between two **elementwise non-negative**
  firing-rate vectors that both sum to exactly 128. It is bounded in [0,1], cannot express
  anti-correlation, reaches 0 only for literally disjoint supports, and is dominated by the largest
  entries — a handful of near-always-on features present in both models pulls it to ≈1 regardless of
  the other 12000. **It saturates high and will look healthy even when `frac_shared` is poor.** The
  docstring's "0 = orthogonal (disjoint feature sets)" oversells it. `PROJECT_STATUS.md:29` reports
  0.857 as "looks fine" — that reading is not supported. Cosine of *mean-centered* usage, or Jaccard
  of the thresholded sets, would be the informative version.
- The dead threshold `1e-3` is a **hardcoded literal** at `:933` while the resampler takes it from
  config at `:461`. They coincide today; changing `resample_dead_threshold` silently decouples the
  diagnostic from what the resampler actually does.
- `train/latent_align_loss` (`:902`) **includes the pre-TopK term and is not multiplied by
  `latent_align_weight`** — its real contribution to the objective is 0.5× the logged number.
- `train/source_model` (`:903`) logs a **string**; wandb stores it as a non-numeric column that will
  not render on a line chart.
- `per_target_losses` keys (`:912-914`) exist only for the source picked that step, so each
  `train/loss_X_to_Y` series is present on roughly half the logged steps — the sparse, gappy series
  visible in the supplied charts.
- **wandb step-collision risk:** `train.py:968` logs with an explicit `step=global_step_actual`,
  while `uni_demo.py:368-369` logs *without* a step (auto-incrementing). If
  `len(dataloader) ≡ 1 (mod 17)` the first train log of the next epoch collides and wandb drops it.
  With `len(dataloader) = 125`: `125 mod 17 = 6`, so **not currently triggered** — but it is latent.

---

#### V12 — Eval scripts diverge from the training config, silently
**Severity: MEDIUM–HIGH · Confidence: High**

`uni_demo.py:373-403` persists everything needed (`config`, `standardization_stats`,
`model_tokens_native`, `spatial_align_to`, `manifest`). Most scripts do not read it back.

| Script | `top_k` | timestep | std. stats | aligner |
|---|---|---|---|---|
| `run_inference_on_images.py` | ckpt ✅ | ckpt ✅ | ckpt ✅ | ckpt ✅ |
| `feature_duplication_diagnostic.py` | ckpt ✅ | ckpt ✅ | ckpt ✅ | ckpt ✅ |
| `cross_model_overlap.py` | **yaml ❌** | ckpt ✅ | ckpt ✅ | ckpt ✅ |
| `dictionary_diagnostic.py` | **yaml ❌** | ckpt ✅ | ckpt ✅ | ckpt ✅ |
| `top_activating_images.py` | **yaml ❌** | ckpt ✅ | ckpt ✅ | **yaml ❌** |
| `dictionary_diagnostic_all_timesteps.py` | **yaml ❌** | n/a | **none, seed=None ❌** | ckpt ✅ |
| `input_rank_diagnostic.py` | n/a | ckpt ✅ | **none, seed=None ❌** | ckpt ✅ |
| `visualize_feature_activations.py` | **yaml ❌** | **hardcoded T−1 ❌❌** | **none, seed=None ❌** | **yaml ❌** |
| `pixart_timestep_autopsy.py` | **yaml ❌** | sweeps | **always None ❌** | **always yaml ❌** |
| `sae_mask.py` | **yaml ❌** | ckpt ✅ | **never applied ❌❌** | **never applied ❌❌** |

**The `_training_global` / `_standardization_stats` trap.** Only
`top_activating_images.load_universal_sae:316-319` sets these attributes.
`visualize_feature_activations.load_checkpoint:54-77` — the loader used by
`pixart_timestep_autopsy.py:137` **and** `pixart_attn1_sanity_check.py:118` — does not. So:
- `pixart_timestep_autopsy.py:139` `eval_g = getattr(model, "_training_global", g)` → **always** the
  live `config.yaml`.
- `pixart_timestep_autopsy.py:150` `standardization_stats=getattr(model, "_standardization_stats",
  None)` → **always `None`** → recomputed. (It does pass `stats_seed=0`, so it is at least
  deterministic, and given an unchanged cache it may coincidentally match — but it never reads the
  persisted stats and never verifies.)

**`top_k` matters a lot here**: it directly sets the nonzero count per token, so it redefines every
firing statistic. `top_k` moved 512 → 128 in a recent commit, and these scripts read live
`config.yaml`, not the checkpoint that produced the numbers.

Two scripts also rank features by **different statistics** while claiming to match:
`run_inference_on_images.py:119` uses `z.abs().mean(...)`, `visualize_feature_activations.py:144`
and `inference.py:111` use `amax`. They will report different "top features" for the same image and
checkpoint.

---

#### V13 — 15× read amplification is the 9 h/run bottleneck (confirms `PROJECT_STATUS.md`)
**Severity: HIGH (velocity) · Confidence: High**

**Files:** `data.py:292-317`, `data.py:395-453`

`__getitem__` loads the full `(15, 1024, 1152)` PixArt tensor; only later, on GPU, does
`_pick_diffusion_slice` take `x[:, 10]`. **14/15 of every read is discarded.** Three additional
amplifiers:
- **`mmap_mode="r"` is silently ignored.** `data.py:292, :302, :315` pass it to `np.load` on a
  **.npz zip archive**. NumPy does not forward `mmap_mode` to `NpzFile`, and `savez_compressed`
  output cannot be memmapped anyway. **Every access is a full zlib inflate.** The code reads as if
  it were lazy; it is not.
- `data.py:317` `torch.from_numpy(npz[key].copy())` — `npz[key]` already returns a fresh decompressed
  array; `.copy()` doubles peak memory and adds a full memcpy per image.
- `_compute_standardization_stats` re-reads **1000 whole combined files** at every startup through
  the same slow path.
- DinoV2-source steps still pay the full PixArt decompress, since both live in one combined npz.

`PROJECT_STATUS.md:42-53`'s diagnosis is correct and the Phase 1 fix (slim single-timestep cache) is
the right call. Note the cache is likely **float16**, not float32 —
`cache_coco_diffusion_activations.py:46-47` only upcasts bfloat16, and the extractor is constructed
with the default `dtype=torch.float16` — so ~35 MB/image, not the 71 MB quoted at
`PROJECT_STATUS.md:47`. (The `create_PixArt_extractor` convenience helper uses bfloat16 and *would*
produce 71 MB — the two paths disagree. Worth confirming against the actual cache.)

---

#### V14 — Checkpoint hygiene
**Severity: MEDIUM · Confidence: High**

- `train.py:970-978` runs `torch.save(model, save_model_path)` at the end of **every epoch** to the
  hardcoded default `./models/universal_sae_final.pt` (`train.py:18`). `uni_demo.py` never overrides
  it, so **every run, regardless of `run_tag`, overwrites the same file 30 times**. It is a
  whole-object pickle (requires `universal_sae.UniversalSAE` at the same module path to load) with
  **no config, no epoch, no standardization stats**. `run_inference_on_images.py:144` assumes a dict
  checkpoint and would `TypeError` on it.
- The good checkpoints (`uni_demo.py:373-403`, `weights/{run_name}/usae_epoch_{n}.pth`) do **not**
  persist: `_usage_ema_*` (plain attributes, not registered buffers → absent from `state_dict`),
  `_train_loss_ema`, **LR scheduler state**, or RNG state. On resume the LR restarts at `base_lr` and
  the usage EMAs restart from scratch (so `resample_dead_features` returns 0 until they repopulate —
  handled gracefully by the guard at `train.py:457`, but the partition diagnostics are blank).
- `standardization_stats` is written via `getattr(dataset, "standardization_stats", {})`, so it
  silently serializes `{}` if `standardize: false`; `_validate_standardization_stats` would then
  raise a bare `KeyError` rather than a clear message.

---

#### V15 — LR schedule is split across two files with an undeclared shared key
**Severity: LOW–MEDIUM · Confidence: High**

Warmup lives in `train.py:218-222, 649-651` (linear, 50 steps); cosine decay lives in
`uni_demo.py:303-310, 358-360` (`CosineAnnealingLR`, `T_max=30`, stepped once per epoch). Both write
`param_group["initial_lr"]` — `_set_optimizer_warmup_lr` via `setdefault`, `CosineAnnealingLR` as its
base-LR store. They agree today (both 5e-4). `CosineAnnealingLR.step()` uses the **recursive** update
(reads and rewrites `group['lr']`), so it composes multiplicatively with the manual warmup writes.
This works only because warmup (50 steps) finishes inside epoch 0 (125 steps). If
`warmup_steps > len(dataloader)` it would silently compound.

Also `train.py:604-639` re-iterates the **entire DataLoader** once per epoch just to peek at one
batch for a warning, tearing down and respawning 8 workers each time.

---

#### V16 — Broken and stale scripts
**Severity: MEDIUM · Confidence: High**

- 🔴 **`pixart_attn1_sanity_check.py` cannot run at all.** `:118` loads via
  `visualize_feature_activations.load_checkpoint`, then `:121-126` raises if
  `getattr(model, "_standardization_stats", None)` is None. That loader **never sets the attribute**,
  so it raises on **every** checkpoint including a freshly-trained one that does contain the stats
  under `ckpt["standardization_stats"]`. The error message ("Retrain or migrate the checkpoint
  first") actively misdirects. Its premise is also false (V8), and it carries a stale comment at
  `:254-257` referring to a `ref_thresh` bug in `pixart_timestep_autopsy.py` that no longer exists.
- 🔴 **`top_activating_images.py main()` raises `TypeError` with default args.** `:26`
  `CHECKPOINT_PATH = None`, `:605` `default=CHECKPOINT_PATH`, `:651-654` `os.path.isfile(checkpoint)`
  → `os.stat(None)` → `TypeError`, which `genericpath.isfile` does not catch (it catches only
  `OSError, ValueError`). The documented auto-discovery path crashes. `sae_mask.py:393` and
  `run_inference_on_images.py:142` both guard correctly; this one does not.
- 🔴 **`sae_mask.py` is functionally wrong.** `:400-409` calls `load_activation_for_image` with no
  `spatial_aligner=` and no standardization step (standardization lives in the *caller* in
  `top_activating_images.py:469-479`, not inside that helper). It feeds **raw, unstandardized,
  unaligned** PixArt activations into an SAE trained on standardized 16×16-pooled ones, then
  `infer_grid_size(act.shape[0])` yields 32 and every masked-patch index is in the wrong coordinate
  system.
- 🟡 **`input_rank_diagnostic.py`**: `ZeroDivisionError` at `:162` (`pr_d/pr_p`) when
  `participation_ratio` is 0.0 — i.e. on **total collapse, the exact case the script exists to
  detect**. Dead `"X_per_feature_std" if False else "rank_90"` at `:173`, whose branch *prints*
  "comparable rank to DinoV2" while never referencing DinoV2. `torch.load` at `:78` without
  `weights_only=False` (torch ≥2.6 flipped the default). Holds every image in `bufs` before
  `torch.cat` at `:131`, and the `max_samples` downsample at `:138-141` happens **after**, so it does
  not help peak RSS (~3 GB at `--n_images 2000`). Stale advice at `:168-169` recommends random
  timesteps — the opposite of the current decision.
- 🟡 **`dictionary_diagnostic_all_timesteps.py`**: docstring `:5-8` says it is "the correct
  measurement when the model was trained with random PixArt timestep sampling" — obsolete under
  `fixed_timestep_idx: 10`. Dead recompute at `:173`. No stats, no seed.
- 🟡 **`inference.py` is dead code** except `print_top_features`. `:68` still hardcodes the last
  timestep; `:109-111` docstring says "average" while the code is `amax`.
- 🟡 **Metric definitions are inconsistent across the three places they exist.** "Model uses feature
  k" means: usage-EMA rate > 1e-3 (`train.py:930-955`), *ever fired on any token of any image*
  (`dictionary_diagnostic.py:267-309`), or *in the per-image top-64 by `amax`*
  (`cross_model_overlap.py:145-161`). Comparing wandb's `partition/score` to the diagnostic's
  `partition_score` is comparing incomparable quantities. The `dictionary_diagnostic` criterion is
  also near-vacuous: with `top_k=128` × 256 tokens × 2000 images, a feature need be selected **once
  in 65M slots** to count as "used", which drives the script into its "the dictionary is shared, your
  scoring metric is the problem" branch almost by construction.

---

### 5. HYPOTHESES (plausible, NOT confirmed)

Kept strictly separate from §4. None of these were traced to a definitive conclusion.

---

#### H1 — The `t=10` choice is not well-supported by the evidence that produced it
**Confidence: Medium-High on the critique · the *direction* of the choice is probably still right**

`pixart_timestep_autopsy.py` measures "peakiness" = mean over the top-8 features of
`max_token|z| / mean_token|z|` (`:90-113`). Three concerns, of which the first two are verified
properties of the code and the third is the actual hypothesis:

1. **(verified) The metric has no spatial content.** `peak` is invariant to any permutation of the
   token axis. A feature firing on one isolated random patch scores maximally; a feature cleanly
   covering a contiguous 8×8 object scores *worse* than one firing on a single patch of sky. It is a
   token-sparsity statistic, not a localization statistic.
2. **(verified) The baseline and the winner are ranked by different statistics.** `:169` computes the
   DinoV2 reference with `by="mean"` (which deliberately selects the features firing on the *most*
   tokens — structurally the *least* peaky), while the winning timestep is selected at `:213` on
   `peak_loc` from `by="localized"` (the *most* peaky). The verdict at `:228-246` then compares
   `best_loc` against a threshold derived from `ref_peak`. That compares the bottom of one
   distribution to the top of another.
3. **(verified) The `WRONG-TIMESTEP` verdict is the unconditional fall-through** (`:243-246`) and its
   text asserts "while others are flat" — a cross-timestep contrast the code never computes. `bt` is
   `argmax`, so **some** timestep always wins. There is no null distribution, no dispersion measure,
   no significance test, and n=1 image per invocation.

**The mathematical concern (this is the hypothesis).** Because TopK puts exactly `top_k` nonzeros in
every token, `peak_k ≥ N / n_k` where `n_k` is the number of tokens feature k fires on, so
`Σ_k 1/peak_k ≤ top_k` identically. Peakiness and "number of active features" are therefore two
views of one conserved quantity — "more features active" mechanically *forces* "higher average
peakiness". Sweeping t sweeps input scale relative to the learned `b_pre` (amplified by V5's
cross-timestep standardization), which changes how many distinct features get selected, which moves
peakiness. **My hypothesis is that the sweep is measuring how widely TopK spreads its fixed budget at
each noise level, not whether the timestep carries semantics.** I did not re-run the autopsy to
confirm.

*One thing I want to explicitly retract before it propagates:* an intermediate analysis claimed the
reported "3617 active, peakiness 6.09" (`config.yaml:41-43`) is arithmetically impossible under
`top_k=128`. That argument only holds if 6.09 came from the `by="localized"` column. The script
prints **both** columns (`:204`) and the quoted DinoV2 reference (1.4) is the `by="mean"` one, so
6.09 is most likely also mean-ranked — in which case it is entirely plausible. **The numbers are not
impossible.** The ranking-mismatch concern (2) stands on its own.

Compounding, and verified: the autopsy ran on the epoch-29 checkpoint whose PixArt encoder was
trained with `fixed_timestep_idx: -1` (t=14), so the sweep pushed 14 out-of-distribution noise levels
through an encoder fitted to one of them. That t=14 scored lowest is arguably *backwards* — a
self-consistency artifact, since `b_pre` was fit to t=14 so those inputs sit closest to the mean and
produce the flattest code.

**Mitigating:** t=10 corresponds to t≈264–329, essentially the DIFT sweet spot (t≈261). The
*direction* is well-motivated by external literature even though this script's evidence for the
specific value over t=9 or t=11 is weak. **Recommendation: keep t=10 as the working default, and do
not spend a cycle re-deriving it until V1–V6 are fixed** — it will need re-deriving anyway once the
inputs change.

---

#### H2 — Duplicate decoder features are caused by the resampler
**Confidence: Medium**

`feature_duplication_diagnostic.py` was added to chase an observation (its docstring `:12-13`) that
three different COCO images returned an identical top-8 feature list. V2(b) — `‖err‖⁴` sampling with
`replacement=True`, up to 4096 draws from 4096 tokens — is a mechanism that would produce exactly
that. **Unconfirmed:** I did not run the diagnostic. It is cheap to test (see Fix 2 verification).

---

#### H3 — The hooked block (8 of 28) is too early to align against DinoV2's final layer
**Confidence: Low-Medium**

Block 8 is ~32% depth. DinoV2's cached tokens are `x_norm_patchtokens` — the **final** layer. Early
DiT blocks carry lower-level features. Plausible contributor to cross-recon failure, but entirely
untested, and it should not be touched until V3/V4 are fixed (a depth sweep on a broken input
pipeline would be uninterpretable).

---

## 6. Ordered repair plan

Dependency-aware. **Do not reorder** — later fixes are unverifiable until earlier ones land.
"Tiny-scale" is specified concretely per step; nothing here trains to convergence.

Before starting, record a baseline so every subsequent claim is comparable: current
`weights/{run_name}/usae_epoch_29.pth`, its `manifest.code_version`, and the wandb run id.

---

### Stage 0 — Restore trust in the measurement instruments (no training, ~1 hour)

Everything downstream is judged by these tools. Fix them before changing anything that affects
results, or you will not be able to tell what your changes did.

**Fix 0.1 — `visualize_feature_activations.py` timestep + stats** *(addresses V1)*
- Rationale: this alone may dissolve the "flat PixArt heatmaps" finding.
- Files: `visualize_feature_activations.py`
- Changes: import `pixart_timestep.resolve_pixart_timestep`; replace `:136`'s
  `t_idx = x.shape[1] - 1` with a resolved index (pass the raw ckpt dict through); pass
  `standardization_stats=ckpt["standardization_stats"]` and `stats_seed` at `:463-471`; build the
  aligner from `ckpt["spatial_align_to"]` not live `g` at `:458`; read `top_k` from the checkpoint's
  own `sae_params` at `:68`; capture and assert on the `load_state_dict` return at `:72`.
- Risks: low. Read-only script.
- Dependencies: none.
- **Verification (tiny-scale):** render the same 3 stems (including `000000562818` from
  `PROJECT_STATUS.md:92`) **twice in the same session** and once at the old t=14.
  Pass = (a) the two t=10 renders are **byte-identical** (proves the stats nondeterminism is gone),
  (b) the t=10 PixArt heatmaps show visibly higher spatial contrast than the t=14 ones. Also print
  `n_active` and max/mean peakiness per render as a number, so this is not purely an eyeball call.
  **This is the single highest-information experiment in the plan. Do it first.**

**Fix 0.2 — Make every eval script read the checkpoint, not live `config.yaml`** *(addresses V12)*
- Files: `dictionary_diagnostic.py`, `cross_model_overlap.py`, `top_activating_images.py`,
  `dictionary_diagnostic_all_timesteps.py`, `input_rank_diagnostic.py`, `pixart_timestep_autopsy.py`
- Changes: the cleanest fix is to make `visualize_feature_activations.load_checkpoint` set
  `model._standardization_stats` and `model._training_global` the way
  `top_activating_images.load_universal_sae:316-319` already does — that repairs
  `pixart_timestep_autopsy.py` and `pixart_attn1_sanity_check.py` for free. Then route `top_k`
  through the checkpoint's `sae_params` everywhere (`feature_duplication_diagnostic.py:165` is the
  correct pattern to copy), and pass `standardization_stats` + `stats_seed` in the three scripts that
  omit them.
- Risks: low.
- **Verification:** run `dictionary_diagnostic.py` on the existing epoch-29 checkpoint twice; assert
  identical output. Assert the loaded `top_k` printed at startup equals the checkpoint's
  `config.sae_params.top_k`, not `config.yaml`'s.

**Fix 0.3 — Repair or quarantine the broken scripts** *(addresses V16)*
- Files: `pixart_attn1_sanity_check.py` (fixed for free by 0.2 — but **update its false premise
  docstring per V8, or delete the script**), `top_activating_images.py:651-654` (guard
  `checkpoint is None`), `sae_mask.py:400-409` (add standardization + aligner, or mark it broken),
  `input_rank_diagnostic.py:162` (guard the division), `:173` (delete the `if False`), `:78`
  (`weights_only=False`).
- Risks: low.
- **Verification:** each script runs to completion with default args on 8 images.

**Fix 0.4 — Fix the wandb metrics** *(addresses V11)*
- Files: `train.py:917, 930-966`
- Changes: delete `latent_sparsity` (constant); make `partition/score` return `inf`/`nan` rather than
  silently switching units when `used_by_all == 0`; add mean-centered usage cosine **and** Jaccard of
  the thresholded used-sets alongside the existing cosine; source the `1e-3` threshold at `:933` from
  `resample_dead_threshold`; log `latent_align_loss` multiplied by `latent_align_weight` (or log both
  and label them); drop the string-valued `train/source_model` or encode it as 0/1.
- Risks: changes chart semantics — old and new runs will not be directly comparable on
  `partition/score`. Note this explicitly in the run tag.
- **Verification:** 20 steps with `use_wandb: true`, confirm every key is finite and numeric.

---

### Stage 1 — Stop training from destroying itself (small training runs)

**Fix 1.1 — Repair dead-feature resampling** *(addresses V2)*
- Rationale: removes the loss spikes and the permanent churn cycle. Blocks any clean read of a
  training curve.
- Files: `train.py:409-527`, `train.py:656-677`, `train.py:383-406`, `config.yaml:91-97`
- Changes, in order of importance:
  1. **Add `fixed_timestep_idx` to the `resample_dead_features` signature and pass it at both the
     call site (`:664-674`) and the `_extract_source_slice` call (`:485-487`).** Without this nothing
     else matters — revived PixArt features stay dead on arrival.
  2. `probs = recon_err.double()` (drop the extra `** 2`) — `recon_err` is already squared L2. Fix
     the docstring at `:443-444` either way.
  3. Use `replacement=False` when `n_dead <= recon_err.numel()`, to stop the duplicate-direction
     collapse.
  4. Lower `resample_max_per_event` from 4096 to ~256–512 (2–4% of the dictionary, not 33%).
  5. Reset the usage EMA to the dead threshold for revived indices so they are not miscounted as
     `used_by_none` for the following ~40 steps.
  6. Reduce `resample_enc_scale` (0.2 → ~0.05) **or** scale by the *achieved* pre-activation on the
     seed token rather than by the alive-encoder-norm — the current formula does not account for the
     cosine-1.0 seeding.
  7. Add `torch.nn.utils.clip_grad_norm_` (start at 1.0) in the training step.
  8. Set `resample_start_step` to a multiple of `resample_interval` (or drop it — it is a no-op).
  9. Fix the `_reset_adam_state_slice` docstring at `:392-394` (~3.2×lr, not ~lr).
- Risks: (3) changes behavior if `n_dead > n_tokens` — keep the `replacement=True` fallback. (4)
  slows dead-feature recovery; watch `used_by_none` over the run. (7) may mask a real instability —
  log the pre-clip grad norm too.
- Dependencies: Fix 0.4 (so the metrics you judge it by are trustworthy).
- **Verification (tiny-scale):** run **2 epochs at N=64 images, batch 8** (16 steps/epoch) with
  `resample_interval=8`, `curriculum_epochs=0`, `use_wandb=false`. Pass criteria:
  (a) `total_loss` at the step **immediately after** each resample event is within **1.2×** the step
  before (today it is 3–5×);
  (b) `used_by_none` is **monotonically non-increasing** across events (today it plateaus — that is
  the churn signature);
  (c) all losses finite;
  (d) the printed `[resample] revived N` count **decreases** across successive events.
  Total runtime with the slim cache from Fix 2.1: a few minutes.

**Fix 1.2 — Make loss weights mean what the config says** *(addresses V9, V10)*
- Files: `train.py:863-869`, `config.yaml:74-77`, `train.py:643-644`
- Changes: either drop the `/ reconstruction_weight_total` normalization (making the weights truly
  absolute) **or** keep it and document in `config.yaml` that these are ratio-only. Recommend
  **keeping the normalization and fixing the comments** — it keeps the loss scale stable across the
  curriculum boundary, which is the more valuable property. Separately, reset
  `model._train_loss_ema` at the curriculum boundary so the epoch-5 discontinuity is visible rather
  than smoothed away, and consider a short LR re-warmup (~50 steps) at that boundary.
- Risks: changing the normalization changes the effective LR — do not combine with any other change.
- Dependencies: Fix 1.1.
- **Verification (tiny-scale):** N=64, batch 8, `curriculum_epochs=1`, 3 epochs. Confirm the
  `total_loss` step at the boundary matches the analytic prediction in V10 (≈8×) to within ~20%, and
  that the EMA now shows a step rather than a ramp. **Change exactly one lever per run**, as
  `PROJECT_STATUS.md:73` already insists.

---

### Stage 2 — Fix the data so cross-model learning is possible at all

This is the stage that actually addresses `loss_PixArt_to_DinoV2 ≈ 0.948`. Stages 0–1 make the
instruments trustworthy; **this** is where the research problem lives. It requires re-caching.

**Fix 2.1 — Unify preprocessing, re-cache PixArt with a DIFT-style single-timestep protocol**
*(addresses V3, V4, V5, V13 — and delivers `PROJECT_STATUS.md` Phase 1)*
- Rationale: one re-cache fixes the pixel-correspondence bug, the null-prompt-hallucination bug, the
  standardization bug, **and** the 9 h → ~40 min I/O bottleneck. These are not separable — they all
  require rewriting the cache, so do them in one pass.
- Files: `DiffusionActivationExtractor.py:642-647, 755-762, 823-885`, `models.py:86-89`,
  `cache_coco_diffusion_activations.py`, `data.py:46-56`
- Changes:
  1. **Unify the crop.** Make DinoV2 and PixArt cover *identical* pixels. Simplest correct option:
     `Resize(short_side)` + `CenterCrop(square)` for **both** (DinoV2 → 224, PixArt → 512). This
     changes DinoV2's cache too, so **both must be re-cached.** Do not skip this — per-token
     alignment is meaningless without it.
  2. **Replace generate-from-noise with DIFT-style single-step extraction:** noise the *clean* latent
     directly to the chosen t (≈264–329), take **one** forward pass, capture the hook. Cache shape
     becomes `(1024, 1152)` — **~2.3 MB/image instead of ~35 MB**, and extraction gets ~15× faster.
  3. Fix the micro-conditioning at `:755-762` to pass **pixel** dims, not latent dims.
  4. Pass `encoder_attention_mask` for the T5 padding.
  5. **Seed the noise per image** (derive from the stem) so the cache is reproducible.
  6. Standardization then computes over the single cached timestep automatically (V5 dissolves).
  7. Set `use_tide: false` (already set) and remove the now-dead 15-timestep handling paths, or leave
     them behind a flag.
- Risks: **highest-risk step in the plan.** Invalidates the existing cache and every existing
  checkpoint. Do it on a branch. Keep the old cache until Stage 2 verification passes.
- Dependencies: Stage 0 (you need working viz to judge the result), Stage 1 (you need a training loop
  that does not self-destruct).
- **Verification (tiny-scale, before re-caching all 2000):**
  1. **Pixel-correspondence check, no model:** for 4 images, render the DinoV2 16×16 grid and the
     PixArt 32×32 grid as overlays on the source image. Confirm by eye that cell (i,j) covers the
     same region. This catches the V3 bug directly and costs minutes.
  2. **Re-cache N=16 images only.** Confirm shapes, dtype, and that the seeded noise reproduces
     byte-identical activations across two runs.
  3. **Representation-quality check, no SAE:** compute the cosine-similarity matrix between PixArt
     patch tokens within one image. If the new extraction is better anchored, neighbouring patches
     should be more similar than distant ones and object regions should be visibly blocky. Compare
     against the same statistic on the old cache. This is the cheapest possible test of "does this
     representation carry spatial content at all" and it is **independent of the SAE**.
  4. Only then re-cache all 2000.

**Fix 2.2 — Fix standardization/pooling ordering** *(addresses V6)*
- Files: `data.py:427-430`, `train.py:705-706` (and the parallel align call sites)
- Changes: apply spatial alignment **before** standardization, so the stats are computed on the
  tensors the SAE actually consumes and both models' inputs are genuinely unit-variance. (Note:
  per-channel affine and token-average-pooling commute exactly, so the *mean* is unaffected — it is
  the **variance** that is currently wrong. `top_activating_images.py:377-381, :479` already uses the
  align-then-standardize order.)
- Risks: changes the loss scale; `loss_*` numbers become incomparable to all prior runs. Say so in
  the run tag.
- Dependencies: Fix 2.1.
- **Verification (tiny-scale):** load 8 batches, print per-model post-align `mean` and `std` over the
  channel axis. Pass = both models within `[0.95, 1.05]` std and `|mean| < 0.05`. **This is a
  numerical assertion, not a judgment call** — and it makes the "≈1.0 = predicting the mean" reading
  of the loss finally valid.

**Fix 2.3 — Create a real held-out split** *(addresses V7)*
- Files: `coco_dataset_setup.py`, `uni_demo.py:203-227`, `data.py`
- Changes: split the 2000 stems 80/20 by a fixed seed written to disk next to
  `selected_images.txt`; compute standardization stats on **train only**; add an eval pass in
  `uni_demo.py` that logs `val/loss_*` and the partition metrics; point every diagnostic at the val
  split by default. `val2017` was used for training, so `train2017` is available for an additional
  fully-independent set later.
- Risks: low, but it will make the headline numbers look worse — that is the point.
- Dependencies: Fix 2.1 (split the new cache, not the old one).
- **Verification:** assert zero stem overlap between the two splits; assert the persisted stats hash
  matches a train-only recomputation.

---

### Stage 3 — Re-measure, then decide (this is `PROJECT_STATUS.md` Phase 1's real decision point)

**Fix 3.1 — Re-run the diagnosis on repaired inputs**
- Dependencies: all of Stages 0–2.
- **This is where the "keep vs. swap PixArt" decision actually gets made**, on trustworthy numbers.
  With the slim cache a full 30-epoch run should be ~40 min, so this is affordable.
- Report: `val/loss_PixArt_to_DinoV2` (the number to beat is 0.948, now on a correctly calibrated
  scale), `used_by_none` trajectory, and re-rendered PixArt vs DinoV2 heatmaps at t=10.
- **Only after this** is it worth revisiting H1 (re-derive the best timestep — now cheap, since a
  slim re-cache per timestep is minutes) or H3 (hook depth sweep).

**Fix 3.2 — Add the metric that measures the actual goal** *(`PROJECT_STATUS.md` Phase 2)*
- The existing metrics measure aggregate co-usage; the stated goal is per-token correspondence.
  Add **cross-model per-token co-fire on the same image** and/or **heatmap IoU between DinoV2 and
  PixArt for shared features**, computed on the **val** split.
- Note this is only meaningful *after* Fix 2.1 — per-token co-fire between grids covering different
  pixels is not interpretable.
- Unify the three conflicting "shared feature" definitions (V16, last bullet) into one helper that
  `train.py`, `dictionary_diagnostic.py`, and `cross_model_overlap.py` all import.

---

## 7. Notes for the next session

- **Do Fix 0.1 first and report the result before doing anything else.** It is ~30 minutes, requires
  no training, and its outcome determines how much of `PROJECT_STATUS.md`'s central finding survives.
- **`PROJECT_STATUS.md` needs updating** once Stage 0 lands: the flat-heatmap premise (V1), the
  attn1/attn2 premise (V8), the `used_by_none` puzzle (V2a), and the `usage_cosine = 0.857 "looks
  fine"` reading (V11) are all now known to be measurement artifacts rather than model behavior.
- **Change one lever per run and state it explicitly**, as `PROJECT_STATUS.md:73` already requires.
  Several fixes here change the loss scale (1.2, 2.2), so runs across those boundaries are not
  comparable — encode the fix stage in `run_tag`.
- The `cosine_reconstruction_loss` path (`train.py:33-42`), `attention_component_loss`
  (`train.py:163-178`), and most of `inference.py` are dead code. Deleting them would reduce the
  surface area for exactly the kind of stale-path drift catalogued in V12/V16, but it is not on the
  critical path.
- Two facts worth not re-deriving: the `spatial_align.py` pooling arithmetic is **correct** (V6), and
  the heatmap reshape in the viz is **correct** (V1). Both look suspicious and both check out.
