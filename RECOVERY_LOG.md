# RECOVERY_LOG.md

Append-only log of fixes applied per `REPAIR_PLAN.md`. Each entry: issue, root cause, files
modified, summary, why it was necessary, verification performed, and remaining uncertainty.

---

## Fix 0.1 — `visualize_feature_activations.py` timestep + stats + state_dict repair

**Addresses:** V1 (BLOCKER)
**Commit:** `6c1a2ee`
**Files modified:** `visualize_feature_activations.py`

### Issue

The visualization script hardcoded `t_idx = x.shape[1] - 1` (the *last* cached PixArt
timestep, index 14) to slice which diffusion timestep's activations to render, while
training pins `fixed_timestep_idx: 10` in `config.yaml`. Every rendered PixArt heatmap
therefore encoded activations the SAE was never trained on. `config.yaml:41-43`
independently records that t=14 has only 288 active features (flat by the project's own
measurement) — the flat PixArt heatmaps motivating `PROJECT_STATUS.md`'s central finding
were an artifact of this bug, not a model behavior.

Two further defects in the same file, in scope for this fix per the plan:
- `CocoActivationDataset` was constructed without `standardization_stats=` or `stats_seed=`,
  so `data.py` recomputed standardization statistics from a fresh, unseeded random sample of
  the cache on every invocation — two renders of the same image with the same checkpoint
  could produce different heatmaps.
- `model.load_state_dict(raw["state_dict"], strict=False)`'s return value was discarded, so
  a name/shape mismatch between the checkpoint and the freshly-constructed model would
  silently leave part of the encoder at random initialization rather than erroring.

### Root cause

`pixart_timestep.py` exists specifically to centralize this decision (its docstring names
this exact failure mode), and every other current eval script
(`dictionary_diagnostic.py`, `cross_model_overlap.py`, `input_rank_diagnostic.py`,
`feature_duplication_diagnostic.py`, `run_inference_on_images.py`, `top_activating_images.py`)
calls `resolve_pixart_timestep(...)`. `visualize_feature_activations.py` never imported it and
carried a stale comment ("matches convention elsewhere in repo") that was no longer true.

### Changes

- Imported `pixart_timestep.resolve_pixart_timestep` and replaced the hardcoded `t_idx` with
  a call to it, threading `ckpt_raw` (the raw checkpoint dict), `config_global` (live
  `config.yaml`'s `global` block, lowest-precedence fallback), and a new `--pixart_timestep`
  CLI override (matching the flag name/semantics used by `dictionary_diagnostic.py`) through
  `encode_image_tokens` and `visualize_stem`.
- `load_checkpoint` now returns `(model, model_tokens_native, raw_ckpt_dict)` instead of a
  2-tuple; `raw_ckpt_dict` is `None` only for the legacy whole-object-pickle checkpoint format
  (which carries no persisted config/stats to begin with).
- `top_k` is now read from the checkpoint's own `config.sae_params.top_k` first, falling back
  to live `config.yaml` only if the checkpoint doesn't have it.
- `model.load_state_dict(...)`'s `(missing, unexpected)` return is now checked; either
  non-empty raises `RuntimeError` naming the offending keys, instead of silently proceeding.
- `main()` now sources `spatial_align_to` and `standardization_stats` from the checkpoint
  (falling back to live config only when the checkpoint lacks them) and passes
  `stats_seed=0` as a deterministic fallback for checkpoints without persisted stats.

### Why necessary

This is Stage 0 / Fix 0.1 in `REPAIR_PLAN.md` — explicitly called out as "the single
highest-information experiment in the plan" since it may dissolve the qualitative half of
`PROJECT_STATUS.md`'s central finding. It also had to land before any other eval-script fix
(V12) could be verified against a trustworthy reference.

### Verification performed

**Environment note:** this sandbox has no GPU, no cached COCO activations, and (initially) no
torch install — the real cache/checkpoints only exist in the project's Google Colab runtime.
Per the hard constraint against full training/pipeline runs, verification was done with a
from-scratch CPU-only Python 3.14 venv (`torch` CPU wheel + numpy/pyyaml/pillow/tqdm installed
into a scratch venv) and a synthetic 1-image cache + tiny real `UniversalSAE` checkpoint
(D=6/5, latent_dim=32, top_k=4, grids 4x4 DinoV2 / 8x8 PixArt) built to match the real
combined-npz schema (`data.py`'s `<MODEL>`, `<MODEL>__sigmas`, `<MODEL>__timesteps` keys) and
checkpoint schema (`uni_demo.py:373-403`'s keys).

Test script: `test_fix01.py` (scratch dir, not part of the repo). Ran the *actual* modified
`visualize_feature_activations.py` end-to-end via `main()`, plus direct calls to
`load_checkpoint` / `resolve_pixart_timestep` for targeted checks:

1. **Timestep resolution** — built a checkpoint with `fixed_timestep_idx=10` and a live
   `config.yaml` deliberately set to a *different* value (3). Resolved timestep = 10 (from
   the checkpoint), not 3 (live yaml) and not 14 (`T-1`, the old hardcoded bug). PASS.
2. **top_k sourcing** — checkpoint's `sae_params.top_k=4` vs. live yaml's `99`. Loaded
   model's `top_k` = 4. PASS.
3. **Determinism** — ran `main()` twice in the same process against the same checkpoint/stem;
   the saved JSON feature reports were byte-identical across both runs. PASS.
4. **Silent state_dict corruption** — constructed a checkpoint whose `state_dict` had a
   renamed key (`saes.PixArt.W_enc.weight` → `...RENAMED_weight`, a name mismatch — the actual
   V1 failure mode, since `strict=False` does not raise on missing/unexpected key names by
   itself). `load_checkpoint` now raises `RuntimeError` naming the missing/unexpected keys.
   Also checked the shape-mismatch case (`latent_dim` doubled) still raises, though this was
   already true pre-fix via PyTorch's own `load_state_dict` behavior. PASS.
5. **Differential check against the pre-fix code** — `git stash`ed the fix and re-ran the
   identical test script unmodified. It crashed (`ValueError: not enough values to unpack`,
   confirming the old 2-tuple return) after printing `[stats] Computing standardisation stats
   for DinoV2 …` / `for PixArt …` on *both* "runs" — i.e., the old code recomputed stats from
   an unseeded random sample every invocation instead of reusing the checkpoint's persisted
   stats, exactly the V1(a) nondeterminism defect. Restored the fix (`git stash pop`) and
   re-ran to confirm all checks pass again. This confirms the test actually discriminates old
   vs. new behavior rather than passing vacuously.

All checks: **PASS** (evidence above; full command transcripts available in this session).

### Remaining uncertainty

- **Not verified against the real cache/checkpoint** — no access to the Colab environment or
  the actual `weights/{run_name}/usae_epoch_29.pth` / `/content/combined_cache` in this
  session. The plan's full acceptance bar for Fix 0.1 ("render the same 3 stems including
  `000000562818`, confirm t=10 heatmaps show visibly higher spatial contrast than t=14, print
  `n_active`/peakiness per render") requires that real environment and is **not yet done** —
  this is a HYPOTHESIS-level gap, not a verified result. The synthetic test proves the *code
  path* is correct (right timestep selected, right stats used, right top_k, loud failure on
  corruption); it cannot by construction demonstrate whether fixing this dissolves the
  qualitative "flat heatmaps" finding, since the synthetic activations are random noise with
  no real spatial structure.
- The legacy whole-object-pickle checkpoint format (`isinstance(raw, UniversalSAE)`) still has
  no persisted timestep/stats/spatial_align_to to source from — that path is unchanged
  (correctly falls back to live config.yaml, matching pre-existing behavior for that format;
  it was not in this fix's scope and is separately covered by V14).
- **Next step:** run this fixed script against the real epoch-29 checkpoint and cache in
  Colab per the plan's stated verification, and report whether the heatmaps' spatial contrast
  and `n_active`/peakiness actually improve at t=10 vs. t=14.

### Addendum — revision after the initial commit (`6c1a2ee` → `79f5bfa`)

The first implementation of this fix changed `load_checkpoint`'s return type from a 2-tuple
to a 3-tuple (to thread the raw checkpoint dict through). That silently broke its two other
existing callers, `pixart_timestep_autopsy.py:137` and `pixart_attn1_sanity_check.py:118`,
both of which destructure `model, model_tokens_native = load_checkpoint(...)`. This was caught
by extending the smoke test to run `pixart_timestep_autopsy.py` end-to-end against the
synthetic checkpoint, which raised `ValueError: not enough values to unpack`.

Revised to follow `REPAIR_PLAN.md` Fix 0.2's explicitly recommended approach instead: keep
`load_checkpoint`'s 2-tuple return, and set `model._standardization_stats` /
`model._training_global` attributes on the returned model (mirroring the pattern
`top_activating_images.load_universal_sae` already uses). Both downstream scripts already
read those attributes via `getattr(model, "...", g)`, so this repairs them without touching
either file — matching the plan's statement that this approach fixes them "for free." Commit
`79f5bfa` has the corrected diff; `6c1a2ee`'s content was superseded, not reverted (both
commits are on this local branch, not pushed).

Re-verified with the same synthetic setup: `model._training_global['fixed_timestep_idx']` and
`model._standardization_stats['PixArt']` are now set and carry the checkpoint's values (not
live yaml's); `resolve_pixart_timestep` still resolves to 10; and a full run of
`pixart_timestep_autopsy.py main()` against the synthetic checkpoint now succeeds, logging
`[stats] Using standardization statistics persisted in the checkpoint` instead of recomputing
from a random sample. All checks pass. `pixart_attn1_sanity_check.py` was not run end-to-end
(it requires loading real PixArt-XL diffusion weights via `PixArtActivationExtractor`, which
is infeasible in this sandbox and would border on the "no full pipeline runs" constraint) —
its dependency on `load_checkpoint`'s attributes was verified directly instead (checks above),
which is the only part this fix touches.

---

## Fix 0.2 — route top_k / standardization stats / spatial aligner through the checkpoint

**Addresses:** V12 (MEDIUM–HIGH)
**Commit:** `3d86c82`
**Files modified:** `dictionary_diagnostic.py`, `cross_model_overlap.py`,
`top_activating_images.py`, `dictionary_diagnostic_all_timesteps.py`, `input_rank_diagnostic.py`

### Issue

Several eval/diagnostic scripts rebuild a `UniversalSAE` from a checkpoint's `state_dict`
(dims, latent size, token counts all correctly sourced from the checkpoint), but read the
model's `top_k` (TopK width), standardization stats, and/or spatial-alignment target from the
*live* `config.yaml` instead of the checkpoint's own saved training config. `config.yaml`'s
`top_k` moved 512 → 128 in a recent commit, so any of these scripts run today against an older
checkpoint would silently reconstruct a wrong-shaped SAE and report firing statistics for a
model that never existed. Per `REPAIR_PLAN.md`'s V12 table, exactly which of {top_k, timestep,
standardization stats, aligner} was broken varied per file:

| Script | top_k | std. stats | aligner |
|---|---|---|---|
| `dictionary_diagnostic.py` | yaml ❌ | ckpt ✅ (already) | ckpt ✅ (already) |
| `cross_model_overlap.py` | yaml ❌ | ckpt ✅ (already) | ckpt ✅ (already) |
| `top_activating_images.py` | yaml ❌ | ckpt ✅ (already) | yaml ❌ |
| `dictionary_diagnostic_all_timesteps.py` | yaml ❌ | none, seed=None ❌ | ckpt ✅ (already) |
| `input_rank_diagnostic.py` | n/a (no model built) | none, seed=None ❌ | ckpt ✅ (already) |

(`pixart_timestep_autopsy.py` and `pixart_attn1_sanity_check.py` needed no further changes —
both call `visualize_feature_activations.load_checkpoint`, which Fix 0.1 already repaired for
them; confirmed by the Fix 0.1 addendum above.)

### Root cause

Each of these scripts independently reimplements "rebuild a UniversalSAE from a checkpoint,"
and each implementation picked a slightly different (and in these 5 cases, wrong) precedence
order between the checkpoint's own persisted config and the live `config.yaml` passed on the
command line. The correct pattern (checkpoint's own value wins, live yaml is only a fallback
for older checkpoints lacking the field) already existed elsewhere in the repo
(`feature_duplication_diagnostic.py:165`, `run_inference_on_images.py`) — these 5 scripts just
hadn't been brought in line with it.

### Changes

- `dictionary_diagnostic.py`, `cross_model_overlap.py`, `dictionary_diagnostic_all_timesteps.py`:
  added `sae_p_ckpt = ckpt_cfg.get("sae_params", {})` alongside the existing `g_ckpt`/`g_file`
  global-block lookup, and changed `top_k=int(sae_p.get("top_k", pick("top_k", N)))` to
  `top_k=int(sae_p_ckpt.get("top_k", sae_p.get("top_k", pick("top_k", N))))` — checkpoint's own
  `sae_params.top_k` now wins, live yaml's `sae_params.top_k` is the fallback, and the legacy
  global-block key is the last resort, unchanged from before.
- `top_activating_images.py`: same `top_k` fix in `load_universal_sae` (added `spc = ckpt_cfg.get("sae_params", {})`,
  changed the `top_k=` line to prefer it). Separately, `main()`'s aligner construction changed
  from `build_spatial_aligner(cfg)` (live yaml's global block) to
  `build_spatial_aligner({"global": eval_g, "model_zoo": cfg.get("model_zoo", {})})` where
  `eval_g = getattr(model, "_training_global", cfg.get("global", {}))` — reusing the
  `_training_global` attribute `load_universal_sae` already sets (checkpoint values merged over
  live yaml), so the alignment target now matches what the checkpoint actually trained with.
- `dictionary_diagnostic_all_timesteps.py`, `input_rank_diagnostic.py`: added
  `standardization_stats=ckpt.get("standardization_stats")` and a deterministic `stats_seed`
  fallback (`pick("stats_seed", 0)` / `g_ckpt.get("stats_seed", g_file.get("stats_seed", 0))`)
  to their `CocoActivationDataset(...)` construction. Previously neither was passed, so
  `data.py` recomputed stats from a fresh, unseeded random 1000-file sample of the cache on
  every invocation of these two scripts.

Explicitly out of scope (left untouched, per the V12 table and to avoid scope creep):
`input_rank_diagnostic.py`'s other known bugs (`ZeroDivisionError` at `pr_d/pr_p`, dead
`if False` branch, `torch.load` without `weights_only=False`) are V16 items for Fix 0.3, not
V12; `sae_mask.py` is also V16/Fix 0.3, not part of Fix 0.2's file list.

### Why necessary

This is Stage 0 / Fix 0.2 in `REPAIR_PLAN.md`, needed so that every downstream diagnostic
(dictionary partitioning, cross-model overlap, top-activating-images, input rank) is measuring
the checkpoint that actually exists rather than a hypothetical one matching today's
`config.yaml`. Explicitly depended on by Fix 0.4 (wandb metrics) and everything in Stage 1+,
which all lean on these diagnostics being trustworthy.

### Verification performed

Same environment caveat as Fix 0.1 (no GPU/Colab access; scratch CPU venv with
torch/numpy/pyyaml/pillow/tqdm/matplotlib). Built a synthetic 3-image cache + checkpoint (same
shapes/schema as Fix 0.1's `test_fix01.py`) where the checkpoint's `sae_params.top_k=4`
deliberately differs from live `config.yaml`'s `sae_params.top_k=99`.

Test script: `test_fix02.py` (scratch dir). For `dictionary_diagnostic.py`,
`dictionary_diagnostic_all_timesteps.py`, and `input_rank_diagnostic.py`: ran each via
`subprocess` (their real CLI entry points) twice against the synthetic checkpoint/cache, since
these three take `--repo_root`/argparse CLI args designed for exactly this. For
`top_activating_images.py`: ran its real CLI (`--skip_coco_labels` to avoid needing COCO
annotation files) twice. For `cross_model_overlap.py` (no argparse — it's written as a
Colab-cell-style script with module-level constants, not a CLI tool): called its
`load_universal_sae(ckpt_path, config_path, device)` function directly, since that's the exact
function this fix changed and the only feasible way to exercise it without running its
`if __name__ == "__main__"` block (which mounts Drive, does GPU-oriented batch processing, etc.
— unrelated to what changed).

Results (all PASS): all 5 scripts report/use `top_k=4` (the checkpoint's value), not `99` (live
yaml's); `dictionary_diagnostic_all_timesteps.py` and `input_rank_diagnostic.py` no longer log
`[stats] Computing standardisation stats...` and produce byte-identical `.npz` output across two
runs (previously nondeterministic); `top_activating_images.py`'s aligner log line shows the
correct `4x4` target grid; all outputs were byte-identical/JSON-identical across repeated runs.

**Differential check against the pre-fix code:** `git stash`ed the fix and re-ran the identical
`test_fix02.py` unmodified. It failed exactly as predicted: `dictionary_diagnostic.py` and
`dictionary_diagnostic_all_timesteps.py` reported `top_k = 99` (leaked from live yaml);
`dictionary_diagnostic_all_timesteps.py`'s output was non-deterministic across the two runs and
logged stats recomputation; `input_rank_diagnostic.py` also recomputed stats;
`cross_model_overlap.py`'s `load_universal_sae` returned `model.top_k == 99`. Restored the fix
(`git stash pop`) and re-ran to confirm all checks pass again — confirms the test discriminates
old vs. new behavior. `git status --short` after restoring showed only the 5 intended files
modified (no stray `__pycache__`/artifact diffs).

**Independent review:** per the working method (fixes touching more than one file get a
subagent review before committing), dispatched a fresh-context subagent to review the diff
against `REPAIR_PLAN.md`'s V12 table and Fix 0.2 spec. It cross-checked each file's diff against
exactly which V12 columns were marked broken for that file, checked the fallback-chain
precedence and guard-clause safety (`ckpt.get("config")` could be `None`/missing `sae_params` on
older checkpoints), and checked for scope creep. Reported zero findings; flagged one latent
pre-existing pattern (an unguarded `ckpt.get("standardization_stats")` could `KeyError` inside
`data.py` for a checkpoint trained with `standardize: false`, saving `{}`) but confirmed it's
not a regression — this diff copies the same pattern already used by the reference-correct
scripts (`feature_duplication_diagnostic.py`, `run_inference_on_images.py`), not a new defect.

### Remaining uncertainty

- Not verified against the real epoch-29 checkpoint / cache (same Colab-access gap as Fix 0.1).
  The synthetic test proves the *code path* correctly prefers checkpoint values; it doesn't
  demonstrate what numbers change on the real checkpoint (whose live-yaml top_k may or may not
  currently disagree with what it was trained on — that's itself unknown without checking).
- The pre-existing `standardization_stats={}` / `KeyError` landmine noted by the reviewing
  subagent (for checkpoints trained with `standardize: false`) is unfixed, matching the rest of
  the codebase's current behavior. Not introduced by this fix; would be a V14/hygiene item if
  addressed.
- `cross_model_overlap.py`'s `if __name__ == "__main__"` block (Drive mounting, full 2000-image
  batch run, matplotlib plotting) was not exercised end-to-end — only the specific function this
  fix changed (`load_universal_sae`) was verified directly.
