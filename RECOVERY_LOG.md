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

---

## Fix 0.3 — repair broken scripts and correct stale/false premises

**Addresses:** V16 (MEDIUM)
**Commit:** `470b7f1`
**Files modified:** `top_activating_images.py`, `sae_mask.py`, `input_rank_diagnostic.py`,
`pixart_attn1_sanity_check.py`

### Issue

Four scripts had bugs ranging from "crashes with default args" to "produces a report of the
wrong coordinate system" to "explains its own output using a false premise":

1. `top_activating_images.py main()`: `CHECKPOINT_PATH = None` is the documented default;
   `os.path.isfile(None)` raises `TypeError`, which `genericpath.isfile`'s
   `except (OSError, ValueError)` does not catch, so the auto-discovery path the docstring
   advertises crashes outright.
2. `sae_mask.py`: called `load_activation_for_image` with no `spatial_aligner=` and applied no
   standardization at all, so it fed raw, unstandardized, native-grid (e.g. 32x32) PixArt
   activations into an SAE trained on standardized, 16x16-pooled ones. `infer_grid_size` then
   reported the wrong grid, and every `--patches`/`--box` index the tool computed pointed at the
   wrong image region.
3. `input_rank_diagnostic.py`: `ZeroDivisionError` in the interpretation section when
   `participation_ratio` is exactly `0.0` — total collapse, the exact condition the script
   exists to detect. Also a dead `"X_per_feature_std" if False else "rank_90"` ternary, and
   `torch.load(...)` without `weights_only=False` (a problem once torch >= 2.6's default
   changed).
4. `pixart_attn1_sanity_check.py`: its docstring, an inline comment, and three separate
   verdict-text branches all asserted or implied that switching a hook from `attn2` to `attn1`
   was being tested. A prior finding (traced through the extractor's full commit history, not
   part of this fix) established that `PixArtActivationExtractor.extract_activations` has always
   hooked the entire last transformer block for PixArt, never `.attn1` or `.attn2` individually,
   and that `HOOK_ATTN_NAME` is dead code on that code path. The script's own `assert
   extractor.HOOK_ATTN_NAME == "attn1"` passes (the attribute really is set to that string) while
   proving nothing about what was actually hooked — a false-confidence check.

### Root cause

(1)-(3) are independent, unrelated bugs in scripts that receive less use/testing than the main
diagnostics — normal script rot. (4) is a stale docstring/verdict left over from before the
extractor's actual hooking behavior was traced; the script was never updated after that finding,
so its explanatory text and its own assert actively mislead a reader into thinking an attn1-only
ablation occurred when it didn't.

### Changes

- `top_activating_images.py`: `if not os.path.isfile(checkpoint)` → `if checkpoint is None or
  not os.path.isfile(checkpoint)`, copying the pattern already used correctly in `sae_mask.py`.
- `sae_mask.py`: added the same checkpoint-preferred `_training_global`/`build_spatial_aligner`
  construction `top_activating_images.py`'s `main()` now uses (Fix 0.2), passed
  `spatial_aligner=` into `load_activation_for_image`, and added a standardization step
  (align-then-standardize, matching `top_activating_images.compute_top_activations`'s order and
  logic verbatim) using `model._standardization_stats`, raising the same clear `ValueError` that
  function raises if a checkpoint lacks persisted stats.
- `input_rank_diagnostic.py`: guarded the `pr_d/pr_p` print behind `if pr_p > 0`, printing "fully
  collapsed (participation_ratio=0)" otherwise; deleted the dead `if False` ternary (always
  evaluated to `"rank_90"`, so this is a behavioral no-op); added `weights_only=False` to
  `torch.load`. Also corrected the interpretation text's stale "train on random timesteps"
  advice (the opposite of the project's current `fixed_timestep_idx` decision) since it's in the
  same block being touched for the crash fix — see "Remaining uncertainty" below for why this is
  flagged as a deliberate scope extension rather than folded in silently.
- `pixart_attn1_sanity_check.py`: corrected the module docstring, the `HOOK_ATTN_NAME` assert's
  comment, a stale comment referencing a `pixart_timestep_autopsy.py` `ref_thresh` bug that no
  longer exists, and all three verdict-text branches (`IMPROVED`, `SAE ISSUE`, `REAL UPSTREAM
  FAILURE`) that phrased their conclusion in terms of "attn1 vs attn2." No control flow, logic,
  or computed values were changed — comments and printed/docstring strings only.

### Why necessary

This is Stage 0 / Fix 0.3 in `REPAIR_PLAN.md`. (1)-(3) block anyone from actually running these
tools (crash) or make their output actively wrong (misaligned masking, an uncaught exception on
the exact input the tool is meant to flag). (4) is a research-integrity issue: a script that
prints a confident "IMPROVED: the attn2->attn1 hook change looks like the fix" verdict when no
such hook change occurred would mislead whoever reads its output into believing an ablation was
tested that wasn't.

### Verification performed

Same environment caveat as Fix 0.1/0.2 (no GPU/Colab access; scratch CPU venv). Built a synthetic
3-image cache + checkpoint (same schema as before). Test script: `test_fix03.py` (scratch dir).

- **`top_activating_images.py`**: since `WEIGHTS_DIR` is a hardcoded module constant (no
  `--weights_dir` CLI flag), ran in-process with the constant monkeypatched to a temp dir holding
  the synthetic checkpoint, and `sys.argv` set with `--checkpoint` *omitted* — the exact bug
  path. Passed (no `TypeError`, checkpoint auto-discovered and used).
- **`sae_mask.py`**: ran via its real CLI (`--source PixArt --patches 0,1`) and checked the
  output JSON's `metadata.grid_size` — `4` (the DinoV2-aligned grid) after the fix.
- **`input_rank_diagnostic.py`**: crafting a genuine `participation_ratio == 0.0` through the
  *real* align+standardize+SVD pipeline turned out to be numerically fragile — an initial attempt
  broadcasting one random nonzero vector across every token/image left ~1e-7-scale float32
  residue after `avg_pool2d`/centering, which sits above `effective_rank`'s `eps=1e-10` threshold
  and produced a spurious rank-1 result instead of a true collapse (caught by checking the actual
  printed `participation_ratio` rather than trusting the setup). Switched to an exact-zero
  activation array (`np.zeros`), which stays exactly representable through centering, and
  confirmed `participation_ratio` prints as `0.0 / 5` and the interpretation section prints the
  guarded "fully collapsed" message with exit code 0 (no crash).
- **Differential check against the pre-fix code**: `git stash`ed the fix and re-ran the identical
  `test_fix03.py` unmodified. All three reproduced exactly: `top_activating_images.py` raised
  `TypeError: _path_isfile: path should be string, bytes, os.PathLike or integer, not NoneType`;
  `sae_mask.py` reported `grid_size=8` (PixArt's raw native grid, wrong); `input_rank_diagnostic.py`
  raised `ZeroDivisionError: division by zero` at the exact `pr_d/pr_p` line REPAIR_PLAN.md's V16
  section describes. Restored the fix (`git stash pop`) and re-ran to confirm all checks pass
  again. `git status --short` showed only the 4 intended files modified.
- **`pixart_attn1_sanity_check.py`**: not runnable end-to-end (requires loading real PixArt-XL
  diffusion weights, infeasible in this sandbox — same constraint noted for Fix 0.1). Verified by
  `ast.parse` (syntax) and manual read-through of every changed string; not exercised at runtime.
- **Independent review**: dispatched a fresh-context subagent to review the diff against
  REPAIR_PLAN.md's V16 catalog. It found one incomplete spot — the `SAE ISSUE` verdict branch in
  `pixart_attn1_sanity_check.py` still said "the raw attn1 activation" unqualified, contradicting
  the file's own new docstring disclaimer — which was corrected in this same commit before
  committing (re-verified with `ast.parse` and the full `test_fix03.py` re-run afterward, both
  still pass). It also flagged the `input_rank_diagnostic.py` advice-text rewrite (see below).

### Remaining uncertainty

- **Scope note, `input_rank_diagnostic.py`**: the reviewing subagent flagged that the corrected
  interpretation-text advice (removing "train on random timesteps," adding a preprocessing-
  unification suggestion) goes beyond Fix 0.3's literal three-line list for this file
  (`:162`/`:173`/`:78`), even though it addresses a separately-documented V16 observation ("stale
  advice... recommends random timesteps — the opposite of the current decision") in the same
  print block already being touched for the crash fix. This is a print-string-only change (zero
  behavioral/logic risk) but is flagged here as a deliberate, judgment-call scope extension
  rather than folded in silently, per the instruction to be explicit about anything beyond the
  fix's literal scope.
- `pixart_attn1_sanity_check.py`'s corrected text was not validated against a real run of the
  script (requires real PixArt-XL weights) — the correction is a careful read-through, not an
  observed-output confirmation. If the extractor's hooking behavior changes in the future (e.g.
  `HOOK_ATTN_NAME` is made to actually take effect for PixArt), this docstring/verdict text would
  need updating again.
- Not verified against real data/Colab (same gap as Fix 0.1/0.2 throughout).

---

## Fix 0.4 — repair wandb training metrics

**Addresses:** V11 (MEDIUM)
**Commit:** `a87413b`
**Files modified:** `train.py`

### Issue

Several wandb keys logged by `train_universal_sae`'s per-step logging block were uninformative,
miscalibrated, or actively misleading:

- `train/latent_sparsity` = `(z == 0).float().mean()`. Hard TopK guarantees exactly `top_k`
  nonzeros per token always, so this was the constant `(latent_dim - top_k) / latent_dim` for
  the entire run — zero information, every run, forever.
- `train/source_model` logged the source model's *name* as a string (e.g. `"PixArt"`). wandb
  stores string-valued keys as a non-numeric column that will not render on a line chart.
- `train/latent_align_loss` logged the **unweighted** per-pair average
  (`latent_align_loss / n_align_pairs`) under a name that implies it's the term's actual
  contribution to the objective — it's multiplied by `latent_align_weight` before being added to
  `loss`, so the logged number understated (or overstated) the real contribution by exactly that
  factor with no way to tell from the chart alone.
- The partition-diagnostic "is this feature used by this model" threshold was hardcoded to
  `1e-3`, completely decoupled from `resample_dead_threshold` (the parameter the dead-feature
  resampler itself uses for the identical concept). Changing `resample_dead_threshold` silently
  stopped matching what this diagnostic reported as dead.
- `partition/score = max_excl / max(used_by_all, 1)`. When `used_by_all == 0` — total
  partitioning between models, the *exact* failure this metric exists to detect — the denominator
  silently became `1`, so the metric's units jumped from a ratio to a raw feature count instead
  of reporting the value it actually is: undefined. A chart reader would see a small-ish finite
  number and not realize the dictionary had completely split.
- `partition/usage_cosine_*` (cosine similarity between two models' per-feature usage-rate
  vectors) is bounded in `[0,1]` since both inputs are non-negative firing rates. It saturates
  high whenever a handful of near-always-on features are shared by both models, regardless of how
  poorly the rest of the ~12k-feature dictionary is shared, and cannot express anti-correlation.

### Root cause

Each metric was written once, independently, without re-deriving what it actually measures once
the surrounding config changed (TopK replacing ReLU+L1, `resample_dead_threshold` being made
configurable, etc.) — the same "metric drift" pattern seen in the eval scripts fixed in Fix 0.2,
just inside the training loop itself this time.

### Changes

All changes are confined to the wandb-logging `if use_wandb and WANDB_AVAILABLE and (batch_idx %
log_every == 0):` block, after `loss`/`optimizer.step()` are already computed — this is a
logging-only change, nothing here feeds back into the loss, forward/backward pass, or optimizer
state:

- Deleted the `train/latent_sparsity` line.
- Replaced `train/source_model` (string) with `train/source_model_idx =
  model.model_names.index(source)` (int, stable ordering set at construction).
- Split into `train/latent_align_loss_unweighted` (the old expression, unchanged) and
  `train/latent_align_loss_weighted` (`latent_align_weight * unweighted`, the actual
  contribution to `loss`).
- Partition "used" threshold: `e > 1e-3` → `e > resample_dead_threshold` (an existing parameter
  of `train_universal_sae`, already in scope).
- `partition/score`: now `max_excl / used_by_all` when `used_by_all > 0`; `float("inf")` when
  `used_by_all == 0` but some model has exclusive usage; `float("nan")` when neither (dictionary
  entirely dead).
- Added `partition/usage_cosine_centered_<A>_vs_<B>` (cosine of each model's usage vector after
  subtracting its own mean — equivalent to a Pearson correlation across features, can go
  negative) and `partition/usage_jaccard_<A>_vs_<B>` (intersection/union of the two
  threshold-crossing boolean sets, `nan` if the union is empty) alongside the existing raw
  cosine, rather than replacing it.

### Why necessary

This is Stage 0 / Fix 0.4 in `REPAIR_PLAN.md`, needed so training runs after this point produce
wandb charts that are actually diagnostic — in particular so `partition/score`'s discontinuity
doesn't hide the one failure mode (total partitioning) the metric exists to catch, and so the
existing `usage_cosine` reading of "0.857, looks fine" (flagged elsewhere as unsupported, see
V11 in the plan) can be cross-checked against a metric that isn't structurally biased toward
looking healthy.

### Verification performed

Same environment caveat as prior fixes (no GPU/Colab; scratch CPU venv, now also with `einops`
installed since `train.py` imports it at module level). Test script: `test_fix04.py` (scratch
dir). Stubbed the `wandb` module in `sys.modules` *before* importing `train.py` (so
`WANDB_AVAILABLE` picks it up), with a fake `wandb.log(d, step=...)` that records every logged
dict — no network/auth needed, and this exercises the real `import wandb` / `WANDB_AVAILABLE`
gate in `train.py`, not a mock of the function under test.

Built a synthetic 8-image cache (same schema as before) and ran `train.train_universal_sae(...)`
for real (`use_wandb=True`, `log_every=1`, `curriculum_epochs=0` so cross-recon/alignment are
active immediately, `resample_dead=True`) against a tiny real `UniversalSAE` (D=6/5,
latent_dim=32, top_k=4) with a real `torch.optim.Adam`, on CPU, for one short "epoch" (4 batches
of 2 images from an 8-image `DataLoader`). Checked every logged dict across all 4 steps:

- `train/latent_sparsity` absent; `train/source_model` (string) absent; `train/source_model_idx`
  present and an `int` at every step.
- `train/latent_align_loss_weighted / train/latent_align_loss_unweighted == 3.0` exactly
  (the `latent_align_weight` passed in), whenever the unweighted term was nonzero.
- Every numeric logged value at every step was finite, *except* `partition/score`, which was
  checked against `partition/used_by_all_models` at the same step: `inf`/`nan` exactly when
  `used_by_all_models == 0`, a plain ratio otherwise — never the other way around.
- `partition/usage_cosine_centered_*` and `partition/usage_jaccard_*` both appeared once both
  models' usage EMAs existed (steps 1-3; absent at step 0, correctly, since only one model's EMA
  exists after a single batch).
- Re-ran the identical training with `resample_dead_threshold=0.9` instead of the default
  `1e-3` and confirmed `partition/used_by_all_models` dropped to `0` across all logged steps
  (a 0.9 firing-rate threshold is realistically uncrossable) — proving the partition diagnostic's
  threshold is actually wired to the parameter rather than still hardcoded.
- **Differential check against the pre-fix code**: `git stash`ed the fix and re-ran the identical
  `test_fix04.py` unmodified. Every discriminating check failed exactly as predicted:
  `train/latent_sparsity` present at every step, `train/source_model` present as a string
  (`"DinoV2"`/`"PixArt"`) with `train/source_model_idx` missing, `usage_cosine_centered_*` and
  `usage_jaccard_*` never appeared, and `used_by_all_models` was identical between the
  `threshold=1e-3` and `threshold=0.9` runs (proving the old hardcoded `1e-3` really was
  decoupled from the parameter). Restored the fix (`git stash pop`) and re-ran to confirm all
  checks pass again.
- **Independent review**: dispatched a fresh-context subagent (required per the working method
  for core-pipeline-file changes) to review the diff against REPAIR_PLAN.md's V11 section. It
  traced `source` back through `_pick_source` to confirm `model.model_names.index(source)` can
  never raise `ValueError` in the production call path (`uni_demo.py`, where both `model_dims`
  and the dataset's `sources` derive from the same `MODEL_ZOO.keys()`), verified the
  mean-centered-cosine and Jaccard math, verified `partition/score`'s three branches, and
  confirmed nothing in the diff touches loss computation, the forward/backward pass, or optimizer
  state. Reported zero findings.

### Remaining uncertainty

- Not verified against a real training run on the actual COCO/PixArt cache or a real wandb
  project (same Colab-access gap as every prior fix) — the synthetic run proves the logging code
  path is correct, not what the real run's `partition/score`/`usage_cosine_centered`/etc. values
  actually look like on trustworthy data.
- Per the plan's own note: this changes chart semantics for `partition/score` (units differ
  around `used_by_all == 0`) and adds/removes/renames several keys, so wandb runs after this fix
  are not directly comparable to runs before it on those specific keys. The plan says to encode
  this in the run tag when the next real training run happens — not done here since no real run
  was launched in this session.
- The verification test used `resample_dead=True` with a very short run (4 steps, `resample_interval=2`)
  specifically to also exercise the resampler in the same pass; it was not the focus of this fix
  (that's V2, a separate Stage 1 item) and was not independently re-verified here beyond "it ran
  without crashing and revived the expected number of features."
