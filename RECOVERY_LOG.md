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

---

## Fix 1.1 — repair dead-feature resampling

**Addresses:** V2 (BLOCKER)
**Commit:** `01dddaa`
**Files modified:** `train.py`, `config.yaml`, `uni_demo.py`

### Issue

`resample_dead_features` (the Bricken et al. dead-feature revival routine, run every
`resample_interval` steps) had five compounding defects that put roughly a third of the shared
12288-wide dictionary into a permanent churn loop instead of converging:

1. **`fixed_timestep_idx` was never threaded into the resampler.** Training pins PixArt to
   timestep 10, but the resampler's internal `_extract_source_slice` call never received
   `fixed_timestep_idx`, so it fell through to `random.randrange(15)`. Revived PixArt features
   were seeded from whichever of the 15 cached timesteps was randomly drawn that event — almost
   never the t=10 distribution training actually samples from — so they were dead on arrival at
   the next resample event, 500 steps later.
2. **Sampling probability was `recon_err ** 4`, not `** 2`.** `recon_err` was already a squared
   L2 norm; squaring it again (contradicting the function's own docstring, which claimed
   "proportional to squared self-reconstruction error") meant a token with 2x the error got 16x
   the sampling probability instead of 4x. Combined with `replacement=True` and up to 4096 draws
   from as few as 4096 tokens, this collapsed revivals onto near-duplicate directions seeded from
   a handful of worst tokens.
3. **Revived encoder rows fired at legitimate-coefficient magnitude.** `W_enc[k] = 0.2 * ref_norm
   * d`, where `d` is the *unit direction of the seed token itself* (cosine 1.0). Trained features
   have small cosine with any given token, so this 0.2x scale did not compensate — revived
   features immediately evicted real, trained features from the top-128 selection.
4. **`resample_max_per_event: 4096` revived up to a third of the 12288-wide dictionary in a
   single step**, compounding (2) and (3).
5. **`resample_start_step: 200` was a silent no-op** — 200 is not a multiple of
   `resample_interval: 500`, so the first eligible step was 500 regardless of this setting.

Separately: **no gradient clipping existed anywhere in the repo**, so a resample-induced gradient
spike had nothing bounding it before `optimizer.step()`.

### Root cause

`resample_dead_features` and its call site in `train_universal_sae` were written once and never
re-derived against the rest of the training loop as it evolved (the `fixed_timestep_idx` pinning
config option, in particular, postdates the resampler and was never plumbed through it). Each
defect is independent but they compound: (1) guarantees revived features start from
out-of-distribution data; (2)+(4) guarantee revivals cluster on a handful of directions instead of
spreading out; (3) guarantees each revival is individually disruptive to the current TopK
selection. Together, a feature revived this way is likely to be re-flagged dead at the *next*
resample event, closing the loop.

### Changes

All in `resample_dead_features` (train.py) and its call site (`train_universal_sae`), per
`REPAIR_PLAN.md` Fix 1.1, in order:

1. Added `fixed_timestep_idx: Optional[int] = None` to `resample_dead_features`'s signature;
   threaded it into the internal `_extract_source_slice(..., fixed_timestep_idx=fixed_timestep_idx)`
   call; passed it from `train_universal_sae`'s call site using the same `fixed_timestep_idx` the
   rest of the function already uses for the main forward pass.
2. `probs = recon_err.double()` (dropped the extra `** 2`); corrected the docstring.
3. `replacement = n_dead > probs.numel()` — sample without replacement whenever there are at
   least as many candidate tokens as features to revive; falls back to `replacement=True` only
   when `n_dead` exceeds the token count.
4. `resample_max_per_event`: `config.yaml` 4096 → 512 (~4% of the dictionary instead of ~33%).
5. After writing the revived weights for a model, the usage EMA for that model at the revived
   indices is set to `dead_threshold + max(dead_threshold, eps)` (strictly above the `> dead_threshold`
   "used" check, handles `dead_threshold=0`), so revived features aren't miscounted as
   `used_by_none` for the ~40 steps it takes the EMA to naturally catch up.
6. `resample_enc_scale`: `config.yaml` 0.2 → 0.05 (the simpler of the plan's two allowed options
   for compensating for cosine-1.0 seeding).
7. Added `torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)` between
   `loss.backward()` and `optimizer.step()`, gated on a new `grad_clip_norm: float = 1.0` parameter
   (`<=0` disables it); the pre-clip norm is logged as `train/grad_norm_preclip` per the plan's own
   risk note ("may mask a real instability — log the pre-clip grad norm too"). Wired through
   `uni_demo.py` (`SAE_PARAMS.get("grad_clip_norm", ...)`) and `config.yaml` (`sae_params.grad_clip_norm: 1.0`).
8. `resample_start_step`: `config.yaml` 200 → 500 (a multiple of `resample_interval`, so it's no
   longer a no-op).
9. Corrected `_reset_adam_state_slice`'s docstring: the first revived Adam update is ~3.2x the
   nominal lr-scaled step (bias correction ≈1 while `m=0.1g`, `v=0.001g²`), not "~lr-scaled... not
   an explosion" as previously claimed.

### Why necessary

This is Stage 1 / Fix 1.1 in `REPAIR_PLAN.md` — the plan calls it a BLOCKER that "knocks the model
off its optimum seven times per run" and keeps "one third of the dictionary... in a churn loop that
can never converge," directly explaining the wandb loss spikes at every resample event and the
`used_by_none = 3389/12288` puzzle noted in `PROJECT_STATUS.md`. It blocks any clean read of a
training curve and must land before Fix 1.2 (loss weight semantics) or Stage 2 (data fixes), both
of which assume a training loop that doesn't self-destruct every 500 steps.

### Verification performed

Same environment caveat as Fixes 0.1-0.4 (no GPU/Colab access). Built a scratch CPU venv (torch
2.13.0+cpu via the `download.pytorch.org/whl/cpu` index, plus numpy/pyyaml/einops/tqdm from PyPI)
since no venv persisted from prior sessions. Two test scripts, both importing and exercising the
*real* `train.py`/`universal_sae.py`, not reimplementations:

**`test_fix11.py`** — isolated checks against `resample_dead_features` directly, using a tiny real
`UniversalSAE` (`model_dims={"DinoV2":6,"PixArt":5}`, `latent_dim=32`, `top_k=4`) and synthetic
activations:
- **Check A (timestep threading):** built PixArt activations where `x[:, t] == t` (constant per
  timestep) so the revived direction is a deterministic function of *which* timestep was used.
  With `fixed_timestep_idx=10` passed, revived `W_dec` columns were byte-identical across three
  different `torch.manual_seed` values (PASS). Monkeypatched `random.randrange` to prove directly
  that the pinned path never calls it (0 calls), while omitting `fixed_timestep_idx` (simulating
  the pre-fix call site) does call it exactly once, and the revived direction matches whichever
  timestep it returned rather than 10 — reproducing the V2a bug mechanism, not just its symptom.
- **Check B (probability formula):** monkeypatched `torch.multinomial` to capture the exact `probs`
  tensor passed in; recomputed `recon_err` independently against a pre-mutation deep copy of the
  model (since the function mutates weights model-by-model, so PixArt's own recon_err must be
  captured before the earlier DinoV2 iteration's weight writes are read back) and confirmed
  `probs == recon_err` exactly (not `recon_err ** 2`). Also confirmed `replacement=False` was
  passed when `n_dead <= token_count`.
- **Check C (EMA reset):** confirmed both models' usage EMAs at the revived indices went from
  `<= dead_threshold` (pre-resample) to `> dead_threshold` (post-resample).
- **Check D (replacement boundary):** confirmed `replacement=False` still holds at the exact
  boundary `n_dead == token_count`.
- All checks: **PASS**.

**`test_fix11_e2e.py`** — end-to-end verification via the real `train_universal_sae` training loop
(not just the resampler in isolation), with `wandb` stubbed (no network), a tiny real `UniversalSAE`
(`latent_dim=64`, `top_k=4` — chosen so TopK competition naturally produces dead features, unlike
a dictionary barely larger than `top_k`), 64 synthetic images, `resample_interval=4`,
`curriculum_epochs=0`, 8 epochs (64 steps):
- All logged losses finite throughout (including the new `train/grad_norm_preclip` key, confirming
  gradient clipping is wired end-to-end).
- 3 resample events fired (steps 4, 32, 60), each reviving only 2 features — **not every interval**.
- Post-resample `train/total_loss` ratio at every event: **0.98x-1.00x** (i.e., the loss *improved*
  or held flat immediately after resampling, not spiked).
- `partition/used_by_none` returned to 0 between events rather than staying elevated.

**Differential check against the pre-fix code** (`git stash` / `git stash pop`, both scripts
re-run before and after): the isolated test's Check A confirmed the pre-fix `resample_dead_features`
signature doesn't even accept a `fixed_timestep_idx` keyword (`TypeError`), directly confirming the
parameter was entirely absent. For the end-to-end comparison, ran a pre-fix-compatible variant
(`test_fix11_e2e_prefix.py`, identical setup minus the new `grad_clip_norm` kwarg,
`resample_enc_scale=0.2`/`resample_max_per_event=0` matching the old defaults) against the stashed
pre-fix code and observed the exact churn signature V2 describes: **a resample event fired at
every single interval with no exceptions** (steps 4, 8, 12, ..., 60 — 15/15 possible events, vs.
3/15 post-fix), reviving exactly 2 features every time, with `partition/used_by_none` **stuck at 2
for the entire second half of the run** (never returning to 0 between events) — the permanent
churn loop V2 describes, reproduced directly on this toy model and resolved by the fix. (Loss
ratios were ~0.98-1.00x in both pre- and post-fix runs on this particular toy scale — this specific
synthetic model's gradient magnitudes were apparently too small for the 3-5x spike V2 predicts to
manifest at this size; the churn-loop signature via `used_by_none`/event-frequency is the
discriminating evidence here, not the loss ratio.)

**CORRECTION — this differential result is confounded, and should not be read as evidence the
churn loop is fixed** (added by a later review pass). Both reported signals (event frequency
15/15 → 3/15, and `used_by_none` returning to 0 between events) are direct mechanical outputs of
change #5 in this very fix — the usage-EMA reset. #5 sets a revived feature's EMA to
`2 × dead_threshold`; at the training loop's EMA decay of 0.95 that takes ~14 steps to fall back
below `dead_threshold`. The test ran at `resample_interval=4`, so #5 **by construction** prevents
a revived feature from being re-flagged dead at the next ~3 events, and **by construction** drives
`used_by_none` to 0 between events. The test measured the metric the change edits, using a metric
the change edits. It cannot distinguish "the churn loop is broken" from "the dead-counter was
reset."

Worse, the confound does not transfer to the real configuration in either direction: at
`resample_interval=501`, `0.95^501 ≈ 6e-12`, so the EMA reset has decayed to nothing long before
the next event and has no effect at all on real-scale event frequency. So the toy run neither
demonstrates the fix nor rules it out at the scale that matters.

What this fix's evidence actually supports, unchanged: the isolated `test_fix11.py` checks
(timestep threading, `probs == recon_err`, `replacement=False`, EMA reset firing) all verify the
individual mechanisms directly and remain valid. The *end-to-end claim* that the churn cycle is
resolved is **not** verified and is downgraded to a hypothesis pending the real run described
below. A non-confounded version of that run needs `resample_interval` set well above the EMA
recovery window (>~60 steps at decay 0.95), or `used_by_none` measured immediately *before* each
resample event rather than between events.

**Independent review:** dispatched a fresh-context subagent to review the diff against
`REPAIR_PLAN.md`'s V2 write-up and Fix 1.1's 9-point prescription. It verified all 9 changes are
present and correctly wired (including tracing that `ema_attrs[model_name]` aliases the real
`model._usage_ema_<name>` tensor so in-place mutation persists without a `setattr`, and that
`grad_norm` is assigned unconditionally before the `use_wandb`-gated log block, ruling out a
`NameError`). Reported zero blocker/high findings; two low-severity style observations (no
`NameError` risk from `grad_norm`'s placement; `uni_demo.py`'s `grad_clip_norm` lookup does a
two-level `SAE_PARAMS.get(..., CONFIG.get(..., 1.0))` fallback while sibling `resample_*` keys only
do a one-level lookup — stylistic only, not a bug).

### Remaining uncertainty

- **Not verified against the real epoch-29 checkpoint / cache or a real Colab training run** (same
  gap as every prior fix in this log). The synthetic end-to-end run demonstrates the churn loop is
  broken on a toy model at toy scale (`latent_dim=64`, `top_k=4`, 64 images); it does not by itself
  prove the effect size on the real 12288/128 dictionary over 2000 images and 30 epochs. Per the
  plan's own Fix 1.1 verification bar, the next session (with Colab/GPU access) should run **2
  epochs at N=64 images, batch 8, `resample_interval=8`, `curriculum_epochs=0`, `use_wandb=false`**
  and confirm: (a) post-resample loss stays within ~1.2x of pre-resample, (b) `used_by_none` is
  monotonically non-increasing across events, (c) all losses finite, (d) revived-count decreases
  across successive events. **REQUIRES EXTERNAL EXPERIMENT** (needs the real cache/GPU, which this
  sandbox does not have) — see command below.
  ```
  # In the Colab/GPU environment, from the repo root, with config.yaml's Fix 1.1 values
  # (resample_interval temporarily overridden to 8, curriculum_epochs to 0, N capped to 64 images):
  python uni_demo.py   # or the project's existing tiny-scale entry point, with the above overrides
  ```
  Expected result: no post-resample loss spikes >1.2x, `used_by_none` trending down, no NaN/Inf.
  Success criteria as stated in `REPAIR_PLAN.md` Fix 1.1's verification section.
- **`resample_enc_scale: 0.05`** was chosen as the simpler of the plan's two allowed remediations
  (lower the scale vs. reformulate it to scale by achieved pre-activation). This is a config value,
  not re-derived from first principles for the real 12288-wide dictionary/128 top_k — it may need
  further tuning once real training data is available; the plan flags this as a "risk: slows
  dead-feature recovery; watch `used_by_none` over the run" tradeoff, not a correctness issue.
- The toy end-to-end model's post-resample loss ratios (~0.98-1.00x) were similar in both the
  pre-fix and post-fix runs, unlike the 3-5x spikes V2 predicts for the real run — most likely
  because this toy model's scale (6/5-dim inputs, 64-wide dictionary) doesn't reproduce the same
  gradient magnitudes as the real 384/1152-dim, 12288-wide dictionary. The churn-loop signature
  (event frequency and `used_by_none` failing to resolve between events) was the reliable
  discriminator at this scale and is reported above instead.
- Fix 1.2 (loss weight normalization / curriculum boundary, V9/V10) explicitly depends on this fix
  landing first per the plan's dependency ordering, and has not yet been started.

---

## Fix 1.2 — make loss weights mean what the config says

**Addresses:** V9 (HIGH), V10 (MEDIUM)
**Commit:** `c294715`
**Files modified:** `train.py`, `config.yaml`

### Issue

Two related misunderstandings about how the training loss is assembled:

- **V9:** `sae_loss = reconstruction_loss / reconstruction_weight_total` is a **weighted average**
  of the self-recon and cross-recon MSE terms, not a weighted sum. `self_weight` and `cross_weight`
  therefore only ever matter as a *ratio* between them — their absolute magnitudes are normalized
  away. `config.yaml:75-76`'s comment framed raising `cross_weight` 1.0 → 2.0 as "back to the old
  2:1 cross/self ratio ... cross-recon is the whole point," which is directionally correct but
  invites the wrong mental model: it does not add cross-recon gradient on top of an unchanged
  self-recon signal. It moves the split from (1/2, 1/2) to (1/3, 2/3) and **shrinks the self-recon
  gradient by a third**. Anyone tuning `cross_weight` expecting "more cross pressure, same self
  pressure" is only trading self-recon away.
- **V10:** At the curriculum boundary (`epoch == curriculum_epochs`), three things change in the
  same step: self-recon's share of the (still-normalized) reconstruction loss drops, cross-recon
  MSE enters at 2/3 weight from an untrained ≈1.0 starting point, and both alignment cosine terms
  (post-TopK and pre-TopK) switch on simultaneously — an analytically-predicted ~8x jump in
  `total_loss`, by construction, not a bug. But `model._train_loss_ema` (the EMA rendered on wandb
  as `train/total_loss_ema`) was never reset at that boundary, so the smoothing blends the jump
  across several steps and the chart shows a ramp instead of the step it actually is — making the
  discontinuity invisible to whoever is reading the training curve.

### Root cause

V9 is a documentation gap, not a code bug: the normalization arithmetic itself is intentional
(kept the loss scale stable), but the comment explaining it implied absolute-magnitude reasoning
that the code doesn't support. V10 is a straightforward oversight: the EMA reset logic was written
before the curriculum-boundary discontinuity was characterized, and nothing was added to handle it
when the curriculum feature was introduced.

### Changes

Per `REPAIR_PLAN.md` Fix 1.2's explicit recommendation ("keep the normalization and fix the
comments" over dropping it, since dropping it "changes the effective LR — do not combine with any
other change"):

1. **No change to the `reconstruction_loss / reconstruction_weight_total` arithmetic.** Added a
   comment directly at that line in `train.py` explaining the weighted-average semantics and why
   the normalization was kept (loss-scale stability across the curriculum boundary, which V10
   depends on). Added a parallel comment in `config.yaml` next to `cross_weight` making the
   ratio-only relationship and the 1.0:2.0 → (1/3, 2/3) split explicit.
2. **Curriculum boundary EMA reset.** Added `model._last_in_curriculum` (a plain attribute, same
   pattern as the pre-existing `_train_loss_ema`/`_usage_ema_*`) tracked across epochs. Right after
   `in_curriculum = epoch < curriculum_epochs` is computed (before the batch loop), if the model
   was in curriculum last epoch and isn't this epoch, `ema_loss` is set to `None` so the first
   logged step of the post-boundary epoch takes the raw, unsmoothed loss value directly instead of
   blending with the pre-boundary EMA.
3. **Explicitly did not add the plan's optional LR re-warmup at the boundary** ("consider a short
   LR re-warmup (~50 steps) at that boundary"). The plan's own V15 documents the existing LR
   schedule as already fragile — warmup logic lives in `train.py`, cosine decay in `uni_demo.py`,
   sharing an undeclared `initial_lr` key, and currently only avoids compounding because warmup
   finishes inside epoch 0. Adding a second warmup trigger at an arbitrary later epoch risks
   interacting with that fragility and is out of scope for this fix; deferred as a judgment call.

### Why necessary

This is Stage 1 / Fix 1.2 in `REPAIR_PLAN.md`, explicitly dependent on Fix 1.1 landing first (a
training loop that doesn't self-destruct every 500 steps has to exist before loss-weight semantics
are worth reasoning about). It matters because both V9 and V10 are the kind of thing that costs a
wasted experiment cycle: someone raising `cross_weight` expecting more cross-model pressure without
realizing they're trading away self-recon (V9), or someone looking at a smooth `total_loss_ema`
ramp at epoch 5 and not realizing a discontinuity is being hidden there, potentially misreading
post-curriculum instability as a separate problem (V10).

### Verification performed

Same environment/caveat as Fix 1.1 (scratch CPU venv, real `train.py`/`universal_sae.py`, `wandb`
stubbed). `test_fix12.py`, using a tiny real `UniversalSAE` (`latent_dim=32`, `top_k=4`) and 16
synthetic images:

- **Check A:** ran 5 epochs with `curriculum_epochs=2`. Confirmed `model._last_in_curriculum`
  correctly reads `False` after the run. Confirmed the **first logged step of epoch 2** (the
  boundary epoch) has `train/total_loss_ema == train/total_loss` exactly (i.e., the EMA was reset
  and recomputed fresh, not blended) — while the first logged step of epoch 1 (still within
  curriculum, no boundary crossed) has `total_loss_ema != total_loss` (a normal blend), confirming
  the reset fires only at the actual transition. PASS.
- **Check B:** confirmed the raw `total_loss` jumps from 0.1094 (last in-curriculum step) to 1.8706
  (first post-curriculum step) — a ~17x jump on this toy model, consistent in direction and order
  of magnitude with V10's analytic ~8x prediction (exact multiplier depends on model/data scale;
  the toy setup isn't expected to reproduce 8x precisely). PASS.
- **Check C:** ran 2 epochs with `curriculum_epochs=0` (cross-recon/alignment active from epoch 0,
  so no boundary ever exists) and confirmed epoch 1's first step is a normal EMA blend, not an
  artificial reset — i.e., the boundary logic doesn't misfire when there's no boundary to cross.
  PASS.

**Differential check against the pre-fix code** (`git stash`/`git stash pop`): re-ran the identical
`test_fix12.py` unmodified against the stashed pre-fix `train.py`. It failed immediately at the
`model._last_in_curriculum` assertion with `AttributeError: 'UniversalSAE' object has no attribute
'_last_in_curriculum'` — confirming the pre-fix code never tracked curriculum-boundary state at
all. Restored the fix (`git stash pop`) and re-ran to confirm all checks pass again.

**Independent review:** dispatched a fresh-context subagent to review the diff against
`REPAIR_PLAN.md`'s V9/V10 write-ups and Fix 1.2's prescription. It traced the boundary-detection
logic through all cases (first-ever call, mid-curriculum, the exact boundary, post-boundary epochs,
`curriculum_epochs=0`) and confirmed correct behavior in each; confirmed the EMA reset produces a
genuinely visible step at the consumption site; confirmed `_last_in_curriculum` follows the
existing plain-attribute pattern with no new checkpointing risk (the checkpoint pickles the whole
model object); confirmed the added comments are factually accurate and that the normalization
arithmetic itself was left untouched (no scope creep into the risk the plan explicitly flagged: "do
not combine [dropping the normalization] with any other change" — this diff didn't touch it at
all). Reported one low-severity observation for awareness, not a defect: if training resumes from a
freshly-constructed model object exactly at the boundary epoch (rather than the pickled checkpoint
that already carries `_last_in_curriculum`), the `getattr` default silently skips that run's reset
— a preexisting limitation shared with `_train_loss_ema`'s identical pattern, not introduced by
this diff, and outside Fix 1.2's scope.

### Remaining uncertainty

- **Not verified against the real epoch-29 checkpoint or a real Colab training run** (same gap as
  every prior fix). The synthetic run confirms the mechanism (reset fires exactly at the boundary,
  produces a visible step, doesn't misfire) but the ~17x jump observed is a toy-model artifact, not
  a measurement of the real run's actual jump size. Per the plan's own Fix 1.2 verification bar,
  the next session (with Colab/GPU access) should run **N=64, batch 8, `curriculum_epochs=1`, 3
  epochs**, and confirm the `total_loss` step at the boundary matches the analytic ~8x prediction
  to within ~20%, and that the EMA now shows a step rather than a ramp on the actual wandb chart.
  **REQUIRES EXTERNAL EXPERIMENT** (needs the real cache/GPU):
  ```
  # In the Colab/GPU environment, from the repo root, with config.yaml's curriculum_epochs
  # temporarily overridden to 1, N capped to 64 images, 3 epochs total:
  python uni_demo.py
  ```
  Expected result: `train/total_loss` at the epoch-1/epoch-2 boundary step jumps to within ~20% of
  the analytic 8x prediction from V10; `train/total_loss_ema` shows a matching step rather than a
  smoothed ramp.
- **The LR re-warmup at the boundary was deliberately not implemented** (see Changes #3 above) —
  this is a scope decision, not an oversight, but it means post-curriculum training still proceeds
  at whatever LR the cosine schedule has decayed to by that epoch, with no re-stabilization step for
  the sudden loss-landscape shift. If post-curriculum training shows instability in a real run, this
  is the first lever the plan suggests trying, currently untried.
- **Stage 1 (Fixes 1.1 + 1.2) is now complete.** Per `REPAIR_PLAN.md`'s dependency ordering, Stage 2
  (Fix 2.1: unify preprocessing + re-cache PixArt; Fix 2.2: standardization/pooling order; Fix 2.3:
  train/val split) is next, and explicitly requires re-caching PixArt activations — this falls
  under the "no dataset recaching / no PixArt activation extraction" constraint on this session and
  is a **REQUIRES EXTERNAL EXPERIMENT** item for the next session with Colab/GPU access, not
  something to attempt here.

---

## Fix 2.1 — unify DinoV2/PixArt preprocessing crop + DIFT-style single-timestep PixArt extraction

**Addresses:** V3 (BLOCKER), V4 (BLOCKER), V13 (HIGH, partially — the I/O-amplification fix; the
mmap/`.copy()` sub-issues in `data.py` are untouched, out of this fix's file list)
**Commit:** `7b6eded`
**Files modified:** `models.py`, `DiffusionActivationExtractor.py`, `cache_coco_diffusion_activations.py`,
`pixart_timestep.py`

### Issue

Three independent, compounding defects in how PixArt activations are produced:

- **V3:** `models.py`'s `DinoV2.__init__` preprocessed images with `transforms.Resize((224, 224))` —
  a tuple, which anisotropically squashes the whole frame to 224×224 with no cropping. PixArt's
  preprocessing (`DiffusionActivationExtractor.py`) already used `Resize(512)` [int — short side to
  512, aspect preserved] + `CenterCrop(512)`. For a typical non-square COCO image, DinoV2's 16×16
  grid covered the *entire* frame (stretched); PixArt's 32×32 grid covered only the center square.
  `latent_align_mode: per_token` and cross-reconstruction both assume grid cell (i,j) is the same
  image location in both models — it wasn't, and the offset varied per image with aspect ratio.
- **V4:** PixArt extraction noised the clean latent to ~99% noise (the scheduler's *first* step,
  sigma≈0.006–0.03 signal fraction) and then generated forward under a null/empty prompt with no
  conditional pass — a DDIM sample only weakly anchored to the real image, very different from (and
  ~15× more expensive than) the DIFT-style protocol (re-noise the clean latent to a moderate t,
  single forward pass) the project's t=10 pin was implicitly imitating. Three further defects in the
  same code path: micro-conditioning (`added_cond_kwargs["resolution"]`) was fed the *latent*
  tensor's H,W (64×64) instead of pixel resolution (512×512) — out-of-distribution for every AdaLN
  modulation in the checkpoint; no `encoder_attention_mask` was passed, so cross-attention attended
  over ~255 T5 pad-embedding positions as real content; the noise (`torch.randn_like`) was
  completely unseeded, so no two extraction runs, ever, produced the same cached activation for the
  same image.
- **V13:** caching all 15 timesteps per image when training only ever reads one (`fixed_timestep_idx:
  10`) is 14/15 of every disk read and I/O transfer wasted.

### Root cause

V3: two preprocessing pipelines written independently for two different models, never unified. V4:
the extraction loop was adapted from a standard "generate an image" DDIM sampling loop rather than
designed for activation *extraction*, so it inherited generation's assumptions (start from near-pure
noise, use a null prompt only as a placeholder for later CFG that was never added) instead of the
DIFT protocol's assumption (start from a moderate, informative noise level, take one pass). V13 is a
direct consequence of caching the full trajectory when only one point on it is ever used.

### Changes

**`models.py`** — `DinoV2.__init__`'s preprocessing changed from `Resize((224, 224))` to
`Resize(224)` [int] + `CenterCrop(224)`, matching PixArt's existing crop strategy exactly. Scoped
strictly to `DinoV2` per `REPAIR_PLAN.md`'s file list (`models.py:86-89`) — `SigLIP`, `CLIP`, `ViT`,
`ResNet`, and `ConvNeXt` in the same file have the identical anisotropic-squash bug but are not part
of the active DinoV2+PixArt pipeline (`models.py`'s own docstring: "the frozen feature-extractor
zoo... nothing in `train.py` imports it" beyond the two models actually in `model_zoo`); left
untouched to avoid scope creep, flagged here for whoever extends to SigLIP/SD per the project's
stated extrapolation goal.

**`DiffusionActivationExtractor.py`** (all changes confined to `PixArtActivationExtractor`, the only
extractor actually used — SD3/FLUX are dead code per `REPAIR_PLAN.md` §7):
- Added `self._pixel_resolution = 512` at `__init__`, next to the `CenterCrop(512)` it must match
  (single source of truth). `_get_transformer_input`'s `added_cond_kwargs["resolution"]` now uses
  this instead of the latent tensor's `H, W` (which are 64×64 after 8× VAE downsampling).
  `aspect_ratio` was left as-is (it's scale-invariant for a uniform downsample, so it was never
  wrong).
- `_encode_null_prompt` now captures the tokenizer's real `.attention_mask` (previously discarded,
  only `.input_ids` was kept) and returns it as `encoder_attention_mask`; `_get_transformer_input`
  threads it into its returned dict, which reaches the transformer automatically since call sites
  already spread it with `**transformer_inputs`.
- Added a `_make_noise(ref, generator=None)` helper: `generator=None` preserves the exact old
  `torch.randn_like` behavior; a single `torch.Generator` seeds the whole batch; a list of one
  `Generator` per batch item produces independently-reproducible per-image noise (sampled row-by-row
  on CPU regardless of `self.device`, so a filename-derived seed reproduces on any machine). Used in
  both the existing multi-step path and the new single-step path.
- Added `single_timestep: Optional[int] = None` and `generator=None` parameters to
  `extract_activations`. When `single_timestep` is given: noise the **clean** latent directly to
  that raw timestep (`alpha = alphas_cumprod[single_timestep]`, the same DDPM forward-process formula
  already used for the old path's `timesteps[0]`, just evaluated at the chosen t instead), take
  **one** forward pass (hook fires once), no `scheduler.step` call, no reverse-diffusion loop. The
  returned `ActivationOutput` has exactly one entry in `activations`/`timesteps`/`sigmas`. When
  `single_timestep=None` (the default), the reverse-diffusion **loop structure** is unchanged —
  see the CORRECTION below for what is *not* unchanged.

**CORRECTION (added by a later review pass; the original wording in this entry was wrong).**
This entry originally claimed the `single_timestep=None` path is "unchanged byte-for-byte from
before this fix — this is purely additive." **That is false.** The loop *body* is unchanged, but
it calls `_get_transformer_input`, which this same fix changed to send `resolution=[512, 512]`
instead of `[64, 64]` and to add `encoder_attention_mask`. Both change what the transformer
computes, on **both** branches. Re-running the old full-trajectory protocol will therefore
**not** reproduce the pre-Fix-2.1 cache.

The code is right and the change is intended — the micro-conditioning and attention-mask defects
were bugs on both paths, so fixing them on both is correct. What was wrong was the claim, and the
verification behind it: the check diffed the *loop bodies* char-for-char, a strictly smaller unit
than "the default path's behavior," and the conclusion was then stated at the larger scope. This
is the same class of error the self-audit below was chartered to catch, in the self-audit's own
output. Practical consequence: old and new PixArt caches are not interchangeable and must not be
mixed in one `combined_cache` directory. Now noted in code at
`cache_coco_diffusion_activations.py`'s `USE_DIFT_SINGLE_TIMESTEP` flag and in
`extract_activations`'s docstring.

**Follow-up fix in the same code path (same review pass):** `_encode_null_prompt` originally
passed the mask only to the transformer's cross-attention; `self.text_encoder(text_inputs_1)` was
still called without `attention_mask=`, so T5's own self-attention still ran over 255 pad
positions. Now fixed (`attention_mask=attention_mask_1`), matching diffusers'
`PixArtAlphaPipeline.encode_prompt`. Low impact — the prompt is `""` for every image, so the
resulting embedding is a dataset-wide constant either way — but it is a real part of the same V4
defect and was left half-done.

**`pixart_timestep.py`** — added `resolve_pixart_raw_timestep(scheduler_timesteps, ckpt=None,
config_global=None, override=None)`, which resolves an index the same way the existing
`resolve_pixart_timestep` does, then maps it through a scheduler's `.timesteps` array to a raw
diffusion timestep (0–999) — the input the new single-timestep extraction path needs, while keeping
the existing `fixed_timestep_idx: 10` config value meaning the same noise level it always has.

**`cache_coco_diffusion_activations.py`** — `cache_diffusion_activations()` gained
`single_timestep` and `seed_from_filenames` parameters. Whether to actually pass them through to a
given extractor is decided via `inspect.signature(extractor.extract_activations).parameters`
introspection, so extractors without these parameters (SD3/FLUX, still on the base class's plain
`extract_activations(self, image)` signature) get a printed warning and silently keep their old
behavior instead of raising `TypeError`. Added `_stem_seed(filename)` (via `zlib.crc32`, since
Python's built-in `hash()` is randomized per-process unless `PYTHONHASHSEED` is fixed) to derive a
deterministic per-image seed. `__main__` now resolves the raw timestep from `config.yaml`'s
`fixed_timestep_idx` via `resolve_pixart_raw_timestep` and passes `seed_from_filenames=True` by
default, behind a `USE_DIFT_SINGLE_TIMESTEP = True` flag documented as switchable back to the old
full-trajectory extraction for scripts that need a genuine timestep sweep (e.g.
`pixart_timestep_autopsy.py`, which needs several distinct timesteps in one cache, not one).

**Deliberate deviation from the plan's literal text, flagged explicitly:** `REPAIR_PLAN.md` Fix 2.1
item 2 says the new cache shape should be `(1024, 1152)` (T fully squeezed out). This implementation
keeps an explicit `T=1` leading dimension — cache shape `(1, 1024, 1152)` — instead. This achieves
the same substance (one real forward pass instead of fifteen, ~15× less data and compute, `V13`
dissolved) while staying **100% compatible** with `train.py`/`data.py`'s existing (B,T,N,D) diffusion
handling (`_extract_source_slice`, `_pick_diffusion_slice`, `resample_dead_features`'s diffusion
branch) without an invasive, GPU-unverifiable rewrite of that already-hardened code. With `T=1`,
`_resolve_fixed_timestep`/`fixed_timestep_idx` clamp any index to `[0, 0]` and trivially always
resolve to index 0 — degrades gracefully, doesn't crash. This was a deliberate scope/risk judgment
call for this session (no GPU access to verify a deeper train.py rewrite), not an oversight.

### Why necessary

This is Stage 2 / Fix 2.1 in `REPAIR_PLAN.md` — the plan calls it the step that "actually addresses
`loss_PixArt_to_DinoV2 ≈ 0.948`" and explicitly the highest-risk step in the whole plan ("invalidates
the existing cache and every existing checkpoint"). V3 alone makes every existing per-token
cross-model claim in `PROJECT_STATUS.md` unverifiable (the two grids never covered the same pixels).
V4 means the cached PixArt "representation of the image" was mostly a hallucination anchored to
~1-3% real signal. Both must be fixed, together, before any re-cache is worth running, since
re-caching is expensive and re-doing it twice (once per bug) wastes the exact 9h→40min I/O win V13
is supposed to deliver.

### Verification performed

**Pixel-correspondence check (REPAIR_PLAN.md's own Fix 2.1 verification step 1), run this session at
the user's explicit request:** built 4 synthetic checkerboard test images (landscape 640×480,
portrait 480×640, wide 800×450, square 500×500 — synthetic rather than real photos, since this
checks the *transform geometry*, not photo content) and computed/rendered both models' crop boxes
using the REAL `preprocess` objects from `models.py` (`DinoV2`, via `torch.hub.load` monkeypatched to
a no-op so no network/weight download occurs) and `DiffusionActivationExtractor.py` (`PixArtActivationExtractor`,
via `_load_pipeline` monkeypatched to a no-op for the same reason — this check only exercises
torchvision `Resize`/`CenterCrop` geometry, no model forward pass). Result, all 4 images: DinoV2's
and PixArt's crop boxes are concentric (centers agree to <1px) with identical coverage fraction of
the short axis. Visually inspected the landscape and portrait overlay PNGs directly (magenta =
DinoV2 224px box, cyan = PixArt 512px box) — the two boxes are visually indistinguishable, confirming
grid cell (i,j) now covers the same normalized region in both models. This is a direct consequence
of the fix (before it, DinoV2's box would have been the *entire* image, not a centered square) — not
independently re-verified against the pre-fix code in this pass, since the structural argument (V3)
was already established without needing a rendered comparison.

**Code-level tests** (scratch CPU venv, same as prior fixes, now also with `torchvision`/`pillow`
installed): three test scripts, all importing and exercising the REAL, unmodified source files:
- `test_fix21_models.py`: instantiated the real `DinoV2` class (torch.hub.load monkeypatched) and
  confirmed `preprocess.transforms[0]` is `Resize` with an **int** size 224 (not a tuple),
  `transforms[1]` is `CenterCrop(224)`; functionally confirmed a 300×500 non-square image comes out
  224×224 via the resize-then-crop path (short side hits 224 before cropping, not squashed
  directly to 224×224).
- `test_fix21_extractor.py`: built a `PixArtActivationExtractor` instance with `__init__` bypassed
  (no real pipeline load) and every model component (`transformer`, `vae`, `scheduler`, `tokenizer`,
  `text_encoder`) replaced by a minimal fake matching the real call signature/shapes, then exercised
  the REAL `extract_activations`/`_make_noise`/`_get_transformer_input`/`_encode_null_prompt`
  methods against those fakes. Confirmed: `single_timestep=None` reproduces the original 15-activation,
  15-`scheduler.step`-call trajectory; `single_timestep=<t>` produces exactly 1 activation and 0
  `scheduler.step` calls; micro-conditioning `resolution` passed to the transformer is `[512, 512]`
  per batch item, not `[64, 64]`; `encoder_attention_mask` reaches the transformer call and correctly
  marks only 1/256 positions real for an empty-string T5 encoding; `_make_noise` is deterministic
  given a matching seed (both single shared Generator and per-image Generator list) and produces
  different noise for a different seed, with a `ValueError` on a generator-count/batch-size mismatch;
  full `extract_activations(single_timestep=..., generator=...)` end-to-end reproduces byte-identical
  activations for the same image+seed and different activations for a different seed.
- `test_fix21_cache_script.py`: confirmed `_stem_seed` is deterministic and filename-sensitive;
  built a fake extractor matching PixArt's new signature and confirmed `cache_diffusion_activations`
  threads `single_timestep`/per-image `generator`s through correctly, writes a cache with `T=1`, and
  that two independent runs with `seed_from_filenames=True` produce identical per-image generator
  seeds; built a second fake extractor matching SD3/FLUX's old signature (`extract_activations(self,
  image)` only) and confirmed the new kwargs are silently and gracefully dropped (a warning printed,
  no `TypeError`), with its normal 3-step trajectory unaffected.

**Differential checks against the pre-fix code** (`git stash`/`git stash pop`, all three test scripts
re-run before and after): `test_fix21_models.py` failed exactly as predicted (`Resize` size was the
tuple `(224, 224)`, not an int); `test_fix21_extractor.py` failed at import (`resolve_pixart_raw_timestep`
did not exist) — the fix's new API surface is entirely additive, so there is no pre-fix equivalent
of the single-timestep/seeding behavior to differentially compare beyond "does it exist at all";
`test_fix21_cache_script.py` similarly failed at import (`_stem_seed` did not exist). All three
restored and re-confirmed passing after `git stash pop`.

**Independent review:** dispatched a fresh-context subagent (required per the working method for
this session's highest-risk, multi-file change) to review the diff against `REPAIR_PLAN.md`'s V3,
V4, V13, and Fix 2.1 sections. It independently traced: the single-timestep branch's noise
formula and absence of `scheduler.step`; `_make_noise`'s three cases; `_pixel_resolution`'s
single-source-of-truth wiring; `encoder_attention_mask`'s full path from tokenizer to transformer
call; byte-for-byte equivalence of the `single_timestep=None` default path to the pre-fix code;
the `inspect.signature` gating's correctness for both PixArt's and SD3/FLUX's actual signatures
(and confirmed it can't mask an unrelated `TypeError`, since there's no try/except involved); that
only `DinoV2` was touched in `models.py` (confirmed via grep that the other 5 model classes still
have the old anisotropic `Resize`); and that `train.py`'s existing timestep-index clamping degrades
gracefully to index 0 for a `T=1` cache, supporting the deliberate-deviation reasoning above.
Reported zero blocker/high findings. Two low-severity notes: `single_timestep`'s type hint should be
`Optional[int]` rather than a bare `int = None` (fixed in this same commit, re-verified with a
re-run of `test_fix21_cache_script.py` afterward, still passes); and that the 512×512-into-a-
1024-trained-checkpoint mismatch V4 also flags is unaddressed by this diff — correctly identified as
a pre-existing plan-level scope choice (Fix 2.1 item 1 explicitly picks PixArt→512), not a
regression introduced here.

### Remaining uncertainty — DEFERRED / REQUIRES EXTERNAL EXPERIMENT

- **The 16-image re-cache test (REPAIR_PLAN.md Fix 2.1 verification step 2) was explicitly deferred
  at the user's direction after a feasibility check.** This sandbox has network access (huggingface.co
  reachable) and disk space (703GB free), but **no GPU** (`nvidia-smi` not found, no CUDA). Actually
  running this test means downloading PixArt-alpha's full pipeline (transformer + VAE + T5-XXL text
  encoder, several GB) and running real transformer forward passes on CPU only, which risks a very
  long or stalled run. Presented this tradeoff to the user; they chose to skip it and defer to Colab.
  **REQUIRES EXTERNAL EXPERIMENT:**
  ```python
  # In a Colab/GPU environment, from the repo root:
  import torch
  from DiffusionActivationExtractor import PixArtActivationExtractor
  from pixart_timestep import resolve_pixart_raw_timestep
  from cache_coco_diffusion_activations import cache_diffusion_activations, _stem_seed

  ext = PixArtActivationExtractor(device="cuda", num_inference_steps=15)
  ext.scheduler.set_timesteps(ext.num_inference_steps, device=ext.device)
  t = resolve_pixart_raw_timestep(ext.scheduler.timesteps.tolist(), config_global={"fixed_timestep_idx": 10})

  cache_diffusion_activations(
      extractor=ext, source_name="PixArt",
      coco_root="/content/coco_data/val2017", cache_root="/content/cache_test16",
      batch_size=2, num_workers=2, image_list=<any 16 filenames>,
      single_timestep=t, seed_from_filenames=True,
  )
  # Then re-run the identical call into a second cache_root and diff the two
  # sets of .npz files byte-for-byte to confirm seeded noise reproduces exactly.
  ```
  Success criteria: cache shape is `(1, 1024, 1152)` per image (not `(15, 1024, 1152)`), dtype
  matches the extractor's configured dtype, and the two independent runs produce byte-identical
  `.npz` activation arrays (proving the per-image seeding actually reproduces on real PixArt
  weights, not just on the fakes this session's tests used).
- **The representation-quality check (Fix 2.1 verification step 3 — cosine-similarity matrix between
  PixArt patch tokens within one image, compared old vs. new extraction) was not run** — same GPU/
  real-weights gap, and it depends on the step-2 re-cache existing first (both old and new cache
  needed for the comparison). **REQUIRES EXTERNAL EXPERIMENT**, to be run immediately after the
  16-image re-cache test above, per `REPAIR_PLAN.md` Fix 2.1's own verification ordering.
- **The full 2000-image re-cache (Fix 2.1 verification step 4) was, obviously, not attempted** — it
  is gated on steps 2 and 3 passing first, per the plan's own ordering, and is explicitly out of this
  session's scope regardless (full-dataset recaching).
- **The DIFT-single-timestep path's actual numerical behavior against real PixArt-XL weights is
  entirely unverified** — the fake-model tests in this session prove the *code path* is correct
  (right formula, right shape, no reverse loop, right conditioning values reach the transformer call),
  not that the resulting activations are qualitatively better-anchored to the real image than the old
  protocol. That claim can only be checked once the deferred re-cache + representation-quality check
  above are run.
- Per `REPAIR_PLAN.md`'s own instruction for this step: "Do it on a branch. Keep the old cache until
  Stage 2 verification passes." This work is on `code-fixes`, not `main`, and is purely additive to
  the existing extraction code — the old cache and old extraction path are untouched and fully
  available, so no existing cache/checkpoint has been invalidated by this commit itself (only an
  actual re-cache run, not yet performed, would do that).

---

## Self-audit — corrections to Fix 1.1 and Fix 2.1, and a closed verification gap

**Commit:** `bc8986b` (comment corrections); integration test below is not a separate commit (test
files live outside the repo, in the session's scratch directory)
**Trigger:** explicit user request, after this session's earlier fixes were already committed, to
audit every comment/docstring/flag written this session against what the code actually executes —
"this codebase has a history of comments describing a fix that the code doesn't actually implement,"
per the user, referencing V8 (the `attn1`/`attn2` hook finding) and the V16 stale-docstring catalog.
This is exactly the failure mode Fix 0.3 fixed once already (in `pixart_attn1_sanity_check.py`'s
docstring); worth checking whether this session's own new comments repeated it.

### What the audit checked, and found

**1. `HOOK_ATTN_NAME = "attn1"` — user's specific concern, checked, confirmed NOT a new issue.**
The user asked to confirm `extract_activations` still hooks the whole block, not `attn1`, despite
`HOOK_ATTN_NAME = "attn1"` being declared. Confirmed: `PixArtActivationExtractor`'s hook line
(`last_block = self._get_last_block(); hook_handle = last_block.register_forward_hook(hook_fn)`)
hooks the block object itself, never `getattr(last_block, self.HOOK_ATTN_NAME)`. This is **V8**,
already documented in `REPAIR_PLAN.md` as a correction to `PROJECT_STATUS.md` (the cache has always
held the full post-block residual stream, `HOOK_ATTN_NAME` is dead code for PixArt) and was not
touched or claimed-fixed by this session's Fix 2.1 work — Fix 2.1 only added a `single_timestep`
parameter and fixed micro-conditioning/attention-mask/noise-seeding, none of which relate to which
submodule is hooked. No regression, no misleading new claim about hook targeting was introduced.

**2. "Unconfirmed caller" — user's specific concern, checked, found a real verification gap and closed it.**
The user pointed out that Fix 2.1's own tests never actually confirmed
`cache_coco_diffusion_activations.py` passes `single_timestep=`/`generator=` to the REAL
`PixArtActivationExtractor.extract_activations` — `test_fix21_cache_script.py` used a hand-written
fake extractor CLASS with the new signature "by construction," and `test_fix21_extractor.py` called
`extract_activations` directly, never through `cache_diffusion_activations()`. Neither proved the
two were actually wired together for the real class. This mattered specifically because
`extract_activations` is decorated with `@torch.no_grad()`, and `cache_diffusion_activations`'s
kwarg-gating relies on `inspect.signature(extractor.extract_activations).parameters` — if the
decorator didn't preserve the signature, the gating would silently disable the new path even though
the method supports it, and nobody would notice since the fallback (dropping the new kwargs with a
printed warning) doesn't raise.
- Checked directly: `inspect.signature` on the real bound method (decorator intact) correctly shows
  `single_timestep`/`generator` as parameters — `torch.no_grad()`'s implementation preserves the
  wrapped signature. Not a bug, but genuinely unconfirmed before this check.
- Wrote a new integration test exercising the REAL `PixArtActivationExtractor` class (only model
  components — transformer/VAE/scheduler/tokenizer — faked; `_load_pipeline` no-op'd, no
  network/GPU) through the REAL, unmodified `cache_diffusion_activations()`. Confirmed on disk: the
  resolved raw timestep reaches the transformer's `timestep` argument on every call;
  `scheduler.step` is never invoked; the cache file has a `T=1` leading dimension; the cache's
  `timesteps` array on disk equals the resolved raw timestep. A differential run with
  `single_timestep=None` confirmed the old 15-step trajectory and `T=15` cache remain unaffected.
  All checks passed — this closes the gap; it was a missing test, not a bug in the code, but the
  user was right that it hadn't actually been confirmed. (First attempt at this test used a
  simplified `FakeBlock`/`FakeTransformer` that never patchified `(B,C,H,W)` into `(B,N,D)` token
  format the way a real DiT block does — this caused a `RuntimeError` in
  `cache_coco_diffusion_activations.py`'s stack/permute logic on the first run, since the fake's
  4-D output didn't match the 3-D `(B,N,D)` shape production code assumes. Fixed the fake to
  patchify/unpatchify like a real block; unrelated to the actual code under test.)

**3. Comment-accuracy sweep across this session's changes found two real inaccuracies (neither
flagged by the earlier per-fix subagent reviews, since those reviewed logic correctness, not
comment precision against independently-rederived math):**

- **`train.py`'s `_reset_adam_state_slice` docstring (Fix 1.1)** claimed a fixed "~3.2x" multiplier,
  reasoning "bias correction is ~1" once `exp_avg`/`exp_avg_sq` are zeroed. This is wrong: `step` is
  NOT reset (by design — it can't be, per-row), so beta2's bias correction `(1 - 0.999^step)` keeps
  whatever value the tensor's *global* step count has already reached, and 0.999's bias correction
  does not approach 1 until thousands of steps in — nowhere near "~1" at this project's actual
  `resample_interval=500` schedule. Numerically verified with real `torch.optim.Adam` (lr=1) at
  every resample step this project actually hits: **1.99x at step 500 (the first resample event),
  2.52x at 1000, 2.79x at 1500, 2.94x at 2000, 3.03x at 2500, 3.08x at 3000, 3.11x at 3500 (the
  last)** — a fixed ~3.2x is the asymptotic value as step→∞, never actually reached in this run.
  Corrected the docstring to state the measured range and explain why it moves. This was a real
  inaccuracy in an already-committed comment — flagged and fixed here, not silently absorbed.
- **`DiffusionActivationExtractor.py`'s `_encode_null_prompt` comment (Fix 2.1)** said "cross-
  attention in every hooked block attends over 255 pad embeddings." Since only ONE block is ever
  hooked (for activation capture — see point 1 above), "every hooked block" is confusing at best,
  reads as implying multiple hooked blocks at worst. The actual mechanism: the attention mask
  affects cross-attention computed in every one of the ~28 transformer blocks during the single
  forward pass; only one block's *output* is captured. Corrected the wording.

**4. Everything else checked and found already accurate** (verified independently, not just
re-read): the DIFT single-timestep branch's noise formula and absence of `scheduler.step`; `_make_noise`'s
three cases; `_pixel_resolution`'s wiring; `encoder_attention_mask`'s full path; ~~the claim that the
`single_timestep=None` default path is byte-for-byte/RNG-stream-identical to the pre-fix code
(verified by extracting and directly diffing the old vs. new loop bodies char-for-char, not just
trusting the earlier subagent review's structural read)~~ — **RETRACTED, see the CORRECTION in Fix
2.1 above.** That check diffed the loop *bodies*, which are indeed identical, and then stated the
conclusion at the level of the whole default *path*, which is not: the loop calls
`_get_transformer_input`, which this fix changed for both branches. This audit pass missed it
because it inherited the earlier entry's framing instead of re-deriving the unit of comparison;
the `resample_enc_scale`/cosine-1.0 claim
in Fix 1.1 (re-derived: `W_enc[k]·(x-b_pre) = enc_scale·‖x-b_pre‖` exactly, by construction);
`resolve_pixart_raw_timestep`'s index→raw-timestep mapping; `HOOK_ATTN_NAME`'s AdaLN-conditioning
comment at a different line (correctly describes conditioning that genuinely applies to all 28
blocks, not a hooked-block conflation).

### Why this matters

Comments and docstrings are read by whoever picks this repair up next; an inaccurate one is worse
than no comment, because it actively misleads rather than leaving a gap someone knows to fill. The
Adam-multiplier correction in particular matters for anyone tuning `resample_interval` or
`resample_enc_scale`: the true post-resample disruption is *smaller* early in a run and *larger*
late in a run, the opposite of what a flat "~3.2x" implies.

### Remaining uncertainty

- No further inaccuracies were found in this pass, but this is not a guarantee none remain — this
  was a targeted audit (every comment this session wrote, checked against independently-rederived
  math or a new/existing test), not an exhaustive formal verification.
- The corrected Adam-multiplier range (1.99x-3.11x) is specific to this project's exact
  `resample_interval=500`-driven step schedule; it would shift if that changes. It does NOT depend
  on the configured `lr` value itself — checked empirically (not just reasoned analytically) at
  `lr` in `{1.0, 0.1, 0.0005}` (this project's actual `lr: 0.0005`), all giving the identical
  1.9855x ratio at step 500 to 4 decimal places, since `eps=1e-8` is negligible next to
  `sqrt(v_hat)` at these gradient magnitudes.

---

## Fix 2.2 — compute standardization stats on the post-alignment distribution

**Addresses:** V6 (HIGH)
**Commit:** `766d12f`
**Files modified:** `data.py`, `uni_demo.py`

### Issue

`config.yaml` sets `spatial_align_to: DinoV2`, so PixArt's native 32×32 token grid is average-pooled
down to DinoV2's 16×16 grid (via `SpatialAligner.align()`, called from `train.py`) before either
model's activations reach the SAE. Standardization statistics (per-channel mean/std, computed in
`data.py`'s `_compute_standardization_stats`) were fit to the RAW, UNPOOLED per-token distribution.
Averaging spatially correlated neighbouring tokens together reduces variance below what any
individual token had — `Var[mean(X1..X4)] = (1 + 3ρ)/4 · Var[X]` for pairwise correlation ρ, which
is `< Var[X]` whenever ρ < 1 — a real statistical property of pooling correlated variables, not a
bug in the pooling arithmetic itself (`spatial_align.py`'s `avg_pool2d` reshape/permute logic is
correct, and was verified correct back in the original diagnostic pass, V6's "what is NOT wrong"
note). A std fit to the unpooled distribution therefore under-normalizes the pooled representation
the SAE actually consumes: `loss_DinoV2_to_PixArt` (identity-mapped target, correctly unit-variance)
and `loss_PixArt_to_DinoV2` (source pooled through a mis-calibrated std) were never on comparable
scales, and neither was validly comparable to the "≈1.0 = predicting the mean" reading the project's
central finding rests on.

### Root cause

`_compute_standardization_stats` was written before spatial alignment existed in the pipeline (or at
least, never updated to know about it) and has no visibility into whether/how its output will be
pooled later in `train.py`. The pooling itself happens in a completely different module (`train.py`,
via a `SpatialAligner` built in `uni_demo.py`) with no shared reference to what statistics were used
upstream.

### Changes

`REPAIR_PLAN.md`'s literal instruction is "apply spatial alignment before standardization." This
implementation achieves the identical numerical outcome with a smaller, lower-risk diff, based on a
mathematical fact checked directly (not assumed): per-channel affine standardization
(`(x - mean) / std`, with `mean`/`std` constant across the token axis) and spatial resampling
(`SpatialAligner.align` — average-pooling for downsampling, nearest/bilinear/bicubic for upsampling)
both act linearly along the token axis and **commute exactly**: `align((x - mean) / std) ==
(align(x) - mean) / std`, for downsampling (pooling is a weighted average with position-independent
affine terms factoring out), upsampling (nearest-neighbor duplication and partition-of-unity
interpolation kernels both preserve constants exactly), and independently for each timestep of a
diffusion source's `(T, N, D)` tensor (alignment only touches the `N`/`D` axes, leaving `T`
untouched, so pooling before vs. after picking a single timestep is identical for whichever timestep
gets picked). Given this, **the actual bug was never about the order the two operations run in at
runtime** — it was that the mean/std constants were fit to the wrong distribution's variance. Fixing
that fixes the numbers regardless of which order the (mathematically-equivalent) operations run in.

Concretely:
- `CocoActivationDataset.__init__` (`data.py`) gained an optional `spatial_aligner=None` parameter
  (duck-typed, avoiding a hard import of `SpatialAligner` — matches the file's existing loose-typing
  style), stored as `self.spatial_aligner`.
- Added `_apply_spatial_align_for_stats(self, act, source)`: returns `act` unchanged if no aligner is
  set or `source` isn't registered with it; for a 2-D vision `(N, D)` activation, adds/removes a
  throwaway batch dim around `SpatialAligner.align` (which requires `(B, N, D)`); for a 3-D diffusion
  `(T, N, D)` activation, calls `align` directly (the leading `T` axis plays the "batch" role
  harmlessly, per the per-timestep-independence argument above).
- `_compute_standardization_stats` calls this right after `_maybe_strip_cls` and before
  `_flatten_tokens_for_stats`, so the Welford accumulator sees the POOLED distribution.
- **`__getitem__` and every one of `train.py`'s ~7 existing `spatial_aligner.align(...)` call sites
  (including inside `resample_dead_features`) are completely untouched** — still standardize (using
  the now-correctly-computed stats) then align, in that literal order, which the commutativity
  argument makes numerically identical to the reverse order.
- `uni_demo.py`: moved the `spatial_aligner = build_spatial_aligner_from_config(...)` block from
  after dataset construction to before it, and passed `spatial_aligner=spatial_aligner` into
  `CocoActivationDataset(...)`. `model_tokens` (the only input `build_spatial_aligner_from_config`
  needs besides live config) was already available earlier in the file, so nothing else needed to
  move.

### Why necessary

This is Stage 2 / Fix 2.2 in `REPAIR_PLAN.md`, explicitly dependent on Fix 2.1 (re-cache) per the
plan's ordering — it "makes the '≈1.0 = predicting the mean' reading of the loss finally valid," a
precondition for any of Stage 3's re-measurement work meaning anything.

### Verification performed

Same scratch CPU venv as prior fixes. `test_fix22.py` built a synthetic combined-npz cache (2000
images, one source "PixArt", native grid 4×4→16 tokens, pooled to target grid 2×2→4 tokens) with a
**known, controllable spatial correlation structure**: each 2×2 block shares one common per-block
value (variance 1) plus independent per-token noise (variance 1) — giving every individual token
variance 2 (ρ=0.5 within a block) and an analytically-predictable pooled variance of `1 + 1/4 =
1.25`. Ran the REAL `CocoActivationDataset`/`SpatialAligner` against this cache:
- **Pre-fix simulation** (`spatial_aligner=None` at construction, reproducing the old behavior):
  stats fit to raw tokens gave std ≈ √2 ≈ 1.41 (matching the per-token variance prediction exactly).
  Standardizing with this std then pooling (mimicking `__getitem__` + `train.py`'s unchanged call
  sequence) gave a post-align std of **~0.79-0.83 — outside `[0.95, 1.05]`**, reproducing the V6 bug
  quantitatively, not just qualitatively.
- **Post-fix** (`spatial_aligner=<the real aligner>` passed to the dataset): stats fit to pooled
  tokens gave std ≈ √1.25 ≈ 1.118 (matching the pooled-variance prediction exactly) — measurably
  smaller than the pre-fix std, as expected. Standardizing with this corrected std then pooling gave
  a post-align std of **0.996-1.006** and mean of **-0.002 to -0.013** — inside `[0.95, 1.05]` and
  `|mean| < 0.05`, **REPAIR_PLAN.md's own Fix 2.2 verification bar, met exactly** (initially missed
  the mean bound at a smaller sample size — 0.0501 vs. the 0.05 cutoff — traced to ordinary sampling
  noise at n=100 images and resolved by widening the check to n=2000, not a code change).
- An earlier draft of this test had a methodology bug (computed variance *within each image's 4
  pooled tokens* via `.var(dim=0)` on a `(4, D)` per-image tensor, then averaged across images — a
  within-group-variance statistic that underestimates the true pooled-distribution variance, since
  subtracting each small group's own empirical mean removes real variance). Caught by cross-checking
  against a from-scratch debug script that flattened across both the image and token axes before
  computing variance, which gave a materially different (and correct) answer; fixed the test to
  match. Recorded here per the standing instruction to flag rather than silently absorb
  test-methodology mistakes.
- **Regression check:** constructing `CocoActivationDataset` with no `spatial_aligner` argument at
  all produces byte-identical stats to explicitly passing `spatial_aligner=None` (`torch.equal` on
  both mean and std) — confirms the new parameter's default preserves old behavior exactly.

**Differential check against the pre-fix code** (`git stash`/`git stash pop`): re-ran the identical
`test_fix22.py` unmodified against the stashed pre-fix `data.py`. Failed immediately with
`TypeError: CocoActivationDataset.__init__() got an unexpected keyword argument 'spatial_aligner'`
— confirms the parameter (and therefore the whole fix) was entirely absent pre-fix. Restored and
re-confirmed all checks pass.

**Independent review:** dispatched a fresh-context subagent to review the diff against
`REPAIR_PLAN.md`'s V6 write-up and to independently re-derive the commutativity argument this diff's
smaller footprint depends on — specifically asked to check downsampling, BOTH upsampling modes, and
the diffusion-timestep-axis case, since if the argument broke in any of those the diff would be
wrong. It confirmed all three algebraically (including bilinear/bicubic interpolation as a
partition-of-unity, a case broader than what this diff needed), confirmed `uni_demo.py`'s reordering
is safe (traced `model_tokens`'s availability), and confirmed via direct `git diff train.py`
inspection (not just the description) that all 7 `spatial_aligner.align(...)` call sites, including
the one inside `resample_dead_features`, are genuinely untouched. Reported one low-severity
observation, not a defect: `_apply_spatial_align_for_stats` pools each of PixArt's 15 cached
timesteps independently, but they're still merged into one Welford accumulator across all 15 — V6 is
fixed, but the pre-existing V5 mismatch (training only ever consumes t=10) means these stats won't
be exactly right for the single t=10 slice until Fix 2.1's re-cache actually lands (`REPAIR_PLAN.md`
already declares this dependency explicitly).

### Remaining uncertainty

- **Not verified against the real cache or a real Colab training run** (same gap as every prior
  fix). The synthetic test's correlation structure (block-shared-value + independent noise) is a
  clean, analytically tractable stand-in for "real DiT patches are spatially correlated" — it proves
  the mechanism and the code path, not the exact ρ (and therefore exact std correction) real PixArt
  activations will need. Per the plan's own Fix 2.2 verification bar, the next session (with
  Colab/GPU access, **after** Fix 2.1's re-cache lands) should run:
  ```
  # load 8 batches from the (re-cached) real data, print per-model post-align mean/std:
  for (acts, meta), _ in itertools.islice(dataloader, 8):
      for name, x in acts.items():
          aligned = spatial_aligner.align(x, source=name) if ... else x
          print(name, aligned.mean().item(), aligned.std().item())
  ```
  Success criteria: both models' post-align std in `[0.95, 1.05]`, `|mean| < 0.05` — **REQUIRES
  EXTERNAL EXPERIMENT** (needs the real, re-cached data).
  ```
  # After Fix 2.1's re-cache lands, load 8 real batches from uni_demo.py's actual dataloader and
  # print per-model post-align mean/std over the channel axis, matching REPAIR_PLAN.md's Fix 2.2
  # verification step exactly.
  ```
- As the subagent review noted, this fix's stats are computed over all 15 cached PixArt timesteps
  pooled together (via `_flatten_tokens_for_stats`), not the single t=10 slice training actually
  uses — correct once Fix 2.1's single-timestep re-cache lands (T=1, so "all timesteps" and "the one
  timestep used" become the same thing), not fully correct against the *current* 15-timestep cache.
  This is the documented Fix 2.1 dependency, not a new gap.
- Per the plan's own risk note: this changes the loss scale for both `loss_DinoV2_to_PixArt` and
  `loss_PixArt_to_DinoV2` — numbers from any future run are not comparable to pre-Fix-2.2 runs.
  Should be encoded in the run tag when the next real training run happens.

---

## Fix 2.3 — real train/val split + held-out evaluation pass

**Addresses:** V7 (HIGH)
**Commit:** `d1990c6`
**Files modified:** `coco_dataset_setup.py`, `data.py`, `train.py`, `uni_demo.py`

### Issue

No train/val split existed anywhere in the repo. `uni_demo.py` built one `CocoActivationDataset`
over all 2000 cached stems with a single shuffled `DataLoader`; every diagnostic pointed at the same
`/content/combined_cache`. The `viz: { set: val }` block in `config.yaml` was dead config — nothing
read it. This meant every number, every heatmap, every top-activating-image, and **the
`fixed_timestep_idx: 10` selection itself** (decided by evaluating a trained checkpoint against its
own training data) were all measured on data the model had already seen. Nothing produced by this
repo was a generalization measurement.

### Root cause

The dataset/training loop was built for a single dataset object from the start; a split was never
added as the project grew, and no diagnostic script was ever pointed at anything but the full cache.

### Changes

- **`coco_dataset_setup.py`**: new `split_train_val(identifiers, save_dir, val_fraction=0.2,
  seed=42, ...)`, following the same idempotent-persist pattern as the existing `select_images()` —
  first call shuffles (fixed seed) and writes `train_stems.txt`/`val_stems.txt` next to
  `selected_images.txt`; later calls reload rather than re-split, so every run, resume, and (once
  updated) diagnostic sees an identical split. **Added a validation the initial implementation
  lacked** (caught by independent review, not by my own first pass): on reload, the persisted
  train∪val stem set is compared against the *current* identifier list; a mismatch raises a
  `RuntimeError` naming exactly how many stems are missing/new and instructing how to regenerate,
  rather than silently training on a stale, skewed split if the cache is ever regenerated with a
  different image set (e.g. after Fix 2.1's re-cache, if that ever changes which images are cached
  rather than just how PixArt's activations are extracted for the same images).
- **`data.py`**: `CocoActivationDataset.__init__` gained an optional `allowed_stems` parameter,
  applied to `self.stems` immediately after cache discovery and strictly *before* the
  standardization-stats block — so a train-only dataset's `_compute_standardization_stats()` never
  sees a val image, and a val-only dataset (see below) never computes its own stats at all.
- **`train.py`**: new `evaluate_universal_sae()` — a `@torch.no_grad()` pass computing self+cross MSE
  for every (source, target) pair the model can structurally reconstruct, reusing
  `train_universal_sae`'s own `_extract_source_slice`/`_extract_target_slice`/`_get_sigmas_bt`/
  `mse_flat`/`_pool_target_for_loss` helpers, with no backward pass, no optimizer step, no
  dead-feature resampling, no curriculum gating, and no latent-alignment loss. Unlike training's
  `_pick_source` (one source per step, to amortize one gradient update), every source present in each
  batch is evaluated, since there's no update cost to amortize. Enters `model.eval()` and restores
  `model.train()` only if the model was already training (checked via `was_training = model.training`
  first). **Also applies the same `model.model_tokens` patch `train_universal_sae` applies for
  spatial-aligned cross-reconstruction** (added after independent review flagged that without it,
  the function would silently skip every cross-model pair if ever called standalone before any
  training epoch had run — idempotent, harmless to apply redundantly, removes the dependency on call
  order entirely rather than relying on "eval always happens to run after training each epoch").
  Returns `{"val/loss_<source>_to_<target>": mean, ..., "val/sae_loss": unweighted_mean,
  "val/n_batches": count}`.
- **`uni_demo.py`**: discovers the full cached stem list (mirroring `CocoActivationDataset`'s own
  anchor-suffix discovery), calls `split_train_val`, builds the main training dataset with
  `allowed_stems=train_stems`, and builds a **separate** `val_dataset`/`val_dataloader` with
  `allowed_stems=val_stems`, `standardization_stats=getattr(dataset, "standardization_stats", None)`
  (train's stats, reused verbatim — never recomputed on val) and the same `spatial_aligner`. Runs
  `evaluate_universal_sae` after every epoch's training call, logging `val/*` to wandb alongside the
  existing epoch-level metrics and printing a one-line summary.

**Explicitly out of scope for this pass**: "point every diagnostic at the val split by default"
(`dictionary_diagnostic.py`, `cross_model_overlap.py`, `top_activating_images.py`,
`input_rank_diagnostic.py`, `pixart_timestep_autopsy.py`, etc.) — `REPAIR_PLAN.md`'s Fix 2.3 file
list names only `coco_dataset_setup.py`, `uni_demo.py`, `data.py`, and updating ~10 diagnostic
scripts (each with its own CLI/dataset-construction conventions, several already touched in Fix
0.2/0.3) is a materially larger, separately-scoped change. The mechanism is now in place and generic
(`allowed_stems=` on any `CocoActivationDataset`, `val_stems.txt` on disk) — pointing a given
diagnostic at val is now a small, mechanical follow-up per script rather than something requiring
new infrastructure.

### Why necessary

This is Stage 2 / Fix 2.3 in `REPAIR_PLAN.md`. It's explicitly the fix the plan flags as making
"the headline numbers look worse — that is the point": a checkpoint that reconstructs its own
training images well says nothing about whether the shared dictionary generalizes, and the
`fixed_timestep_idx: 10` decision itself was never validated against held-out data.

### Verification performed

Same scratch CPU venv as prior fixes. `test_fix23.py` built a synthetic 200-image combined-npz cache
(DinoV2 (8,6) + PixArt (15,8,5), matching the real schema) and exercised the REAL
`split_train_val`/`CocoActivationDataset`/`evaluate_universal_sae`:
- **Split correctness**: 80/20 sizes exactly (160/40 for 200 images), zero stem overlap between
  splits, confirmed both at the raw split-function level and again after being threaded through two
  separate `CocoActivationDataset` instances (`set(train_ds.stems) & set(val_ds.stems) == set()`).
- **No leakage**: `val_dataset` constructed with `standardization_stats=train_ds.standardization_stats`
  reused those stats **verbatim** (`torch.equal` on both mean and std tensors) rather than
  recomputing — confirmed by inspecting the log output too (`"[stats] Using standardization
  statistics persisted in the checkpoint."` printed for val, `"[stats] Computing standardisation
  stats..."` printed only for train).
- **`evaluate_universal_sae` correctness**: ran it against a tiny real `UniversalSAE` mid-"training"
  (`model.train()` called first) and confirmed (a) every model parameter is bit-for-bit unchanged
  before vs. after the call (`torch.equal` on every parameter), (b) no parameter accumulated a
  gradient, (c) `model.training` is restored to `True` afterward, (d) by monkeypatching
  `model.eval`/`model.train` to record calls, confirmed both are actually invoked (not just claimed
  in a docstring) — `model.eval()` at least once, `model.train()` at least once to restore state, (e)
  all returned loss values (including cross-model pairs, confirming `can_cross_reconstruct`
  structurally worked) are finite, and `val/sae_loss` exactly equals the mean of the individual pair
  losses.
- **New split/cache mismatch validation**: confirmed a changed identifier set (simulating a re-cache
  that adds/removes images) correctly raises `RuntimeError` naming the mismatch, while an unchanged
  identifier set with only a different `seed` argument correctly reloads the persisted split
  unaffected (seed is irrelevant once a split is persisted — this is intentional idempotency, not a
  bug: re-running with a "different" seed must NOT silently reshuffle an already-committed split).

**Differential check against the pre-fix code** (`git stash`/`git stash pop`): re-ran the identical
`test_fix23.py` unmodified against the stashed pre-fix code. Failed immediately at import
(`ImportError: cannot import name 'split_train_val' from 'coco_dataset_setup'`) — confirms the
entire mechanism was absent pre-fix. Restored and re-confirmed all checks pass, then re-ran
`test_fix11.py`/`test_fix12.py`/`test_fix22.py` as a regression check on `train.py`'s other recent
changes (Fix 1.1/1.2/2.2) — all still pass, confirming `evaluate_universal_sae`'s addition didn't
disturb `train_universal_sae`.

**Independent review:** dispatched a fresh-context subagent to review the diff against
`REPAIR_PLAN.md`'s V7 write-up and Fix 2.3's spec. It traced the `__init__` order in `data.py` to
confirm the `allowed_stems` filter genuinely runs before stats computation; confirmed `val_dataset`'s
construction always takes `_validate_standardization_stats`'s reuse path, never
`_compute_standardization_stats`'s fresh-computation path; confirmed `evaluate_universal_sae` touches
neither `model._train_loss_ema` nor any `model._usage_ema_*` attribute (both are written only inside
`train_universal_sae`'s own loop); confirmed via `git diff --stat` that no diagnostic script was
touched; confirmed `val_dataset` does pass `spatial_aligner=spatial_aligner`. It found: (1) **Medium**
— the split-validation gap described above, fixed in this same commit and re-verified; (2) **Low,
plausible** — the `model_tokens` patch call-order dependency, fixed in this same commit (the
redundant patch inside `evaluate_universal_sae` itself) and re-verified via the full test re-run;
(3) **Low** — `uni_demo.py`'s stem-discovery logic duplicates (rather than shares) the identical
three-line formula already in `data.py`'s `__init__`; confirmed no actual drift risk under any
current config (both compute the exact same anchor suffix from the same `combined_npz`/`sources`
inputs), a maintainability note rather than a defect, left as-is to avoid refactoring
`CocoActivationDataset`'s discovery logic into a shared helper as an unrelated change.

### Remaining uncertainty — DEFERRED / REQUIRES EXTERNAL EXPERIMENT

- **Not verified against the real cache or a real Colab training run** (same gap as every prior fix).
  The synthetic test proves the split/filtering/no-leakage/eval-pass mechanism is correct; it doesn't
  produce the actual `val/loss_PixArt_to_DinoV2` number for the real project. Per the plan's own
  verification bar for this fix ("assert zero stem overlap between the two splits; assert the
  persisted stats hash matches a train-only recomputation"), the next session (with Colab/GPU access)
  should, after running `uni_demo.py` once to generate the real split and checkpoint:
  ```python
  # From the repo root, after a real uni_demo.py run has created
  # <path_to_cache>/train_stems.txt and val_stems.txt:
  train_stems = open(f"{cache_root}/train_stems.txt").read().split()
  val_stems = open(f"{cache_root}/val_stems.txt").read().split()
  assert not (set(train_stems) & set(val_stems))
  # Recompute stats train-only from scratch and diff against the checkpoint's
  # persisted standardization_stats to confirm they match exactly (not just
  # "close" -- this dataset's stats computation is deterministic given stats_seed):
  from data import CocoActivationDataset
  recomputed = CocoActivationDataset(
      cache_root=cache_root, sources=sources, combined_npz=True, standardize=True,
      allowed_stems=train_stems, stats_seed=<same stats_seed as the real run>,
  ).standardization_stats
  ckpt = torch.load(checkpoint_path, weights_only=False)
  for source in recomputed:
      assert torch.allclose(recomputed[source]["mean"], ckpt["standardization_stats"][source]["mean"])
      assert torch.allclose(recomputed[source]["std"], ckpt["standardization_stats"][source]["std"])
  ```
  Success criteria: both asserts pass (zero overlap; stats hash-equivalent to a from-scratch
  train-only recomputation), and `val/loss_*` metrics appear in the real wandb run.
- **Pointing existing diagnostics at the val split is deliberately unstarted** (see Changes above) —
  tracked here as follow-up, not forgotten. `dictionary_diagnostic.py`, `cross_model_overlap.py`,
  `top_activating_images.py`, `input_rank_diagnostic.py`, and `pixart_timestep_autopsy.py` are the
  concrete candidates; each needs `allowed_stems=<val_stems from disk>` threaded into its own
  `CocoActivationDataset` construction, following the exact pattern this fix established.
  Re-deriving `fixed_timestep_idx` itself on held-out data (H1 in `REPAIR_PLAN.md`) depends on this.
- **`evaluate_universal_sae` runs every epoch unconditionally** — for the real 2000-image dataset (400
  val images) this is a full extra forward pass per epoch on top of training; not expected to be
  expensive relative to training itself, but not measured against real data/timing in this session.

---

## Fix 3.2 — unify feature-usage definitions, add per-token co-fire metric

**Addresses:** V16 (last bullet, MEDIUM), Stage 3's "add the metric that measures the actual goal"
**Commit:** `363a1b0`
**Files modified:** `train.py`, `dictionary_diagnostic.py`, `cross_model_overlap.py`
**Files added:** `feature_usage.py`

### Issue

Two related problems: (1) "does this model use dictionary feature k" was answered three
incompatible ways across the codebase — `train.py`'s partition diagnostic used a usage-EMA rate
threshold (continuous, whole-training-run), `dictionary_diagnostic.py` used "fired on ≥1 token of
≥1 image in a fixed sample" (ever/never), `cross_model_overlap.py` used "in the per-image top-K by
max|activation|, for ≥1 image" (top-K membership) — so a "used" count from one script was never
comparable to another's, and `dictionary_diagnostic.py` comparing wandb's `partition/score` to its
own `partition_score` was comparing incommensurable quantities. (2) Every existing cross-model
metric (`partition/usage_cosine`, `dictionary_diagnostic.py`'s per-image Jaccard/cosine,
`cross_model_overlap.py`'s top-K Jaccard) aggregates over the token axis *before* comparing the two
models — but the project's stated goal ("the same dictionary features fire on the same content
across models") is explicitly a per-token/per-position claim, which none of the existing metrics can
detect even in principle.

### Root cause

Each script's "used"/"overlap" logic was written independently as each was created, with no shared
reference implementation — the same pattern of drift already seen and partially fixed in V12 (eval
scripts reading live config instead of the checkpoint) and V8 (stale hook-target assumptions).

### Changes

**New `feature_usage.py`** — three functions, all pure-tensor, no dependency on any other module in
this repo (checked: only imports `typing.Optional` and `torch`, so no circular-import risk with
`train.py`/`dictionary_diagnostic.py`/`cross_model_overlap.py` importing it):
- `compute_feature_usage(activations, criterion, threshold=1e-3, top_k=None) -> (K,) bool`: one
  function, three criteria (`"rate_above_threshold"`, `"ever_fired"`, `"top_k_per_sample"`), each
  reproducing exactly one of the three scripts' historical definitions. Calling it with the same
  criterion from two different scripts is now guaranteed identical (one implementation), instead of
  two independent reimplementations that could silently drift.
- `per_token_cofire_jaccard(z_a, z_b) -> scalar`: the actual new per-token metric. Given two
  ALREADY spatially-aligned `(B, N, K)` latent codes (meaningless otherwise — REPAIR_PLAN.md V3),
  computes Jaccard(active-feature-set) at each `(batch, token)` position independently, then
  averages — as opposed to every prior metric's per-image or per-corpus aggregate.
- `feature_heatmap_iou(z_a, z_b, feature_idx, threshold=0.0) -> float`: IoU of one feature's
  active-token mask between two aligned models across a batch. **Implemented but not wired into any
  call site in this commit** — the plan's Fix 3.2 wording is "per-token co-fire and/or heatmap IoU,"
  and per-token co-fire alone satisfies that bar. Left available for a future diagnostic (e.g. a
  per-feature heatmap-IoU histogram in `cross_model_overlap.py`, analogous to its existing per-image
  Jaccard histogram) without needing new infrastructure.

**`train.py`**:
- `evaluate_universal_sae` (Fix 2.3's held-out eval pass) now also computes
  `per_token_cofire_jaccard` for every pair of models present in each val batch — gated on
  `spatial_aligner is not None` (per-token comparison is meaningless without a common grid), so this
  metric is computed on the **val split**, per the plan's explicit instruction. Each source's `z`
  (already computed for the reconstruction-loss part of the function) is stashed in a per-batch
  dict; after the source/target loop, every unordered pair is compared via
  `per_token_cofire_jaccard`, returned as `val/cofire_jaccard_<A>_vs_<B>`. A defensive shape-mismatch
  guard (should never fire under the current pipeline, since `spatial_aligner` forces every source to
  the same `target_n_tokens` and all sources in one batch share the same `B`) now **prints a warning**
  before skipping, rather than silently continuing — added after independent review flagged that the
  original silent `continue` would hide a real bug if this guard ever actually fired.
- The pre-existing partition-diagnostic "used" computation
  (`e > resample_dead_threshold`, from Fix 0.4) now calls
  `compute_feature_usage(e, criterion="rate_above_threshold", threshold=resample_dead_threshold)` —
  behaviorally identical, but the actual boolean-threshold logic now lives in one shared place.

**`dictionary_diagnostic.py`**: two inline reimplementations replaced with calls to the shared
function — the "did feature k fire on any token of this image" check (`(z_d != 0).any(dim=(0,1))` →
`compute_feature_usage(z_d, criterion="ever_fired")`) and the per-image top-K selection
(`torch.topk(scores_d, k=...)` → `compute_feature_usage(scores_d.unsqueeze(0),
criterion="top_k_per_sample", top_k=...)` then `.nonzero()` to recover indices — both call sites'
results are immediately converted to a `set()`, so the change from score-ordered to index-ordered
output is inconsequential).

**`cross_model_overlap.py`**: `top_feature_set`'s per-image top-K selection changed the same way.

### Why necessary

This is Stage 3's "add the metric that measures the actual goal" in `REPAIR_PLAN.md`
(`PROJECT_STATUS.md` Phase 2) plus V16's last bullet. The existing aggregate-co-usage metrics can
report a healthy-looking `usage_cosine` or partition score while the actual, stated research goal
(per-token semantic correspondence) is completely absent — this gives the project at least one
metric that can't be fooled that way, once real (post-Fix-2.1) data exists to run it on.

### Verification performed

Same scratch CPU venv as prior fixes. Two test scripts against the REAL `feature_usage.py`/
`train.py`:
- **`test_fix32.py`**: unit-level checks on all three `compute_feature_usage` criteria (including a
  direct equivalence check against the literal old formulas — `torch.equal` against
  `(z != 0).any(dim=(0,1))` for `ever_fired`, and a set-equality check against per-row `torch.topk`
  unions for `top_k_per_sample` — not just "looks plausible," byte/set-identical to what the code
  used to compute), plus error-path checks (wrong input rank, missing `top_k`, unknown criterion).
  `per_token_cofire_jaccard`: identical inputs → exactly 1.0; fully disjoint active sets → exactly
  0.0; a hand-computed partial-overlap case (`{0,1}` vs `{1,2}`) → exactly 1/3; both-empty active
  sets → 0.0, not NaN; shape mismatch raises. `feature_heatmap_iou`: a hand-computed
  partially-overlapping case → exactly 0.5; a feature that never fires for either model → 0.0, not a
  division error.
- **`test_fix32_eval_wiring.py`**: exercised the REAL `evaluate_universal_sae` end-to-end. Confirmed
  no `cofire_jaccard` key appears when `spatial_aligner=None`; confirmed exactly one
  `val/cofire_jaccard_DinoV2_vs_PixArt` key appears (correctly alphabetized) when a real
  `SpatialAligner` is passed, with a finite value in `[0, 1]`. **Strongest check**: built a second
  model where PixArt's `W_enc`/`b_pre` were copied from DinoV2's, fed the *identical* input tensor to
  both "models," and confirmed the resulting `cofire_jaccard` is **exactly 1.0** — a genuine
  ground-truth sanity check (identical encoder + identical input must produce identical firing
  patterns at every token), not just "some plausible-looking number."

**Differential check against the pre-fix code** (`git stash -u`/`git stash pop`, since
`feature_usage.py` is a new untracked file that needed explicit inclusion): re-ran both test scripts
unmodified against the stashed pre-fix code. `test_fix32.py` failed immediately at import
(`ModuleNotFoundError: No module named 'feature_usage'`); `test_fix32_eval_wiring.py`'s Check A
(no-aligner case) still passed trivially (nothing to break there), but Check B failed exactly as
predicted (`val/cofire_jaccard_DinoV2_vs_PixArt` key genuinely absent pre-fix). Restored and
re-confirmed both pass, then re-ran `test_fix11.py`/`test_fix12.py`/`test_fix22.py`/`test_fix23.py`
as a full regression check on `train.py`'s accumulated changes across this session — all still pass.

**Independent review:** dispatched a fresh-context subagent to review the diff plus the new file
against V16's last bullet and Fix 3.2's spec. It independently re-verified the `ever_fired` and
`top_k_per_sample` substitutions are mathematically/set-equivalent to the exact original formulas at
the exact shapes each call site actually uses (not just "similar looking"); confirmed `zs_by_source`
is populated on both the diffusion and non-diffusion branches of the per-source loop; confirmed the
pairwise-enumeration pattern is correct and would generalize cleanly to 3+ models without further
code changes; confirmed no circular import; confirmed the `dictionary_diagnostic.py`/
`cross_model_overlap.py` edits are surgical (only the described lines). It flagged one **medium**
finding — the shape-mismatch guard silently `continue`d instead of logging, which would hide a real
bug if it ever actually fired (effectively dead code today, since `spatial_aligner` and shared-batch
`B` make the mismatch condition unreachable under the current pipeline) — fixed in this same commit
(added a `print` warning before the `continue`) and re-verified with both test scripts re-run
afterward. One informational note: `feature_heatmap_iou` is implemented but genuinely unused/unwired
anywhere in this commit — noted above as a deliberate scope choice (per-token co-fire alone satisfies
Fix 3.2's "and/or" requirement), not an oversight.

### Remaining uncertainty — DEFERRED / REQUIRES EXTERNAL EXPERIMENT

- **Not verified against the real cache/checkpoint or a real Colab training run** (same gap as every
  prior fix). All verification used synthetic tensors and toy models; the actual
  `val/cofire_jaccard_DinoV2_vs_PixArt` VALUE for the real project — and whether it tells a different
  story than `partition/usage_cosine`'s "0.857, looks fine" reading the plan already flags as
  unsupported — is unmeasured. Per `REPAIR_PLAN.md`'s own framing (this metric is "only meaningful
  after Fix 2.1" since per-token comparison requires the pixel-correspondence bug, V3, to be fixed
  first), this is explicitly a **REQUIRES EXTERNAL EXPERIMENT** item gated on Fix 2.1's actual
  re-cache having run:
  ```
  # After a real Fix 2.1 re-cache and at least one training epoch, in the Colab/GPU
  # environment: val/cofire_jaccard_DinoV2_vs_PixArt is now logged to wandb automatically
  # by uni_demo.py's existing epoch loop (Fix 2.3) -- no separate script needed. Compare
  # its trend across epochs against partition/usage_cosine_DinoV2_vs_PixArt's trend.
  ```
  Expected result / success criteria: a finite value in `[0, 1]` every epoch; whether it's
  meaningfully lower than `usage_cosine` (confirming the aggregate metric was overstating shared
  structure, per the plan's H1-adjacent concern) is the actual research question this metric exists
  to answer, not something to predict in advance.
- **`feature_heatmap_iou` is unwired** (see above) — a natural follow-up is a per-feature heatmap-IoU
  histogram in `cross_model_overlap.py`, alongside its existing per-image Jaccard histogram, for
  whichever features `compute_feature_usage(criterion="rate_above_threshold")` marks as used by both
  models on the real (post-Fix-2.1) checkpoint.
- **This is the last item in `REPAIR_PLAN.md`'s ordered repair plan that doesn't require GPU/real
  data to implement.** Fix 3.1 ("re-run the diagnosis on repaired inputs") is explicitly a
  re-measurement step with no code to write — it *is* the external experiment, gated on Fixes 2.1-2.3
  having actually been run for real. All Stage 0-3 code changes REPAIR_PLAN.md calls for are now
  complete on this branch.

---

## Review pass — correctness audit of Fixes 0.1–3.2, plus 8 follow-up fixes

**Addresses:** findings from a full re-read of `REPAIR_PLAN.md` + `RECOVERY_LOG.md` against the
actual code (`git diff 03b80ff..HEAD`), at the user's request to check that the changes were
correct and helpful.
**Files modified:** `DiffusionActivationExtractor.py`, `cache_coco_diffusion_activations.py`,
`train.py`, `feature_usage.py`, `uni_demo.py`, `config.yaml`, `dictionary_diagnostic.py`,
`dictionary_diagnostic_all_timesteps.py`, `cross_model_overlap.py`, `input_rank_diagnostic.py`,
`visualize_feature_activations.py`, `PROJECT_STATUS.md`, `RECOVERY_LOG.md`

### What the audit confirmed

Every prescribed change in Fixes 0.1–3.2 is present and does what the log claims. Verified by
tracing the mechanism, not by confirming a diff exists — including: `load_checkpoint` keeps its
2-tuple return on **both** branches (the early-return refactor doesn't reintroduce the bug the
Fix 0.1 addendum fixed); `ema = ema_attrs[model_name]; ema[dead_idx] = ...` genuinely aliases the
live `model._usage_ema_*` tensor; the single-timestep sigma formula is byte-identical to
`_get_ddim_sigmas`; `__main__` builds the extractor with `num_inference_steps=15`, so
`resolve_pixart_raw_timestep` maps cache index 10 to the *same* raw `t` the old cache's index 10
held (the likeliest place for silent drift, and it doesn't drift); `evaluate_universal_sae`'s
source/target/pool/MSE path mirrors the training path line-for-line, so `val/loss_*` really is
comparable to `train/loss_*`; and Fix 3.2's `top_k_per_sample` substitutions are exactly
equivalent (`.abs()` is a no-op on `z.abs().amax(...)` scores).

### Problems found and fixed

1. **A materially false claim in Fix 2.1, repeated by the self-audit.** The log said the
   `single_timestep=None` path was "unchanged byte-for-byte... purely additive," verified by
   diffing the loop bodies. The loop bodies *are* identical, but they call
   `_get_transformer_input`, which the same fix changed for both branches. Corrected in place at
   both sites (see the CORRECTION blocks in Fix 2.1 and the self-audit), and noted in code at
   `USE_DIFT_SINGLE_TIMESTEP` and in `extract_activations`'s docstring. This is the same class of
   error the self-audit was chartered to catch, occurring inside the self-audit's own output — the
   audit inherited the earlier entry's framing instead of re-deriving the unit of comparison.
2. **Fix 1.1's end-to-end differential result is confounded by Fix 1.1's own change #5.** Both
   reported signals are mechanical outputs of the usage-EMA reset at `resample_interval=4`.
   Downgraded to a hypothesis in place, with a description of what a non-confounded run needs. No
   code change — the fix is fine, the evidence for it wasn't.
3. **T5 self-attention was still unmasked.** Fix 2.1 threaded `attention_mask` to the
   transformer's cross-attention but left `self.text_encoder(text_inputs_1)` unmasked, so T5's own
   self-attention still ran over 255 pad positions. Now passes `attention_mask=attention_mask_1`,
   matching diffusers' `PixArtAlphaPipeline.encode_prompt`. Low impact (the prompt is `""` for
   every image, so the embedding is a dataset-wide constant either way) but half a fix is worse
   than a documented gap.
4. **`grad_clip_norm: 1.0` was invisible exactly where it matters.** `train/grad_norm_preclip` is
   logged only inside the wandb-gated block, and the plan's own tiny-scale verification runs
   specify `use_wandb=false`. Since 1.0 is a starting value rather than a measured one, a real
   pre-clip norm 10–100× above it would act as a large silent LR reduction across the whole run.
   Added a once-per-process console warning when the pre-clip norm exceeds 10× the threshold,
   printing the actual rescale factor. Value left at 1.0 per the plan; **read the warning/metric
   before trusting it.**
5. **Val dataloader ran single-threaded** (`num_workers=0`) over 400 images every epoch, against a
   cache where every read is a full zlib inflate of a ~35MB array (V13) — adding wall-clock to the
   exact bottleneck Stage 2 exists to remove. Now matches the train worker count.
6. **`per_token_cofire_jaccard` shipped without a chance baseline.** Hard TopK makes the raw value
   small by construction (~0.0052 at `top_k=128`/`K=12288`), so `0.02` is 4× chance but reads as
   "basically nothing." Added `per_token_cofire_jaccard_chance` (computed from *observed*
   per-token active counts, so it stays correct under any selection rule) and a `..._lift` ratio,
   both logged per pair. The lift is the figure to read.
7. **Fix 2.2's `spatial_aligner=` was only wired in `uni_demo.py`.** Harmless today because Fix 0.2
   makes every eval script use the checkpoint's persisted stats — but any fallback to
   recomputation (older checkpoint, missing stats) would silently fit stats to the unpooled
   distribution, i.e. reintroduce V6 for that run. Wired into all five eval scripts that build
   both an aligner and a dataset. Pure safety net: zero behavior change whenever checkpoint stats
   exist.
8. **V2(e) was unaddressed and unmentioned.** `resample_interval: 500` is even and `_pick_source`
   alternates on `global_step % 2`, so *every* resample event landed on a DinoV2-source step for
   the entire run. `resample_interval` (and `resample_start_step`) 500 → **501**, so event parity
   alternates. This is a real behavior change — events now fire at 501/1002/1503/… — and should be
   noted in the next `run_tag`.
9. **`_ensure_bt` corrupts the shape of a T=1 cache — a real bug, not in `REPAIR_PLAN.md`, and
   armed specifically by Fix 2.1.** `values.squeeze()` with no argument drops *every* size-1 dim.
   Harmless on the 15-timestep cache, but on the `(1, 1024, 1152)` cache Fix 2.1 produces the
   collated `(B, 1)` timestep tensor collapses to `(B,)` and is then re-expanded to **`(B, B)`**,
   and at `B == 1` it collapses to 0-d and raises `ValueError`. Two consequences:
   - **Latent crash:** any run where `len(dataset) % batch_size == 1` dies on the last batch once
     the DIFT cache lands. Today's numbers (1600 train / 400 val, batch 16) happen to divide
     evenly, so it would not have fired immediately — it would have fired the first time anyone
     changed the image count, batch size, or `val_fraction`.
   - **Silent wrong-shape:** at `B > 1` the `(B, B)` tensor made `t_bt[:, idx]` return *image 0's*
     timestep for every image. That is numerically invisible today only because every image in a
     single-timestep cache shares the same raw `t` — a coincidence, not a guarantee.
   Rewritten to normalize without a blanket squeeze. Verified across all six real `(B, T)` shapes
   plus the three defensive shapes (bare `(T,)`, `(1, T)`, 0-d scalar) with no regressions, and
   confirmed that distinct per-image timesteps now survive (`[100,200,300,400]` in, same out —
   the old version returned `[100,100,100,100]`).
10. **`apply_topk`'s docstring claimed a straight-through estimator that does not exist.**
   `REPAIR_PLAN.md` §2 identified this ("The docstring calls it a 'straight-through trick'; it is
   not") but no Fix had it in scope, so it survived the whole repair. `mask` comes from
   `torch.zeros_like` and carries no grad, so `d(out)/d(z_pre)` is exactly `mask` — gradient
   reaches selected indices only and is exactly zero elsewhere, which is the opposite of what
   straight-through means. Docstring and inline comment corrected, with the reason
   `pre_topk_align_weight` exists spelled out (it is the only learning signal unselected features
   get). Comment-only; zero behavior change.
11. **`PROJECT_STATUS.md` had never been updated**, despite `REPAIR_PLAN.md` §7 calling for it once
   Stage 0 landed. It was still presenting the flat-heatmap evidence (V1), the `used_by_none`
   "puzzle" (V2a), the `usage_cosine = 0.857` "looks fine" reading (V11), and the `0.948`
   calibration (V5/V6) as model findings. Rewritten: findings marked retracted/qualified inline
   rather than deleted (the observations were real, the attributions weren't), the phased plan
   marked superseded with current state and an explicit next-five-steps list, and the "run this
   next" block repointed from `pixart_timestep_autopsy.py` to Fix 0.1's render.

### Verification performed

No GPU/cache in this environment, same constraint as every prior entry. All modified files
`ast.parse` clean. Change 9 (`_ensure_bt`) was verified by reimplementing both the old and new
logic in numpy and running every shape the pipeline can produce — the old version reproduced the
`(B, B)` corruption and the `B=1` crash exactly as described, the new one returns `(B, T)` for all
of them and preserves per-image timestep values. Changes 3, 5, 7, 8 are one-to-few-line edits
whose correctness is structural
and was checked by reading the call sites (aligner built before dataset in all five scripts;
`attention_mask` is the documented kwarg for `T5EncoderModel.forward`; `num_workers` reads the
same config key train does; `_clamp` handles the new interval identically). Changes 4 and 6 are
additive: 4 is a print guarded by `getattr(model, "_warned_grad_clip", False)` and cannot raise
(`float(nan) > x` is `False`, so it won't fire on a NaN norm); 6 adds new dict keys and cannot
affect the loss, the existing keys, or any control flow, with the `chance_val > 0` guard
returning `nan` rather than dividing by zero.

### Remaining uncertainty

- **Unchanged and unchangeable here: nothing in this project has been validated against real
  data.** All nine fixes plus these eight follow-ups rest on synthetic CPU tests. The audit raises
  confidence that the *code* is right; it says nothing about the research question.
- **Change 8 (`resample_interval: 501`) is untested at any scale** — it is a config value, not a
  code path, and the parity argument is arithmetic, but it does move every resample event.
- Change 4's 10× warning threshold is arbitrary. It exists to make the situation visible, not to
  decide it.
- V13's `mmap_mode` / `.copy()` amplifiers in `data.py` remain unfixed (out of every fix's file
  list so far, and largely mooted by Fix 2.1's re-cache — but only once that re-cache runs).
- **New failure mode introduced by Fix 2.2, left deliberately as a loud crash:** with
  `use_class_tokens: true`, `_apply_spatial_align_for_stats` now passes a 257-token DinoV2
  activation to `SpatialAligner.align`, which raises `ValueError: ... expected N=256 (grid 16x16),
  got N=257. If your activations include a CLS/register token, strip it before calling align()`.
  `config.yaml` sets `use_class_tokens: false`, so this cannot fire today. Not guarded, on
  purpose: per-token spatial alignment with a CLS token in the sequence is genuinely ill-defined,
  the message is self-explanatory and actionable, and silently skipping alignment there would
  reintroduce V6 for that run.
- Change 9 fixes the shape corruption; it does **not** mean the T=1 path is validated end to end.
  That still needs the real 16-image re-cache from Fix 2.1's own verification ordering.

---
