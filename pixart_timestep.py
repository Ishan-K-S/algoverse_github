"""
One place to decide which PixArt timestep everything looks at.

Training pins the timestep through global.fixed_timestep_idx in config.yaml,
but every eval script used to hardcode "last timestep" on its own. That drift
is how we scored a run at t=14 while judging a config that meant something
else, so the resolution order here always starts from the checkpoint the run
actually produced.

Order of precedence:
  1. an explicit override (a CLI flag someone passed on purpose)
  2. fixed_timestep_idx saved inside the checkpoint
  3. fixed_timestep_idx from config.yaml
  4. the last timestep, which is the old behaviour

Negative indices count back from the end, so -1 is the last timestep.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _clamp(idx: int, total_steps: int) -> int:
    if idx < 0:
        idx = total_steps + idx
    return max(0, min(idx, total_steps - 1))


def _training_global(ckpt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(ckpt, dict):
        return {}
    cfg = ckpt.get("config")
    if not isinstance(cfg, dict):
        return {}
    g = cfg.get("global")
    return g if isinstance(g, dict) else {}


def resolve_pixart_timestep(
    total_steps: int,
    ckpt: Optional[Dict[str, Any]] = None,
    config_global: Optional[Dict[str, Any]] = None,
    override: Optional[int] = None,
) -> int:
    """
    Return the timestep index to slice out of a (T, N, D) PixArt activation.

    `ckpt` is the raw torch.load dict, `config_global` is the "global" block of
    config.yaml. Both are optional; pass whatever the caller happens to have.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")

    if override is not None:
        return _clamp(int(override), total_steps)

    saved = _training_global(ckpt).get("fixed_timestep_idx")
    if saved is not None:
        return _clamp(int(saved), total_steps)

    if config_global is not None:
        from_yaml = config_global.get("fixed_timestep_idx")
        if from_yaml is not None:
            return _clamp(int(from_yaml), total_steps)

    return total_steps - 1
