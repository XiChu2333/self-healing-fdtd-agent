from __future__ import annotations
from typing import Dict, Any, List, Tuple

# Constrained action space (paper Sec. III-C).
# Bounds keep repairs physically meaningful and terminate the loop.
BOUNDS = {
    "resolution": (5, 80),
    "dpml": (0.5, 4.0),
    "sim_time": (50.0, 2000.0),
}

# Action schema exposed to the LLM controller (single source of truth).
ACTION_SCHEMA: Dict[str, Dict[str, Any]] = {
    "increase_resolution": {
        "param_key": "new_resolution", "sim_key": "resolution", "cast": int,
        "doc": "Raise grid density to reduce discretization/dispersion error.",
    },
    "increase_dpml": {
        "param_key": "new_dpml", "sim_key": "dpml", "cast": float,
        "doc": "Thicken absorbing boundary to suppress artificial reflections.",
    },
    "increase_sim_time": {
        "param_key": "new_sim_time", "sim_key": "sim_time", "cast": float,
        "doc": "Extend runtime so transients decay (needed for high-Q structures).",
    },
    "disable_refine_confirmation": {
        "param_key": None, "sim_key": None, "cast": None,
        "doc": "Skip the refinement-confirmation step (use sparingly).",
    },
}


def validate_action(cfg: Dict[str, Any], action: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Schema + bounds guardrail. Returns (sanitized_action, warnings).

    Raises ValueError only if the action type is unknown or the required
    parameter is missing/non-numeric (caller may then fall back).
    """
    warnings: List[str] = []
    t = action.get("type")
    if t not in ACTION_SCHEMA:
        raise ValueError(f"Unknown action type: {t!r}")
    spec = ACTION_SCHEMA[t]
    out = {"type": t, "reason": str(action.get("reason", ""))[:300]}
    if spec["param_key"] is None:
        return out, warnings

    raw = action.get(spec["param_key"])
    try:
        val = spec["cast"](raw)
    except (TypeError, ValueError):
        raise ValueError(f"Action {t}: missing/invalid {spec['param_key']}={raw!r}")

    lo, hi = BOUNDS[spec["sim_key"]]
    clamped = min(max(val, lo), hi)
    if clamped != val:
        warnings.append(f"{spec['param_key']} clamped {val} -> {clamped} (bounds {lo}..{hi})")
    cur = cfg["sim"].get(spec["sim_key"])
    if cur is not None and clamped <= spec["cast"](cur):
        warnings.append(f"{t} does not increase {spec['sim_key']} ({cur} -> {clamped})")
    out[spec["param_key"]] = clamped
    return out, warnings


def apply_action(cfg: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
    cfg2 = {**cfg, "structure": dict(cfg["structure"]), "sim": dict(cfg["sim"]), "quality": dict(cfg["quality"])}

    t = action.get("type")
    if t == "increase_resolution":
        cfg2["sim"]["resolution"] = int(action["new_resolution"])
    elif t == "increase_dpml":
        cfg2["sim"]["dpml"] = float(action["new_dpml"])
    elif t == "increase_sim_time":
        cfg2["sim"]["sim_time"] = float(action["new_sim_time"])
    elif t == "disable_refine_confirmation":
        cfg2["quality"]["require_refine_confirmation"] = False
    else:
        raise ValueError(f"Unknown action type: {t}")

    return cfg2
