from __future__ import annotations
import os
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


def extract_f_res(freq: np.ndarray, flux: np.ndarray, qcfg: Dict[str, Any]) -> Tuple[float, float]:
    """Resonance proxy with band-edge masking.

    Gaussian-source flux vanishes at the band edges, so an unmasked argmin
    trivially locks onto the window edge (spurious). We exclude a margin
    (default 15%) on each side and support two modes:
      "dip"  : argmin |flux| (reflection dip, e.g. patch/stack)
      "peak" : argmax |flux| (radiated/resonant peak, e.g. cavity)
    """
    mode = qcfg.get("resonance_mode", "dip")
    margin = float(qcfg.get("band_margin", 0.15))
    n = len(freq)
    lo, hi = int(n * margin), n - int(n * margin)
    mag = np.abs(flux[lo:hi])
    idx = lo + int(np.argmax(mag) if mode == "peak" else np.argmin(mag))
    on_edge = idx in (lo, hi - 1)
    return float(freq[idx]), float(np.abs(flux[idx])), on_edge


def parse_outputs(cfg: Dict[str, Any], run_dir: str, log_path: str, spectrum_path: str, returncode: int) -> Dict[str, Any]:
    obs: Dict[str, Any] = {
        "run_dir": run_dir,
        "returncode": returncode,
        "status": "FAILED" if returncode != 0 else "OK",
        "failure_type": None,
        "evidence": [],
        "metrics": {}
    }

    # surface solver-log warnings as evidence (hints for the controller)
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("warning:", "error=")):
                    obs["evidence"].append(line.strip()[:300])

    # hard fail
    if returncode != 0 or (not os.path.exists(spectrum_path)):
        obs["failure_type"] = "HARD_FAIL"
        obs["evidence"].append("solver error or missing spectrum output")
        return obs

    df = pd.read_csv(spectrum_path)
    freq = df["freq"].to_numpy(dtype=float)
    flux = df["refl_flux"].to_numpy(dtype=float)

    # nan/inf
    bad = ~np.isfinite(flux)
    nan_ratio = float(bad.mean())
    obs["metrics"]["nan_ratio"] = nan_ratio
    if nan_ratio > float(cfg["quality"]["max_nan_ratio"]):
        obs["status"] = "FAILED"
        obs["failure_type"] = "QUALITY_FAIL_NAN"
        obs["evidence"].append(f"nan_ratio={nan_ratio:.3f}")
        return obs

    # resonance proxy: minimum magnitude point
    f_res, mag_at_res, on_edge = extract_f_res(freq, flux, cfg["quality"])
    obs["metrics"]["f_res"] = f_res
    obs["metrics"]["min_refl_mag"] = mag_at_res
    if on_edge:
        obs["evidence"].append(
            "warning: resonance proxy lies on the masked band edge; "
            "no interior spectral feature found - proxy may be invalid")

    # mark quality "uncertain" if we require confirmation via refinement
    if bool(cfg["quality"]["require_refine_confirmation"]):
        obs["status"] = "QUALITY_UNCERTAIN"
        obs["failure_type"] = "NEED_REFINE_CONFIRMATION"
        obs["evidence"].append("requires second run to confirm stability")
        return obs

    # otherwise pass
    obs["status"] = "PASS"
    return obs


def stability_check(cfg: Dict[str, Any], f_prev: float, f_now: float) -> Tuple[bool, float]:
    drift = abs(f_now - f_prev) / max(abs(f_prev), 1e-9)
    return drift < float(cfg["quality"]["drift_threshold"]), float(drift)
