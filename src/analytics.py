"""Analytical references and convergence estimators (paper Secs. IV-V).

tmm_reflectance      : transfer-matrix reflectance of a lossless dielectric
                       stack at normal incidence (reference for Case B2).
richardson_extrapolate: resolution-convergence estimate for a scalar metric
                       (e.g., f_res) from two grid refinements.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np


def tmm_reflectance(freqs: np.ndarray, layers: List[Dict[str, float]],
                    n_in: float = 1.0, n_out: float = 1.0) -> np.ndarray:
    """Reflectance R(f) of a planar stack at normal incidence.

    freqs are in Meep units (c=1), layers = [{"eps": e, "thickness": d}, ...].
    """
    freqs = np.asarray(freqs, dtype=float)
    R = np.empty_like(freqs)
    for i, f in enumerate(freqs):
        k0 = 2 * np.pi * f
        M = np.eye(2, dtype=complex)
        for layer in layers:
            n = np.sqrt(float(layer["eps"]))
            d = float(layer["thickness"])
            delta = k0 * n * d
            m = np.array([[np.cos(delta), 1j * np.sin(delta) / n],
                          [1j * n * np.sin(delta), np.cos(delta)]])
            M = M @ m
        num = (M[0, 0] + M[0, 1] * n_out) * n_in - (M[1, 0] + M[1, 1] * n_out)
        den = (M[0, 0] + M[0, 1] * n_out) * n_in + (M[1, 0] + M[1, 1] * n_out)
        r = num / den
        R[i] = np.abs(r) ** 2
    return R


def richardson_extrapolate(v_coarse: float, v_fine: float,
                           h_coarse: float, h_fine: float,
                           order: int = 2) -> Tuple[float, float]:
    """Richardson-extrapolated value and estimated error of the fine result.

    h is the grid spacing (1/resolution). Assumes v(h) = v* + C h^p.
    Returns (v_star, est_error_of_fine).
    """
    t = (h_coarse / h_fine) ** order
    v_star = (t * v_fine - v_coarse) / (t - 1.0)
    return float(v_star), float(abs(v_star - v_fine))


def spectral_l2_error(sim_vals: np.ndarray, ref_vals: np.ndarray) -> float:
    """Normalized L2 error between simulated and reference spectra."""
    sim_vals = np.asarray(sim_vals, float)
    ref_vals = np.asarray(ref_vals, float)
    denom = np.linalg.norm(ref_vals)
    return float(np.linalg.norm(sim_vals - ref_vals) / (denom if denom > 0 else 1.0))
