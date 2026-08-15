from __future__ import annotations
import os
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass
class RunOutputs:
    run_dir: str
    log_path: str
    spectrum_path: str
    returncode: int


# ---------------------------------------------------------------- meep backend
def _run_meep_real(cfg: Dict[str, Any], run_dir: str, log_lines: list) -> None:
    import meep as mp
    from src.geometries import GEOMETRY_BUILDERS

    sim_p = cfg["sim"]
    if not sim_p.get("verbose_solver", False):
        try:
            mp.verbosity(0)  # silence meep's per-run banner output
        except Exception:
            pass
    kind = cfg["structure"].get("kind", "patch_2d")
    spec = GEOMETRY_BUILDERS[kind](cfg)

    simulation = mp.Simulation(
        cell_size=mp.Vector3(sim_p["cell_sx"], sim_p["cell_sy"], 0),
        boundary_layers=[mp.PML(sim_p["dpml"])],
        geometry=spec["geometry"],
        sources=spec["sources"],
        resolution=int(sim_p["resolution"]),
    )
    refl = simulation.add_flux(float(sim_p["fcen"]), float(sim_p["df"]),
                               int(sim_p["nfreq"]), mp.FluxRegion(**spec["flux_region"]))

    # physical diagnostics, sampled during the run:
    #  - structure probe: truncated ring-down (high-Q structures)
    #  - boundary probe (just inside the PML): late-time field persistence
    #    indicating under-absorbing boundaries
    probe_pt = spec.get("probe_pt", spec["flux_region"]["center"])
    bnd_pt = mp.Vector3(-0.5 * float(sim_p["cell_sx"]) + float(sim_p["dpml"]) + 0.1, 0)
    amps: list = []
    bnd_amps: list = []

    def _probe(sim):
        amps.append(abs(sim.get_field_point(mp.Ez, probe_pt)))
        bnd_amps.append(abs(sim.get_field_point(mp.Ez, bnd_pt)))

    T = float(sim_p["sim_time"])
    simulation.run(mp.at_every(T / 50.0, _probe), until=T)

    def _ratio(a):
        peak = max(a) if a else 0.0
        return (a[-1] / peak) if peak > 0 else 0.0

    decay_ratio = _ratio(amps)
    thr = float(cfg["quality"].get("max_decay_ratio", 1e-3))
    log_lines.append(f"decay_ratio={decay_ratio:.3e}\n")
    if decay_ratio > thr:
        log_lines.append(
            f"warning: fields not decayed at termination "
            f"(decay_ratio={decay_ratio:.3e} > {thr}); "
            f"spectrum may be corrupted by truncated ring-down\n")

    # late-window mean vs peak at the boundary probe: persistent late-time
    # energy next to the absorber points at PML reflection
    if len(bnd_amps) >= 10:
        peak_b = max(bnd_amps)
        late = sum(bnd_amps[-10:]) / 10.0
        bnd_ratio = (late / peak_b) if peak_b > 0 else 0.0
        log_lines.append(f"boundary_late_ratio={bnd_ratio:.3e}\n")
        if bnd_ratio > float(cfg["quality"].get("max_boundary_ratio", 5e-3)):
            log_lines.append(
                f"warning: field persists near the absorbing boundary "
                f"(boundary_late_ratio={bnd_ratio:.3e}); possible PML "
                f"reflection, consider thicker PML\n")

    freqs = np.array(mp.get_flux_freqs(refl), dtype=float)
    refl_flux = np.array(mp.get_fluxes(refl), dtype=float)
    pd.DataFrame({"freq": freqs, "refl_flux": refl_flux}).to_csv(
        os.path.join(run_dir, "spectrum.csv"), index=False)
    log_lines.append(f"geometry_kind={kind}\n")


# ---------------------------------------------------------------- mock backend
def _run_mock(cfg: Dict[str, Any], run_dir: str, log_lines: list) -> None:
    """Synthetic solver for development/testing without meep.

    Emulates the failure modes of each benchmark case so the full
    observe-diagnose-act loop (controllers, guardrails, batch statistics)
    can be exercised on any machine. NOT used for paper results.
    """
    sim_p = cfg["sim"]
    kind = cfg["structure"].get("kind", "patch_2d")
    res = float(sim_p["resolution"])
    dpml = float(sim_p["dpml"])
    sim_time = float(sim_p["sim_time"])
    rng = np.random.default_rng(int(cfg.get("mock_seed", 0)) + int(res) + int(dpml * 10))

    freqs = np.linspace(sim_p["fcen"] - sim_p["df"] / 2,
                        sim_p["fcen"] + sim_p["df"] / 2, int(sim_p["nfreq"]))

    # resolution-dependent resonance (first-order-like convergence error)
    f_true = float(cfg["structure"].get("mock_f_true", sim_p["fcen"]))
    f_eff = f_true + 0.35 / res  # drift shrinks as resolution grows
    width = 0.05
    mode = cfg.get("quality", {}).get("resonance_mode", "dip")

    # kind-specific failure effects act on the apparent extremum location
    if kind == "resonator_2d":
        t_needed = float(cfg["structure"].get("mock_decay_time", 600.0))
        if sim_time < t_needed:
            # truncated ring-down -> apparent peak jitters across trials;
            # resolution refinement alone cannot stabilize it
            severity = t_needed / sim_time - 1.0
            f_eff += rng.normal(0.0, 0.03 * severity)
            log_lines.append("warning: fields not decayed at termination\n")
    elif kind == "cavity_near_pml":
        d_needed = float(cfg["structure"].get("mock_dpml_needed", 1.5))
        if dpml < d_needed:
            # boundary-reflection interference -> apparent peak shifts
            # deterministically with (resolution, dpml); only a thicker
            # PML removes the shift
            f_eff += 0.035 * np.sin(res * 1.3 + dpml * 7.0)
            log_lines.append(
                "warning: field persists near the absorbing boundary "
                "(mock); possible PML reflection, consider thicker PML\n")

    if mode == "peak":
        flux = 0.15 + 0.9 * np.exp(-((freqs - f_eff) / width) ** 2)
    else:
        flux = 1.0 - 0.8 * np.exp(-((freqs - f_eff) / width) ** 2)
    flux += rng.normal(0, 0.005, freqs.shape)

    if kind == "radiator_near_pml":
        d_needed = float(cfg["structure"].get("mock_dpml_needed", 2.0))
        if dpml < d_needed:
            # spurious boundary reflection -> NaN patches in the spectrum
            n_bad = int(len(freqs) * 0.1 * (d_needed - dpml))
            bad = rng.choice(len(freqs), size=max(n_bad, 1), replace=False)
            flux[bad] = np.nan
            log_lines.append("warning: possible boundary reflection near PML\n")
    elif kind == "multilayer_stack":
        if res < float(cfg["structure"].get("mock_res_needed", 20)):
            flux += rng.normal(0, 0.05, freqs.shape)  # under-resolved layers

    pd.DataFrame({"freq": freqs, "refl_flux": flux}).to_csv(
        os.path.join(run_dir, "spectrum.csv"), index=False)
    log_lines.append(f"geometry_kind={kind} (mock)\n")


# --------------------------------------------------------------------- driver
def run_meep(cfg: Dict[str, Any], run_dir: str) -> RunOutputs:
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.txt")
    spectrum_path = os.path.join(run_dir, "spectrum.csv")

    start = time.time()
    returncode = 0
    log_lines = []
    backend = cfg["sim"].get("backend", "meep")

    try:
        if backend == "mock":
            _run_mock(cfg, run_dir, log_lines)
        else:
            _run_meep_real(cfg, run_dir, log_lines)
        log_lines.insert(0, "STATUS=OK\n")
        log_lines.append(f"runtime_sec={time.time()-start:.2f}\n")
        log_lines.append(f"resolution={cfg['sim']['resolution']}\n")
        log_lines.append(f"dpml={cfg['sim']['dpml']}\n")
        log_lines.append(f"sim_time={cfg['sim']['sim_time']}\n")
    except Exception as e:
        returncode = 1
        log_lines.append("STATUS=ERROR\n")
        log_lines.append(f"error={repr(e)}\n")
        log_lines.append(traceback.format_exc() + "\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    return RunOutputs(run_dir=run_dir, log_path=log_path, spectrum_path=spectrum_path, returncode=returncode)
