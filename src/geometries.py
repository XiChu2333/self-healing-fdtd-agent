"""Geometry builders for the benchmark suite (paper Sec. V).

Each builder returns a spec dict consumed by meep_runner:
  cell (sx, sy), geometry list, source spec, flux-monitor spec.
Builders import meep lazily so that mock mode works without meep installed.

Cases:
  patch_2d          - conference baseline (Case A)
  multilayer_stack  - Bragg-like stack, transfer-matrix reference (Case B2,
                      resolution-sensitive)
  resonator_2d      - high-Q dielectric cavity, slow field decay (Case B1,
                      sim_time-sensitive)
  radiator_near_pml - line source close to the absorbing boundary (Case B3,
                      dpml-sensitive)
"""
from __future__ import annotations
from typing import Dict, Any


def _mp():
    import meep as mp
    return mp


def build_patch_2d(cfg: Dict[str, Any]):
    mp = _mp()
    s, sim = cfg["structure"], cfg["sim"]

    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, float(s["substrate_height"]), mp.inf),
                 center=mp.Vector3(0, -2.0),
                 material=mp.Medium(epsilon=float(s["substrate_eps"]))),
        mp.Block(size=mp.Vector3(float(s["patch_width"]), float(s["patch_height"]), mp.inf),
                 center=mp.Vector3(float(s["patch_center_x"]), float(s["patch_center_y"])),
                 material=mp.perfect_electric_conductor),
    ]
    src = mp.Source(src=mp.GaussianSource(frequency=float(sim["fcen"]), fwidth=float(sim["df"])),
                    component=mp.Ez,
                    center=mp.Vector3(-6, float(s["patch_center_y"])),
                    size=mp.Vector3(0, 2.0))
    flux = dict(center=mp.Vector3(-5, float(s["patch_center_y"])), size=mp.Vector3(0, 2.0))
    return {"geometry": geometry, "sources": [src], "flux_region": flux}


def build_multilayer_stack(cfg: Dict[str, Any]):
    """Quarter-wave-like stack normal to x; analytic reference available
    via src.analytics.tmm_reflectance."""
    mp = _mp()
    s, sim = cfg["structure"], cfg["sim"]
    # layers: list of {"eps": float, "thickness": float}, stacked along +x from x0
    x = float(s.get("stack_start_x", -1.0))
    geometry = []
    for layer in s["layers"]:
        th = float(layer["thickness"])
        geometry.append(mp.Block(
            size=mp.Vector3(th, mp.inf, mp.inf),
            center=mp.Vector3(x + th / 2, 0),
            material=mp.Medium(epsilon=float(layer["eps"]))))
        x += th

    src_x = float(s.get("source_x", -0.5 * sim["cell_sx"] + sim["dpml"] + 1.0))
    src = mp.Source(src=mp.GaussianSource(frequency=float(sim["fcen"]), fwidth=float(sim["df"])),
                    component=mp.Ez, center=mp.Vector3(src_x, 0),
                    size=mp.Vector3(0, float(sim["cell_sy"]) - 2 * float(sim["dpml"])))
    flux = dict(center=mp.Vector3(src_x + 0.5, 0),
                size=mp.Vector3(0, float(sim["cell_sy"]) - 2 * float(sim["dpml"])))
    return {"geometry": geometry, "sources": [src], "flux_region": flux}


def build_resonator_2d(cfg: Dict[str, Any]):
    """High-eps dielectric block cavity; long ring-down demands sufficient
    sim_time before the spectrum is trustworthy."""
    mp = _mp()
    s, sim = cfg["structure"], cfg["sim"]
    geometry = [mp.Block(
        size=mp.Vector3(float(s["cavity_w"]), float(s["cavity_h"]), mp.inf),
        center=mp.Vector3(0, 0),
        material=mp.Medium(epsilon=float(s["cavity_eps"])))]
    src = mp.Source(src=mp.GaussianSource(frequency=float(sim["fcen"]), fwidth=float(sim["df"])),
                    component=mp.Ez, center=mp.Vector3(-0.25 * float(s["cavity_w"]), 0))
    flux = dict(center=mp.Vector3(-0.5 * float(sim["cell_sx"]) + float(sim["dpml"]) + 0.5, 0),
                size=mp.Vector3(0, 2.0))
    return {"geometry": geometry, "sources": [src], "flux_region": flux,
            "probe_pt": mp.Vector3(0.15 * float(s["cavity_w"]), 0)}


def build_radiator_near_pml(cfg: Dict[str, Any]):
    """Line source placed close to the cell edge; thin PML produces spurious
    boundary reflections in the recorded spectrum."""
    mp = _mp()
    s, sim = cfg["structure"], cfg["sim"]
    gap = float(s.get("source_edge_gap", 1.5))  # distance from PML inner edge
    src_x = -0.5 * float(sim["cell_sx"]) + float(sim["dpml"]) + gap
    src = mp.Source(src=mp.GaussianSource(frequency=float(sim["fcen"]), fwidth=float(sim["df"])),
                    component=mp.Ez, center=mp.Vector3(src_x, 0), size=mp.Vector3(0, 1.0))
    flux = dict(center=mp.Vector3(src_x + 1.0, 0), size=mp.Vector3(0, 2.0))
    return {"geometry": [], "sources": [src], "flux_region": flux}


def build_cavity_near_pml(cfg: Dict[str, Any]):
    """Moderate-Q dielectric cavity placed close to the absorbing boundary
    (Case B4). With a thin PML, boundary reflections interfere with the
    cavity response and corrupt the radiated spectrum; the correct repair
    is thickening the PML, not refining the grid or extending runtime."""
    mp = _mp()
    s, sim = cfg["structure"], cfg["sim"]
    w = float(s["cavity_w"]); h = float(s["cavity_h"])
    if "cavity_center_x" in s:
        # absolute position, independent of dpml (so PML repairs do not
        # relocate the structure -- required for valid reference audits)
        cx = float(s["cavity_center_x"])
    else:
        gap = float(s.get("pml_gap", 0.7))  # cavity edge to PML inner edge
        cx = -0.5 * float(sim["cell_sx"]) + float(sim["dpml"]) + gap + w / 2
    geometry = [mp.Block(size=mp.Vector3(w, h, mp.inf),
                         center=mp.Vector3(cx, 0),
                         material=mp.Medium(epsilon=float(s["cavity_eps"])))]
    src = mp.Source(src=mp.GaussianSource(frequency=float(sim["fcen"]), fwidth=float(sim["df"])),
                    component=mp.Ez, center=mp.Vector3(cx + 0.1 * w, 0))
    # radiated flux monitored on the far (well-absorbed) side of the cell
    flux = dict(center=mp.Vector3(0.5 * float(sim["cell_sx"]) - float(sim["dpml"]) - 0.5, 0),
                size=mp.Vector3(0, 2.0))
    return {"geometry": geometry, "sources": [src], "flux_region": flux,
            "probe_pt": mp.Vector3(cx + 0.15 * w, 0)}


GEOMETRY_BUILDERS = {
    "patch_2d": build_patch_2d,
    "multilayer_stack": build_multilayer_stack,
    "resonator_2d": build_resonator_2d,
    "radiator_near_pml": build_radiator_near_pml,
    "cavity_near_pml": build_cavity_near_pml,
}
