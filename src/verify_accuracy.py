"""Accuracy audit for accepted results (guards against spurious convergence).

For each config: run one high-accuracy reference simulation (boosted
resolution + sim_time), then compare each controller's ACCEPTED f_res
(from batch episode.json files) against the reference.

Usage:
  python -m src.verify_accuracy results/episodes/<stamp> configs/*.json
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import pandas as pd

from src.meep_runner import run_meep
from src.parser_quality import parse_outputs

REF_RESOLUTION = 60
REF_SIM_TIME = 1500.0
REF_DPML = 3.0


def reference_f_res(cfg_path: str, out_root: str) -> float:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    name = cfg.get("name") or os.path.splitext(os.path.basename(cfg_path))[0]
    cfg["name"] = name
    cfg["sim"]["resolution"] = REF_RESOLUTION
    cfg["sim"]["sim_time"] = REF_SIM_TIME
    # PML must fit the cell: cap at 1/4 of the smaller cell dimension
    dpml_cap = 0.25 * min(float(cfg["sim"]["cell_sx"]), float(cfg["sim"]["cell_sy"]))
    cfg["sim"]["dpml"] = max(float(cfg["sim"]["dpml"]), min(REF_DPML, dpml_cap))
    cfg["quality"]["require_refine_confirmation"] = False

    run_dir = os.path.join(out_root, f"reference_{name}")
    spectrum_cached = os.path.join(run_dir, "spectrum.csv")
    if os.path.exists(spectrum_cached):
        from src.parser_quality import extract_f_res
        df = pd.read_csv(spectrum_cached)
        f_ref, _, on_edge = extract_f_res(df["freq"].to_numpy(float),
                                          df["refl_flux"].to_numpy(float), cfg["quality"])
        print(f"[reference] {name}: f_res={f_ref:.5f} (cached)"
              + ("  WARNING: proxy on band edge!" if on_edge else ""))
        return f_ref
    out = run_meep(cfg, run_dir=run_dir)
    obs = parse_outputs(cfg, run_dir, out.log_path, out.spectrum_path, out.returncode)
    f_ref = obs["metrics"].get("f_res")
    if f_ref is None:
        raise RuntimeError(f"reference run failed for {name}: {obs['evidence']}")
    print(f"[reference] {name}: f_res={f_ref:.5f} (res={REF_RESOLUTION}, T={REF_SIM_TIME})")
    return f_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes_dir", help="e.g. results/episodes/20260702-013524")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out", default="results/reference")
    ap.add_argument("--check-reference", action="store_true",
                    help="also run a res=80/T=2000 reference and report its "
                         "deviation from the res=60/T=1500 reference "
                         "(reference self-convergence check)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    refs, qcfgs = {}, {}
    for cfg_path in args.configs:
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        refs[name] = reference_f_res(cfg_path, args.out)
        with open(cfg_path, "r", encoding="utf-8") as f:
            qcfgs[name] = json.load(f)["quality"]

    if args.check_reference:
        # self-convergence: rerun each reference at res=80 / T=2000 in a
        # sibling directory and report the relative deviation
        global REF_RESOLUTION, REF_SIM_TIME
        REF_RESOLUTION, REF_SIM_TIME = 80, 2000.0
        hi_dir = args.out.rstrip("/") + "_hi"
        os.makedirs(hi_dir, exist_ok=True)
        print("\n[reference self-convergence check: res=80, T=2000]")
        for cfg_path in args.configs:
            name = os.path.splitext(os.path.basename(cfg_path))[0]
            f_hi = reference_f_res(cfg_path, hi_dir)
            dev = abs(f_hi - refs[name]) / abs(refs[name])
            print(f"  {name}: ref60={refs[name]:.5f} ref80={f_hi:.5f} "
                  f"deviation={dev:.4%}")

    rows = []
    for ep in glob.glob(os.path.join(args.episodes_dir, "*", "*", "*", "episode.json")):
        d = json.load(open(ep))
        r = d["result"]
        if not r["success"]:
            continue
        name, ctrl = r["config"], r["controller"]
        if name not in refs:
            continue
        # accepted f_res = last trial's spectrum
        last = os.path.join(os.path.dirname(ep), f"trial_{r['trials']}", "spectrum.csv")
        if not os.path.exists(last):
            continue
        df = pd.read_csv(last)
        from src.parser_quality import extract_f_res
        f_acc, _, _ = extract_f_res(df["freq"].to_numpy(float),
                                    df["refl_flux"].to_numpy(float), qcfgs[name])
        rows.append({"config": name, "controller": ctrl,
                     "f_accepted": f_acc, "f_reference": refs[name],
                     "rel_error": abs(f_acc - refs[name]) / abs(refs[name])})

    if not rows:
        import sys
        parent = os.path.dirname(args.episodes_dir.rstrip("/"))
        avail = glob.glob(os.path.join(parent, "*")) if parent else []
        sys.exit(f"No episodes found under {args.episodes_dir!r}.\n"
                 f"Available: {avail}\n"
                 "Hint: the directory stamp must match the batch summary filename.")

    out = pd.DataFrame(rows)
    summary = (out.groupby(["config", "controller"])
                  .agg(mean_rel_error=("rel_error", "mean"),
                       max_rel_error=("rel_error", "max"),
                       n=("rel_error", "size")).reset_index())
    path = os.path.join(args.out, "accuracy_summary.csv")
    summary.to_csv(path, index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nSaved: {path}")
    print("\nInterpretation: rel_error >~ drift_threshold (0.01) on a PASSED "
          "episode indicates spurious convergence (accepted but wrong).")


if __name__ == "__main__":
    main()
