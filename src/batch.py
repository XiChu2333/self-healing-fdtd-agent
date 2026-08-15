"""Batch benchmark runner (paper Sec. V-C).

Runs configs x controllers x repeats, aggregates success rate, trials to
convergence, and wall time into results/summary.csv (+ raw runs.jsonl).

Example:
  python -m src.batch configs/sample_patch_2d.json configs/resonator_2d.json \
      --controllers rule sweep llm --repeats 10 --out results
"""
from __future__ import annotations
import argparse
import json
import os
import time

import pandas as pd

from src.agent_loop import run_episode
from src.controllers import make_controller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="config JSON paths")
    ap.add_argument("--controllers", nargs="+",
                    default=["rule", "rule-diag", "sweep", "llm"])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    # fail fast: validate every controller (e.g., missing API key) BEFORE
    # spending solver time on the earlier ones
    for ctrl_name in args.controllers:
        make_controller(ctrl_name)

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw_path = os.path.join(args.out, f"runs-{stamp}.jsonl")
    rows = []

    with open(raw_path, "w", encoding="utf-8") as raw:
        for cfg_path in args.configs:
            with open(cfg_path, "r", encoding="utf-8") as f:
                base_cfg = json.load(f)
            base_cfg.setdefault("name", os.path.splitext(os.path.basename(cfg_path))[0])

            for ctrl_name in args.controllers:
                for rep in range(args.repeats):
                    cfg = json.loads(json.dumps(base_cfg))  # deep copy
                    cfg["mock_seed"] = rep  # only affects mock backend noise
                    controller = make_controller(ctrl_name)
                    tag = f"{stamp}/{cfg['name']}/{ctrl_name}/rep{rep:02d}"
                    r = run_episode(cfg, controller, run_root=os.path.join(args.out, "episodes"),
                                    tag=tag, verbose=False)
                    r["repeat"] = rep
                    raw.write(json.dumps(r, default=str) + "\n")
                    rows.append({"config": cfg["name"], "controller": ctrl_name, "repeat": rep,
                                 "success": r["success"], "trials": r["trials"],
                                 "wall_time_sec": r["wall_time_sec"]})
                    fail_info = "" if r["success"] else f"  <- {r.get('last_failure')}: {'; '.join(r.get('last_evidence', []))}"
                    print(f"{cfg['name']:>20} | {ctrl_name:>5} | rep {rep:02d} | "
                          f"{'PASS' if r['success'] else 'FAIL'} in {r['trials']} trial(s){fail_info}")

    df = pd.DataFrame(rows)
    summary = (df.groupby(["config", "controller"])
                 .agg(success_rate=("success", "mean"),
                      mean_trials=("trials", "mean"),
                      std_trials=("trials", "std"),
                      mean_wall_time=("wall_time_sec", "mean"),
                      n=("success", "size"))
                 .reset_index())
    summary_path = os.path.join(args.out, f"summary-{stamp}.csv")
    summary.to_csv(summary_path, index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nSaved: {summary_path}\nRaw episodes: {raw_path}")


if __name__ == "__main__":
    main()
