from __future__ import annotations
import argparse
import json
import os
import time
from typing import Dict, Any, Optional

from src.meep_runner import run_meep
from src.parser_quality import parse_outputs, stability_check
from src.actions import apply_action, validate_action
from src.controllers import make_controller


def run_episode(cfg: Dict[str, Any], controller, run_root: str = "runs",
                tag: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """One self-healing episode (observe-diagnose-act until PASS or budget).

    Returns a result dict used by both the CLI and the batch runner.
    """
    max_trials = int(cfg["quality"]["max_trials"])
    history = []
    prev_f_res: Optional[float] = None
    prev_res: Optional[int] = None
    t_start = time.time()

    ts = time.strftime("%Y%m%d-%H%M%S")
    base_run_dir = os.path.join(run_root, tag or f"{ts}-{controller.name}")
    os.makedirs(base_run_dir, exist_ok=True)

    result = {"controller": controller.name, "config": cfg.get("name", "unnamed"),
              "success": False, "trials": 0, "wall_time_sec": 0.0,
              "run_dir": base_run_dir, "final_sim": None}

    for t in range(1, max_trials + 1):
        result["trials"] = t
        run_dir = os.path.join(base_run_dir, f"trial_{t}")
        out = run_meep(cfg, run_dir=run_dir)
        obs = parse_outputs(cfg, run_dir, out.log_path, out.spectrum_path, out.returncode)

        if verbose:
            print(f"[trial {t}] status={obs['status']} failure_type={obs['failure_type']} metrics={obs.get('metrics', {})}")

        cur_res = int(cfg["sim"]["resolution"])
        if prev_f_res is not None and "f_res" in obs.get("metrics", {}):
            ok, drift = stability_check(cfg, prev_f_res, obs["metrics"]["f_res"])
            obs["metrics"]["drift"] = drift
            if cur_res == prev_res:
                # same-resolution pair (produced by a runtime/PML repair):
                # it assesses that parameter only and must NOT certify grid
                # convergence -- require a resolution confirmation next
                obs["status"] = "QUALITY_UNCERTAIN"
                obs["failure_type"] = "NEED_REFINE_CONFIRMATION"
                obs["evidence"].append(
                    f"same-resolution pair (drift={drift:.4f}): parameter "
                    f"repair assessed, but grid convergence unconfirmed; "
                    f"resolution refinement required before acceptance")
                if verbose:
                    print(f"[trial {t}] same-res pair (drift={drift:.4f}); grid confirmation required")
            elif ok:
                obs["status"], obs["failure_type"] = "PASS", None
                if verbose:
                    print(f"[trial {t}] PASS (drift={drift:.4f})")
            else:
                obs["status"], obs["failure_type"] = "FAILED", "STABILITY_FAIL"
                obs["evidence"].append(f"drift={drift:.4f} > threshold")
                if verbose:
                    print(f"[trial {t}] stability fail (drift={drift:.4f})")

        if "f_res" in obs.get("metrics", {}):
            prev_f_res = obs["metrics"]["f_res"]
            prev_res = cur_res

        if obs["status"] == "PASS":
            result["success"] = True
            result["final_sim"] = dict(cfg["sim"])
            break

        if t == max_trials:
            break

        action = controller.decide(obs, cfg, history)
        action, warns = validate_action(cfg, action)
        for w in warns:
            obs["evidence"].append(f"guardrail: {w}")
        history.append({"trial": t, "obs": obs, "action": action})
        cfg = apply_action(cfg, action)

        if verbose:
            print(f"[trial {t}] action={action}")

    result["wall_time_sec"] = time.time() - t_start
    if not result["success"]:
        result["last_failure"] = obs.get("failure_type")
        result["last_evidence"] = (obs.get("evidence") or [])[:2]
    if hasattr(controller, "stats"):
        result["llm_stats"] = dict(controller.stats)

    with open(os.path.join(base_run_dir, "episode.json"), "w", encoding="utf-8") as f:
        json.dump({"result": result,
                   "history": [{"trial": h["trial"],
                                "failure_type": h["obs"].get("failure_type"),
                                "evidence": h["obs"].get("evidence"),
                                "metrics": h["obs"].get("metrics"),
                                "action": h["action"]} for h in history]},
                  f, indent=2, default=str)

    if verbose:
        print(("PASS" if result["success"] else "FAILED") + f" after {result['trials']} trial(s). Result dir: {base_run_dir}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Self-healing CEM simulation loop")
    ap.add_argument("config", help="path to config JSON")
    ap.add_argument("--controller", default="rule",
                    choices=["rule", "rule-diag", "rule-diag2", "sweep",
                             "llm", "llm-nodecay"])
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("name", os.path.splitext(os.path.basename(args.config))[0])

    controller = make_controller(args.controller)
    result = run_episode(cfg, controller, run_root=args.run_root)
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
