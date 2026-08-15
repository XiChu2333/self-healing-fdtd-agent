"""Decision controllers for the self-healing loop.

Three interchangeable policies (paper Sec. V-C ablation):
  - RuleController : deterministic baseline (ported from the conference MVP)
  - SweepController: naive uniform refinement (lower baseline)
  - LLMController  : LLM decision-maker over a constrained JSON action space,
                     with schema/bounds guardrails and rule fallback

All controllers implement decide(obs, cfg, history) -> action dict.
"""
from __future__ import annotations
import json
import os
import re
import urllib.request
from typing import Dict, Any, List, Optional

from src.actions import ACTION_SCHEMA, BOUNDS, validate_action


class RuleController:
    """Deterministic repair policy (conference-version behavior)."""

    name = "rule"

    def decide(self, obs: Dict[str, Any], cfg: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        sim = cfg["sim"]

        if obs["failure_type"] == "NEED_REFINE_CONFIRMATION":
            new_res = max(int(sim["resolution"]) * 2, int(sim["resolution"]) + 10)
            return {"type": "increase_resolution", "new_resolution": min(new_res, BOUNDS["resolution"][1]),
                    "reason": "confirm stability via refinement"}

        if obs["failure_type"] == "HARD_FAIL":
            if len(history) == 0:
                return {"type": "increase_sim_time", "new_sim_time": float(sim["sim_time"]) * 1.5,
                        "reason": "extend time to allow decay"}
            return {"type": "increase_dpml", "new_dpml": float(sim["dpml"]) + 1.0,
                    "reason": "increase PML thickness"}

        return {"type": "increase_resolution", "new_resolution": int(sim["resolution"]) + 10,
                "reason": "improve numerical accuracy"}


class DiagRuleController(RuleController):
    """Diagnostic-aware rule policy (reviewer-requested baseline).

    Same as RuleController, but inspects the evidence strings for the
    physical diagnostics and routes the repair accordingly:
      - field-decay warning     -> increase sim_time
      - boundary/PML warning    -> increase dpml
    This isolates the contribution of structured diagnostics from the
    contribution of the LLM decision-maker.
    """

    name = "rule-diag"

    def decide(self, obs: Dict[str, Any], cfg: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        sim = cfg["sim"]
        ev = " ".join(obs.get("evidence") or []).lower()
        if "not decayed" in ev:
            return {"type": "increase_sim_time", "new_sim_time": float(sim["sim_time"]) * 2.0,
                    "reason": "diagnostic: fields not decayed -> extend runtime"}
        if "boundary" in ev or "pml" in ev:
            return {"type": "increase_dpml", "new_dpml": float(sim["dpml"]) + 1.0,
                    "reason": "diagnostic: boundary reflection -> thicken PML"}
        return super().decide(obs, cfg, history)


class DiagRuleV2Controller(RuleController):
    """Steelman diagnostic-aware rule policy (budget-limited detours).

    Improves on DiagRuleController with the obvious post-hoc fix for its
    deadlocks: each diagnostic may trigger its corresponding repair AT MOST
    ONCE per episode (decay -> runtime, boundary -> PML); once a detour
    budget is spent, the policy returns to the base grid-confirmation
    behavior regardless of whether the warning persists.
    """

    name = "rule-diag2"

    def __init__(self):
        self.detour_used = {"decay": False, "boundary": False}

    def decide(self, obs: Dict[str, Any], cfg: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        sim = cfg["sim"]
        ev = " ".join(obs.get("evidence") or []).lower()
        if "not decayed" in ev and not self.detour_used["decay"]:
            self.detour_used["decay"] = True
            return {"type": "increase_sim_time", "new_sim_time": float(sim["sim_time"]) * 2.0,
                    "reason": "diagnostic detour (budget 1): fields not decayed -> extend runtime"}
        if ("boundary" in ev or "pml" in ev) and not self.detour_used["boundary"]:
            self.detour_used["boundary"] = True
            return {"type": "increase_dpml", "new_dpml": float(sim["dpml"]) + 1.0,
                    "reason": "diagnostic detour (budget 1): boundary reflection -> thicken PML"}
        return super().decide(obs, cfg, history)


class SweepController:
    """Naive baseline: always refine resolution by a fixed step."""

    name = "sweep"

    def __init__(self, step: int = 10):
        self.step = step

    def decide(self, obs: Dict[str, Any], cfg: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"type": "increase_resolution",
                "new_resolution": int(cfg["sim"]["resolution"]) + self.step,
                "reason": "uniform refinement sweep"}


SYSTEM_PROMPT = """You are a numerical-stability controller supervising FDTD \
electromagnetics simulations (Meep). Your job: given a structured observation of \
the last run, choose ONE repair action that most directly addresses the failure.

Failure taxonomy and typical causes:
- HARD_FAIL: solver crash or missing output. Often insufficient runtime (fields \
not decayed) or unstable configuration.
- QUALITY_FAIL_NAN: non-finite values in the spectrum. Numerical instability; \
consider discretization or boundary absorption.
- NEED_REFINE_CONFIRMATION: first run at coarse settings; refine the grid to \
confirm the result is converged.
- STABILITY_FAIL: resonance drift between refinement steps exceeded threshold; \
discretization is not yet converged.
Evidence strings may contain hints (e.g., 'field decay', 'PML', 'boundary \
reflection') - prefer the parameter that matches the physical cause.

Allowed actions (respond with EXACTLY one JSON object, no prose, no code fences):
{action_docs}

Parameter bounds: {bounds}

Rules:
- The chosen numeric value MUST be strictly greater than the current value.
- Prefer the smallest change likely to fix the failure (cost grows ~ resolution^3).
- Output format: {{"type": "<action>", "<param>": <value>, "reason": "<short>"}}"""


def _action_docs() -> str:
    lines = []
    for name, spec in ACTION_SCHEMA.items():
        p = f', "{spec["param_key"]}": <number>' if spec["param_key"] else ""
        lines.append(f'- {{"type": "{name}"{p}}}: {spec["doc"]}')
    return "\n".join(lines)


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in LLM output: {text[:200]!r}")
    return json.loads(m.group(0))


class LLMController:
    """LLM policy with guardrails.

    Provider selected by env: ANTHROPIC_API_KEY or OPENAI_API_KEY
    (override with LLM_PROVIDER=anthropic|openai, LLM_MODEL=<model>).
    Malformed/invalid outputs are retried up to max_retries, then the
    rule policy takes over (guardrail fallback). All decisions and
    fallbacks are recorded in self.stats for the reliability analysis
    (paper Sec. VI).
    """

    name = "llm"

    def __init__(self, model: Optional[str] = None, temperature: float = 0.2,
                 max_retries: int = 2, timeout: int = 60):
        self.provider = os.environ.get("LLM_PROVIDER") or (
            "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else
            "openai" if os.environ.get("OPENAI_API_KEY") else None)
        if self.provider is None:
            raise RuntimeError("LLMController: set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        self.model = model or os.environ.get("LLM_MODEL") or (
            "claude-sonnet-4-5" if self.provider == "anthropic" else "gpt-4o-mini")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.fallback = RuleController()
        self.stats = {"calls": 0, "malformed": 0, "invalid": 0, "fallbacks": 0, "clamps": 0}

    # ---- prompt ----
    def _messages(self, obs: Dict[str, Any], cfg: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        hist = [{"trial": h["trial"], "failure_type": h["obs"].get("failure_type"),
                 "action": {k: v for k, v in h["action"].items() if k != "reason"}}
                for h in history[-5:]]
        user = {
            "current_sim_parameters": cfg["sim"],
            "observation": {k: obs.get(k) for k in ("status", "failure_type", "evidence", "metrics")},
            "previous_trials": hist,
        }
        return json.dumps(user, default=str)

    # ---- provider calls (stdlib only; no SDK dependency) ----
    def _call(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps({
                    "model": self.model, "max_tokens": 300, "temperature": self.temperature,
                    "system": system, "messages": [{"role": "user", "content": user}],
                }).encode(),
                headers={"content-type": "application/json",
                         "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                         "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())["content"][0]["text"]
        else:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps({
                    "model": self.model, "max_tokens": 300, "temperature": self.temperature,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                }).encode(),
                headers={"content-type": "application/json",
                         "authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]

    # ---- policy ----
    def decide(self, obs: Dict[str, Any], cfg: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        system = SYSTEM_PROMPT.format(action_docs=_action_docs(), bounds=json.dumps(BOUNDS))
        user = self._messages(obs, cfg, history)

        for attempt in range(1 + self.max_retries):
            self.stats["calls"] += 1
            try:
                raw = self._call(system, user)
                action = _extract_json(raw)
            except (ValueError, json.JSONDecodeError):
                self.stats["malformed"] += 1
                continue
            except Exception:
                self.stats["malformed"] += 1  # network/API error counts against reliability
                continue
            try:
                action, warns = validate_action(cfg, action)
            except ValueError:
                self.stats["invalid"] += 1
                continue
            if warns:
                self.stats["clamps"] += 1
            action["controller"] = "llm"
            return action

        self.stats["fallbacks"] += 1
        action = self.fallback.decide(obs, cfg, history)
        action["controller"] = "rule-fallback"
        return action


class LLMNoDecayController(LLMController):
    """Observation-ablation variant: identical to LLMController, but the
    field-decay diagnostic is stripped from the evidence before the LLM
    sees it. Isolates the causal contribution of rho_decay to the LLM's
    repair-direction choice (reviewer-requested ablation)."""

    name = "llm-nodecay"

    def decide(self, obs: Dict[str, Any], cfg: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        obs2 = dict(obs)
        obs2["evidence"] = [e for e in (obs.get("evidence") or [])
                            if "decay" not in e.lower()]
        # history passed to the prompt is rebuilt from action summaries only,
        # so no decay information leaks through it
        return super().decide(obs2, cfg, history)


CONTROLLERS = {"rule": RuleController, "rule-diag": DiagRuleController,
               "rule-diag2": DiagRuleV2Controller,
               "sweep": SweepController, "llm": LLMController,
               "llm-nodecay": LLMNoDecayController}


def make_controller(name: str, **kwargs):
    if name not in CONTROLLERS:
        raise ValueError(f"Unknown controller {name!r}; choose from {list(CONTROLLERS)}")
    return CONTROLLERS[name](**kwargs)
