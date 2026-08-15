# Self-Healing FDTD Agent

Diagnostic-guided repair of FDTD electromagnetic simulations using a
constrained LLM controller.

This repository accompanies the manuscript *"Diagnostic-Guided Repair of
FDTD Simulations Using a Constrained LLM Controller"* (X. Chu and Y. Hou,
submitted to the *IEEE Journal on Multiscale and Multiphysics Computational
Techniques*, 2026), which extends our ICCEM 2026 conference paper
*"A Lightweight Agentic Framework for Self-Healing Computational
Electromagnetics Simulations."*

## What it does

Computational electromagnetics (CEM) simulations silently fail when numerical
parameters (grid resolution, absorbing-boundary thickness, runtime) are
mis-set: a standard drift-based convergence check can report success while the
accepted result is wrong by several percent. This framework wraps the
[Meep](https://meep.readthedocs.io/) FDTD solver in a closed
observe-diagnose-act loop:

1. **Observe** - each run is parsed into a structured observation: failure
   type, resonance proxy with band-edge masking, resonance drift, a
   field-decay diagnostic, and a boundary-persistence diagnostic.
2. **Diagnose & act** - a controller maps the observation to one bounded
   repair action (increase resolution, thicken PML, or extend runtime).
   Guardrails validate every action against a JSON schema and parameter
   bounds, with deterministic fallback.
3. **Certify** - acceptance requires resonance drift below threshold across a
   pair of runs that differ in spatial resolution; repairs of other
   parameters never certify convergence on their own.
4. **Audit** - accepted results can be verified against self-converged
   high-accuracy references (`src/verify_accuracy.py`).

## Controllers

| Name | Policy |
|---|---|
| `llm` | LLM decision-maker (Anthropic or OpenAI) over the structured observation, with schema/bounds guardrails and rule fallback |
| `llm-nodecay` | Observation ablation: the field-decay diagnostic is hidden from the LLM |
| `rule` | Deterministic baseline: refinement confirmation, then resolution increases |
| `rule-diag` | Unconditional diagnostic routing (decay -> runtime, boundary -> PML) |
| `rule-diag2` | Budget-limited diagnostic routing (each detour at most once per episode) |
| `sweep` | Uniform resolution refinement |

## Benchmark cases (`configs/`)

| Config | Failure mode exercised |
|---|---|
| `sample_patch_2d` | baseline, discretization-limited |
| `multilayer_stack` | under-resolved layers; transfer-matrix analytical reference (`src/analytics.py`) |
| `radiator_near_pml` | thin PML (benign in practice; accuracy-validation case) |
| `resonator_2d` | high-Q truncated ring-down: drift criterion accepts wrong results unless runtime is repaired |
| `cavity_near_pml` | misleading (red-herring) decay diagnostic |
| `pml_stress` | sub-cell PML; boundary diagnostic routes to absorber repair |

## Install

```bash
conda install -c conda-forge pymeep numpy pandas matplotlib
```

For the LLM controller, set one of:

```bash
export ANTHROPIC_API_KEY=...   # default model: claude-sonnet-4-5
export OPENAI_API_KEY=...      # default model: gpt-4o-mini
# optional: LLM_PROVIDER=anthropic|openai, LLM_MODEL=<model-id>
```

## Usage

Single episode:

```bash
python -m src.agent_loop configs/resonator_2d.json --controller llm
```

Full benchmark with statistics (success rate, trials, wall time):

```bash
python -m src.batch configs/*.json --controllers rule rule-diag sweep llm llm-nodecay --repeats 10 --out results
```

Accuracy audit against self-converged references:

```bash
python -m src.verify_accuracy results/episodes/<stamp> configs/*.json --out results/reference --check-reference
```

### Mock backend

Set `"backend": "mock"` inside the `"sim"` block of any config to run the
full loop without Meep installed. The mock solver emulates each case's
failure mode; it is for development only, never for reported results.

## Citation

Until the journal version is available, please cite the conference paper:

```bibtex
@inproceedings{chu2026selfhealing,
  author    = {Chu, Xi and Hou, Yupeng},
  title     = {A Lightweight Agentic Framework for Self-Healing
               Computational Electromagnetics Simulations},
  booktitle = {Proc. 2026 IEEE Int. Conf. Computational Electromagnetics (ICCEM)},
  address   = {Shanghai, China},
  month     = apr,
  year      = {2026}
}
```

## License

MIT
