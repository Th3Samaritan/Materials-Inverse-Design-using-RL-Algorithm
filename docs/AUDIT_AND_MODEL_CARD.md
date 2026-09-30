# Audit and legacy model card

Audit date: **8 September 2026**. Status: **synthetic research prototype; industrial accuracy unverified**.

## What the project does

The application maps a fixed chemical composition and six desired properties to candidate heat-treatment schedules. It does not currently optimize composition, component geometry, a full furnace temperature trajectory, or a multi-step factory route.

Inputs: 15 composition coordinates in wt% and five process coordinates: austenitizing temperature in °C, soak in hours, quench category, tempering temperature in °C and tempering hold in hours. Outputs: UTS/YS in MPa, elongation/reduction of area in %, HRC, and impact energy in J. Charpy specimen geometry, test temperature, orientation and notch details are absent from the predictor.

The forward surrogate is an ensemble of five MLPs with hidden widths 256/256/128. Normalization buffers are present and active in the inspected checkpoint. The inverse policy uses composition/property encoders, a transformer decoder, continuous mixture-density heads and a Gumbel categorical head. Candidates are scored by the forward ensemble.

The project contains SAC-style experiments, but the inspected current `training/config.py` and `training/train_policy.py` implement differentiable optimization through a frozen forward model with an NLL anchoring loss. The SAC improvement plan is historical context, not an accurate description of every exported checkpoint. Exact training-code-to-checkpoint provenance remains incomplete.

## Local evidence inspected

In this repository: app source, README, requirements, checkpoint metadata, training log, KNN comparison and case-study CSVs.

In `C:/Users/MAST/Documents/MAST Consolidate/MAST-RL`:

- `data/synthetic.py`, `data/loader.py`, `utils/steel_physics.py`.
- `training/config.py`, relevant forward/policy training code, `models/actor.py`, historical `RL_IMPROVEMENTS_PLAN.md`.
- Relevant generator/training cells in `MAST_Kaggle_Training.ipynb` and `outputs/output 2/MAST_Kaggle_Training_fixed(2).ipynb`.
- Exported bundle checkpoint hashes.

This was a targeted source and artifact audit, not a claim that every notebook output or experiment in the parent directory was reviewed. No dataset with independently measured industrial outcomes was established by this inspection. External files were inspected read-only.

## Checkpoint identity

| Artifact | SHA-256 |
|---|---|
| `checkpoints/forward_model.pt` | `6b32d4a2b6e2fed2e128d2082e204839d2995f5f635629b39ae15c579b86e449` |
| `checkpoints/policy.pt` | `ff074ba6d75414d17cc78d8bec7934401b603813c342b345d6490741c9593fc4` |

Both hashes exactly match `MAST-RL/MAST_model_bundle (1)/checkpoints`. This establishes artifact identity, not the exact training revision or experimental validity.

Policy property means are approximately `[1418.72, 1061.33, 9.44, 13.92, 44.30, 37.66]`; standard deviations `[229.81, 164.49, 6.70, 6.98, 8.78, 26.46]` in property order. New model loading requires finite, positive scales and strictly compatible weights.

## Findings

| Finding | Evidence and consequence | Disposition |
|---|---|---|
| Synthetic training/evaluation | Generator uses heuristic composition/process/property equations; splits shuffle synthetic rows. | Treat legacy metrics as simulator agreement; require measured tests. |
| Inverted cooling effect | Both inspected notebooks and `steel_physics.py` contain the expression below. | Document and test a bounded transformation primitive. Existing weights unchanged. |
| H13 silently altered | Si=1, Cr=5.15, Mo=1.35, V=1 exceed bounds .5/2/.6/.3; `np.clip` changed them. | Remove H13 from legacy choices; reject it in core validation. |
| Missing geometry and quench state | Nominal generator section is 25 mm; predictor lacks section, agitation, bath temperature and loading inputs. | Restrict modeled geometry; new geometry-aware model required. |
| Clamp breaks Ac3 margin guarantee | `Ac3 + delta` is clamped to 750–1100 °C and can violate a 30–150 °C margin. | Screen actual output, not merely intended parameterization. |
| Mixed-unit ranking fallback | Old app could rank raw errors without normalization metadata. | Require validated normalization metadata. |
| Uncertainty overstated | Ensemble disagreement had no measured coverage validation. | Label ensemble spread, not statistical confidence. |
| Missing constraints and abstention | Top candidates displayed even far from targets. | Add screening, no-candidate results and tolerance status. |
| Stale results | Input edits could display old candidates against new targets. | Invalidate results on request/settings changes. |
| Unrestricted deserialization | Uploaded files used `weights_only=False`. | Restricted tensor loading and strict state dictionaries. |
| Incorrect startup instruction | README referred to nonexistent `app.py`. | Document actual entry point. |

### Phase-fraction defect

The old computation defines a cooling-dependent sigmoid `g`, then calculates:

```text
f_mart_max = clip(1 - exp(-0.011 * (Ms - 25)) * g, 0, 0.99)
```

As cooling ratio becomes small, `g` approaches zero and this approaches .99 rather than zero. It can decrease when cooling accelerates. At Ms=350 °C and C=.4 wt%, ratio .01 gives approximately .9702 martensite after retained-austenite subtraction; ratio 100 gives approximately .9525. These are calculations of the source expression, not measurements.

The new primitive applies `available_austenite * (1 - exp(-k * max(Ms-T, 0)))`. It conserves available phase and returns zero above Ms. **This is only one corrected component:** available austenite must come from independently validated diffusional transformation kinetics. Changing a parenthesis alone cannot qualify the generator.

Further concerns require metallurgical review: ideal critical diameter does not use its extracted carbon variable; the grain-size-factor direction warrants checking; cooling noise uses global NumPy state despite a generator RNG elsewhere; quench labels substitute for measured cooling curves. Do not accept empirical constants solely because comments name established relationships.

## Existing metrics

Stored KNN MAE: approximately 58.95 MPa UTS, 41.68 MPa YS, .963 percentage points elongation, 1.561 percentage points reduction of area, 2.198 HRC and 3.644 J impact energy. These inherited CSV values were not recomputed against experimental data.

The case-study CSV includes a 4340-like 650 MPa UTS target with approximately 1144 MPa predicted UTS. This demonstrates a target miss, not a physical lower-strength limit. The old README's unsupported universal lower-strength assertion was removed.

An average mixing MPa, percentages, hardness and joules is not a meaningful single industrial accuracy claim. Report each property in its own unit against an agreed engineering tolerance.

## Intended use and limits

Use this model to reproduce legacy experiments, compare candidate generators, inspect constraints and plan validation. It does not establish material certification, guaranteed recipes, part-size transfer, fatigue/creep performance, calibrated confidence, global optimality, physical infeasibility, actual cycle-time/energy savings or accuracy for other material families.

Before a replacement model is released, archive its dataset, immutable code revision, simulator version, units/feature order, seeds, grouped splits, hyperparameters, calibration set and review outcome. Distinguish legacy and replacement predictions in every report.
