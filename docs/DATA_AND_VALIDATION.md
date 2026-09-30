# Data, physics validation and model release

## Evidence separation

Every record must identify whether it is **experimental**, **simulation** or **synthetic**. Never relabel simulation as experiment, or infer evidence quality from a realistic-looking number. Keep laboratory measurements, empirical-generator outputs, DFT and continuum calculations distinguishable during training and reporting. A model trained on mixed fidelity needs explicit fidelity and discrepancy treatment.

The current `mast.data` validator supports numeric feature/property records with explicit units and provenance. It is a starting contract, not a complete LIMS, materials ontology or automated rights checker. Source/license text records a declaration; a reviewer must verify commercial reuse rights.

## Minimal numeric record

The input to `mast.train_baseline` is a JSON array of records shaped as follows. Placeholder numbers are intentionally null; fill them from actual measurements rather than treating this as sample training data.

```json
{
  "record_id": "specimen-result-id",
  "group_id": "independent-heat-lot-or-campaign-id",
  "family": "Low-alloy steel",
  "source": "traceable-test-report-reference",
  "source_type": "experimental",
  "license": "documented-owner-permission-or-license",
  "test_method": "exact-method-edition-specimen-orientation-and-conditions",
  "features": {
    "carbon": {"value": null, "unit": "wt%"},
    "section_diameter": {"value": null, "unit": "mm"},
    "austenitize_temperature": {"value": null, "unit": "degC"},
    "austenitize_hold": {"value": null, "unit": "h"}
  },
  "properties": {
    "UTS": {"value": null, "unit": "MPa"}
  }
}
```

The validator rejects missing provenance, duplicate IDs, exact duplicate source observations, nonfinite/nonnumeric measurements and missing units. The baseline rejects mixed units, mixed feature/property schemas, mixed test methods, mixed families and nonexperimental records. It does not guess unit conversions or impute missing observations.

Actual production ingestion must also retain raw report/file hashes, heat certificate, specimen geometry and location, date, operator/instrument, calibration, measurement uncertainty, censored values, replicates, process history, starting condition, supplier, site, consent and corrected-record lineage. Add named categorical fields and curve/tensor types before using polymer chemistry, layups or time-series data. The numeric baseline must not encode arbitrary categories as continuous numbers without an explicit modeling decision.

## Splitting and leakage

`grouped_split` reserves approximately 20% of independent groups for test, 20% for validation and the remainder for training using a deterministic seed. Groups must correspond to a meaningful independent unit (heat, lot or campaign), not a unique row ID assigned solely to bypass grouping. Five groups is a software minimum, not enough evidence for generalization by itself.

For deployment evaluation, prefer forward-in-time future lots and hold out entire sites/furnaces/chemistry regions where transfer is claimed. Specimens from the same heat, near-duplicate papers, augmented rows and repeat simulations must remain together. Audit source overlap manually and programmatically. Fit scalers, imputers and feature selection on training data only; tune on validation; open final test once after design decisions are fixed.

## Measured-data baseline

```powershell
.\.venv\Scripts\python.exe -m mast.train_baseline path-to-measured-records.json --output outputs/baseline
```

The command evaluates a fixed random forest against a training-mean predictor. It writes per-property MAE in original units, group assignments, a seed, dataset hash and limitations. It does not serialize/deploy a predictor or overwrite legacy weights. Unit tests use artificial fixtures only to test software behavior; they are not experimental validation and are not delivered as a measured dataset.

Next baselines should include nearest-neighbor retrieval, calibrated Gaussian processes, gradient boosting and an appropriate physical model. Add model complexity only if it improves independently held-out performance or acquisition efficiency. Compare RL and other inverse optimizers only after forward-model usefulness is demonstrated.

## Physics verification before retraining

For steels: check composition balance, phase-fraction sum and nonnegativity, transformation limits above/below critical temperatures, cooling-rate sensitivity, section-size scaling, heat equation residuals, boundary conditions, latent-heat treatment and grain/precipitation evolution. Test against documented reference cases, not only a synthetic generator that shares the implementation. The added martensite function tests one bound and limiting-case relationship; it does not validate the full chain.

For composites: check positive-definite stiffness, consistent orientation transformations, constituent/void fractions, degree of cure in [0,1], energy balance and calibrated cure kinetics. For polymers/elastomers: check correct formulation basis, temperature units, admissible strain-energy response, viscosity/crosslink limits and test-rate dependence. Limit every physical law to its calibrated regime; “physics-informed” is not a guarantee of correctness.

When using simulated labels, archive solver/database version, input hash, convergence/residual checks, mesh/time-step sensitivity, boundary conditions and failed jobs. Validate numerical accuracy separately from constitutive/material-model accuracy. A converged wrong model is still wrong.

## Uncertainty and target semantics

Ensemble variance measures disagreement, not total predictive error. Quantify measurement variability, model discrepancy and extrapolation separately. Calibrate intervals on an independent calibration set; report empirical coverage and width by property, family and relevant subgroup. Split conformal methods require exchangeability assumptions; batch/time/domain shift can break nominal coverage. Do not claim joint six-property coverage from six marginal intervals without a joint correction or calibrated joint procedure.

Production requests should support property intervals, minimum/maximum requirements, hard constraints and optional preferences. Current UI uses six point targets and relative tolerances; its score treats overshoot and undershoot symmetrically. That is unsuitable for every customer requirement and must be extended before such use. “No match in this search” is distinct from proven infeasibility.

Validate out-of-domain detection on withheld chemistries, geometries and process conditions. Training bounds are a necessary screening rule, not a complete OOD detector: combinations inside a box may still be unsupported. Unknown operating conditions should trigger abstention or an explicit experiment request.

## Prospective pilot protocol

1. Agree on one grade/process family, component/section envelope, equipment and test conditions.
2. Define per-property acceptance intervals and the business outcome before looking at model proposals.
3. Freeze a baseline method, target suite, model version and experiment budget.
4. Select small batches containing promising feasible candidates, informative candidates and baseline/control runs; engineer-review every proposal.
5. Randomize run order where practical, retain replicates and log deviations and failed runs.
6. Measure actual properties and complete cycle/energy/cost data, not surrogate predictions alone.
7. Compare all-property pass rate, trials to success, constraint violations, uncertainty coverage and cost against current practice.
8. Review discrepant cases, revise models with versioned data, and repeat on new lots before expanding scope.

Sample size and replication require a study design using expected variance and required effect size. They cannot be justified by a generic fixed number in this repository.

## Release gates

A candidate model needs an audited data/model card; independently held-out metrics; calibrated uncertainty and OOD results; engineering acceptance thresholds; prospective evidence; reproducible training artifacts; named owner; approval/rollback records; and monitoring for drift. No universal error threshold is assumed here. A customer must define what “accurate enough” means for each property and application.

A future commercial service must additionally verify authentication/authorization, tenant isolation, audit retention, quotas, reproducible job retries, recovery, license obligations and safe import behavior. A unit-test pass does not establish any of these production qualities by itself.
