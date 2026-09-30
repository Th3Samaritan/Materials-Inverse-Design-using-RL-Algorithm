# Research source register

Accessed 8 September 2026. This is a targeted research review, not an exhaustive systematic literature review. Dates below refer to publication when identifiable; software documentation is a living source. Search crawl dates were not treated as publication dates. Source assertions and MAST engineering recommendations are distinguished in the roadmap.

| Source | Type/date | Why it matters | Limit on inference |
|---|---|---|---|
| [AIMATDESIGN](https://www.nature.com/articles/s41524-025-01894-x) | Primary research, npj Computational Materials, 2026 | Knowledge-augmented RL and comparison with classical/active-learning methods. | Does not validate MAST's steel simulator or show RL always wins. |
| [MatterGen](https://www.microsoft.com/en-us/research/blog/mattergen-a-new-paradigm-of-materials-design-with-generative-ai/) | Primary developer research account, 16 January 2025 | Conditional inorganic crystal generation. | Not a manufacturing-history predictor. |
| [Generative active learning for polymer rheology](https://www.nature.com/articles/s41524-025-01900-2) | Primary research, 15 December 2025, 2026 volume | Architecture-aware inverse design of viscosity profiles using simulation. | Computational evidence, not demonstrated factory formulation performance. |
| [Composite cure constrained Bayesian optimization](https://arxiv.org/abs/2505.24775) | Primary preprint, 30 May 2025 | Cure optimization with numerical simulation and constraints. | Preprint/simulation limitations; resin and equipment transfer need testing. |
| [Materials Project methodology](https://docs.materialsproject.org/methodology/materials-methodology/overview) | Official technical documentation | Scope of first-principles materials calculations. | Atomistic calculated properties are not finished-part mechanical qualification. |
| [pycalphad](https://pycalphad.org/docs/latest/) | Official technical documentation | CALPHAD equilibrium framework and extension path. | Database, kinetic modeling and licensing remain separate needs. |
| [BoTorch constraints](https://botorch.org/docs/constraints) | Official technical documentation | Distinguishes parameter and outcome constraints. | An optimizer cannot compensate for an inaccurate surrogate. |
| [Constrained multi-objective BO tutorial](https://botorch.org/docs/v0.16.1/tutorials/constrained_multi_objective_bo) | Official versioned tutorial | Implementable acquisition-method reference for a future module. | BO is researched here, not yet integrated. |
| [2026 AI/ML smart manufacturing roadmap](https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing) | NIST publication record, 3 July 2026 | Industrial integration, data and reliability challenges. | General manufacturing context, not MAST customer validation. |
| [Citrine](https://citrine.io/) | Vendor website | Existing materials-informatics product positioning and enterprise expectations. | Vendor claims are not independently verified performance evidence. |
| [MaterialsZone](https://www.materials.zone/) | Vendor website | Materials-data and R&D workflow positioning. | No inference about market size, prices or measured comparative accuracy. |
| [Thermo-Calc 2026b](https://thermocalc.com/news/thermo-calc-2026b-release-overview/) | Official vendor release documentation, 2026 | Existing material/process/property workflows and integration context. | Commercial licenses and independent validation still required. |

Additional research surfaced but was not used as a foundation for implementation claims: [target-oriented Bayesian optimization](https://www.nature.com/articles/s41524-025-01704-4), [InvDesFlow-AL](https://www.nature.com/articles/s41524-025-01830-z), and [DFT uncertainty across databases](https://arxiv.org/abs/2007.01988). These are useful follow-up reading, not evidence that current MAST models are validated.

No source dataset, proprietary thermodynamic database, commercial material-property table or external model was downloaded into training during this work. Public accessibility does not settle redistribution or commercial-use rights.
