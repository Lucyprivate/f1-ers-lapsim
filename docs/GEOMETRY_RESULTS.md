# Geometry registration and physical-model research (0.9.0)

Study `spielberg_geometry_registration_2026_02` used frozen source `5df3736ef4c20720bcfc7430a070abd4ecbd5203`.
This is a finite calibration and control-response investigation. No teacher or learned actor is adopted, no lap-time gain is claimed, and real SOC and complete FIA validation remain pending.

## Calibration-only registration and geometry

Geometry gate: **False**. No boundary-stable calibration route improves unbiased position CV by at least 1%
Registration uses race laps 3–6 only. Each held-out fold excludes that lap from the shared transform and speed support; 20/40/80 m denote second-derivative regularization lengths.

| Held-out lap | Phase (m) | Clock (s) | Position fit RMSE (m) | Optimizer residual calls | Local phase / clock uncertainty | Held-out outside fraction |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 3 | 182.2699 | -0.035797 | 7.3448 | 20 | 0.6563 m / 0.010351 s | 0.1875 |
| 4 | 179.7789 | -0.017970 | 9.7810 | 20 | 0.8759 m / 0.013792 s | 0.1898 |
| 5 | 180.9182 | -0.023771 | 9.8106 | 20 | 0.8851 m / 0.013987 s | 0.1831 |
| 6 | 179.0564 | 0.000977 | 8.5086 | 20 | 0.7696 m / 0.012120 s | 0.1919 |

Local residual-based uncertainties do not include correlated GPS or systematic boundary errors. A smooth, closed curve does not establish that the inherited historical corridor represents the 2026 racing line.
Baseline positional cross-validation RMSE under the same fold registration: 2.3225 m.

| Regularization length (m) | Position CV RMSE (m) | All-fold / final gate | Failure evidence |
| ---: | ---: | --- | --- |
| 20 | 0.8774 | False | continuous_curve_touches_or_crosses_width_polygon; leaves_nearest_segment_inherited_boundaries |
| 40 | 1.1438 | False | continuous_curve_touches_or_crosses_width_polygon; leaves_nearest_segment_inherited_boundaries |
| 80 | 1.5518 | False | continuous_curve_touches_or_crosses_width_polygon; leaves_nearest_segment_inherited_boundaries |

Candidate specifications and failed-geometry diagnostics remain in the private artifact set with their SHA-256 identities. No speed residual selected a route.

## Physical refit disposition

Status: `withheld_geometry_gate`. No positional route passed every fold and continuous geometry gate
Reserved screen points: 0; viable points: unavailable; fitting evaluations: 0.
Development laps 7/9 loaded by this study: **False**. They remain historical development data, not a new test.
Model fitting requires full physical and public-rule feasible trajectories; cold-lap SOC and endpoint repeatability remain separate sustainable-control gates.

## Cold and actual carried control response

Status: `directional_diagnostics_completed`. Reserved single-lap calls: 132 / 132.
| Boundary | Complete physical / public responses | Measurable responses | Changed phase schedules | Maximum speed response RMSE (km/h) |
| --- | ---: | ---: | ---: | ---: |
| cold | 32 / 32 | 12 | 10 | 2.898327 |
| carried_late | 32 / 32 | 16 | 14 | 2.878434 |

Phase changes are reported as hybrid directional responses. Numerical refinement and half-amplitude checks do not establish a smooth Jacobian or a globally optimal sustainable policy.

| Baseline boundary | Lap time (s) | Mean calibration RMSE (km/h) | S1 / S2 / S3 speed RMSE (km/h) | Calibration accuracy |
| --- | ---: | ---: | --- | --- |
| cold | 71.179241 | 20.8301 | 21.9018 / 19.2526 / 21.6276 | False |
| late_actual_carry | 70.440791 | 20.3488 | 20.4992 / 18.9909 / 21.6276 | False |
| late_response_reference | 70.440791 | 20.3488 | 20.4992 / 18.9909 / 21.6276 | False |

These accuracy scores use calibration laps 3–6; no development score is silently substituted. SOC here is ideal DC-equivalent model energy. The JSON evidence reports per-lap SOC/endpoint closure and all actual-carry crossing residuals.

**Sector definitions:** calibration speed RMSE and its paired energy table use telemetry timing boundaries projected onto the frozen route. Separate fixed-map energy sectors are also retained; the two partitions are not interchangeable.

| Boundary / telemetry sector | Deployment (MJ) | Harvest (MJ) | Net stored energy (MJ) |
| --- | ---: | ---: | ---: |
| cold / S1 | 0.263557 | 1.061461 | 0.797903 |
| cold / S2 | 1.909700 | 2.083242 | 0.173542 |
| cold / S3 | 0.771520 | 0.932251 | 0.160731 |
| late_actual_carry / S1 | 1.545759 | 1.081063 | -0.464696 |
| late_actual_carry / S2 | 1.757997 | 2.062474 | 0.304477 |
| late_actual_carry / S3 | 0.771520 | 0.932251 | 0.160731 |
| late_response_reference / S1 | 1.545759 | 1.081063 | -0.464696 |
| late_response_reference / S2 | 1.757997 | 2.062474 | 0.304477 |
| late_response_reference / S3 | 0.771520 | 0.932251 | 0.160731 |

Saved numerical comparisons: 40; valid complete-trajectory pairs: 40. Numerical checks passed: **True**.
Output resampling (10/5/2.5 m) and internal integration refinement are separate; each adjacent pair retains the 0.10 s and 0.5 km/h limits.

## Evidence scope and reproducibility

New SAC attempts: **0**. Full teacher search: **0**. FP2 stays sealed. Browser interaction was not tested. No thresholds, parameter bounds, or historical labels were relaxed to declare success.

The [previous research report](RESEARCH_RESULTS.md) and `evidence/research/` describe the earlier frozen 0.8.0 experiment and 0.8.1 corrections; those outcomes are historical evidence rather than new 0.9.0 validation.

Manifest SHA-256: `677a580021e57c9759208662e59c15e1ea26f6b986381d4ea25a50c97aec22c7`.
Freeze SHA-256: `fa65c8c6c5ebd6f902d17a1465affb8794433d0d54c7fd53b7b4618f63df72e3`.
Exact terminal-stage file hashes and compact evidence are in `evidence/geometry/summary.json`. Raw telemetry, complete trajectories, and candidate point arrays are excluded from the public snapshot.

## Interpretation and limitations of the 0.9.0 evidence

All three candidate lines improve held-out positional cross-validation error. The baseline is **2.323 m**; the 20, 40 and 80 m regularization candidates reach **0.877, 1.144 and 1.552 m**, respectively. Nevertheless, every candidate fails the continuous inherited-boundary gate, including its held-out fits. The study therefore selects no replacement line and withholds physical refitting. Improved position prediction is evidence for a smoother common positional reference; it does not establish a valid racing line or explain the speed residuals.

The unchanged ideal-control baseline also fails the prediction-accuracy gate. Its reproduced cold-lap mean calibration speed RMSE is **20.830 km/h**, with Sector 1/2/3 RMSE **21.902/19.253/21.628 km/h**. On the sixth physically carried lap these become **20.349 km/h** and **20.499/18.991/21.628 km/h**. These are comparisons against the calibration laps 3–6; development laps 7/9 were not loaded for this new study. Thus both the candidate geometry gate and the baseline prediction-accuracy gate fail. No two-path accuracy pass or real-car improvement follows from reproducing the old control trajectory.

Model-fitting eligibility and sustainable-control qualification are separate. A complete, physically feasible autonomous trajectory that passes the declared public-rule subset may enter physical fitting even if its cold-lap energy or endpoint closure fails. Sustainable-control qualification additionally requires nonnegative model energy change, the declared speed/DC/rule-state endpoint tolerances, and the actual six-lap carried-state check. A responsive perturbation, a faster partial trajectory, or a legal single lap does not establish a qualified faster sustainable policy. Replay has no measured electrical state: its ERS/public-rule and sustainable-control fields are inapplicable, and incomplete replay receives no formal speed RMSE.

### Energy and sector definitions

The energy coordinate in this study is **ideal DC-equivalent usable model energy** in the frozen 2–4 MJ working interval. It is not measured battery SOC. The cold baseline starts at 2 MJ and ends at about 3.132176 MJ; the sixth carried lap changes from about 3.134226 to 3.134738 MJ. A small DC accounting residual checks the model ledger arithmetic. It cannot validate battery losses, actual SOC, transmission state, fuel use, or the unpublished FIA ECU behavior. The public-rule subset and the engineering energy/closure safeguards are reported separately.

The saved carried-baseline numerical diagnostics report the following net model energy changes:

| Maximum internal step | Net model energy change |
| --- | --- |
| 0.02 s / 1 m | +512.410 J |
| 0.01 s / 0.5 m | +820.957 J |
| 0.005 s / 0.25 m | +254.377 J |

The largest adjacent change, **566.580 J**, exceeds the finest-resolution positive margin of **254.377 J**. Speed/time convergence can pass while a robust positive energy margin remains unestablished. One carried refinement also changes the executed hybrid FSM event schedule. These are saved numerical diagnostics under the existing rule and tolerance definitions; they do not introduce a new formal energy rule or change a saved gate. They limit the strength of the sustainability claim.

The compact evidence keeps two sector partitions with explicit labels:

| Partition | Purpose | Where to read it |
| --- | --- | --- |
| Observed telemetry sector timings projected onto the frozen route, separately for each calibration lap | Pair speed RMSE and time error with model energy over the same comparison windows | Public `baseline_calibration` cards: `mean_calibration_partition_sector_energy`; individual energy records remain in the hashed private per-calibration-lap `sector_energy` records |
| Fixed simulator-map sector boundaries | Describe the simulated lap on its declared map partitions | Public `baseline_calibration` cards: `fixed_geometry_partition_sector_energy`; private attempt metrics: top-level `sector_energy` |

They are not interchangeable. For example, the cold baseline's fixed-map Sector 1 takes about **19.691 s**, whereas the calibration-lap-3 projected Sector 1 prediction takes about **17.552 s**. Averaging the four calibration energy partitions means repartitioning the same simulated baseline trajectory four ways; it does not represent four independent SOC observations. Net model energy change and within-lap energy swing answer different questions. The within-lap range is retained as `energy_swing_mj` in the hashed private attempt metrics; the compact public cards omit that field.

### Post hoc localization of the inherited-boundary failure

The continuous gate already failed before the following localization was computed. This is a **post hoc diagnostic**, outside the frozen formal study manifest. It changes no geometry score, candidate choice, threshold, or gate, performs no new fit or integrator execution, and accesses neither original telemetry nor FP2.

All three full-calibration candidate curves exit and re-enter the same inner/right inherited polygon near its scaled model coordinate **1210 m**. The pairs occur at boundary edges 278/279 and spline segment 694; they are away from the control-line seam and the source-parameter seam, and neither crossing is within 0.1 m of a polygon miter vertex. The baseline has no such crossing.

| Regularization | Exit → re-entry on candidate arc length (m) | Outside arc length (m) | Estimated maximum penetration into the explicit inherited polygon (m) |
| --- | --- | --- | --- |
| 20 m | 1209.421 → 1210.064 | 0.644 | 0.060 |
| 40 m | 1207.307 → 1208.525 | 1.218 | 0.140 |
| 80 m | 1206.282 → 1207.629 | 1.348 | 0.165 |

Maximum penetration is sampled at approximately 0.001 m spacing within each localized excursion. It is an estimate against the explicit inherited miter polygon, not a surveyed road boundary. The formal 0.5 m nearest-segment sample grid reported positive minimum clearance for the 20 m candidate; a post hoc 0.1 m grid finds four outside samples and a minimum clearance of about −0.031 m. This explains why a positive sampled clearance cannot replace the continuous containment gate. The nearest-segment corridor and explicit miter polygon are different inherited boundary representations, and their distances must not be conflated.

The inherited corridor, calibration GPS errors, and the actual 2026 racing line remain uncertain. These model intersections do not identify a real 2026 track-limit violation. Position/geometry consistency alone also cannot prove that the route caused the observed dynamics error. The remaining source of speed error is not identifiable from this study.

Post hoc provenance and compact localization are retained in [`evidence/geometry/boundary_posthoc.json`](../evidence/geometry/boundary_posthoc.json), derived from the task-local `work/geometry_boundary_posthoc.json` with its exact source-byte hash retained. The compact file omits point coordinates and candidate point arrays. Its own publication hash is recorded separately in `evidence/geometry/publication_addendum.json`; it is outside the formal manifest. The original experiment source commit is `5df3736ef4c20720bcfc7430a070abd4ecbd5203`.

| Evidence item | SHA-256 |
| --- | --- |
| Task-local post hoc source JSON | `e5790abeea2d36bbbd3e64dfba1a83ba42408684587796771d5cc5ebaa0e5c5e` |
| Post hoc helper | `4d726ecbfbc540bb970fcea7ecfab18ed1f3b0ce163b5528bc461e1e66f9c458` |
| Frozen geometry module | `e0cf623bfccd16d9bfdab3c0e8062facac992776710a8573cc6c6debafab6af3` |
| Inherited centerline input | `277b2854c4323021b10b6b63ebfe6a6cc971f494c0728202e910d67d7d65dc8c` |
| Frozen baseline route input | `dd8516946053f5fdd4a5fe0b2f3df1bfe10d90fea84fe2ce125da4763da7fc15` |
| Full 20 m candidate | `34b164e86b98db3dbc697ec144fdf4e060172df80650621345457cca62b2211a` |
| Full 40 m candidate | `30375adf26ec73c5ff9eff9c3e2950f4954b7e4e99788dee0ae490b633fa660d` |
| Full 80 m candidate | `8a96fdc9f5d3831e080cad905c45fdc7287637c50f884e40104e905858c080cc` |

### Publication and reproduction scope

The archived D actor uses its original **34-feature** observation. The current research diagnostic environment uses **36 features**, adding episode initial speed and initial DC power to that archived schema. The new diagnostic study does not train or adopt a new actor or teacher. The archived 120-episode D experiment and the closed parent study's 192-point physical screen/35 teacher attempts are separate historical results.

The public Git snapshot supports source inspection and synthetic regression tests. Reproducing the historical-input scores additionally requires the omitted authorized historical inputs and private artifact set. The canonical experiment ledger is outside the repository and tied to the local protocol identity; a restored ledger snapshot is evidence of past access, not a fresh budget or permission to repeat sealed scoring. Saved hashes and stage receipts establish evidence integrity; they do not certify passing geometry, prediction accuracy, full FIA compliance, or real SOC.

No new SAC episode, full teacher search, learned-policy adoption, or lap-time gain is claimed. FP2 remains sealed and browser interaction was not tested.

## Final bounded-response audit

All 132 reserved executions completed with physical/public-rule feasible trajectories. Of those, 111 passed the existing single-lap control qualification; only the baseline was tested over six actual carried laps. None of the 132 calibration accuracy evaluations passed. Mean speed RMSE spans 19.901–20.867 km/h, while Sector 1/2/3 ranges are 20.314–21.994 / 17.742–19.281 / 21.588–21.667 km/h. These calibration comparisons are diagnostic repetitions on laps 3–6, not independent validation samples.

All 40 adjacent numerical comparisons passed the declared speed/time checks. All 44 frozen numerical executions retained the original action hashes. The 16 half-amplitude runs completed, with nine meeting single-lap control qualification. The largest scaling discrepancy between twice the half-amplitude response and the full-amplitude response is 0.173319 s and 0.288039 MJ. No amplitude-linearity threshold was preregistered; this does not establish a smooth Jacobian, an adopted redistribution dimension or an optimal sustainable policy.

### Portable saved-evidence verification

The frozen object-only leaf parser rejects a valid list-root mass-sensitivity JSON. The additive `tools/verify_geometry_evidence.py` entry accepts strict JSON values for leaf diagnostics while still requiring object roots for the summary, freeze and manifest. It retains byte pins, path checks, duplicate-key/nonfinite rejection and inventory checks. Original tracked experiment Python/configuration bytes remain unchanged. Use this current entry after restoring the derived result tree:

```sh
python tools/verify_geometry_evidence.py RESTORED_RESULTS/geometry_registration_2026_02
```

This command checks saved bytes and never starts a simulation or reopens a scoring budget. The frozen `verify_geometry_artifacts.py` module is retained byte-for-byte as part of the frozen source history; its list-root restriction is not used by the current portable entry.
