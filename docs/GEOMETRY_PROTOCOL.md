# Geometry and input consistency study (0.9.0)

Study `spielberg_geometry_registration_2026_02` is a new, bounded child of the closed `spielberg_open_source_control_2026_01` study. Its budget does not replace the parent's 192 physics points, 35 teacher attempts, or zero new SAC attempts. FP2 remains sealed. Source, configuration, historical inputs, dependency installation records and Python are frozen from a clean Git commit before execution.

## Positional diagnosis

Only race laps 3–6 enter geometry estimation. Each of four leave-one-lap-out folds estimates one proper similarity transform, common control-line phase and location clock from its three training laps. Registration is limited to the original phase ±100 m, clock ±1 s, raw units per metre 8–12 and 120 objective evaluations. The withheld lap does not provide a training speed-integral boundary. Limited continuation from the training lap's own two endpoint samples is explicitly recorded as diagnostic support; it cannot extend replay inputs.

One common route family fits periodic lateral offsets with at most 2 m node spacing, equal per-lap positional loss and a periodic second-derivative regularizer. The fixed 20, 40 and 80 m values are regularization lengths, not the earlier Gaussian standard deviations. Bounds encode inherited left/right widths. Continuous geometry still needs independent closure, true arc-length, boundary and curvature checks; bounded nodes alone do not certify a spline between them. No clipping of an invalid fitted route establishes passing geometry.

Every fold and final fit must pass. Position-only validation needs at least 1% improvement over the inherited baseline; values within 1% prefer the smoother candidate. No predicted speed residual, per-lap origin, distance stretching or development-lap registration is used. Historical TUM widths are not surveyed 2026 track limits.

The curvature checks follow the methodological warning in the fixed [TUM minimum-time documentation](https://raw.githubusercontent.com/TUMFTM/global_racetrajectory_optimization/a9995e2f5407f22eb7fb9dceac2b71a35276bb41/opt_mintime_traj/Readme.md); that project's performance is not an accuracy claim for this simulator. Project-owned implementations do not import upstream solver source.

## Separate model and control gates

`GateReport` records input coverage, completion, physical execution, the public rule subset, accuracy, per-lap energy, endpoint repeatability and actual six-carry continuity independently. Replay electrical/control gates are inapplicable. A complete physically feasible autonomous cold lap can enter fitting without a periodic speed/DC/FSM endpoint; the public rule shield remains mandatory. Sustainable control additionally requires nonnegative per-lap energy, endpoint repeatability and actual carried-state continuity.

Replay uses only observed initial speed. Throttle is interpolated in real time and brake is held from the previous sample. Actual raw coverage is a terminal integration event: `input_exhausted` preserves a real accepted prefix, without holding the final control beyond its support. Partial laps receive no formal RMSE or invented sector time.

Only passing geometry permits one 64-point Sobol screen, seed 2026, for the aggregate tire model with independent brake demand. Parameters retain the existing bounds: friction 1.2–2.4, drag area 0.6–2.0 m², downforce area 1.8–6.0 m², replay equivalent power 350–732.5 kW, and brake gain 0.1–1.0. Three distinct complete physical/public-rule feasible starts are mandatory before the three fits, each capped at 300 iterations and 600 evaluations. The original equal-path normalized speed, lap time and sector objective remains unchanged. Ten local parameter sensitivities are descriptive; they cannot select a new candidate.

Route and parameters freeze before the single historical development assessment on laps 7/9. These are development laps, not a new test. Formal accuracy remains average full-lap RMSE ≤15 km/h, every lap-time error ≤2%, every sector-time error ≤1 s. Output resampling at 10/5/2.5 m and internal refinements (0.02 s, 1 m), (0.01 s, 0.5 m), (0.005 s, 0.25 m) are separate tests, with adjacent differences ≤0.10 s and ≤0.5 km/h.

## Bounded control response

The old qualified ideal-energy baseline is used independently of the new route. Six actual carried executions establish a boundary pool. Cold and actual late carried boundaries receive a baseline and sixteen ±0.025 block perturbations. At most four complete responsive blocks per boundary, in block order, receive ±0.0125 probes and internal refinements. Baseline output/internal numerical checks are separate. The total maximum is 132 single-lap integrator calls. Every invocation claims a durable key and numbered slot first; failures, interruptions, clones and branch switches cannot restore that budget.

The real FSM, 1000 m tail guard and causal energy reserve remain active. A phase-changing perturbation is reported as a directional hybrid response, not a smooth Jacobian. This study adopts no new teacher, runs no CasADi search and permits zero SAC episodes. Local lack of cold response does not establish global or carried-state optimality.

## Offline commands and limits

```sh
python -m f1ers_v7.research.cli geometry-audit --root . --data-root HISTORICAL_ROOT --ledger-root SHARED_LEDGER
python -m f1ers_v7.research.cli physics-refit --root . --data-root HISTORICAL_ROOT --ledger-root SHARED_LEDGER
python -m f1ers_v7.research.cli control-response --root . --data-root HISTORICAL_ROOT --ledger-root SHARED_LEDGER
python -m f1ers_v7.research.cli verify-geometry --root . --data-root HISTORICAL_ROOT --offline
```

The canonical ledger is workspace-specific and external to the repository. Results live in ignored `results/geometry_registration_2026_02`; Git contains current source and compact evidence only. Archived datasets, FIA PDFs and private FP2 windows are not published. Restored ledger snapshots are audit evidence, never a fresh authorization. A terminal stage cannot be rerun; a verification reads frozen results without optimizing or scoring late observations again.

Defaults remain 800 kg, 400 kW equivalent ICE wheel power and a 2–4 MJ ideal energy window. Diagnostic 780/820 kg cases describe a declared ±20 kg mass assumption, not measured fuel load. True brake pressure, SOC, power split, transmission and the 2026 surveyed racing corridor remain unknown. Browser interaction and computational validation are recorded separately; browser validation is not performed by these commands.
