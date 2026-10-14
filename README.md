# F1 ERS Lab — v7

A research simulator for causal ERS deployment control at Spielberg. It uses an implicit vehicle integrator, a rule shield, imitation learning, and bounded SAC experiments.

This repository contains the current `f1ers_v7` package. Supporting integration and learning components live under `f1ers_v7/_core`. The public package excludes older standalone project versions. Original local Git history and ZIP archives are retained separately.

The latest frozen experiment completed **120 SAC episodes** across three seeds. The selected checkpoint matched the analytical teacher. Its last-two-lap mean after six physically carried laps was **70.44079066 s**. **The experiment showed no learning gain.** Historical development speed RMSE was **19.62589671 km/h**, above the unchanged 15 km/h gate. The model is not released as an accurate real-car predictor.

The actor controls positive deployment. Negative signed wheel demand requests maximum regeneration through the same tire, power, energy, and slew constraints. It does not add braking force. A 34-feature observation includes episode initial energy. Speed, energy and rule state carry across the control line without within-lap resets.

The implementation checks the public-rule subset separately from engineering safeguards. Full FIA ECU behavior, real SOC, losses and transmission remain unverified. The registered FP2 test remains sealed.

```sh
python -m pip install -e .
python -m f1ers_v7 audit-v7 --root . --offline
python tools/verify_public_snapshot.py
python -m unittest discover -s tests -v
```

The package version is **0.7.1**. This update groups runtime dependencies and documents the existing frozen experiment. It reports no new training result. `run-v7` and `verify-v7` need private frozen artifacts omitted from the repository. They refuse to fetch or substitute those artifacts automatically.

See [results](docs/RESULTS.md), [reproducibility](docs/REPRODUCIBILITY.md) and [notices](NOTICE.md). MIT applies to project-owned code only; historical telemetry, official FIA documents and large experiment artifacts are omitted.
