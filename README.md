# F1 ERS Lab

A research simulator for causal ERS deployment control at Spielberg, with an implicit vehicle integrator, a rule shield, imitation learning and bounded SAC experiments.

This repository contains the current `f1ers_v7` package. Supporting integration and learning components live under `f1ers_v7/_core`; older standalone project versions are not included. Original local Git history and ZIP archives are retained separately.

The archived D experiment completed **120 SAC episodes** across three seeds. Its selected checkpoint reproduced the analytical teacher: **70.44079066 s** for the last-two-lap mean after six physically carried laps. **No learning gain was demonstrated.** Historical development speed RMSE was **19.62589671 km/h**, above the unchanged 15 km/h gate.

The actor controls positive deployment. Negative signed wheel demand requests maximum regeneration through the same tire, power, energy and slew constraints; it does not add braking force. A 34-feature observation includes episode initial energy. Speed, energy and rule state carry across the control line without within-lap resets.

The public-rule subset is enforced separately from engineering safeguards. Full FIA ECU behavior, real SOC, losses and transmission remain unverified. The registered FP2 test remains sealed.

```sh
python -m pip install -e .
python -m f1ers_v7 audit-v7 --root . --offline
python tools/verify_public_snapshot.py
python -m unittest discover -s tests -v
```

Current research completed 192 physics screening points without viable fit starts, then stopped teacher search after 35 attempts on a diagnosed surrogate-unit defect. **No new SAC episode was started and no gain was demonstrated.** Version 0.8.1 fixes the units, zero-authority proposals and exact replay verification; the corrected formal optimizer has not been rerun. See the [research results](docs/RESEARCH_RESULTS.md).

Version **0.8.1** adds optional open-source-method research: independent brake requests, axle tire budgets, storage/auxiliary loss ledgers, bounded CasADi teacher proposals, 36-feature observations and NumPy versus SB3-aligned SAC audits. Optional modes keep the historical defaults. The model is not released as an accurate real-car predictor.

```sh
python -m pip install -e ".[research]"
python -m f1ers_v7.research.cli audit --root . --offline
```

The research CLI provides `audit`, `physics`, `teacher`, `learning` and `verify` stages. See the [research protocol](docs/RESEARCH_PROTOCOL.md) and [upstream methods](docs/OPEN_SOURCE_METHODS.md). Formal runs require explicit read-only historical inputs and the shared external budget ledger. Archived `run-v7` and `verify-v7` still require their omitted private artifacts.

See [results](docs/RESULTS.md), [reproducibility](docs/REPRODUCIBILITY.md) and [notices](NOTICE.md). MIT applies to project-owned code only; historical telemetry, official FIA documents and large experiment artifacts are omitted.
