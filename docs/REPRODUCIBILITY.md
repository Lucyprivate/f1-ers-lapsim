# Reproducibility

## Public package and frozen provenance

The public layout contains one current package, `f1ers_v7`, with supporting code under `f1ers_v7/_core`. Namespace migration, dependency consolidation and English documentation change the source files and hashes. A checkout of this public repository is therefore **not** the original frozen checkout and cannot claim exact historical verification from source alone.

The reported D experiment was frozen before execution:

| Receipt | SHA / identifier |
|---|---|
| Original local source commit | `9be703eadc8a8b25f2b0b28fd225e0f0952d6210` |
| Original D manifest SHA256 | `6c3ec2c07c1e4eddf41af7ee4c4c0300221f8666ca659f311e19a65f17fa97f4` |
| Selected checkpoint SHA256 | `c72a7137cf919786ab4e0f4b3f59614b372f616f4654b749add2740b21b27cd1` |

These identify separately retained local artifacts. The original history, manifests, data access ledger and ZIP archives remain preserved outside this public release. Their presence is not implied by the table.

## What can be checked publicly

Install the current package and run its included tests:

```sh
python -m pip install -e .
python -m f1ers_v7 audit-v7 --root . --offline
python tools/verify_public_snapshot.py
python -m unittest discover -s tests -v
```

Use Python 3.12. The runtime versions are pinned in `requirements-lock.txt`. The audit reports the latest 34-feature deployment controller without loading historical telemetry or starting training. The inventory check verifies the published file list and SHA256 hashes; its receipt is separate from the original experiment freeze.

`python -m f1ers_v7 run-v7 --root . --offline` and `verify-v7` are guarded experiment commands. A source-only checkout exits with a missing-private-prerequisites error before creating a run, fetching data or opening FP2. Recover the authorized original artifacts separately to run those commands; no download bundle is included here.

The tests check code behavior. They do not establish a real-car model fit or reproduce the omitted historical telemetry experiment. Exact reproduction of the original D result requires its original frozen source, declared dependencies, geometry/data artifacts, checkpoints and manifest verification. New runs of the reorganized package require a new source/configuration/data/dependency freeze and separate result identity.

## Fixed experiment contract

The reported run used 800 kg, 400 kW equivalent ICE wheel power and an ideal 2–4 MJ energy window. Formal integration used a 0.02 s time-step ceiling, 1 m internal-distance ceiling and 5 m output spacing. The engineering geometry margin was 1 s, decision notice 0.5 s and deployment tail guard 1000 m; these are model safeguards, not new FIA rules.

Three seeds each received 20 cold and 20 warm SAC episodes. BC used 600 epochs followed by three 200-epoch DAgger rounds. Qualification carried six real simulated laps, including energy and rule state, without policy-specific periodic shooting or within-lap state resets. Failed complete laps and accepted failure prefixes were retained; partial laps were excluded from formal accuracy scoring.

The registered new-session protocol is 2026 Austrian FP2, driver 63, earliest qualifying MEDIUM stint: three early laps for one bounded condition adaptation and three later laps for one final scoring access. The shared ledger persists across branches/clones. Local accuracy gates have not passed, so that test remains unopened. This public release neither opens it nor creates a fresh scoring opportunity.
