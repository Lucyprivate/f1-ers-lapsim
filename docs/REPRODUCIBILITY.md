# Reproducibility

## Public package and frozen provenance

The public layout contains one current package, `f1ers_v7`. Supporting code lives under `f1ers_v7/_core`. Namespace migration, dependency consolidation and English documentation change the source files and hashes. This public checkout is **not** the original frozen checkout. Its source alone cannot establish exact historical verification.

The D experiment used a source freeze set before execution:

| Receipt | SHA / identifier |
|---|---|
| Original local source commit | `9be703eadc8a8b25f2b0b28fd225e0f0952d6210` |
| Original D manifest SHA256 | `6c3ec2c07c1e4eddf41af7ee4c4c0300221f8666ca659f311e19a65f17fa97f4` |
| Selected checkpoint SHA256 | `c72a7137cf919786ab4e0f4b3f59614b372f616f4654b749add2740b21b27cd1` |

These identify separately retained local artifacts. The original history, manifests, data access ledger, and ZIP archives remain outside this public release. The table does not imply their presence.

<a id="what-can-be-checked-publicly"></a>

## Public checks

Install the current package.
Run its included tests:

```sh
python -m pip install -e .
python -m f1ers_v7 audit-v7 --root . --offline
python tools/verify_public_snapshot.py
python -m unittest discover -s tests -v
```

Use Python 3.12. `requirements-lock.txt` pins the runtime versions. The audit reports the latest 34-feature deployment controller without loading historical telemetry or starting training. The inventory check compares the published file list and SHA256 hashes. Its receipt is separate from the original experiment freeze.

Guards protect the `python -m f1ers_v7 run-v7 --root . --offline` and `verify-v7` experiment commands. A source-only checkout exits with a missing-private-prerequisites error. It does so before it creates a run, fetches data, or opens FP2. Recover the authorized original artifacts separately to run those commands. This repository includes no download bundle.

The tests check code behavior. They do not establish a real-car model fit or reproduce the omitted historical telemetry experiment. Exact reproduction needs the D result's original frozen source, declared dependencies, geometry/data artifacts, checkpoints, and manifest verification. New runs need a new source/configuration/data/dependency freeze and a separate result identity.

## Fixed experiment contract

The reported run used 800 kg, 400 kW equivalent ICE wheel power, and an ideal 2–4 MJ energy window. Formal integration used a 0.02 s time-step ceiling, 1 m internal-distance ceiling, and 5 m output spacing. The engineering geometry margin was 1 s. Decision notice was 0.5 s, and the deployment tail guard was 1000 m. These are model safeguards, not new FIA rules.

Three seeds each received 20 cold and 20 warm SAC episodes. BC used 600 epochs. Three 200-epoch DAgger rounds came next. Qualification carried six real simulated laps with energy and rule state. It used no policy-specific periodic shooting or within-lap state resets. The experiment kept failed complete laps and accepted failure prefixes. Formal accuracy scoring excluded partial laps.

The registered new-session protocol is 2026 Austrian FP2, driver 63, in the earliest qualifying MEDIUM stint. Three early laps allow one bounded condition adaptation. Three later laps allow one final scoring access. The shared ledger persists across branches/clones. Local accuracy gates did not pass. That test remains unopened. This public release neither opens it nor creates a fresh scoring opportunity.
