# Open-source control research (current source 0.8.1)

The original frozen checkout, tags, data and deliveries are unchanged. This
branch extends the current package through `f1ers_v7.research`. Optional physical
modes default off. Archived D results remain historical evidence; they are not
results of the new algorithms.

## Running the stages

Use Python 3.12 and the pinned dependencies. The core installation stays small;
the research extra adds PyTorch, Stable-Baselines3, Gymnasium and CasADi.

```sh
python -m pip install -e ".[research]"
python -m f1ers_v7.research.cli audit --root . --offline
python -m f1ers_v7.research.cli audit --root . --data-root /authorized/historical/root --offline
```

The historical root must contain original Spielberg TUM CSVs, processed track
and metadata, calibration-only registration, and the original race telemetry
window. It is read-only. No old experiment result directory is a prerequisite.
The raw window contains historical laps 3-9; processed calibration and
development views are separate. There is no FP2 acquisition or unsealing entry
point in this research runner.

For formal execution, commit all source/configuration files and pass
`--data-root`, `--ledger-root`, and one result `--name` to each stage:

```sh
python -m f1ers_v7.research.cli physics --root . --data-root DATA --ledger-root LEDGER --offline
python -m f1ers_v7.research.cli teacher --root . --data-root DATA --ledger-root LEDGER --offline
python -m f1ers_v7.research.cli learning --root . --data-root DATA --ledger-root LEDGER --backend both --offline
python -m f1ers_v7.research.cli verify --root . --data-root DATA --offline --replay
```

`LEDGER` is the canonical external `audit/research-control` directory, beside the
pre-existing `audit/test-access` ledger. Its absolute location is inherited from
the original workspace protocol. A clone cannot create a fresh experimental
identity by changing this argument. Adapting the public source for a different
workspace is a new study, not continuation of this registered experiment.

Each formal stage has an exclusive start receipt. An interrupted stage retains
its artifacts and cannot be restarted to reset a search or episode budget.
Automatic resumption of an interrupted optimizer is not implemented. Episode
attempts are spent before reset, even when an exception or physical failure
occurs. Source, data, configuration, Python/dependency versions and installed
distribution RECORD hashes are frozen. Continuation requires the same clean
commit and dependencies. Reports and trajectories are ignored by Git.

## Qualification and selection

Only race laps 3-6 enter diagnostics and fits. Laps 7/9 remain reused development
data; 10-12 are not new tests. Neither diagnostic velocities nor observed future
speeds update an integrated state. The accuracy thresholds remain 15 km/h mean
whole-lap RMSE, 2% per-lap time error and 1 s per-sector time error. Partial laps
have no formal accuracy score.

Physics candidates follow baseline, independent braking request, then axle
budget. Each has 64 seeded Sobol calibration trials; fewer than three complete
physical starts stops fitting. Three eligible starts each have at most 300
iterations and 600 evaluations. Axle/loss values are engineering assumptions,
not extra fitted degrees of freedom. Every accepted prefix and failure is saved.

The learning line keeps the qualified ideal-energy model fixed. A changed
physics candidate does not replace it midway. A model-only comparison after
accuracy failure is explicitly authorized, but physical/numerical/differential
failure stops training. The 36 observations expose initial speed, energy and DC
closure targets for an approved boundary pool; arbitrary hidden initial FSM
histories are rejected. Full ECU Markov sufficiency is not claimed.

The teacher optimizes actions only: three starts, 16 equal arc-length parameter
blocks and one 32-block refinement. Commands are sampled on the unchanged 5 m
driver decision grid; a parameter boundary therefore takes effect at its next
driver decision. CasADi multiple shooting uses a frozen baseline conditional
mode schedule and approximate block dynamics. Blocks intersecting HOLD/LIMITED
keep their baseline action. It neither optimizes FSM transitions nor certifies
the exact rules. Every proposed action sequence is filtered through the actual
integrator and rule shield for six physically carried laps. The global exact
search cap is 600, including failed attempts. Final certification and numerical
audits are counted separately. Adoption requires a gain above 0.10 s plus
numerical qualification. Global optimality and infinite-horizon sustainability
are not established by a finite six-lap check.

The same 0.10 s adoption cutoff applies to learned candidates. Smaller observed
physical gains remain diagnostics until numerical certification; the adopted
winner may remain BC. The report does not call an unaudited smaller-gain actor
physically or numerically failed.

The NumPy/SB3-aligned comparison shares BC initialization, normalization,
fixed temperature, second-based reward scaling and attempted-episode budget.
The SB3 adapter deliberately differs from stock `SAC.learn`; those differences
are recorded. A constant teacher produces an analytical clone. DAgger is not
reported without an expert that replans from actual learner states.

## Verification limits

`verify` checks stored artifacts, input hashes, the canonical run claim and the
frozen Git tree. `--replay` also recomputes the baseline six carried laps. It does
not spend fresh optimizer/training budgets or rescore a final test. Complete
optimizer/training replay is not claimed by an artifact-hash check.

Public Git contains source and small summaries only. Telemetry, FIA documents,
checkpoints, large trajectories and private archives are excluded. Browser
interaction verification and real SOC/full FIA ECU certification remain pending.


## Closed study and read-only artifact verification

The `spielberg_open_source_control_2026_01` run is closed with a controlled
teacher stop and zero new SAC attempts. Its source is frozen0.8.0 at
`1cff56a38630c49fc56fcfafe9a0e65d43a6665b`. Current0.8.1 fixes the diagnosed
proposal/serialization/metadata defects. It has not repeated the formal search.
Do not delete ledgers, rename the run or switch clones to reset the closed
identity. A separately registered correction experiment must preserve the
shared test identity and previous attempted budgets.

The current source-sensitive research CLI intentionally reports source drift
against the old freeze. To check restored artifacts from current source without
executing old or new models:

```sh
python tools/verify_public_snapshot.py
python tools/verify_research_artifacts.py --artifacts /restored/opensource_control_2026_01 --completion /restored/completion.json
```

The standalone verifier uses the tracked `evidence/research/summary.json` as its
pin. It checks exact manifest/freeze hashes, inventory and optional completion
payload, rejects path/link/tamper cases, and neither modifies artifacts nor
consumes training budget. It establishes stored byte identity only. Private
telemetry and FIA documents are not included in public Git or the new research
artifact archive. Authorized historical inputs remain necessary for simulation.

For source-sensitive saved-score verification, use a detached checkout of the
frozen commit and the archived research CLI **without** `--replay`. The old
`--replay` has a tuple/list comparison defect; the preserved independent replay
helper compared persisted certificates and reproduced all six laps byte for
byte. The archive includes that helper and its receipt. Source0.8.1 fixes the
comparison without changing the old result or relaxing tolerances. The new
completion receipt and tracked summary independently pin the final manifest.
