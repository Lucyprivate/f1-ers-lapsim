# Open-source research results and corrections

Current source is **0.8.1**. The empirical study was frozen at **0.8.0**, commit
`1cff56a38630c49fc56fcfafe9a0e65d43a6665b`, and stopped after a proposal-model
unit defect was identified. No new SAC episodes were started. The source
corrections are tested separately; this is not a restarted or successful formal
optimization under 0.8.1.

## What was executed

The three declared physics candidates completed **192 Sobol points**: 768 replay
trajectories and 192 autonomous trajectories. Every candidate had zero complete
two-path viable starts, so fitting and candidate development scoring were
withheld. All partial trajectories remain unscored. The axle and loss cases
are diagnostic sensitivities, not accepted model improvements. Their fallback
parameters differ from the qualified ideal-energy control baseline.

Teacher search spent **35 attempts**: 34 completed qualifications, of which
17 passed and 17 failed the per-lap energy requirement. One attempt was
interrupted after its first complete lap. That lap was retained; its in-flight
second-lap memory trace was not retained, and the interrupted candidate has no
formal score. The best retained objective remains **70.4407906618 s**, the
baseline. The planned full search did not complete. The durable stop consumes
the attempt and does not provide a new scoring or optimization opportunity.

The learning stage correctly returned `withheld_teacher_or_physics_gate`.
NumPy and SB3 each spent **0 new SAC attempts** and produced **0 learner
certifications**. The 120-episode maximum was reserved, not consumed. Archived
D's 120 episodes belong to a different earlier study.

## Accuracy and energy

These are saved baseline predictions against historical observations, not a
new vehicle model selected from development data. The post-stop development
score uses only historical laps 7/9 and does not select a candidate.

| Baseline metric | Calibration 3-6 | Historical development 7/9 |
|---|---:|---:|
| Whole-lap mean speed RMSE (km/h) | 20.3488 | 19.6259 |
| Sector 1 speed RMSE (km/h) | 20.4992 | 19.0768 |
| Sector 2 speed RMSE (km/h) | 18.9909 | 19.9659 |
| Sector 3 speed RMSE (km/h) | 21.6276 | 19.6528 |
| Maximum absolute lap-time error (%) | 2.0935 | 2.7101 |
| Maximum absolute sector-time error (s) | 1.9211 | 1.9863 |

The unchanged whole-lap RMSE <=15 km/h, every-lap time error <=2%, and
every-sector time error <=1 s gates fail. No partial replay accuracy substitutes
for these gates. No model or SAC gain is demonstrated.

The baseline passed six actual carried laps with exact control-line speed,
energy, DC power and FSM continuity. Its cold first lap is 71.1792412829 s;
later laps are 70.4407906618 s with approximately **512.410 J net stored-energy
gain per lap** in the ideal DC-equivalent model.

| Mean sector energy (MJ), historical 7/9 boundaries | Deploy | Harvest | Net |
|---|---:|---:|---:|
| Sector 1 | 1.545759 | 1.121336 | -0.424423 |
| Sector 2 | 1.757997 | 2.026731 | +0.268734 |
| Sector 3 | 0.771520 | 0.927721 | +0.156201 |

Exact values and calibration-sector accounting are in the [summary](../evidence/research/summary.json).
Sector balance residuals are about 1e-14 MJ. A small ideal-energy surplus is
not evidence of real battery-loss or auxiliary-load robustness.

## Findings and implemented self-corrections

Joint-force replay failures in the two aggregate models have zero remaining
longitudinal capacity and zero actual longitudinal force. The data support
investigating lateral overshoot, route/driver mismatch and earlier braking;
they do not establish excessive executed braking as the cause. Independent
brake demand alone did not rescue a complete lap in this finite screen. The
axle model also rejected all autonomous screens; the inherited aggregate
driving envelope is not established as compatible with its axle assumptions.
See [physics falsification](RESEARCH_PHYSICS.md).

Two cold-lap CasADi references have all-zero free response columns. The third
has one responsive column uniquely fixed by terminal-speed equality. Thus all
three proposals lack useful feasible action sensitivity. They are not evidence
of successful energy/time optimization. The frozen proposal also converted MJ
to a numeric kW value where the speed equation required W, understating response
by 1000. These findings triggered the controlled stop. See [surrogate
falsification](RESEARCH_SURROGATE.md).

Source 0.8.1 uses an explicit **1e6 MJ-to-J** conversion and rejects zero useful
action dimension before solver construction. A two-column regression verifies
that speed-neutral redistribution can change interior motion/time; an
independent mechanical-work scale checks the SI conversion. Energy/box
constraints can still remove feasibility from a positive-dimensional proposal.
Cold fixed-phase authority does not imply six-carry optimality.

The original frozen replay verifier compared in-memory tuple metadata with JSON
lists and falsely returned failure. An independent six-lap recomputation at the
unchanged frozen source produced a certificate **byte-identical** to the saved
baseline. Source 0.8.1 compares persisted JSON exactly; tests still reject a
single-float-ULP or metadata change. Learner certification metadata now counts
actual certificates rather than using an unconditional true flag.

## Verification and limits

Current source regression: **145 run, 143 passed, 2 skipped, 0 failed**. The two
link-creation tests lack Windows privileges; symlink/junction creation was not
experimentally verified. Ten NumPy/autograd differential fixtures pass at about
4.55e-13 maximum error. The baseline cold-first and carried-late numerical
checks pass: maximum adjacent internal time change is 0.00264846 s and speed
RMSE is below 0.0067 km/h; output resampling is recorded separately.

The standalone byte verifier checked **2612 declared artifact files**, exact
inventory, manifest/freeze hashes and the independent canonical completion
receipt. It does not simulate a car or establish accuracy. The public summary
pins the stopped result manifest; use [verification instructions](RESEARCH_PROTOCOL.md).
Fresh optimizer/training replay is not claimed. The corrected 0.8.1 formal
optimization remains unrun; the closed protocol identity cannot be restarted.

The original local repository remains at
`b418dd27b071c413477b646db0c813350ad357ba`. FP2 stays sealed. Full FIA ECU behavior,
actual SOC, real-car gains, global/infinite-horizon optimality and browser
interaction verification remain unverified. Parameters and bounds were not
expanded, and no median-error metric was substituted for failed RMSE gates.
