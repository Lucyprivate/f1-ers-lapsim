# Conservative cold-lap surrogate: authority falsification

This diagnosis uses saved artifacts only. It spends no optimization, simulation, or training budget and does not access FP2. This diagnosis was produced during the active 0.8.0 run, which subsequently stopped. The completed stop receipt and immutable manifest are pinned in [the public summary](../evidence/research/summary.json). Current 0.8.1 fixes the SI conversion and refuses zero useful action dimension; its corrected optimizer has not been formally rerun.

Frozen source commit: `1cff56a38630c49fc56fcfafe9a0e65d43a6665b`. Protocol: `spielberg_open_source_control_2026_01`.

## Main finding

All three cold first-lap surrogates have zero useful feasible action sensitivity. This explains why the saved CasADi proposals provide no evidence of an energy/time optimum. It does not prove that the exact controller or another control family is optimal.

| Constant start | Pinned / 16 | Free NORMAL blocks | Nonzero free coefficients (MJ per deployment fraction) | Sensitive dimensions after terminal speed equality |
|---|---:|---|---|---:|
| +0 | 13 | 1, 15, 16 | 0 (all free columns) | 0 |
| -0.25 | 12 | 1, 12, 15, 16 | block 12: 0.981202442781166 | 0 |
| +0.25 | 12 | 1, 14, 15, 16 | 0 (all free columns) | 0 |

## Formula and constraints

Source: [frozen 0.8.0 teacher, lines 75-138](https://github.com/Lucyprivate/f1-ers-lapsim/blob/1cff56a38630c49fc56fcfafe9a0e65d43a6665b/f1ers_v7/research/teacher.py#L75-L138). Each of 16 blocks has equal arc length. The stored deployed DC energy defines `c_i = deploy_i / ((u_ref_i+1)/2)`. With `f_i=(u_i+1)/2`, the squared-speed perturbation accumulates `A_i*(f_i-f_ref_i)`, where `A_i=2*0.95*c_i*1000*ds_i/(dt_i*800*max(mean(vref_i,vref_i+1),1))`. Harvest is fixed, and `e[i+1]=e[i]-c_i*f_i+harvest_i`. Any block intersecting a saved HOLD or LIMITED phase pins `u_i` to its reference. The terminal equality is `v_end=v_initial`; energy stays in [2,4] MJ and must end no lower than it starts.

The phase set uses `state_after_decide` at decision-start positions. This is the frozen approximation used by the source, not a reconstruction of every internal substep transition.

## Exact free-column evidence

### Start +0: candidate 0001 → proposal 0002

| Free block | Arc interval (m) | Deploy (MJ) | Fixed harvest (MJ) | c_i (MJ / fraction) | A_i (m²/s² / fraction) | Saved proposal u |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.000000000–267.812242594 | 0 | 0.650409100913864 | 0 | 0 | 0 |
| 15 | 3749.371396313–4017.183638907 | 0 | 0.210404684885757 | 0 | 0 | 0 |
| 16 | 4017.183638907–4284.995881501 | 0 | 0 | 0 | 0 | 0 |

Cold reference lap: 71.179241282859 s, deploy 2.9447778055159 MJ, harvest 4.07695421520181 MJ. Endpoint speed minus initial: 0 m/s. Maximum proposal change on any nonzero coefficient column: 3.14001457178502e-16.

Saved exact qualification: reference passed=True, saved laps=6, objective=70.44079066181877; proposal passed=True, saved laps=6, objective=70.44079066181877. These are finite executed qualifications, not proof of surrogate/global optimality.

### Start -0.25: candidate 0003 → proposal 0004

| Free block | Arc interval (m) | Deploy (MJ) | Fixed harvest (MJ) | c_i (MJ / fraction) | A_i (m²/s² / fraction) | Saved proposal u |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.000000000–267.812242594 | 0 | 0.650409100913864 | 0 | 0 | 0 |
| 12 | 2945.934668532–3213.746911126 | 0.367950916042937 | 0 | 0.981202442781166 | 2.3426688829234 | -0.25000067372768647 |
| 15 | 3749.371396313–4017.183638907 | 0 | 0.210398486182879 | 0 | 0 | 0 |
| 16 | 4017.183638907–4284.995881501 | 0 | 0 | 0 | 0 | 0 |

Cold reference lap: 71.193179650371 s, deploy 2.63351687477819 MJ, harvest 4.032553724711 MJ. Endpoint speed minus initial: 5.07209563238575e-09 m/s. Maximum proposal change on any nonzero coefficient column: 6.73727686473224e-07.

Only block 12 has a nonzero free coefficient. Summing the squared-speed recursion and applying `v_end=v_initial` forces `u12 = u_ref + 2*(v_initial²-vref_end²)/A12 = -0.25000067372762153`. The saved proposal is -0.25000067372768647; its roughly 6.74e-7 change closes reference endpoint roundoff rather than creating useful optimization authority.

Saved exact qualification: reference passed=True, saved laps=6, objective=70.78344216364873; proposal passed=True, saved laps=6, objective=70.78344234864599. These are finite executed qualifications, not proof of surrogate/global optimality.

### Start +0.25: candidate 0005 → proposal 0006

| Free block | Arc interval (m) | Deploy (MJ) | Fixed harvest (MJ) | c_i (MJ / fraction) | A_i (m²/s² / fraction) | Saved proposal u |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.000000000–267.812242594 | 0 | 0.650409100913864 | 0 | 0 | 0.17911400226433852 |
| 14 | 3481.559153720–3749.371396313 | 0 | 0.721193937767233 | 0 | 0 | 0.17911400226433852 |
| 15 | 3749.371396313–4017.183638907 | 0 | 0.210398485742256 | 0 | 0 | 0.17911400226433852 |
| 16 | 4017.183638907–4284.995881501 | 0 | 0 | 0 | 0 | 0.17911400226433852 |

Cold reference lap: 71.426263612476 s, deploy 2.66171238082655 MJ, harvest 4.0584243671713 MJ. Endpoint speed minus initial: 5.47272804851673e-09 m/s. Maximum proposal change on any nonzero coefficient column: 2.53053689114324e-10.

Saved exact qualification: reference passed=False, saved laps=2, objective=None; proposal passed=False, saved laps=2, objective=None. These are finite executed qualifications, not proof of surrogate/global optimality.

## Source-unit limitation

The saved deployment integral uses `positive_dc_kw*dt_s/1000`, so `c_i` is in MJ. The source computes `power_delta = c_i/dt_i*1000*eta*(f_i-f_ref_i)`, a quantity in kW, but inserts its numerical value directly into `2*power_delta*ds/(mass*speed)`, whose SI squared-speed increment requires watts. The written speed sensitivity is therefore 1000 times smaller than the SI mechanical-power expression (`c_i/dt_i*1e6`). For the only nonzero free column (start −0.25, block 12), the source-derived A is 2.342668882923403 m²/s² per fraction; the SI-consistent counterpart would be 2342.668882923403.

These tables reproduce the original frozen source as written. Its results remain unchanged; current 0.8.1 source corrections are reported separately. This unit limitation does not change the authority finding: zero columns remain zero, and a single sensitive column remains uniquely constrained by the terminal-speed equality. The tiny roundoff-closing proposal change would have a different magnitude under an SI-correct expression. This is another reason not to interpret the proposals as validated physical energy/time optimization.

## Cold versus carried limitation

The cold start is at the 2 MJ floor, so the first free block has no recorded deployment. Later laps carry energy above that floor, which can make the same action block physically active. Proposals may therefore alter later executed laps through columns that were inactive in the cold surrogate. The final 1000 m tail guard makes the late free deployment columns zero here; it is an experimental control-family restriction, not a FIA requirement. Exact six-carry qualification must remain separate from surrogate evidence.

## Baseline request and execution diagnostics

Raw baseline action is exactly zero throughout, with zero raw [-1,1] saturations. It asks for half of the request cap before filters on nonnegative wheel demand. The accepted maximum DC is 175 kW, compared with 350 kW public maxima; absence of raw saturation does not imply unused legal deployment authority, since energy, geometry, tail and FSM constraints restrict requests and execution.

| Saved baseline lap | Nonnegative / negative wheel decisions | Zero averaged DC on nonnegative demand | Energy-filtered requests | Geometry-filtered requests | Tail decisions | Net energy (MJ) |
|---:|---|---|---:|---:|---:|---:|
| 1 | 722 / 135 | 446 (34.856101 s) | 69 | 349 | 170 | 1.13217640968591 |
| 6 | 722 / 135 | 315 (26.021353 s) | 31 | 372 | 170 | 0.000512409967911687 |

Filter categories overlap. Phase counts/energy are grouped by phase after decision and use whole decision averages, so they do not resolve transitions within a decision. A positive executed DC with zero outgoing request can reflect committed HOLD/ramp/slew. Conversely, a positive request with zero averaged execution is not by itself a failure. Saved predecision public caps, filtered requests and averaged executed powers are distinct quantities.

The integration kernel checks beginning, midpoint and end joint tire force, wheel demand, energy reserve, electrical cap and slew. Existing force-excess/mechanical-residual fields are retained in the JSON. No new vehicle simulation or exact force-cap reconstruction was performed; the evidence does not isolate every zero-power sample to one cap or subtract an assumed constant ICE power from an unknown internal wheel-force headroom.

## Reporting boundary

Call the CasADi component a fixed-phase conservative proposal generator whose useful cold-lap action authority was falsified. Do not call these proposals evidence of successful energy/time optimization. Registered exact pattern search remains an independent finite comparison capable of changing pinned blocks and actual FSM timing. Any gain or absence of gain must be qualified by exact saved six-carry and numerical gates.

No global optimum, no-better-policy result, infinite-horizon stability, real-vehicle gain, full FIA ECU compliance or actual SOC validation is established. A future separately preregistered surrogate could expose action sensitivity across carried boundaries; this frozen run must remain unchanged.

## Artifact provenance

The accompanying JSON records SHA-256 and byte length for each saved first lap, start, qualification and source/config file read. Files were rehashed immediately after parsing to reject changing artifacts. The companion JSON is a during-run diagnostic snapshot. The final immutable stopped-run manifest and completion receipt are now pinned in [the public summary](../evidence/research/summary.json).

Companion: [surrogate evidence](../evidence/research/surrogate.json).
