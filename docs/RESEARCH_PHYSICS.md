# Frozen physics falsification results (0.8.0)

The frozen study screened 64 identical Sobol parameter points for each of three declared model candidates: aggregate baseline, independent replay braking, and independent replay braking with axle budgets. All 192 points failed the complete two-path feasibility gate. Each candidate correctly stopped before fitting; zero fitting evaluations were spent. Development was loaded upstream but was not scored or used for candidate selection. No candidate was selected, numerical qualification was withheld, and formal model release failed. No formal accuracy score was assigned to any partial replay.

| Candidate | Replay joint-force failures | Certified replay stops | Full physical autonomous runs | Qualified autonomous runs |
|---|---:|---:|---:|---:|
| Aggregate baseline | 240 | 16 | 64 | 0 |
| Independent replay braking | 239 | 17 | 64 | 0 |
| Independent braking + axles | 242 | 14 | 0 | 0 |

All 768 replay trajectories terminated before a complete lap. In the aggregate candidates, every joint-force failure had zero remaining longitudinal capacity and zero actual longitudinal force. This is evidence of excessive speed for the modeled lateral demand, rather than excessive applied braking force. Independent demand remains distinct from executed capability, but the direct change did not produce a viable full-lap point in this finite screen. The stronger input mapping sometimes led to a real stop under the held brake switch; those stopping events were retained at their actual time and position.

The axle substitution also rejected all 64 autonomous screens and all nine fixed axle sensitivity cases. The nominal sensitivity stopped near 253.65 m. Its rear remaining longitudinal capacity was zero while rear lateral force already exceeded its load-dependent friction limit; front braking continued. This exposes an incompatibility between the aggregate safety driver and the declared axle/load/lateral-sharing model. It does not identify the true car's axle parameters or prove that the axle model improves prediction. An axle-consistent driving envelope would require a separately frozen experiment.

The independent nearest-centerline-segment boundary check found no outside samples. Minimum clearance was 0.66724 m, versus 0.47849 m for the inherited nearest-node check. The 0.18875 m difference does not validate the observed racing line or rescue previously rejected alternative routes; no geometry was changed.

Loss sensitivities used the explicit diagnostic fallback: 800 kg, 400 kW ICE, friction 1.75, drag area 1.15 m², downforce area 3.6 m², replay equivalent power 600 kW and brake gain 0.5. They used the fixed cold boundary 77.7942395576 m/s and 2 MJ. These parameters differ from the qualified ideal-energy learning baseline, so their times must not be presented as that baseline's gains or loss penalties.

At zero auxiliary load, efficiencies 1.00, 0.98 and 0.95 each completed six actual carried laps with exact physical boundary continuity and passed request-proxy reconstruction. All failed qualification: the first cold lap missed speed closure by about 1.27860 m/s; efficiencies 0.98 and 0.95 also lost stored energy on individual later laps. Diagnostic last-two means were 70.83072, 70.86225 and 70.97960 s respectively. The 0.95 case had a positive last-two average energy change but a negative final-lap change, demonstrating why average energy balance cannot replace the per-lap gate. Every 50 W or 100 W auxiliary cold case stopped at the actual storage floor at t=0 and s=0. That result follows from the exact-floor cold-start assumptions and absent initial charging; it is not evidence that a real F1 car cannot start.

DC-side recharge counters and storage-loss/auxiliary ledgers remained separate. Maximum accumulated storage accounting error in the complete zero-auxiliary sensitivity cases was below 1.8×10⁻¹³ MJ. All failure trajectories were retained. No sensitivity case was numerically qualified, used for candidate selection, or substituted into training. FP2 remains sealed. Zero viable Sobol points is a finite screening outcome, not a proof that no feasible parameter set exists within the fixed bounds.

Frozen code: `1cff56a38630c49fc56fcfafe9a0e65d43a6665b`.
Physics report SHA-256: `82fce6229b3062b8941f86f891c93a6070d9a96c0ed5a923438627e7d08b01a1`.
