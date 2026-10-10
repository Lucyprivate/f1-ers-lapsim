# Latest frozen results

These results describe the original local D experiment, not a new test of the reorganized public package.

| Item | Result |
|---|---:|
| Selected checkpoint | `D2_seed_2026_best` |
| Cold first lap | 71.17924128 s |
| Last-two mean, six actual carried laps | 70.44079066 s |
| SAC episodes completed | 120 |
| Seeds | 2026, 2027, 2028 |
| Retained six-lap qualifications | 18 passed / 19 examined |
| Learning gain over analytical teacher | None |

The best retained SAC policies matched the cloned teacher. Other retained policies were slower or failed the per-lap non-deficit condition. Selection required all six physical/control gates before ranking by the last-two mean; observed speed residuals were not a selection input. Unsaved intermediate policy weights were not exhaustively qualified.

## Accuracy remains below the release gates

The historical development laps 7 and 9 are reused development data, not a new test. Calibration uses laps 3–6.

| Metric | Calibration | Historical development | Gate |
|---|---:|---:|---:|
| Mean whole-lap speed RMSE, km/h | 20.3488 | 19.6259 | ≤15 |
| Sector 1 speed RMSE, km/h | 20.4992 | 19.0768 | Diagnostic |
| Sector 2 speed RMSE, km/h | 18.9909 | 19.9659 | Diagnostic |
| Sector 3 speed RMSE, km/h | 21.6276 | 19.6528 | Diagnostic |
| Maximum absolute lap-time error, % | 2.0935 | 2.7101 | ≤2 per lap |
| Maximum absolute sector-time error, s | 1.9211 | 1.9863 | ≤1 per sector |

Numerical convergence passed with frozen raw demands and initial conditions. The largest adjacent internal-refinement differences were **0.002151 s** in lap time and **0.005995 km/h** in speed RMSE, below the 0.10 s / 0.5 km/h limits. Output resampling was tested separately.

The selected policy passed six-lap ideal-energy, speed/DC/FSM closure and engineering-proxy checks. Its warm energy surplus was only **512.410 J/lap**. A separate illustrative 50 W auxiliary load would change that to about **−3.010 kJ/lap**. All four frozen mass/power sensitivity cases had at least one deficit lap. Neither true SOC sustainability nor an infinite-horizon cycle is established.

Accuracy failed; FP2 remains sealed and formal release remains blocked. Replay accuracy, complete FIA ECU certification and global optimality are not established by this control experiment.
