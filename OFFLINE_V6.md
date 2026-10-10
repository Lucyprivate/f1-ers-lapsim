# V6 离线恢复与复验

交付为诊断研究，不代表模型精度放行。先看 RESULTS_V6_R7.md 与 RESULTS_V6.md；原公开子集基准70.6725秒未通过附加代理门槛，71.3313秒为三个冻结裕量中通过暖态附加门槛的有限研究候选。冷态强门槛、真实精度、完整FIA、真实SOC与FP2验收仍未通过或待完成。

在新的目录从 `F1_ERS_Lab_v6.bundle` 克隆并检出交付修补标签 `v6-0.6.1`。模型及实验版本仍为0.6.0，原标签保持不变；0.6.1只修复冻结历史访问凭证的交付许可。ZIP本身不恢复源码；源码必须已与同一发布提交一致。使用另行发布的 `F1_ERS_Lab_v6.zip.sha256` 中的完整散列：

```powershell
python tools/restore_v6.py --root . --archive /absolute/F1_ERS_Lab_v6.zip --sha256 PUBLISHED_SHA256
```

恢复工具先校验整个ZIP和所有路径，只恢复 `results/austria_2026_v6` 及依赖wheel。所有已追踪源码只校验、不覆盖；任一不同文件、链接、越界路径、Windows路径别名或FP2数组会拒绝。再次恢复应为幂等。

计算环境使用Windows Python3.12。可创建独立虚拟环境，用恢复的wheel安装：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install --no-index --find-links vendor/wheels -r requirements-lock.txt
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:TEMP=Join-Path (Get-Location) '.cache/test-tmp'
$env:TMP=$env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

复验主实验和R7。主复验使用 `save=False` 避免改写发布后的验证凭证；两者只重跑已保存策略的末圈，不搜索或训练，也不打开FP2：

```powershell
.venv/Scripts/python.exe -c "from pathlib import Path; from f1ers_v6.verification import verify; print(verify(Path('.'),save=False))"
.venv/Scripts/python.exe tools/run_r7.py --repo . --verify results/austria_2026_v6/R7
```

计算清单和所有训练/失败轨迹散列均核对，固定初态下的速度、动作、终态、代理审计及分段指标可精确复现。随机SAC、DAgger和教师搜索没有重新运行；其结果由原始散列库存验证。主实验354项制品与R7独立76项制品分开冻结，外层日志封存说明保留原返回清单。后续报告与审查由交付清单另行固定。

报告是自包含HTML，可以直接打开。若要重建报告，将报告依赖装入单独目标目录，避免改变计算环境：

```powershell
.venv/Scripts/python.exe -m pip install --no-index --find-links vendor/reporting-wheels --target .cache/reporting-deps -r requirements-reporting-v6.txt
.venv/Scripts/python.exe tools/report_v6.py .
.venv/Scripts/python.exe tools/append_r7_report.py --root .
```

`audit-v6`、`run-v6`、`verify-v6` 入口仍可用。重跑研究须新名称、干净Git工作树及独立源码固定；已有正式结果不可覆盖。原冻结v2–v5制品仍在原下载包，不属于本轮忽略制品恢复白名单。

发布记录固定源码提交、结果提交、完整Git bundle和ZIP散列。独立克隆/恢复/离线安装/全部回归/双实验精确复验结果另存于发布包旁的验证文件，避免为写入后验结果重新生成已经固定的ZIP。浏览器视觉验证与计算验证分开；前者未实测。
