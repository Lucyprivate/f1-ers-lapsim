# F1 ERS Lab v4

本轮将真实时间输入事件、联合轮胎力和段内停车定位纳入共用车辆内核，再验收真实输入重放与自主预测。策略训练继续暂停。

## 仓库与基线

本地仓库使用 `main` 保存导入基线，`baseline-v3-0.3.0` 标记原字节，`improve/v4-dynamics` 保存本轮实现。署名仅在仓库内配置为 `F1 ERS Lab <f1ers@local.invalid>`；无远端。

原始 v2/v3 交付在工作区 `outputs` 中冻结。v3 的127项测试和程序复现通过；审查发现段末联合轮胎力漏检，以及重放输入未在真实切换时刻拆步。因此旧 `execution_feasible` 结果只表示当时的检查结果，不能证明整个轨迹符合新物理检查。v4 保留这一历史说明，不修改旧交付。

Git 追踪历史数据、代码、协议、参数和紧凑摘要；大型结果、依赖wheel、环境、FP2原始窗口及测试数组不入库。正式运行记录干净的 `code_commit`，结果随后单独提交。离线制品必须与源代码版本配套，Git bundle只包含已追踪文件。

## 离线运行

需要Python 3.12。依赖与v3相同，提供在 `vendor/wheels` 中，散列见 `vendor/wheels_manifest.json`。在仓库目录执行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install --no-index --find-links vendor/wheels -r requirements-lock.txt
./run.ps1 -Mode audit-v4
./run.ps1 -Mode test-v4
./run.ps1 -Mode run-v4 -Offline
./run.ps1 -Mode verify-v4
```

已保存正式实验时，使用 `verify-v4`，不覆盖或重复拟合原实验。新研究使用独立实验名；新实验名不能产生新的FP2最终测试机会。离线验证重新计算固定参数的校准与开发轨迹，核对失败轨迹及全部冻结散列，仅核对已保存最终测试评分。

要复核历史v3，在基线标签建立独立检出，并从固定v3 ZIP恢复忽略的结果及wheel：

```powershell
git worktree add --detach ../baseline-v3 baseline-v3-0.3.0
.venv/Scripts/python.exe -m f1ers_v4 restore-artifacts --root ../baseline-v3 --archive <v3交付ZIP路径>
```

恢复入口校验固定SHA256，只恢复允许的忽略制品，不覆盖追踪源码。之后在基线检出运行 `python -m unittest discover -s tests -v` 及 `python -m f1ers_v3 verify-v3`。当前v4根入口配置已更新，原 `verify-v3` 必须在基线检出运行。

## 物理模型与数值设置

车辆质量默认800kg，825/850kg只作敏感性分析。摩擦、阻力、下压力、重放动力与制动强度是等效参数。真实压力、SOC、动力分配及准确赛车线未知；赛道采用冻结TUM路线的周期样条。

Numba时间域隐式中点使用连续几何，首/中/末三点共同限制有符号轮端力。段内离散机械功与动能变化一致。输入时间节点和空间规则边界显式拆步；停车记录实际位置与时间，求根失败另记 `numerical_fail`。

油门线性插值，刹车开关前值保持。`replay_brake_gain` 范围0.1–1.0，仅映射重放制动需求；固定纵横向能力轴比例1。重放只使用初始观测速度，无途中重置、未来速度制动或ERS标签。自主驾驶每5米依据固定几何产生原始需求，预注册抓地安全比例0.96；该值不按开发误差调整。默认ERS需求为零，内核仍验证外部因果需求的能量、来源、功率、再生/摩擦预算与降功率储备；零需求结果不构成主动ERS策略收益证据。

内部时间步上限0.02秒、距离上限1米，单步/两半步误差合格后接受两半步。完整设置与阈值见 `config/protocol_austria_2026_v4.json`。10/5/2.5米输出重采样单独验收；内部积分收敛比较0.02/0.01/0.005秒及1/0.5/0.25米上限。相邻细化圈时差≤0.10秒，速度差RMSE≤0.5km/h。

## 校准与最终测试

正赛3–6圈拟合，7/9圈开发验收，10–12圈为历史诊断。固定种子2026的64点Sobol筛选必须找到三个完整、物理可行起点，才开始三起点校准。每起点最多300次迭代、600次评估。所有失败计算保留压缩轨迹；部分圈不计算正式精度。

两路径分别要求平均速度RMSE≤15km/h，每圈圈时绝对误差≤2%，每段绝对误差≤1秒。数值收敛、物理、参数收敛和开发门槛全部通过后才获取FP2遥测。

目标固定为2026奥地利站Practice2、车手63、MEDIUM，按原质量规则取最早≥6个合格完整圈的stint。前三圈适配，仅抓地与重放动力缩放0.85–1.15；晚三圈在第二次冻结后只评分一次。普通开发加载器只接收独立早圈视图。

测试访问凭证位于仓库外的工作区 `audit/test-access`。配置中绑定该持久路径和固定会话身份；克隆、切换分支、更换实验名及中断都不重置机会。把仓库移到其他机器时可离线验证；最终测试仍需保留既有访问凭证及受信任的共享位置，不能自动创建新的测试身份。

## 交付阅读顺序

查看 `RESULTS_V4.md` 与 `results/austria_2026_v4/report.html`，再检查 `gates.json`、`calibration.json`、`numerical_audit.json`、冻结配置和 `verification.json`。计算验证与浏览器验证独立记录；未实测项目明确标记。Git bundle与制品ZIP分别提供散列，下载后先核对，再恢复或克隆。
