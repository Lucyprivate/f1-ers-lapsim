# v5：结构误差归因与有限候选

开发分支为 `research/v5-falsification`。v2/v3/v4 的源码、原配置、数据、标签与交付包保持冻结；v5 新增分析和驾驶适配层，继续调用原 `f1ers_v4.vehicle.vehicle_step`、Numba 求根、停车定位、联合轮胎预算及电量保护。原根入口和项目元数据不改写，避免破坏 v4 源码散列。

## 离线运行

使用原 Python 3.12 环境和已冻结依赖。冷环境可通过 `vendor/wheels` 与 `requirements-lock.txt` 离线安装。

```powershell
./run-v5.ps1 -Mode diagnose-v5 -Offline
./run-v5.ps1 -Mode test-v5
./run-v5.ps1 -Mode run-v5 -Offline
./run-v5.ps1 -Mode verify-v5
```

也可执行 `python -m f1ers_v5 diagnose-v5`、`run-v5 --offline` 和 `verify-v5`。正式实验要求干净工作树，且禁止覆盖已有候选、冻结配置或正式实验。独立新实验可由模块入口 `--name` 指定；更换实验名不创建新的 FP2 测试身份。

`results/austria_2026_v5` 保存诊断、位置候选、每个动力学候选的 64 点筛选、失败轨迹、Huber 对照、数值与质量敏感性、冻结证书和报告。完整失败轨迹以 gzip JSON 保存，部分圈不计算正式精度。Git 仅追踪源码、协议和紧凑结果摘要；大型轨迹和报告保持忽略。

下载包采用仓库根相对路径；完整历史由同名 Git bundle 提供。独立恢复可按以下顺序进行，先核对单独发布的 ZIP/bundle SHA256，再克隆、恢复允许的忽略制品、离线安装依赖：

```powershell
git clone F1_ERS_Lab_v5.bundle f1-ers-lab-v5
Set-Location f1-ers-lab-v5
git switch research/v5-falsification
$publishedV5Sha = (Get-Content ../F1_ERS_Lab_v5.zip.sha256).Split(' ')[0]
python tools/restore_v5.py --root . --zip ../F1_ERS_Lab_v5.zip --sha256 $publishedV5Sha
python -m venv .venv
./.venv/Scripts/python.exe -m pip install --no-index --find-links vendor/wheels -r requirements-lock.txt
./run-v5.ps1 -Mode test-v5
./run-v5.ps1 -Mode verify-v5
```

恢复工具在任何写入前检查完整 ZIP 散列、全部成员和全部目标；只恢复 v4/v5 结果及离线依赖，源码成员跳过，晚圈原始与加工数组拒绝。同一内容可重复恢复，已有不同内容会被拒绝。命令输出每个成员的哈希、大小和恢复动作，可保存为独立恢复记录。计算验证与浏览器交互验证各自报告。

## 证据与有限扩展

所有结构诊断和几何选择只接收正赛圈 3–6；开发视图仅含圈 7/9。原位置、真实时间、有效掩码和样本来源保留。运动学和推断轮端力是带采样、量化、位置和参数敏感性的诊断区间，不是观测制动压力、轮胎摩擦或 ICE/ERS 分配。

几何仅从校准位置拟合共用周期横向偏移，Gaussian 标准差为 20/40/80 米。按留一圈原始位置误差比较；误差相差不足 1% 选更平滑者。每折和全拟合都必须满足闭合、真实弧长、沿用 TUM 边界与曲率稳定检查，否则回退原线路。规则及扇区经过同一非线性 source/arc-length 映射，不按新旧长度伸缩。

基线制动请求是一个全局增益乘剩余纵向能力；独立候选是增益乘当前总法向载荷所对应的能力，然后由原内核截断到可执行的联合预算。增益均为 0.1–1.0 的等效参数。仅当至少三个校准圈在不确定性控制后支持前强后弱，才拟合因果松刹时间 0.1–5 秒；它只读取已过去的刹车开启时间。

牵引建立时间 0.05–1 秒仅在至少三个校准圈的完整自主轨迹支持出弯系统性超速且其他解释不足时启用。它作用于正的 signed wheel demand，使目标加速度无法绕过；负需求、真实重放油门保持原逻辑。自适应重试不会推进驾驶器时钟。数值审计冻结原始驾驶决策与时间输入；状态依赖的请求力不被误称为固定轮端力轨迹，也不用裁剪后的执行力重建控制。

每个进入正式拟合的候选固定 Sobol64、种子2026，必须有至少三个完整、物理可行起点。原平方损失实验采用三起点、每起点最多300次迭代和600次评估；Huber 敏感性对照是独立的同预算实验，只改变速度损失（分界15 km/h），不用于正式参数选择。没有可行起点时两种优化都不启动，也不通过鲁棒损失为部分圈评分。

正式模型依原等权归一化开发目标选择，1%内优先更少参数、再按预注册候选顺序。800kg质量、原参数边界、5米输出及原积分容差保持不变；质量敏感性不参与选择。两路径独立要求RMSE≤15km/h、每圈圈时误差≤2%、每段误差≤1秒。中位与95%分位误差是补充诊断，所有点继续计入正式RMSE。

自主路径保留400kW内燃机与默认零MGU-K请求。推断直线动力只用于检验该假设是否可辨识，不拟合ERS策略来吸收真实动力分配未知。电气字段通过表示规则及保护检查，不代表验证真实SOC或赛车ERS策略。

## 测试隔离与冻结

使用原固定FP2/车手63/MEDIUM协议身份及仓库外同一个账本；v5没有新的评分机会。本地全部门槛通过前不获取FP2遥测。若通过，前三圈仍仅适配抓地、重放动力缩放及坐标/时间配准；固定形状、制动结构、响应参数不再拟合，晚三圈只评分一次。

v5证书扩展原v4证书，校验新增源码、配置、候选、诊断和失败制品。`verify-v5`重算每个候选的固定参数校准与开发轨迹，核对报告门槛与冻结证书，读取已保存最终评分，绝不重开晚圈。筛选、优化过程、数值审计与质量敏感性核对冻结散列，不重新优化；新实验必须使用新的本地结果目录。训练和策略收益结论继续暂停。

## APEX 方法借鉴及边界

固定参考版本为 `8720bf9effb7054c7ef24c2842a5e87c65f08ee2`：

- [FALSIFICATION：结构残差与诚实报告](https://github.com/yussypu/apex/blob/8720bf9effb7054c7ef24c2842a5e87c65f08ee2/docs/FALSIFICATION.md)
- [校准实现：Huber与RMS、中位数补充指标](https://github.com/yussypu/apex/blob/8720bf9effb7054c7ef24c2842a5e87c65f08ee2/src/apex/calibration.py)
- [几何实现](https://github.com/yussypu/apex/blob/8720bf9effb7054c7ef24c2842a5e87c65f08ee2/src/apex/geometry.py)

借鉴的是可被证伪的诊断方法。其同圈拟合、线路重建及报告不能作为本项目留出圈/跨会话精度证据；其具体残差数值不转化为本项目收益承诺。未复制其最近邻速度匹配或丢弃时间/Source的加工方法。浏览器交互与计算验证分别记录，未实测部分明确标注。
