# 2026 F1 充放电研究原型

第三阶段独立实现与复现入口见 [README_V3.md](README_V3.md)。当前双路径精度、物理检查和跨会话状态见 `results/austria_2026_v3/report.html`。以下保留第一、第二阶段历史记录；第三阶段收益和训练必须等待全部固定门槛通过。

首阶段：奥地利 Spielberg、二维固定行驶路线、可持续单圈。公开遥测用于校准和独立验证；机器学习控制 MGU-K 请求功率，物理与规则层决定可执行功率。

第二阶段因果执行与精度验收已完成，见 [README_V2.md](README_V2.md) 和 `results/austria_2026_v2/report.html`。第二阶段精度未达标，按预注册协议未启动正式学习训练。下表为第一阶段历史结果，不能与新协议直接比较。原始第一阶段源码和结果仍保存在此前的独立下载包。

## 已执行的实验结果

| 控制器 | 仿真圈速 |
|---|---:|
| 无 ERS 对照 | 73.987 s |
| 启发式基线 | 71.258 s |
| SLSQP 数值控制基线 | 69.723 s |
| 保留的 BC/DAgger，三个种子均值 | 69.792 s |

三个种子的最佳合法学习策略分别为 70.015、69.689、69.673 s，平均较启发式快 1.466 s（2.057%），种子间标准差为 0.158 s。每个种子完成 60 轮 SAC；本次 SAC 未改善 BC/DAgger，因此保留的模型均来自模仿学习。部分学习策略略快于数值教师，与教师仅搜索有限控制族有关，不能据此宣称超过全局最优。

独立验证圈 7、9 的速度 RMSE 分别为 19.639、17.384 km/h，圈速误差分别为 −0.996%、−1.581%。校准搜索达到 20 次迭代上限，未报告收敛；当前仅通过物理一致性检查，仍是粗略等效模型。上述收益属于该模型，不能直接解释成真实赛车收益。

交付验证包括 31 项单元测试、三个保存模型的圈速精确复现、来源散列和报告脚本语法检查。应用浏览器权限校验不可用，因此未完成页面视觉及交互实测。

## 查看与运行

实验完成后，直接打开 `results/austria_2026/report.html`。报告完全自包含，无需外部脚本或网络；可切换控制器、沿赛道查看速度/电机功率/储能，并比较独立验证误差。

当前任务已配置局部 Python 环境。Windows PowerShell 中运行：

```powershell
.\run.ps1 test
.\run.ps1 run -Episodes 60 -BcEpochs 600
.\run.ps1 serve
```

启动本地预览后，访问 `http://127.0.0.1:8765/austria_2026/report.html`。

在其他机器上，使用 Python 3.12 创建局部环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\run.ps1 run
```

已有原始数据与处理缓存时，运行不需要联网。`fetch` 获取缺失公开数据；`python -m f1ers run --refresh` 显式刷新数据。下载受 OpenF1 速率限制，缓存保存原文件及来源校验信息。

## 数据与实验协议

- 2026 Austrian GP Race，OpenF1 session 11315；同一车手的完整干净圈按圈分成校准集和验证集，不进行随机遥测行切分。
- TUM 中心线、宽度与固定最小曲率路线用于赛道建模。最小曲率路线不等于最短圈速路线。
- FIA 该场 Power Unit Information 与赛事资料用于规则配置；规则来源、坐标映射及官方文件冲突见 `SOURCES.md` 与数据审计。
- 遥测坐标只有近似精度。沿程距离由遥测积分和赛道配准得到，配准仅使用校准圈。
- 原始遥测中的速度、圈速和分段时间属于实测量；仿真 trace 中的速度、圈速、SOC、MGU-K 功率及等效车辆参数属于模型量。公开遥测不能唯一分解 ICE 与电机输出。

## 模型与控制器

车辆采用点质量与准稳态闭环圈速模型，包含驱动功率、空气阻力、下压力相关摩擦椭圆、制动和电能回收。ICE 与 MGU-K 共同改变纵向速度；不是把固定真实速度曲线重放一遍。

电功率与能量在 DC 侧统一计算，驱动/回收效率只在机械与电侧转换时使用。`soc_fraction` 表示储能在允许使用的 4 MJ 窗口中的归一化位置，不能当成真实电池总容量的百分比。记录原始请求、执行动作、能量守恒残差及约束投影幅度。可持续性投影减少部署预算，不在终点添加虚拟电量。该执行层通过离线整圈重放和二分搜索选取正部署缩放；报告收益包含这一执行层，学习策略本身尚未证明在线可持续。SAC 训练环境仅执行因果的瞬时约束裁剪与终点欠电惩罚。

控制器包括启发式、SLSQP 数值控制基线、模仿学习与连续 Soft Actor-Critic。当前数值基线用 SLSQP 搜索分段部署和回收幅度，报告控制族、离散化与收敛状态，并非全量轨迹最优解。机器学习使用真实反馈状态、前方速度包络、剩余电量及回收预算；NumPy 实现 MLP 反向传播、Adam、重参数化 Gaussian actor、双 Q、目标网络和经验回放。

每个随机种子分别保存行为克隆、SAC 最后候选及最佳合法策略。若 SAC 微调使结果变差，保留之前更好的模型，并在报告中明确其来源。数值优化结果不宣称为全局最优解。

行为克隆增加最多三轮 DAgger：采集学习器闭环产生的状态，由当前赛道位置对应的冻结数值专家提供训练标签，再聚合训练。标签仅用于训练，运行策略是数值神经网络，不执行专家动作查表。该阶段验证单赛道拟合，不证明新赛道泛化。

最佳 checkpoint 可重新加载：

```python
import json
from pathlib import Path
from f1ers.data import load_dataset
from f1ers.model import Simulator, VehicleParams
from f1ers.learning import load_policy, evaluate_policy

root = Path.cwd()
experiment = root / "results/austria_2026"
dataset = load_dataset(root)
vehicle = VehicleParams.from_dict(json.loads((experiment / "vehicle.json").read_text()))
sim = Simulator(dataset["track"], dataset["rules"], vehicle)
training = json.loads((experiment / "models/training_summary.json").read_text())
policy = load_policy(experiment / training["selected_policy"])
result = evaluate_policy(sim, policy)
print(result["summary"])
```

## 结果文件

| 文件 | 内容 |
|---|---|
| `results/austria_2026/report.html` | 可交互赛道和算法比较 |
| `experiment.json` | 完整状态、规则、数据审计及训练配置 |
| `summary.json` | 各算法圈速、能量和可行性摘要 |
| `validation.json` | 整圈留出的速度、圈速和分段误差 |
| `trace_*.csv` | 沿程仿真结果，可导入其他工具 |
| `models/seed_*/` | 可重新加载的权重、BC/SAC 训练曲线及选择记录 |

除报告外，上表文件均位于同一实验结果目录。

## 完成范围与真实性边界

当前交付单赛道完整实验和可复用模块接口。新增赛道需要新的几何、赛事规则与遥测审计；全赛道结果及留一赛道泛化不以当前单赛道实验代替。

模型采用固定车辆/轮胎与干地条件，忽略坡度和倾角，Overtake 关闭。未知主动空气动力区段及未公开的 FIA DOC058 状态机另列为近似/未实现，报告的约束通过仅针对已实现的公开约束子集。

独立遥测验证用于说明等效模型能达到的精度，误差包含真实用电、燃油、风和轮胎状态差异。不能将模拟储能或功率描述成车队真实遥测。所有精度与算法收益以保存的实测实验结果为准。

第三方来源及许可见 `SOURCES.md` 和 `third_party/`。原始数据的权利和使用条件保留在原来源；本项目研究结果不声称为 FIA 或 F1 官方产品。
