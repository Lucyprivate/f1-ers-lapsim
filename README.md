# f1-ers-lapsim

2026 F1 Spielberg 圈速与 ERS 控制研究：车辆动力学、公开规则约束、数值教师、BC/DAgger 与 SAC。最新成果为 **v7 诊断版本**，模型精度尚未通过正式门槛。

## 最新结果

| 指标 | 冻结结果 |
|---|---:|
| 最快已测试且通过六圈连续资格与数值检查的暖态圈 | 70.440791 s |
| 同一控制器的冷态首圈 | 71.179241 s |
| 理想 DC 等效储能暖态净变化 | +512.410 J/圈 |
| 新增 SAC 回合（3 种子，冷/暖条件） | 120 |
| SAC 相对保留 BC 基线的圈速增益 | 未取得 |
| 历史开发圈 7/9 平均速度 RMSE | 19.625897 km/h，未通过 15 km/h 门槛 |
| 开发圈 S1 / S2 / S3 速度 RMSE | 19.076825 / 19.965902 / 19.652757 km/h |
| 最大圈时误差 / 分段误差 | 2.710122% / 1.986265 s，均未通过 |

六圈之间真实携带速度、电量、DC 功率及规则状态，不在圈界重置。六圈资格不能证明无限周期可持续或全局最优。+512 J 的余量仅相当于约 7.274 W；假设 50 W 辅助损耗便会出现电量亏损。使用 800 kg、400 kW 等效 ICE 轮端动力、2–4 MJ 理想储能，参数敏感性中仍有可持续性失败。

当前通过的是实现的公开规则子集、额外工程请求保护及数值检查；完整 FIA ECU 期间、真实 SOC 损耗和跨会话 FP2 测试仍待完成。正赛 7/9 圈已反复用于开发，不能称为新测试。所有正式精度门槛保持原值，`formal_release=false`。

## 代码与成果

- [v7 方法与接口](README_V7.md)、[结构审计](docs/V7_AUDIT.md)
- [A/B/C/D 冻结摘要与验证凭证](summaries/v7/)
- [公开发布范围与复现限制](docs/GITHUB_PUBLICATION.md)
- [逐文件原始来源散列](docs/PUBLICATION_SOURCE.json)
- [数据与规则来源](SOURCES.md)、[第三方声明](NOTICE.md)
- v2–v6 历史方法保留在各版本 README；[最早阶段说明](docs/README_PHASES_1_TO_3.md)只作历史记录。

本仓库是从本地冻结提交 `b418dd27b071c413477b646db0c813350ad357ba` 导出的公开源码快照。原完整历史、标签和交付包保存在本地。公开快照不包含完整 FIA PDF/页面图、原始/派生遥测缓存、训练权重、wheel、大型轨迹或原 ZIP/bundle；保留相关 URL、散列、配置及紧凑结论。新增公开发布文档不修改冻结算法。

## 快速检查（不会训练或访问 FP2）

已验证数值环境为 Windows + Python 3.12；其他系统未实测。在线安装锁定依赖，然后从仓库根运行：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe tools/verify_public_snapshot.py
.venv/Scripts/python.exe -m f1ers_v7 audit-v7 --root . --offline
.venv/Scripts/python.exe -m unittest discover -s tests -p "test_v7_*.py" -v
```

`audit-v7`只检查基础协议，不能替代 D 阶段 34 通道、物理轨迹或精度验收。源码快照的合成回归结果在 [PUBLIC_VERIFICATION.json](docs/PUBLIC_VERIFICATION.json)；原冻结交付曾通过 472 项回归及独立克隆的四个精确复验，两者验证范围不同。

## 正式实验与精确复验

正式 `run-v7` / `verify-v7` 以及 B/C/D 脚本依赖授权取得的完整缓存、保存权重、冻结依赖和原计算提交历史。源码快照不能单独复现这些被冻结的实验；步骤与条件见 [发布说明](docs/GITHUB_PUBLICATION.md)。`--offline` 不会补出缺失制品。FP2 保持未打开，公开发布和克隆不增加评分机会。

## 许可

自有源码的开源许可证等待项目所有者选择；当前公开展示不授予额外复用许可。第三方资料、数据及其派生内容沿用原来源的权利与许可，见 [NOTICE.md](NOTICE.md)。本项目与 F1、FIA 或任何车队无隶属或官方认证关系。
