# v4 方法、来源与假设

本轮使用同一组冻结历史公开来源，详细URL、访问记录、许可及规则证据继承 `SOURCES_V3.md`、`SOURCES_V2.md`、`data/manifest.json` 和 `data/v3/protocol_lock.json`。没有把参考项目的误差或圈速结果当成本项目的收益承诺。

| 参考 | 使用的方法 | 本轮落实 |
|---|---|---|
| [Fastest-lap](https://github.com/juanmanzanero/fastest-lap) | 分开验证几何、轮胎包络、车辆动态 | 固定连续路线，共用首/中/末联合轮胎力内核 |
| [TUM 圈速论文](https://mediatum.ub.tum.de/doc/1647512/1647512.pdf) | 圈时与分段误差分别验收，诊断制动与弯心 | 两路径分别评分，制动起点、减速与弯心误差单列 |
| [TUM 轨迹优化](https://github.com/TUMFTM/global_racetrajectory_optimization/blob/master/opt_mintime_traj/Readme.md) | 连续、平滑曲率影响数值求解 | 周期样条按真实弧长查询，输出网格与积分细化分开检查 |
| [FastF1 原始实现](https://raw.githubusercontent.com/theOehrly/Fast-F1/master/fastf1/core.py) | 区分连续与离散通道、原始与插值来源 | 油门线性、刹车前值保持，在真实时间事件拆步 |

时间域隐式中点的机械功等式来自本项目公开的离散推导：`Δs=(v0+v1)Δt/2`、`mΔv=(Fx−D)Δt`，因此 `ΔK=(Fx−D)Δs`。实现同时保存求根误差、阶段轮胎力与电量账本。它是工程近似；三阶段证书与步长收敛检查是数值验收，不替代连续车辆真值。

v3程序可复现，但漏检端点联合力及真实输入事件。冻结历史交付保持原字节，本轮报告补充其物理判定限制。

本轮没有真实制动压力、SOC、MGU-K分配或精确赛车线。刹车增益为剩余轮胎能力的全局等效比例，不称为压力；参数只在预注册校准圈内拟合。沿用800kg，质量变化只作敏感性分析。

FP2元数据与圈质量只能在本地门槛通过并冻结后查询；真实访问结果保存在新v4缓存，测试数据不提交Git。没有足够数据时保持“新会话验收待完成”，不替换目标。

来源和许可证继续随交付保存，仓库仅用于本地研究，未建立远端或公开发布。
