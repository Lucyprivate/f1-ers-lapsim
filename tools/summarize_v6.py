"""Write a small reviewable summary after calculations, never select a policy."""
import argparse
import json
from pathlib import Path

def main(root):
    root=Path(root).resolve()
    output=root/'results/austria_2026_v6'
    e=json.loads((output/'experiment.json').read_text(encoding='utf-8'))
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    cal=e['calibration']['aggregate']; dev=e['development']['aggregate']
    rows=e['contiguous_laps']['rows']
    text=f'''# V6 控制研究结果

完成 R0–R6 七轮实际实验及日志审查；SAC 共180个训练 episode，179个完成物理整圈、1个在约2750米因 `deployment_slew` 终止。九组最优 SAC 全部保留 episode 0 的 BC，未证明 SAC 增益或收敛。部分圈经过时间不计为圈时。

原声明公开子集下获选 **{e['selected']['name']}**；连续六圈末两圈平均 **{e['contiguous_laps']['stable_window_mean_time_s']:.6f}秒**，末圈{e['selected_summary']['lap_time_s']:.6f}秒。相对同一模型零 K 候选快{e['tested_model_gain_s']:.6f}秒。这是有限保留候选的模型结果，不是全局最优、真实 FIA 合规或真实赛车收益证明。

| 评分数据 | 全圈 RMSE km/h | S1 RMSE | S2 RMSE | S3 RMSE | 最大圈时误差 % | 最大分段误差秒 |
|---|---:|---:|---:|---:|---:|---:|
'''
    for label,a in [('校准3–6圈',cal),('历史开发7/9圈',dev)]:
        text+='| '+label+' | '+' | '.join(f'{v:.6f}' for v in [a['mean_speed_rmse_kmh'],*a['sector_speed_rmse_kmh'],a['max_abs_lap_error_pct'],a['max_abs_sector_error_s']])+' |\n'
    text+='''
原门槛仍为 RMSE≤15 km/h、每圈圈时误差≤2%、每段误差≤1秒；**精度未通过**。没有放宽阈值、按残差删点或开启 FP2 最终测试。重放物理模型缺陷未在本轮修复，本轮仅研究自主 ERS；不将 v6 的自主结果用作重放通过证据。

| 连续圈 | 圈时秒 | 圈初 MJ | 圈末 MJ | 净变化 MJ | 原声明子集及周期门槛 |
|---|---:|---:|---:|---:|---|
'''
    for r in rows:
        text+=f"| {r['lap']} | {r['time_s']:.6f} | {r['initial_energy_mj']:.9f} | {r['terminal_energy_mj']:.9f} | {r['energy_change_mj']:.10f} | {r['per_lap_certified']} |\n"
    text+='''
真实末状态直接接入下一圈，速度、储能、DC和削减状态不重置；只重置每圈回收计数。以上为理想 DC 等效量，默认800 kg、400 kW ICE、2–4 MJ工作区间。50 W辅助负载使净零稳态约亏0.0035 MJ/圈，真实 ES 损耗未知；不能宣称真实 SOC 已闭合，也不证明无限周期稳定。

已确认的根因与修正：

- 冻结自主驾驶器使用仅 ICE 的固定目标加速度，K多数替代ICE；动力响应驾驶层使实际功率影响速度，沿用 v4 积分内核。
- 负加速度可能来自阻力；按有符号轮端需求识别再生，再生与摩擦共享轮胎预算。
- 旧学习环境、状态与检查点和新积分路径不一致；v6统一训练/评分执行器，使用31维状态和版本检查。
- 终点免费重置电量或削减期间会制造虚假收益；所有候选实际连续六圈，按末两圈均时排名。
- SAC双Q互相接近仍可同时错误。MC初始化与终点零势函数联合对照改善部分末次表现，但Q与实际回报仍差约32秒；所有能量势函数完整episode之和为零，没有改写圈时目标。

规则上界、有效请求和实际供给分开审查。C5.12.4约束driver maximum power demand削减，不要求实际K功率恒定；C5.12.7另约束实际DC下降。历史可达K峰值代理会与能量限制后的请求脱节，新增HOLD+LIMITED工程一致性门槛存在失败；**原声明子集获选末圈也未通过该附加门槛**。这个额外门槛不追溯改写冻结证书，也不是FIA违规证据。完整DOC-058期间定义与真实需求映射仍待完成。[赛事有效技术Issue19](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_19_-_2026-06-25.pdf)

数值验收冻结原始轮端需求和ERS动作，分别检查10/5/2.5米输出与(.02秒,1米)/(.01秒,.5米)/(.005秒,.25米)积分细化。各对照与失败轨迹见交互报告，数值通过不能替代真实遥测精度通过。

数据纪律：只用历史正赛3–6圈做结构判断和控制诊断，7/9圈为已使用开发数据，10–12圈不是新测试。独立审查曾提前读取一个SAC对照的7/9圈用于诊断报告，未用于奖励、训练、R6配置或候选选择；该访问已披露。FP2共享账本和冻结历史包保持不变。

## 离线复现与文件

- `audit-v6`：根因、输入与规则范围审计。
- `run-v6`：在干净工作树开启独立新实验；已有实验不可覆盖。
- `verify-v6`：验证全部正式散列并以相同真实携带初态重跑获选轨迹和分段评分。随机训练及教师搜索不重跑，全部失败日志与检查点通过散列验证。
- `python -m unittest discover -s tests`：回归验证。
- `tools/restore_v6.py`：从独立Git bundle克隆后，以发布ZIP散列恢复允许的忽略制品；拒绝越界、源代码覆盖及FP2数组。
- `results/austria_2026_v6/report.html`：自包含速度、功率、能量和局部误差图。

独立克隆使用相同Windows Python3.12、`requirements-lock.txt`及ZIP内`vendor/wheels`。报告工具使用单独目标环境和`vendor/reporting-wheels`，不修改计算依赖。仓库无远端，项目专用署名；v2/v3/v4/v5源码、数据和标签未改写。

正式R0–R6源码提交：`'''+e['provenance']['code_commit']+'''`。外层日志结束后只封存run.log散列，原返回清单逐字保留为manifest_at_return.json；未改计算。后续审查和报告由交付清单另行固定。浏览器视觉验收未实测；计算、恢复及独立克隆验证单独记录。

仍待完成：真实线路/曲率与动力误差归因、整圈重放、公开代理的可达性保证、真实FIA ECU期间/传动/气动验证、真实ES损耗与SOC，以及冻结门槛全部通过后才可打开的FP2条件迁移。下一轮学习应先检验critic回报尺度、稳态初态分布和硬保护后的执行动作学习，继续保留原精度门槛。
'''
    r7=output/'R7'/'experiment.json'
    if r7.exists():
        v=json.loads(r7.read_text(encoding='utf-8'))
        text+='\n## R7 日志驱动修正\n\n已另行冻结几何前视保护：在已知固定几何的低正轮端需求区前，提前限制新部署请求，保持与实际降功率仍由原状态机执行。三种全局裕量独立验证六圈；不读取未来遥测、不按失败点设局部掩码、不增加SAC训练。原BC2027 raw轨迹的提前终止被修复。冷态更强代理门槛的失败单独保留，受控暖机候选不能称为完整冷态强子集通过。R7结果、源码提交、数值与代理审查见R7/experiment.json及R7/verification.json。\n'
    (root/'RESULTS_V6.md').write_text(text,encoding='utf-8')
    (output/'results.md').write_text(text,encoding='utf-8')
    print('Wrote RESULTS_V6.md and results.md; no calculation or policy changed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
