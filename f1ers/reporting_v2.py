"""Causal-protocol report, retaining v1 output as a labelled history only."""
import html
import json
import csv
import numpy as np
from pathlib import Path
from .reporting import TEMPLATE,serializable


def export_trace_v2(path,result):
    """Keep the shield certificate and request/execution differences portable."""
    keys=["s_m","x_m","y_m","speed_mps","time_s","dt_s","energy_mj","soc_fraction","recharge_mj",
        "mguk_dc_kw","mguk_mechanical_kw","ice_mechanical_kw","wheel_power_kw","friction_brake_kw",
        "ax_mps2","ay_mps2","aero_mode","requested_actions","shielded_actions","executed_actions",
        "reserve_mj","deployable_credit_mj","energy_floor_mj","guard_reasons","segment_converged"]
    n=len(result["s_m"])
    columns={}
    for key in keys:
        if key not in result:continue
        values=np.array([json.dumps(v,ensure_ascii=False) for v in result[key]]) if key=="guard_reasons" else np.asarray(result[key])
        if values.ndim==1 and len(values)>=n:columns[key]=values
    with path.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.writer(handle);writer.writerow(["row_type",*columns])
        writer.writerows([["segment",*[v[i] for v in columns.values()]] for i in range(n)])
        # A terminal state has no segment power or acceleration. Blank fields
        # keep that distinction explicit rather than inventing end-point power.
        terminal={"s_m":float(result["s_m"][-1]+np.diff(result["s_m"])[-1]) if n>1 else None,
            "x_m":result.get("x_m",[None])[0] if n and result["summary"].get("completed_lap") else None,
            "y_m":result.get("y_m",[None])[0] if n and result["summary"].get("completed_lap") else None,
            "speed_mps":result["summary"].get("speed_end_mps")}
        for key in ("time_s","energy_mj","soc_fraction","recharge_mj","energy_floor_mj"):
            if key in columns and len(result[key])==n+1:terminal[key]=result[key][-1]
        writer.writerow(["terminal",*[terminal.get(k,"") for k in columns]])


def render_report_v2(path:Path,bundle:dict):
    template=TEMPLATE
    template=template.replace("Spielberg 能量管理实验室","Spielberg 因果闭环实验室")
    template=template.replace("二维赛道 · 可持续单圈 · 公开遥测验证","同一逐段执行器 · 保守电量保护 · 冻结遥测测试")
    template=template.replace("校准参考策略验证","新测试集车辆精度")
    template=template.replace("相对启发式</div>","相对参考（模型）</div>")
    template=template.replace("text('lap',fmt(s.lap_time_s,3)+' s');text('gain',fmt(lap(C[0].result)-s.lap_time_s,3)+' s');",
        "text('lap',s.feasible?fmt(s.lap_time_s,3)+' s':'无效圈');text('gain',D.gates.benefit_claim_allowed&&s.feasible&&summary(C[0].result).feasible?fmt(lap(C[0].result)-s.lap_time_s,3)+' s':'结论暂停');")
    template=template.replace('${fmt(s.lap_time_s,3)}</td>',"${s.feasible?fmt(s.lap_time_s,3):'无效圈'}</td>")
    template=template.replace('${fmt(lap(C[0].result)-s.lap_time_s,3)}</td>',"${D.gates.benefit_claim_allowed&&s.feasible&&summary(C[0].result).feasible?fmt(lap(C[0].result)-s.lap_time_s,3):'—'}</td>")
    template=template.replace("index=Math.max(0,Math.min(n-1,index));", "if(!n){text('lap','无有效轨迹');text('gain','结论暂停');['track','ggv','speed','power','soc'].forEach(id=>setupCanvas(id));return}index=Math.max(0,Math.min(n-1,index));")
    template=template.replace("部署投影系数","离线整圈缩放")
    template=template.replace("${fmt(s.projection_alpha,3)}","未使用")
    old="可持续投影通过离线全圈重放选择部署缩放，报告收益包含此执行层；学习策略本身尚未证明在线可持续。投影系数与原始请求保留在数据中。数值基线是在有限控制族中找到的解，未证明全局最优。"
    new="所有正式结果调用同一因果逐段执行器，部署只使用本圈已回收能量并保留降功率储备。无离线整圈缩放；末速或执行约束失败的圈不参与有效圈速比较。数值基线仅搜索有限控制族。"
    template=template.replace(old,new)
    template=template.replace("验证对象为校准后的车辆与启发式参考策略。优化策略的实测量对比另存 validation.json；真实电机功率未知，不能验证模拟充放电真值。分段边界若由遥测积分估计会在数据中标记。",
        "新测试圈在模型与训练配置冻结后一次性计算误差。验证对象为校准参考策略；真实电机功率未知，不能验证充放电真值。分段边界来自实测时间/距离插值，保留近似误差。")
    gate=bundle["gates"]
    status="精度门槛通过" if gate["final_test"]["passed"] else "精度门槛未通过 · 收益结论暂停"
    panel='<section class="panel spacer"><h2>'+status+'</h2><p>开发集与新测试集分别报告；收敛、物理可行和观测精度不是同一条件。</p><pre>'+html.escape(json.dumps(serializable(gate),ensure_ascii=False,indent=2))+'</pre></section>'
    if bundle["training"].get("skipped"):
        panel+='<section class="panel spacer"><h2>本次未启动正式学习训练</h2><p>'+html.escape(bundle["training"]["reason"])+'</p><p>BC/DAgger 和固定/自动温度 SAC 的因果训练接口已实现并单独测试；本报告不会把未执行训练记成结果。</p></section>'
    history=bundle.get("history",{})
    panel+='<section class="panel spacer"><h2>第一阶段历史记录</h2><p>旧成绩含不同执行协议与离线部署缩放，不能直接解释为本阶段收益或损失。</p><pre>'+html.escape(json.dumps(history,ensure_ascii=False,indent=2))+'</pre></section>'
    template=template.replace('<div class="notes">',panel+'<div class="notes">')
    payload=json.dumps(serializable(bundle),ensure_ascii=False,allow_nan=False).replace("<","\\u003c").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    path.write_text(template.replace("__DATA__",payload),encoding="utf-8")
