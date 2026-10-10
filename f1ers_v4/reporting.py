"""Small self-contained interactive report plus complete retained CSV traces."""
from __future__ import annotations
import csv
import html
import json
from pathlib import Path
import numpy as np
from f1ers_v3.artifacts import jsonable


def export_comparison(output, comparisons):
    output = Path(output)
    for role, comparison in comparisons.items():
        items = [("autonomous", comparison["autonomous"]["result"])] + [
            (f"replay_lap_{number}", r) for number, r in comparison["replay"]["results"].items()]
        for name, result in items:
            with (output/f"trace_{role}_{name}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                fields = ["s_m", "speed_mps", "time_s", "wheel_force_n", "terminal_status", "guard_reason"]
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                s = result.get("s_m", []); speed = result.get("speed_mps", []); t = result.get("time_s", [])
                forces = result.get("wheel_force_n", [])
                for i in range(len(s)):
                    writer.writerow({"s_m": s[i], "speed_mps": speed[i], "time_s": t[i] if i < len(t) else "",
                        "wheel_force_n": forces[i] if i < len(forces) else "", "terminal_status": "accepted"})
                writer.writerow({"s_m": result.get("terminal_s_m", 0),
                    "speed_mps": result.get("terminal_speed_mps", 0), "time_s": result.get("terminal_time_s", 0),
                    "terminal_status": result.get("terminal_status", "unknown"),
                    "guard_reason": json.dumps(jsonable(result.get("summary", {}).get("guard_failures", [])), ensure_ascii=False)})


def render_report(path, experiment, comparisons, dataset=None):
    payload = {"experiment": experiment, "curves": {}}
    for role, c in comparisons.items():
        payload["curves"][role] = {"autonomous": {k: c["autonomous"]["result"].get(k) for k in
             ("s_m", "speed_mps", "terminal_s_m", "summary")},
             "replay": {n: {k: r.get(k) for k in ("s_m", "speed_mps", "terminal_s_m", "summary")}
                        for n, r in c["replay"]["results"].items()},
             "observed": [{"s_m": l["s_m"], "speed_mps": l["speed_mps"],
                            "summary": {"length_m": dataset["track"]["length_m"]}}
                           for l in (dataset or {}).get("laps", []) if l["role"] == role]}
    encoded = json.dumps(jsonable(payload), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    text = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>F1 ERS v4 模型验收</title>
<style>body{font:16px system-ui;margin:32px auto;max-width:1120px;padding:0 20px;color:#152238;background:#f5f7fa}h1{font-size:30px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}td,th{padding:10px;border-bottom:1px solid #dce2eb;text-align:left}table{border-collapse:collapse;width:100%}.fail{color:#b02532}.pass{color:#167044}canvas{width:100%;height:300px}pre{overflow:auto;font-size:13px;max-height:350px}select{padding:8px}</style>
<h1>F1 ERS · v4 模型验收</h1><p>等效车辆模型 · 真实输入重放与自主预测分别验收</p>
<section><h2>放行状态</h2><div id="gates"></div><p>策略训练暂停；未通过的门槛保留，不放宽阈值。</p><p id="cross"></p></section>
<section><h2>完整圈精度</h2><table><thead><tr><th>数据</th><th>路径</th><th>完整圈</th><th>平均速度 RMSE</th><th>最大圈时误差</th><th>最大分段误差</th></tr></thead><tbody id="metrics"></tbody></table></section>
<section><h2>速度轨迹</h2><select id="role"><option value="development">开发圈</option><option value="calibration">校准圈</option></select> <span>灰：观测；蓝：自主预测；橙：重放。部分轨迹只显示实际完成距离。</span><canvas id="plot" width="1050" height="300"></canvas></section>
<section><h2>历史结果说明与模型边界</h2><p>v3 已通过程序复现，但当时漏检段末联合轮胎力，且重放未在真实输入切换处拆步；其物理可行性结论已在本轮补充说明。</p><p>制动强度、摩擦、动力均为等效参数。真实制动压力、SOC、动力分配和精确赛车线未知，车辆质量默认800 kg。</p></section>
<section><h2>冻结配置与数值验收</h2><pre id="details"></pre></section><script>const D=PAYLOAD;
const E=D.experiment, labels={data:'数据审查',convergence:'参数收敛',physical:'物理约束',grid:'数值收敛',development:'开发精度',new_test:'新会话测试'};
document.querySelector('#gates').innerHTML=Object.entries(E.gates).map(([k,v])=>`<p class="${v?'pass':'fail'}">${labels[k]||k}：${v?'通过':'未通过 / 待完成'}</p>`).join('');
document.querySelector('#cross').textContent=E.cross_session.interpretation||E.cross_session.reason;
for(const [r,c] of Object.entries(E.local)){for(const m of ['replay','autonomous']){const q=c[m],o=q.gate.observed;const row=document.createElement('tr');const vals=[r==='development'?'开发':'校准',m==='replay'?'重放':'自主',`${q.metrics.filter(a=>a.valid).length}/${q.expected_laps}`,o.mean_speed_rmse_kmh?.toFixed(2)??'不评分',o.max_abs_lap_error_pct===undefined?'不评分':o.max_abs_lap_error_pct.toFixed(2)+'%',o.max_abs_sector_error_s===undefined?'不评分':o.max_abs_sector_error_s.toFixed(3)+' s'];for(const v of vals){const td=document.createElement('td');td.textContent=v;row.append(td)}document.querySelector('#metrics').append(row)}}
document.querySelector('#details').textContent=JSON.stringify({parameters:E.vehicle,budget:E.report.budget,screening_feasible:E.report.screening_feasible_count,numerical:E.numerical,code_commit:E.code_provenance.code_commit},null,2);
function draw(){const canvas=document.querySelector('#plot'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,1050,300);const c=D.curves[document.querySelector('#role').value],curves=[...c.observed.map(a=>[a,'#a4aab3']),[c.autonomous,'#245abb'],...Object.values(c.replay).map(a=>[a,'#db7a23'])];const xmax=Math.max(1,...curves.map(([a])=>a.summary.length_m||a.terminal_s_m||Math.max(0,...a.s_m)));ctx.strokeStyle='#ced6e2';ctx.strokeRect(45,15,985,250);ctx.fillStyle='#35465e';ctx.fillText('km/h',3,18);ctx.fillText('距离 m',980,293);for(const [a,col]of curves){ctx.strokeStyle=col;ctx.beginPath();for(let i=0;i<a.s_m.length;i++){const x=45+985*a.s_m[i]/xmax,y=265-250*a.speed_mps[i]*3.6/360;i?ctx.lineTo(x,y):ctx.moveTo(x,y)}ctx.stroke()}}document.querySelector('#role').onchange=draw;draw();</script></html>'''
    Path(path).write_text(text.replace("PAYLOAD", encoded), encoding="utf-8")
