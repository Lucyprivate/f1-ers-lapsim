"""Self-contained dual-path report; incomplete laps stay visibly incomplete."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from .artifacts import jsonable, write_json


def export_trace(path, result):
    """N segment rows and a separate terminal state; no invented terminal power."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(result.get("s_m", []))
    columns = ["node", "mode", "ers_validated", "s_m", "time_s", "speed_mps", "dt_s", "ax_mps2",
               "wheel_power_kw", "equivalent_drive_kw", "equivalent_brake_force_n",
               "mguk_dc_kw", "energy_mj", "recharge_mj", "guard_reason"]
    summary = result.get("summary", {})
    history = result.get("guard_diagnostics", [])
    accepted = [row for row in history if row.get("step_certified")]
    rejected = [row for row in history if not row.get("step_certified")]
    def reasons(row):
        return json.dumps(jsonable(row.get("guard_reasons", row.get("failures", []))), ensure_ascii=False)
    def at(key, i):
        value = result.get(key)
        if value is None or i >= len(value):
            return ""
        item = value[i]
        if item is None:
            return ""
        return json.dumps(jsonable(item), ensure_ascii=False) if isinstance(item, (dict, list, tuple)) else item
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i in range(n):
            row = {key: at(key, i) for key in columns if key not in ("node", "mode", "ers_validated")}
            row["node"] = i
            row["mode"] = result.get("mode", summary.get("mode", ""))
            row["ers_validated"] = result.get("ers_validated", summary.get("ers_validated", ""))
            row["guard_reason"] = reasons(accepted[i]) if i < len(accepted) else ""
            writer.writerow(row)
        times = result.get("time_s", [])
        if len(times) == n + 1:
            fallback_distance = result["s_m"][-1] + float(result.get("ds_m", 0)) if n else 0.
            row = {"node": n, "s_m": summary.get("distance_completed_m",
                      summary.get("length_m", fallback_distance)),
                   "time_s": times[-1], "speed_mps": summary.get("speed_end_mps", "")}
            row["mode"] = result.get("mode", summary.get("mode", ""))
            row["ers_validated"] = result.get("ers_validated", summary.get("ers_validated", ""))
            row["guard_reason"] = reasons(rejected[-1]) if rejected else summary.get("failure_exception", "")
            for key in ("energy_mj", "recharge_mj"):
                row[key] = at(key, n)
            writer.writerow(row)


def _gate_rows(bundle):
    rows = []
    for candidate in bundle.get("candidates", []):
        report = candidate["report"]
        for mode, path in candidate.get("development", {}).items():
            if not isinstance(path, dict) or "gate" not in path:
                continue
            gate = path["gate"]
            observed = gate.get("observed", {})
            def number(key, digits=3):
                value = observed.get(key)
                return f"{value:.{digits}f}" if value is not None and np.isfinite(value) else "不完整 / 无可评分圈"
            rows.append([report.get("candidate", candidate.get("candidate", "")),
                         "输入重放" if mode == "replay" else "自主预测",
                         number("mean_speed_rmse_kmh"), number("max_abs_lap_error_pct"),
                         number("max_abs_sector_error_s"), "通过" if gate.get("passed") else "未通过"])
    cross = bundle.get("cross_session", {})
    for mode, path in cross.get("evaluation", {}).items():
        if isinstance(path, dict) and "gate" in path:
            obs = path["gate"].get("observed", {})
            rows.append(["FP2 后续测试", "输入重放" if mode == "replay" else "自主预测",
                         *[f"{obs[k]:.3f}" if obs.get(k) is not None else "不完整 / 无可评分圈"
                           for k in ("mean_speed_rmse_kmh", "max_abs_lap_error_pct", "max_abs_sector_error_s")],
                         "通过" if path["gate"].get("passed") else "未通过"])
    return rows


def render_report(output, bundle):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
                   for row in _gate_rows(bundle))
    payload = json.dumps(jsonable(bundle), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    page = TEMPLATE.replace("__PAYLOAD__", payload).replace("__ROWS__", rows)
    status = bundle.get("status", "待验收")
    if bundle.get("diagnostic_selection"):
        status += "。没有合格候选，默认展示仅供诊断，不代表模型选择通过。"
    page = page.replace("__STATUS__", html.escape(status))
    (output / "report.html").write_text(page, encoding="utf-8")
    write_json(output / "report_data.json", bundle)
    return output / "report.html"


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spielberg v3 · 双路径精度与跨会话验收</title>
<style>
:root{color-scheme:dark;--bg:#0c1220;--panel:#151e30;--line:#2a3852;--text:#e5edf9;--muted:#abbad1;--orange:#f4ab59;--blue:#65baf2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1380px;padding:28px;margin:auto}h1{font-size:29px;margin:6px 0 14px}h2{font-size:19px;margin:0 0 12px}p{margin:8px 0}.muted{color:var(--muted)}.tag{color:var(--blue);letter-spacing:2px;font-size:12px}.status{padding:14px 18px;border-left:4px solid var(--orange);background:#252234;border-radius:8px;margin:18px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid .card{margin:0}canvas{width:100%;height:270px;display:block}#track{height:340px}.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:center}select{background:#0c1525;color:var(--text);border:1px solid var(--line);padding:8px;border-radius:7px}input{accent-color:var(--blue)}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:10px;text-align:left}th{color:var(--muted)}.scroll{overflow:auto}.metric{display:flex;gap:24px;flex-wrap:wrap;margin:12px 0}.metric strong{color:var(--orange)}.error{color:#ffb2a7}.legend{display:flex;gap:20px;color:var(--muted);font-size:13px}.legend b{font-weight:500;color:var(--blue)}.legend i{font-style:normal;color:var(--orange)}details{margin:12px 0}pre{white-space:pre-wrap;word-break:break-word;background:#0c1525;padding:12px;border-radius:8px;font-size:12px;max-height:440px;overflow:auto}a{color:var(--blue)}@media(max-width:800px){main{padding:16px}.grid{grid-template-columns:1fr}h1{font-size:23px}}
</style></head><body><main>
<div class="tag">SPIELBERG · PUBLIC DATA · V3</div>
<h1>双路径车辆精度与跨会话验收</h1>
<p class="muted">真实驾驶输入重放用于诊断等效车辆；自主预测单独接受策略训练门槛。</p>
<div class="status" id="status">__STATUS__</div>
<section class="card"><h2>固定精度门槛</h2><p>完整圈平均速度 RMSE ≤15 km/h · 每圈圈时误差 ≤2% · 每段时间误差 ≤1秒</p>
<div class="scroll"><table><thead><tr><th>候选 / 数据</th><th>路径</th><th>速度 RMSE km/h</th><th>最大圈时误差 %</th><th>最大分段误差 s</th><th>结论</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="muted">v2历史新测试速度 RMSE 20.640 km/h。v3驾驶输入、对齐和执行协议有变化，历史值仅作背景，不构成同协议收益比较。</p></section>
<section class="card"><div class="controls"><label>候选 <select id="candidate"></select></label><label>路径 <select id="mode"><option value="replay">真实输入重放</option><option value="autonomous">模型自主预测</option></select></label><label>开发圈 <select id="lap"></select></label><label><input id="observed" type="checkbox" checked> 显示观测</label></div>
<div id="traceStatus" class="metric"></div><div class="legend"><b>● 模拟</b><i>● 公开观测</i></div></section>
<div class="grid"><section class="card"><h2>速度曲线</h2><canvas id="speed"></canvas></section><section class="card"><h2>纵向与横向加速度</h2><canvas id="accel"></canvas><p class="muted">同一车辆积分内核；曲率和轮胎能力来自冻结模型。</p></section></div>
<div class="grid" style="margin-top:16px"><section class="card"><h2>赛道进度</h2><canvas id="track"></canvas><div class="controls"><button id="play">播放</button><input id="progress" type="range" min="0" max="1000" value="0" style="flex:1"><span id="cursor"></span></div></section><section class="card"><h2>电量 / 等效动力</h2><canvas id="energy"></canvas><p id="energyNote" class="muted"></p></section></div>
<section class="card"><h2>验收与诊断记录</h2><div id="records"></div></section>
<section class="card"><h2>解释边界</h2><p>质量默认800 kg。未知燃油、轮胎、真实制动压力、SOC、动力分配和精确赛车线仍可能造成误差。重放的等效动力不代表真实ICE或MGU-K功率；重放不验证ERS或SOC。</p><p>参数和规则仅对应公开数据及保守实现子集；任何收益只属于通过验收后的同协议模拟实验。</p><p id="uiVerification" class="muted">计算验证与页面交互验证分别记录。</p></section>
</main><script id="data" type="application/json">__PAYLOAD__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const candidates=D.candidates||[], laps=(D.local_dataset||{}).laps||[];
let selected=null, result=null, currentLap=null, timer=null;
function option(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.appendChild(o)}
candidates.forEach((c,i)=>option($('candidate'),i,(c.report||{}).candidate||c.candidate||('候选'+i)));
laps.filter(l=>l.role==='development').forEach(l=>option($('lap'),l.lap_number,'第'+l.lap_number+'圈'));
if(D.selected_candidate_index!=null)$('candidate').value=String(D.selected_candidate_index);
function canvas(id){const el=$(id),dpr=window.devicePixelRatio||1,w=el.clientWidth,h=el.clientHeight;el.width=w*dpr;el.height=h*dpr;const ctx=el.getContext('2d');ctx.scale(dpr,dpr);ctx.fillStyle='#151e30';ctx.fillRect(0,0,w,h);return{ctx,w,h}}
function chart(id,series,xlabel,ylabel){const{ctx,w,h}=canvas(id),pad={l:50,r:14,t:15,b:34};let xs=[],ys=[];series.forEach(s=>{s.x.forEach((x,i)=>{if(Number.isFinite(x)&&Number.isFinite(s.y[i])){xs.push(x);ys.push(s.y[i])}})});ctx.font='12px system-ui';ctx.fillStyle='#abbad1';if(!xs.length){ctx.fillText('此路径没有可显示的数据',20,36);return}let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(xmax<=xmin)xmax=xmin+1;if(ymax<=ymin)ymax=ymin+1;const ym=(ymax-ymin)*.08;ymin-=ym;ymax+=ym;const xx=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),yy=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);ctx.strokeStyle='#2a3852';for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4;ctx.beginPath();ctx.moveTo(pad.l,yy(y));ctx.lineTo(w-pad.r,yy(y));ctx.stroke();ctx.fillText(y.toFixed(1),5,yy(y)+4)}ctx.fillText(xlabel,w/2,h-6);ctx.fillText(ylabel,5,11);series.forEach(s=>{ctx.strokeStyle=s.color;ctx.lineWidth=1.7;ctx.beginPath();let drawing=false;s.x.forEach((x,i)=>{const y=s.y[i];if(!Number.isFinite(x)||!Number.isFinite(y)){drawing=false;return}if(!drawing){ctx.moveTo(xx(x),yy(y));drawing=true}else ctx.lineTo(xx(x),yy(y))});ctx.stroke()})}
function findResult(){selected=candidates[Number($('candidate').value)];currentLap=(selected?.dataset?.laps||laps).find(l=>String(l.lap_number)===$('lap').value);if(!selected||!currentLap){result=null;return}const p=(selected.development||{})[$('mode').value]||{};result=$('mode').value==='autonomous'?p.result:(p.results||{})[String(currentLap.lap_number)];if(!result&&Array.isArray(p.results))result=p.results.find(r=>r.lap_number===currentLap.lap_number)?.result||null}
function terminalSeries(r){const x=[...(r.s_m||[])],v=[...(r.speed_mps||[])];const sm=r.summary||{};if(x.length&&sm.speed_end_mps!=null){x.push(sm.distance_completed_m??sm.length_m??((selected.dataset||{}).track||D.local_dataset?.track||{}).length_m);v.push(sm.speed_end_mps)}return{x,v}}
function drawTrack(){const{ctx,w,h}=canvas('track'),t=(selected?.dataset||D.local_dataset||{}).track||{},xs=t.x_m||[],ys=t.y_m||[];if(!xs.length){ctx.fillStyle='#abbad1';ctx.fillText('无路线数据',20,30);return}const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),scale=Math.min((w-40)/(xmax-xmin),(h-40)/(ymax-ymin)),xx=x=>20+(x-xmin)*scale,yy=y=>h-20-(y-ymin)*scale;ctx.strokeStyle='#667d9e';ctx.lineWidth=3;ctx.beginPath();xs.forEach((x,i)=>i?ctx.lineTo(xx(x),yy(ys[i])):ctx.moveTo(xx(x),yy(ys[i])));ctx.closePath();ctx.stroke();const ratio=Number($('progress').value)/1000, n=(result?.s_m||[]).length,idx=Math.max(0,Math.min(n-1,Math.floor(ratio*Math.max(0,n-1))));const distance=n?result.s_m[idx]:0,length=t.length_m||1,ti=Math.min(xs.length-1,Math.floor(distance/length*xs.length));ctx.fillStyle='#f4ab59';ctx.beginPath();ctx.arc(xx(xs[ti]),yy(ys[ti]),6,0,2*Math.PI);ctx.fill();$('cursor').textContent=n?Math.round(distance)+' m · '+(result.speed_mps[idx]*3.6).toFixed(1)+' km/h':'无轨迹'}
function draw(){findResult();const mode=$('mode').value;const p=(selected?.development||{})[mode]||{},m=(p.metrics||[]).find(x=>x.valid!==false&&x.lap_number===currentLap?.lap_number);$('traceStatus').textContent=result?((result.summary||{}).completed_lap?'完整圈':'无效 / 未完成圈')+(m?' · RMSE '+m.speed_rmse_kmh.toFixed(3)+' km/h · 圈时误差 '+m.lap_error_pct.toFixed(3)+'%':' · 不计入完整圈精度评分'):'无可用轨迹';$('traceStatus').className='metric '+(result?.summary?.completed_lap?'':'error');const rv=result?terminalSeries(result):{x:[],v:[]},series=[{x:rv.x,y:rv.v.map(x=>x*3.6),color:'#65baf2'}];if($('observed').checked&&currentLap)series.push({x:currentLap.s_m||[],y:(currentLap.speed_mps||[]).map(x=>x*3.6),color:'#f4ab59'});chart('speed',series,'距离 m','km/h');const s=result?.s_m||[];chart('accel',[{x:s,y:(result?.ax_mps2||[]).map(x=>x/9.81),color:'#65baf2'},{x:s,y:(result?.ay_mps2||[]).map(x=>x/9.81),color:'#f4ab59'}],'距离 m','G');if(mode==='replay'){chart('energy',[{x:s,y:result?.equivalent_drive_kw||[],color:'#65baf2'}],'距离 m','等效 kW');$('energyNote').textContent='ERS、SOC不适用。曲线仅表示重放驾驶输入对应的等效轮端动力。'}else{const e=result?.energy_mj||[],ex=[...s];if(e.length===s.length+1)ex.push(rv.x[rv.x.length-1]);chart('energy',[{x:ex,y:e,color:'#65baf2'}],'距离 m','MJ');$('energyNote').textContent='因果执行：初始2 MJ、保底2 MJ、容量4 MJ，部署只使用已经回收的能量。'}drawTrack()}
['candidate','mode','lap','observed'].forEach(id=>$(id).addEventListener('change',draw));$('progress').addEventListener('input',drawTrack);$('play').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放';return}$('play').textContent='暂停';timer=setInterval(()=>{$('progress').value=(Number($('progress').value)+4)%1001;drawTrack()},30)});window.addEventListener('resize',draw);
const records=[['网格收敛',D.mesh],['跨会话质量与访问',D.cross_session],['校准与参数灵敏度',candidates.map(c=>c.report)],['燃油质量敏感性',D.mass_sensitivity],['训练放行',D.training],['来源与信息边界',D.audit],['验证记录',D.verification]];records.forEach(([title,value])=>{const details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent=title;pre.textContent=JSON.stringify(value??{},null,2);details.append(summary,pre);$('records').appendChild(details)});$('uiVerification').textContent=D.verification?.browser_visual_verification||'页面交互验证尚未完成；计算验证单独记录。';draw();
</script></body></html>'''
