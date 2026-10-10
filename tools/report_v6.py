"""Post-experiment scientific Plotly report; no effect on control calculations."""
from pathlib import Path
import sys,gzip,json,html,hashlib
ROOT=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'.cache/reporting-deps'))
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from f1ers_v4.data import load_local_dataset

OUT=ROOT/'results/austria_2026_v6'
def read(p):
    if p.suffix=='.gz':
        with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(p.read_text(encoding='utf-8'))
def fmt(v,d=3):return '—' if v is None else f'{v:.{d}f}'
def time_label(summary):
    if not summary:return '—'
    label=fmt(summary.get('lap_time_s'),6)
    return label if summary.get('completed_lap') else label+' (终止前经过时间，非圈时)'
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

e=read(OUT/'experiment.json'); result=read(OUT/'selected_trajectory.json.gz')
dataset=load_local_dataset(ROOT); selected=result['summary']; s=result['internal_trace']['s_m']; trace=result['internal_trace']
neutral=read(OUT/'R0_representative.json.gz')
observed=[l for l in dataset['laps'] if l['role']=='calibration']; grid=np.asarray(observed[0]['s_m'])
real=np.asarray([np.interp(grid,l['s_m'],l['speed_mps'])*3.6 for l in observed])
fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.06,
    subplot_titles=('Speed versus calibration telemetry (laps 3–6)','Executed DC power: deployment positive, harvest negative','Ideal DC-equivalent energy (actual ES losses unknown)'))
fig.add_trace(go.Scatter(x=grid,y=real.mean(0),name='Observed mean 3–6',line={'color':'#334155','width':2}),row=1,col=1)
fig.add_trace(go.Scatter(x=grid,y=real.max(0),showlegend=False,line={'width':0}),row=1,col=1)
fig.add_trace(go.Scatter(x=grid,y=real.min(0),name='Observed range',fill='tonexty',fillcolor='rgba(71,85,105,.14)',line={'width':0}),row=1,col=1)
fig.add_trace(go.Scatter(x=neutral['internal_trace']['s_m'],y=np.asarray(neutral['internal_trace']['speed_mps'])*3.6,name='Frozen v4 neutral',line={'color':'#94a3b8'}),row=1,col=1)
fig.add_trace(go.Scatter(x=s,y=np.asarray(trace['speed_mps'])*3.6,name=e['selected']['name']+' (carried lap 6)',line={'color':'#2563eb','width':2}),row=1,col=1)
fig.add_trace(go.Scatter(x=np.asarray(s)[1:],y=trace['dc_kw'],name='Executed CU-K DC',line={'color':'#0284c7'}),row=2,col=1)
fig.add_trace(go.Scatter(x=s,y=trace['energy_mj'],name='Ideal energy',line={'color':'#059669'}),row=3,col=1)
fig.add_hline(y=2.,line_dash='dash',line_color='#dc2626',row=3,col=1)
fig.add_hline(y=4.,line_dash='dash',line_color='#dc2626',row=3,col=1)
fig.update_layout(height=960,template='plotly_white',margin={'l':65,'r':35,'t':70,'b':50},legend={'orientation':'h','y':1.12},hovermode='x unified')
fig.update_yaxes(title_text='km/h',row=1,col=1);fig.update_yaxes(title_text='kW DC',row=2,col=1);fig.update_yaxes(title_text='MJ',row=3,col=1)
fig.update_xaxes(title_text='Frozen continuous route distance (m)',row=3,col=1)
figure_html=fig.to_html(full_html=False,include_plotlyjs=True,config={'displaylogo':False,'responsive':True})
stages=[];sector_values=[]
for r in e['rounds']:
    a=(r.get('calibration') or {}).get('aggregate',{})
    if 'sector_speed_rmse_kmh' in a:stages.append(r['round']);sector_values.append(a['sector_speed_rmse_kmh'])
stages.append('Selected lap 6');sector_values.append(e['calibration']['aggregate']['sector_speed_rmse_kmh'])
bars=go.Figure()
for j in range(3):bars.add_trace(go.Bar(name=f'Sector {j+1}',x=stages,y=[v[j] for v in sector_values]))
bars.update_layout(height=410,template='plotly_white',title='Calibration sector speed RMSE (all samples)',barmode='group',yaxis_title='km/h')
bars_html=bars.to_html(full_html=False,include_plotlyjs=False,config={'displaylogo':False})
stage_rows=[]
for r in e['rounds']:
    summary=r.get('summary') or {}; a=(r.get('calibration') or {}).get('aggregate',{})
    stage_rows.append([r['round'],r['change'],fmt(summary.get('lap_time_s')),fmt(a.get('mean_speed_rmse_kmh')),
        '/'.join(fmt(x) for x in a.get('sector_speed_rmse_kmh',[])),fmt(a.get('max_abs_sector_error_s'))])
sectors=e['calibration']['metrics'][0]['sector_energy']
sector_rows=[[x['sector'],fmt(x['time_s']),fmt(x['deploy_mj'],6),fmt(x['harvest_mj'],6),fmt(x['net_energy_mj'],6),fmt(x['balance_residual_mj'],10)] for x in sectors]
error_rows=[]
for role in ['calibration','development']:
    for m in e[role]['metrics']:
        error_rows.append([role,m['lap_number'],fmt(m['speed_rmse_kmh']),'/'.join(fmt(x) for x in m['sector_speed_rmse_kmh']),
            fmt(m['lap_error_pct']),'/'.join(fmt(x) for x in m['sector_error_s'])])
learning_rows=[]
for identifier in ('R5','R6'):
    row=next(x for x in e['rounds'] if x['round']==identifier)
    for c in row['cases']:
        learning_rows.append([identifier,c['seed'],c['temperature_mode'],c['episodes_completed'],c['best_episode'],
            time_label(c.get('best_summary')),time_label(c.get('last_summary')),
            c['sac_improvement_found'],fmt(c['final_alpha'],6),len(c['training_failures'])])
candidate_rows=[[c['name'],fmt(c['stable_window_mean_time_s']),fmt(c['continuity']['rows'][-1]['terminal_energy_mj'],9)]
    for c in sorted(e['multi_cycle_certified_candidates'],key=lambda c:c['stable_window_mean_time_s'])]
cal=e['calibration']['aggregate'];dev=e['development']['aggregate']
intro=f'''<h1>V6 可持续充放电控制研究</h1><p>已执行 R0–R6 七轮实验。获选 <strong>{html.escape(e['selected']['name'])}</strong>；连续验证末两圈均时 <strong>{fmt(e['contiguous_laps']['stable_window_mean_time_s'],6)} 秒</strong>。这是保留候选中的模型结果，不是全局最优或真实赛车收益证明。</p>
<div class="cards"><div>模型内相对零 K<br><strong>{fmt(e['tested_model_gain_s'],6)} 秒</strong></div><div>校准 / 开发 RMSE<br><strong>{fmt(cal['mean_speed_rmse_kmh'])} / {fmt(dev['mean_speed_rmse_kmh'])} km/h</strong></div><div>连续 / 数值检查<br><strong>{e['gates']['contiguous_sustainable']} / {e['gates']['numerical']}</strong></div><div>完整 FIA / 新测试<br><strong>未验证 / 未打开</strong></div></div>
<p class="warn">原精度门槛继续保留：平均 RMSE ≤15 km/h、每圈圈时误差≤2%、每段时间误差≤1秒。获选策略的模型精度是否通过：校准 {e['gates']['calibration_fidelity']}，开发 {e['gates']['development_fidelity']}。部分圈不评分；没有按残差删样本。</p>'''
body=intro+'<h2>已确认的根因</h2><p>旧自主驾驶器以仅 ICE 的速度包络固定目标加速度，K 多数替代 ICE，圈速梯度几乎为零。负加速度也可能来自阻力，不能据此请求再生。旧学习环境与新积分器、观测和动作定义不同，旧检查点不兼容。终点电量或功率限制状态若重置，会制造虚假的跨圈收益。v4/v5 尚未训练 ML，原精度偏差不能归因于 SAC。</p>'
body+='<h2>逐段速度、执行功率和能量</h2>'+figure_html+bars_html
body+='<h2>七轮实际改进</h2>'+table(['轮次','实际改动','代表冷圈秒','整体 RMSE','S1 / S2 / S3 RMSE','最大分段时间误差秒'],stage_rows)
body+='<h2>获选连续末圈的逐圈观测比较</h2>'+table(['角色','观测圈','RMSE km/h','S1 / S2 / S3 RMSE','圈时误差 %','S1 / S2 / S3 时间误差秒'],error_rows)
body+='<h2>按冻结观测分段的能量收支</h2>'+table(['段','秒','部署 MJ','回收 MJ','净电量 MJ','账本残差 MJ'],sector_rows)
body+='<p>上表使用校准观测的冻结时间距离边界；规则区段另使用既定 FIA→连续弧长映射。分段能量波动是计划中的充放电，整圈净变化与积分残差分别报告。50 W 辅助电耗敏感性会使理想净零圈约亏 0.0035 MJ/圈；真实 ES 损耗尚不能辨识，不能宣称真实电池 SOC 已闭合。</p>'
body+='<h2>SAC 的最好与最后候选</h2>'+table(['轮次','种子','温度','实际 episode','最好 episode','最好秒','最后秒','优于 BC','末 alpha','训练失败'],learning_rows)
body+='<p>最好 episode 为 0 表示保留 BC；它不是 SAC 改进。soft Q、熵和训练 return 不用于合法圈速排名。训练采用固定初速，正式评估采用策略自身周期速度求解；二者同执行器，但初态未必相同。R6 为预注册 MC 初始化与 λ=1秒/MJ 势函数联合对照，不能单独归因。</p>'
body+='<p>实际完成180个训练 episode：179个完成物理整圈，1个在约2750米因 deployment_slew 终止。物理完整不等于速度、电量和规则状态周期闭合。SAC 九组最好候选均回退至 episode 0 的 BC；Q 与 Monte Carlo 回报仍有明显偏差，不能宣称收敛。</p>'
body+='<h2>六圈连续验收的保留候选</h2>'+table(['候选','末两圈均时秒','第六圈末能量 MJ'],candidate_rows)
body+=table(['连续圈','圈时秒','圈初 MJ','圈末 MJ','净变化 MJ','50W敏感净变化 MJ','每圈认证'],
    [[r['lap'],fmt(r['time_s'],6),fmt(r['initial_energy_mj'],9),fmt(r['terminal_energy_mj'],9),
      fmt(r['energy_change_mj'],10),fmt(r['auxiliary_adjusted_change_mj'],6),r['per_lap_certified']]
     for r in e['contiguous_laps']['rows']])
body+='<p>所有圈从真实末状态继续执行，速度、储能、DC、FSM不重置；仅每圈回收计数重置。六圈有限验证不是无限周期稳定性证明。未保留教师试验与 ML 检查点完整归档，但未穷尽连续排名。</p>'
body+='<h2>固定原始需求的数值验收</h2>'+table(['比较类型','相邻细化','圈时差秒','速度差 RMSE km/h','通过'],
    [[group,i+1,fmt(r['lap_time_difference_s'],6),fmt(r['speed_rmse_kmh'],6),r['passed']]
     for group,rows in e['numerical']['comparisons'].items() for i,r in enumerate(rows)])
body+='<h2>冻结参数的敏感性</h2>'+table(['参数','值','圈时秒','模型内周期合法','校准 RMSE'],
    [[r['field'],r['value'],fmt(r['summary']['lap_time_s'],6),r['summary']['feasible'],
      fmt(r['calibration']['aggregate'].get('mean_speed_rmse_kmh'))] for r in e['sensitivity']])
body+='<h2>规则与复现范围</h2><p>事件重现使用奥地利赛事当日有效技术 Issue19/运动Issue7；DOC-058 的实际期间定义、转速和传动、真实SOC、精确主动气动及ICE燃油流量仍未验证。Hard Boundary 实现的是声明的公开子集；full_fia_verified 始终 false。默认 800 kg、400 kW、2–4 MJ，敏感性不参与排名。v2–v5 原交付与标签未改，FP2共享账本未解封。</p>'
body+='<p>FIA C5.12.4 约束 driver maximum power demand 的削减，不要求实际 K 恒定。本模型的历史可达 K 峰值代理、能量约束后的有效请求、实际供给分开记录。补充的 HOLD+LIMITED 可达代理一致性审查发现冷态和R0–R6获选末圈的缺口；该获选末圈不满足附加工程门槛。这不是 FIA 违规证据。原冻结证书仍按原声明子集解释，完整 FIA 验证继续待完成。详见 selected_proxy_supplement.json。</p>'
body+='<p>独立审查曾在最终汇总前读取一个 SAC 对照的历史开发圈7/9用于报告诊断；该读取未用于奖励、训练、R6配置或候选选择。开发圈已被多阶段使用，不称为新测试。FP2最终测试仍未打开。</p>'
body+='<p>正式源码提交 '+html.escape(e['provenance']['code_commit'])+'。校准与开发的分段评分可由 verify-v6 离线重跑；随机训练和搜索由完整散列库存验证，不假称重训练。浏览器视觉验收状态：未实测。结果与来源见 experiment.json、manifest.json、review/ 和 RULES_V6.md。</p>'
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V6 sustainable control</title><style>body{font-family:system-ui,Arial,sans-serif;margin:32px auto;max-width:1300px;padding:0 24px;color:#172033;line-height:1.65;background:#f8fafc}h1{font-size:32px}h2{margin-top:42px}p{max-width:1100px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.cards div{padding:18px;background:white;border:1px solid #dbe3ee;border-radius:8px}.cards strong{font-size:20px}.warn{background:#fff7ed;padding:18px;border-left:4px solid #ea580c}table{width:100%;border-collapse:collapse;font-size:14px;background:white;display:block;overflow-x:auto}th,td{text-align:left;padding:11px;border:1px solid #e2e8f0;white-space:normal}th{background:#eaf0f7}@media(max-width:850px){.cards{grid-template-columns:1fr 1fr}body{padding:0 10px;margin:15px}}</style>'+body+'</html>'
(OUT/'report.html').write_text(page,encoding='utf-8')
summary={'selected':e['selected']['name'],'stable_time_s':e['contiguous_laps']['stable_window_mean_time_s'],
    'calibration':cal,'development':dev,'gates':e['gates'],'code_commit':e['provenance']['code_commit'],
    'actual_sac_episodes':sum(c['episodes_completed'] for r in e['rounds'] if r['round'] in ('R5','R6') for c in r['cases']),
    'sac_improvements':sum(c['sac_improvement_found'] for r in e['rounds'] if r['round'] in ('R5','R6') for c in r['cases']),
    'scope':e['fastest_scope'],'global_optimality_proven':False,'full_fia_verified':False,
    'auxiliary_50w_sustainable':e['contiguous_laps']['auxiliary_50w_all_laps_nonnegative']}
if (OUT/'selected_proxy_supplement.json').exists():
    summary['additional_surrogate_hold_and_limited_request_gate_passed']=read(OUT/'selected_proxy_supplement.json')['additional_surrogate_hold_and_limited_request_gate_passed']
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
