"""Build a self-contained scientific report without altering sealed runs."""
from pathlib import Path
import sys, json, gzip, html
import numpy as np


def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8-sig')) as stream:
        return json.load(stream)


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    sys.path.insert(0, str(root/'.cache/reporting-deps'))
    sys.path.insert(0, str(root))
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots
    from f1ers_v4.data import load_local_dataset
    from f1ers_v7.experiment import digest
    if output.exists():
        raise ValueError('Do not overwrite delivery report')
    runs = []
    for name in ('austria_2026_v7', 'austria_2026_v7B', 'austria_2026_v7C', 'austria_2026_v7D'):
        folder = root/'results'/name
        experiment = read(folder/'experiment.json')
        winner = experiment['winner']
        calibration = read(folder/'selected_calibration.json') if (folder/'selected_calibration.json').is_file() else {'aggregate': {}, 'metrics': []}
        development = read(folder/'development_score.json') if (folder/'development_score.json').is_file() else {'aggregate': {}}
        saved_trace = read(folder/winner['result_directory']/'lap_6.json.gz') if winner else None
        runs.append({'name': name, 'folder': folder, 'experiment': experiment, 'winner': winner,
            'calibration': calibration, 'development': development, 'trace': saved_trace,
            'manifest_sha256': digest(folder/'manifest.json')})
    # A fast physical qualification alone cannot replace integration convergence.
    eligible = [r for r in runs if r['winner'] and r['trace']
        and r['winner']['qualification'].get('passed') is True
        and r['experiment']['gates'].get('numerical_convergence') is True
        and np.isfinite(r['winner']['qualification']['last_two_mean_time_s'])]
    if not eligible:
        raise ValueError('No retained winner passed both six-lap qualification and numerical convergence')
    fastest = min(eligible, key=lambda r: r['winner']['qualification']['last_two_mean_time_s'])
    selected, trace = fastest['trace']['summary'], fastest['trace']['internal_trace']
    # Summarize already frozen scoring ledgers; do not reintegrate electrical power.
    saved_calibration_metrics = [metric for metric in fastest['calibration']['metrics']
        if metric.get('valid') is True and metric['lap_number'] in (3,4,5,6)]
    calibration_boundaries_complete = (len(saved_calibration_metrics)==4 and
        {metric['lap_number'] for metric in saved_calibration_metrics} == {3,4,5,6})
    sector_energy_means = []
    for sector in (1,2,3):
        entries = [entry for metric in saved_calibration_metrics
            for entry in metric.get('sector_energy', [])
            if entry.get('sector') == sector and entry.get('complete') is True]
        row = {'sector':sector, 'available':bool(calibration_boundaries_complete and len(entries)==4),
            'saved_calibration_laps':[metric['lap_number'] for metric in saved_calibration_metrics],
            'boundary_definition':'Each observed calibration lap has its own already saved sector boundaries',
            'prediction_trace':fastest['winner']['result_directory']+'/lap_6.json.gz'}
        if row['available']:
            row.update(deploy_mj=float(np.mean([entry['deploy_mj'] for entry in entries])),
                harvest_mj=float(np.mean([entry['harvest_mj'] for entry in entries])),
                net_energy_mj=float(np.mean([entry['net_energy_mj'] for entry in entries])),
                energy_balance_residual_mj=float(np.mean([entry['balance_residual_mj'] for entry in entries])),
                max_abs_energy_balance_residual_mj=float(np.max(np.abs([entry['balance_residual_mj'] for entry in entries]))),
                saved_boundary_sample_count=len(entries))
        sector_energy_means.append(row)
    steady_summaries = fastest['winner']['qualification']['lap_summaries'][-2:]
    steady_times = np.asarray([summary['lap_time_s'] for summary in steady_summaries],float)
    steady_net = np.asarray([summary['energy_end_mj']-summary['energy_start_mj'] for summary in steady_summaries],float)
    steady_stability = {'physical_laps':[5,6], 'samples':len(steady_summaries), 'std_ddof':0,
        'lap_time_mean_s':float(np.mean(steady_times)), 'lap_time_span_s':float(np.ptp(steady_times)),
        'lap_time_std_s':float(np.std(steady_times)),
        'net_energy_mean_mj':float(np.mean(steady_net)), 'net_energy_span_mj':float(np.ptp(steady_net)),
        'net_energy_std_mj':float(np.std(steady_net)),
        'source':'Already saved six-carry qualification lap_summaries, final two physical laps',
        'new_scoring_or_simulation':False, 'stationary_or_infinite_horizon_proven':False}
    colors = ['#536a78', '#2d82b7', '#d58228', '#854c8c']
    data = load_local_dataset(root)
    charts = []
    speed = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['冻结路线坐标上的速度', '相对校准圈观测的速度残差'])
    for lap in data['laps']:
        if lap['role'] == 'calibration':
            speed.add_trace(go.Scatter(x=lap['s_m'], y=np.asarray(lap['speed_mps'])*3.6,
                name=f"观测正赛圈 {lap['lap_number']}", line={'width':1, 'color':'#b2bdc6'}), row=1, col=1)
    for r, color in zip(runs, colors):
        if not r['trace']:
            continue
        tt = r['trace']['internal_trace']; ss = np.asarray(tt['s_m']); vv = np.asarray(tt['speed_mps'])*3.6
        speed.add_trace(go.Scatter(x=ss, y=vv, name=r['name'].replace('austria_2026_', ''), line={'color':color}), row=1, col=1)
        for lap in data['laps']:
            if lap['role'] == 'calibration':
                speed.add_trace(go.Scatter(x=lap['s_m'], y=np.interp(lap['s_m'],ss,vv)-np.asarray(lap['speed_mps'])*3.6,
                    name=r['name']+f" residual lap{lap['lap_number']}", line={'color':color,'width':1}, showlegend=False), row=2, col=1)
    speed.update_yaxes(title_text='km/h', row=1, col=1); speed.update_yaxes(title_text='预测−观测 km/h', row=2, col=1)
    speed.update_xaxes(title_text='固定路线距离 m', row=2, col=1); charts.append(speed)
    sectors = make_subplots(rows=1,cols=2,subplot_titles=['历史开发圈 7/9：分段速度 RMSE', '校准圈 3–6：分段平均偏差'])
    for r,color in zip(runs,colors):
        if r['development']['aggregate']:
            sectors.add_trace(go.Bar(x=['S1','S2','S3'],y=r['development']['aggregate']['sector_speed_rmse_kmh'],name=r['name'],marker_color=color),row=1,col=1)
        if r['calibration']['metrics']:
            bias=np.mean([m['sector_signed_bias_kmh'] for m in r['calibration']['metrics']],axis=0)
            sectors.add_trace(go.Bar(x=['S1','S2','S3'],y=bias,name=r['name'],marker_color=color,showlegend=False),row=1,col=2)
    sectors.update_yaxes(title_text='km/h'); charts.append(sectors)
    energy = make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=['已选六圈中第六圈的理想 DC 等效电量', '实际 DC 功率与规则期间'])
    energy.add_trace(go.Scatter(x=trace['s_m'],y=trace['energy_mj'],name='DC-equivalent E'),row=1,col=1)
    energy.add_trace(go.Scatter(x=trace['s_m'][1:],y=trace['dc_kw'],name='actual DC kW'),row=2,col=1)
    energy.add_hline(y=2,line_dash='dash',row=1,col=1); energy.add_hline(y=4,line_dash='dash',row=1,col=1)
    energy.update_yaxes(title_text='MJ',row=1,col=1); energy.update_yaxes(title_text='kW',row=2,col=1)
    energy.update_xaxes(title_text='路线距离 m',row=2,col=1); charts.append(energy)
    carried=make_subplots(rows=1,cols=2,subplot_titles=['六个实际连续圈的圈时', '每圈理想能量净变化'])
    for r,color in zip(runs,colors):
        if not r['winner']:
            continue
        ss=r['winner']['qualification']['lap_summaries']
        carried.add_trace(go.Scatter(x=list(range(1,7)),y=[s['lap_time_s'] for s in ss],name=r['name'],line={'color':color}),row=1,col=1)
        carried.add_trace(go.Scatter(x=list(range(1,7)),y=[1e6*(s['energy_end_mj']-s['energy_start_mj']) for s in ss],name=r['name'],line={'color':color},showlegend=False),row=1,col=2)
    carried.add_hline(y=0,row=1,col=2,line_dash='dash'); carried.update_yaxes(title_text='s',row=1,col=1); carried.update_yaxes(title_text='J / lap',row=1,col=2)
    charts.append(carried)
    fig=go.Figure()
    tail_rows = read(root/'results/austria_2026_v7C/tail_guard_rows.json')
    for row in tail_rows:
        metrics=row['calibration_diagnostic']['aggregate']
        if metrics:
            fig.add_trace(go.Bar(x=['S1','S2','S3'],y=metrics['sector_speed_rmse_kmh'],
                name=f"尾段 {row['tail_guard_m']:g} m ({'可行' if row['qualification']['passed'] else '被拒绝'})"))
    fig.update_layout(title='独立尾段工程限制对照：失败候选的 RMSE 仅作诊断',yaxis_title='校准 RMSE km/h'); charts.append(fig)
    for fig in charts:
        fig.update_layout(template='plotly_white',height=560,font={'family':'Arial, sans-serif','size':14},margin={'l':65,'r':30,'t':65,'b':60})
    b=read(root/'results/austria_2026_v7B/experiment.json')
    d=read(root/'results/austria_2026_v7D/experiment.json')
    sensitivity_path = root/'results/austria_2026_v7D/sensitivity.json'
    sensitivity_rows = []
    for saved in read(sensitivity_path) if sensitivity_path.is_file() else []:
        summaries = saved.get('lap_summaries', [])
        sixth = next((summary for summary in summaries if summary.get('physical_lap_index')==5),None)
        sixth_complete = bool(sixth and sixth.get('completed_lap') is True
            and sixth.get('execution_feasible') is True
            and isinstance(sixth.get('lap_time_s'),(int,float))
            and np.isfinite(sixth['lap_time_s']) and sixth['lap_time_s']>0)
        last = summaries[-1] if summaries else {}
        sensitivity_rows.append({'mass_kg':saved['mass_kg'], 'ice_power_kw':saved['ice_power_kw'],
            'saved_lap_count':len(summaries),
            'complete_physical_laps':sum(summary.get('completed_lap') is True
                and summary.get('execution_feasible') is True for summary in summaries),
            'has_complete_lap6':sixth_complete,
            'lap6_time_s':sixth['lap_time_s'] if sixth_complete else None,
            'lap6_net_energy_mj':sixth['energy_end_mj']-sixth['energy_start_mj'] if sixth_complete else None,
            'last_saved_lap_number':last['physical_lap_index']+1 if 'physical_lap_index' in last else None,
            'last_saved_sustainable':last.get('sustainable'), 'last_saved_status':last.get('status'),
            'saved_lap_numbers_with_negative_net_energy':[summary['physical_lap_index']+1
                for summary in summaries if summary['energy_end_mj']-summary['energy_start_mj']<0],
            'physical_control_line_continuity':saved.get('physical_control_line_continuity'),
            'controller_reoptimized':False, 'formal_qualification_or_accuracy_rescored':False})
    r7_development = read(root/'results/austria_2026_v6/R7/development_score.json')['aggregate']
    r7_rmse = r7_development['mean_speed_rmse_kmh']
    c_rmse = runs[2]['development']['aggregate']['mean_speed_rmse_kmh']
    tail_comparison = {row['tail_guard_m']: row for row in tail_rows if row['family'] == 'B_selected'}
    tail_delta = abs(tail_comparison[700.]['qualification']['last_two_mean_time_s']-
        tail_comparison[1000.]['qualification']['last_two_mean_time_s'])
    learning_rows=[]
    critic_rows=[]
    for family in ('austria_2026_v7B', 'austria_2026_v7D'):
            pattern = 'worker_receipt.json' if family.endswith('v7D') else '*metadata.json'
            for path in sorted((root/'results'/family).rglob(pattern)):
                relative = path.relative_to(root/'results'/family)
                if 'inputs' in relative.parts:
                    continue
                m=read(path)
                if family.endswith('v7B') and ('algorithm' not in m or 'seed' not in m):
                    continue
                stage=relative.parts[0]
                fallback_seed = relative.parts[1] if len(relative.parts)>1 else None
                learning_rows.append({'stage':family.replace('austria_2026_','')+'/'+stage,'seed':m.get('seed', fallback_seed),'metadata_file':path.relative_to(root).as_posix(),
                    'episodes_requested':m.get('episodes_requested',0),'episodes_attempted':m.get('episodes_attempted'),
                    'episodes_completed':m.get('episodes_completed',0),
                    'physical_full_laps_completed':m.get('physical_full_laps_completed'),'best_episode':m.get('best_episode'),
                    'gain_over_bc_s':m.get('gain_over_bc_s'),'sac_improvement_found':m.get('sac_improvement_found',False)})
                if family.endswith('v7D') and stage in ('D3','D4'):
                    warmup = m.get('critic_initialisation', {})
                    history_path = path.parent/'sac_history.json'
                    history = read(history_path) if history_path.is_file() else []
                    # The frozen D history uses `critic`, not `critic_diagnostic`.
                    saved_diagnostics = [record for record in history if isinstance(record.get('critic'),dict)]
                    last_diagnostic = saved_diagnostics[-1] if saved_diagnostics else {}
                    critic_rows.append({'stage':stage, 'seed':m.get('seed',fallback_seed),
                        'q_vs_mc_before_rmse_s':warmup.get('before',{}).get('q_vs_mc_rmse_s'),
                        'q_vs_mc_after_rmse_s':warmup.get('after',{}).get('q_vs_mc_rmse_s'),
                        'last_saved_training_episode':last_diagnostic.get('episode'),
                        'q_vs_mc_last_rmse_s':last_diagnostic.get('critic',{}).get('q_vs_mc_rmse_s'),
                        'worker_receipt':path.relative_to(root).as_posix(),
                        'history_file':history_path.relative_to(root).as_posix() if history_path.is_file() else None,
                        'history_critic_field':'critic', 'convergence_proven':False})
    numbers=[]
    for r in runs:
        development_aggregate=r['development']['aggregate'] or {}; c=r['calibration']['aggregate'] or {}
        numbers.append({'experiment':r['name'],'source_commit':r['experiment']['source_commit'],
            'manifest_sha256':r['manifest_sha256'],'winner':r['winner']['name'] if r['winner'] else None,
            'last_two_time_s':r['winner']['qualification']['last_two_mean_time_s'] if r['winner'] else None,
            'fastest_comparison_eligible':r in eligible,
            'calibration':c,'development':development_aggregate,'gates':r['experiment']['gates']})
    facts={'schema_version':7,'fastest_retained_tested_experiment':fastest['name'],
        'actual_new_SAC_episodes_completed':sum(r['experiment'].get('actual_new_SAC_episodes_completed',0) for r in runs),
        'actual_new_SAC_episodes_requested':sum(r['experiment'].get('actual_new_SAC_episodes_requested',0) for r in runs),
        'D_actual_new_SAC_episodes_attempted':d.get('actual_new_SAC_episodes_attempted'),
        'completed_episode_interpretation':'Terminated and saved training episode, including physical terminal failure; not a certified legal lap',
        'fastest_requires_both_qualification_and_numerical_convergence':True,
        'fastest_saved_calibration_sector_energy_means':sector_energy_means,
        'fastest_six_carry_final_two_stability':steady_stability,
        'historical_R7_development_rmse_kmh':r7_rmse,'C_development_rmse_kmh':c_rmse,
        'C_minus_R7_development_rmse_kmh':c_rmse-r7_rmse,
        'C_same_controller_tail_700_vs_1000_mean_time_difference_s':tail_delta,
        'C_350_qualification_passed':tail_comparison[350.]['qualification']['passed'],
        'D_retained_candidate_qualifications':[{'name':candidate['name'],
            'retained_checkpoint':candidate.get('retained_checkpoint'),
            'qualification_passed':candidate['qualification']['passed']}
            for candidate in d['candidates'] if candidate.get('retained_checkpoint') in ('best','last')],
        'experiments':numbers,'learning':learning_rows,'browser_visual_validation':'not_measured',
        'D_saved_critic_MC_diagnostics_by_seed':critic_rows,
        'D_frozen_controller_parameter_sensitivity':{
            'controller':d['winner']['name'] if d['winner'] else None,
            'saved_source':'results/austria_2026_v7D/sensitivity.json',
            'controller_reoptimized':False, 'formal_qualification_or_accuracy_rescored':False,
            'baseline_six_lap_qualification_generalized_to_varied_parameters':False,
            'rows':sensitivity_rows},
        'A_critic_comparison_scope':'S6 cached isolated critic experiment, not newly executed A/B SAC or D training outcomes',
        'FP2_accessed':False,'full_FIA_proven':False,'true_SOC_proven':False,
        'global_optimality_proven':False,'infinite_horizon_proven':False,
        'all_trained_checkpoints_six_lap_qualified':False,
        'limitations':['D retained best and last checkpoints were loaded for six-lap qualification; unsaved intermediate updates were not exhaustively qualified',
            'Historical development was already reused; not a new test',
            'D fixes the initial-energy objective alias, not complete Markov sufficiency; initial speed closure still depends on an unobserved initial speed',
            'Negative-wheel raw actor actions are physical noops under D; inherited SAC entropy and approximate Q gradients can still affect the shared network',
            'Ideal DC-equivalent storage excludes real ES losses and auxiliaries']}
    def table(rows,cols):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(c,'')))+'</td>' for c in cols)+'</tr>' for row in rows)+'</tbody></table>'
    def display(value, decimals=3):
        return round(float(value),decimals) if value is not None and np.isfinite(value) else '未评分'
    comparison=[{'实验':r['experiment'],'后两圈均时 s':display(r['last_two_time_s'],6),
        '开发平均 RMSE km/h':display(r['development'].get('mean_speed_rmse_kmh')),
        '相对 R7 RMSE 增量 km/h':display(r['development'].get('mean_speed_rmse_kmh')-r7_rmse) if r['development'] else '未评分',
        '最大圈时误差 %':display(r['development'].get('max_abs_lap_error_pct')),
        '最大分段误差 s':display(r['development'].get('max_abs_sector_error_s')),
        '精度通过':r['gates']['development_accuracy'],'数值通过':r['gates']['numerical_convergence'],
        '进入最快比较':r['fastest_comparison_eligible']} for r in numbers]
    warm=fastest['winner']['qualification']['lap_summaries'][-1]
    surplus=(warm['energy_end_mj']-warm['energy_start_mj'])*1e6
    parasitic=surplus/warm['lap_time_s']
    body='<h1>Spielberg v7：可达请求、连续多圈与价值学习</h1>'
    body+='<p class="bad">这是公开规则子集下的模型研究交付。真实精度门槛仍独立判定；完整 FIA ECU、真实 SOC 与跨会话测试均未验证。</p>'
    body+=f'<p>在六圈资格与数值收敛均通过的已保留候选中，最快控制器为 {html.escape(fastest["winner"]["name"])}，后两圈均时 {fastest["winner"]["qualification"]["last_two_mean_time_s"]:.6f} 秒。速度精度仍按独立门槛验收；这只是有限候选比较，不是全局最优证明。</p>'
    body+=table(comparison,list(comparison[0]))
    body+=f'<p class="bad">历史 R7 开发平均速度 RMSE 为 {r7_rmse:.7f} km/h；已完成 C 为 {c_rmse:.7f} km/h，增加 {c_rmse-r7_rmse:.7f} km/h。公开子集及工程控制门槛的改善没有带来这项精度的改善。R7 未通过完整冷态附加门槛，不能因此恢复为合格控制器；D 的实际最终评分与门槛单独列在上表。</p>'
    body+='<h2>物理与学习归因</h2><ol><li>旧冷态 HOLD/LIMITED 历史请求超过能源可达代理：完整期间审计发现 1070 次缺口查询，最大 75.679 kW。修复预留保持、完整降功率斜坡和决策通知能量；不得使用电量不足作为 DC 斜率豁免。</li><li>旧观察遗漏历史参考与削减目标。新观察显式记录两者，33维检查点独立版本化。</li><li>固定尺度同步变换 Q、奖励、失败罚分及熵。A 的 S6 缓存隔离 critic 对照中，Q–MC误差约40秒降至16–18秒；这只属于该缓存诊断，不代表 A/B 执行了新 SAC，也不能泛化到 D。D 的实际初始化与训练末次诊断另列下表。</li><li>真实完成一圈但亏电的轨迹原先可能得到快速圈时奖励。现在保留物理轨迹并赋予原100秒约束失败罚分。</li><li>单圈教师最优不足以证明连续可行。数值候选必须真实携带六圈全部状态，观察残差不进入控制目标。</li></ol>'
    body+='<p>BC 的满再生请求存在约5.7–7.0 kJ的近似缺口，正部署也改变了收支，不能归结为单一原因。D 把负轮端需求的最大再生请求交给硬约束层，actor仅选择正部署，并将本圈初始电量加入34维观察。常数 raw=0 的教师表示正轮段部署请求0.5、负轮段最大请求再生。显式零均值初始化已解析复现该常数，真实600+3×200监督更新仍训练方差及共享隐藏层，却不能解释为学到了更快的均值控制；实际再生仍受原联合轮胎力、电量容量、功率与规则约束。</p>'
    body+=f'<p>B 新增 SAC 已终止并留存 {b["actual_new_SAC_episodes_completed"]} episodes；D 请求 {d["actual_new_SAC_episodes_requested"]}、开始 {d.get("actual_new_SAC_episodes_attempted", "未记录")}、终止并留存 {d["actual_new_SAC_episodes_completed"]} episodes。completed 包含物理失败后正常终止的训练轨迹，不能称为合法完整圈；表中另列 full physical laps，它们仍需独立合法性及可持续资格。本轮请求、开始及终止总量均不得超过120。best_episode=0 是保留的 BC，不算 SAC 收益。</p>'
    body+=table(learning_rows,['stage','seed','episodes_requested','episodes_attempted','episodes_completed','physical_full_laps_completed','best_episode','gain_over_bc_s','sac_improvement_found'])
    body+='<p>D 各 SAC 条件/seed 任务已保存的 Q–MC 误差如下，直接读取 worker_receipt 中 critic_initialisation.before/after，以及 sac_history 最后保存的 critic 诊断。before/after 使用同一初始化轨迹；末次诊断来自当时训练轨迹，其 actor、状态分布及 soft MC 回报可能已改变，不能把三列当作同一验证集上的单调收敛曲线，也不能代替速度 RMSE、六圈资格或数值收敛。未保存的诊断标为未评分。</p>'
    critic_table = [{'条件':row['stage'], 'seed':row['seed'],
        'warmup 前 Q–MC RMSE s':display(row['q_vs_mc_before_rmse_s'],4),
        'warmup 后 Q–MC RMSE s':display(row['q_vs_mc_after_rmse_s'],4),
        '末次保存诊断 episode':row['last_saved_training_episode'],
        '末次 Q–MC RMSE s':display(row['q_vs_mc_last_rmse_s'],4)} for row in critic_rows]
    body+=table(critic_table,['条件','seed','warmup 前 Q–MC RMSE s','warmup 后 Q–MC RMSE s','末次保存诊断 episode','末次 Q–MC RMSE s'])
    body+='<p>D 对实际保留且可读取的 best 和 last 检查点重新加载并尝试六圈资格，缺失或失败单独保留。冷组单圈评估固定教师首圈边界，暖组固定教师第六圈起始完整边界，两者均禁止速度 shooting；最后所有候选统一从同一冷态 E2 起点实际携带六圈排名。未保存的中间更新权重没有遍历资格，单边界 best 也不能视为所有更新权重中的连续冷起最优。</p>'
    body+='<h2>分段精度与能量平稳度</h2><p>只完整物理圈评分，所有残差保留。原门槛为完整圈平均速度 RMSE≤15 km/h、每圈圈时误差≤2%、每段圈时误差≤1秒；三段速度 RMSE 单独作诊断，没有新增每段 RMSE≤15 的硬门槛。校准3–6圈；7/9圈是历史开发，不是新测试。前两段偏快与第三段偏慢不能相互抵消后宣布通过。</p>'
    body+='<p>以下三段能量读自最快合格实验已经保存的 selected_calibration 评分：同一已选第六圈预测轨迹，按校准观测圈3–6各自保存的分段边界切分后取平均。各圈边界略异，四组账本不是四条独立预测；没有重新积分或重新评分。net 为段末减段初的理想储能变化，正值表示净充电；balance residual=net+deploy−harvest。</p>'
    energy_rows = [{'段':f'S{row["sector"]}', '保存边界组数':row.get('saved_boundary_sample_count',0),
        'deploy MJ':display(row.get('deploy_mj'),6), 'harvest MJ':display(row.get('harvest_mj'),6),
        'net MJ':display(row.get('net_energy_mj'),6),
        '平均平衡残差 MJ':f'{row["energy_balance_residual_mj"]:.3e}' if row['available'] else '未保存',
        '最大绝对残差 MJ':f'{row["max_abs_energy_balance_residual_mj"]:.3e}' if row['available'] else '未保存'}
        for row in sector_energy_means]
    body+=table(energy_rows,list(energy_rows[0]))
    stability_rows = [{'保存指标':'第5/6实际圈圈时 s',
        '均值':f'{steady_stability["lap_time_mean_s"]:.9f}',
        'span':f'{steady_stability["lap_time_span_s"]:.3e}',
        'std (ddof=0)':f'{steady_stability["lap_time_std_s"]:.3e}'},
        {'保存指标':'第5/6实际圈净理想储能变化 J',
        '均值':f'{steady_stability["net_energy_mean_mj"]*1e6:.6f}',
        'span':f'{steady_stability["net_energy_span_mj"]*1e6:.3e}',
        'std (ddof=0)':f'{steady_stability["net_energy_std_mj"]*1e6:.3e}'}]
    body+='<p>六圈真实携带中，最后两圈的已保存时间与净能量描述统计如下。span 为最大减最小，std 使用 ddof=0；只有两个样本，不能据此证明稳态分布、数值收敛或无限周期可持续性。</p>'
    body+=table(stability_rows,list(stability_rows[0]))
    for index,fig in enumerate(charts):
        body+=fig.to_html(full_html=False,include_plotlyjs=True if index==0 else False)
    body+=f'<p>所选暖圈理想能量净变化 {surplus:.3f} J；等价持续损耗余量约 {parasitic:.3f} W。它只描述这一理想暖圈的余量，没有实测储能损耗与辅助负载，不能据此确认真实可持续性。六圈携带不证明无限周期；实际SOC≥初值仍待真实参数验证。</p>'
    body+='<h3>冻结 D 控制器的参数敏感性</h3><p>下表只读取已保存 sensitivity 摘要，四组参数使用同一已选 D 控制器，没有重新优化，也未重新进行完整资格或正式精度验收。完整圈数只要求完成物理圈且 execution_feasible；车辆积分 status=accepted 与 sustainable 是不同指标。负净ΔE圈号由每个保存圈的 E_end−E_start 直接计算，只作算数摘要；末圈净充电不能证明前六圈均可持续。基线参数的六圈资格不能外推到这些参数。缺少完整第六圈时，不给出第六圈时间、净能量或伪造正式精度；末圈状态指最后实际保存的圈。</p>'
    sensitivity_table = [{'质量 kg':row['mass_kg'], 'ICE轮端 kW':row['ice_power_kw'],
        '完整物理圈数':row['complete_physical_laps'],
        '完整第6圈时间 s':display(row['lap6_time_s'],6),
        '完整第6圈净ΔE MJ':display(row['lap6_net_energy_mj'],9),
        '保存圈中负净ΔE圈号':row['saved_lap_numbers_with_negative_net_energy'],
        '最后保存圈号':row['last_saved_lap_number'], '末圈 sustainable':row['last_saved_sustainable'],
        '末圈 status':row['last_saved_status'], '真实控制线连续':row['physical_control_line_continuity']}
        for row in sensitivity_rows]
    body+=table(sensitivity_table,['质量 kg','ICE轮端 kW','完整物理圈数','完整第6圈时间 s','完整第6圈净ΔE MJ','保存圈中负净ΔE圈号','最后保存圈号','末圈 sustainable','末圈 status','真实控制线连续'])
    body+='<h2>验证与局限</h2><p>输出10/5/2.5m与内部时间/距离步细化独立检查；原始驾驶需求、动作和初态冻结。验证重放已选轨迹与已保存评分，不重新选择。浏览器视觉交互尚未实测；计算与独立bundle恢复验证另有回执。</p><p>保留800kg、400kW、2–4MJ假设。质量780/820kg和功率380/420kW敏感性仅作诊断，不进入选择。新的尾段对照只检验同一控制器直接改变工程限制；失败不证明长保护必需。未打开FP2，未添加自由赛车线或逐弯参数。</p>'
    body+=f'<p>C 的同一 B 控制器在700m和1000m尾段限制下，后两圈均时相差 {tail_delta:.3g} 秒，数值上等同；这不能证明1000m限制必需。350m候选的六圈资格为 {tail_comparison[350.]["qualification"]["passed"]}，即使其速度 RMSE 更好，也不能越过能量、周期或合法性边界进入最快合格比较。该实验只检验这一个控制器族，不能推断其他控制器所需的最短限制。</p>'
    body+='<p>仍有局限：旧33维观察遗漏初始电量，合成反例证明相同观察可获不同终点罚分；D的第34维修复了这一目标别名，却没有证明完整 Markov 充分性。例如终端速度闭合仍依赖未显式观察的初始速度。合成反例不能证明两条完整真实轨迹均可达，也不能单独证明SAC失败原因。负轮端需求下D的raw actor动作是物理无效动作，继承SAC仍对它计算熵及近似Q梯度，有限critic可能学到虚假的刹车动作依赖，并通过共享网络影响正部署；全再生映射的收益不能归因于SAC学会了刹车。</p>'
    body+='<h2>规则来源与冻结身份</h2><p>使用奥地利站时点的 FIA Technical Issue19 (2026-06-25) 及赛事 Power Unit Information；DOC-058 ECU期间定义不完整。工程可达请求不是“实际MGU-K必须恒定”的规则。<a href="https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_19_-_2026-06-25.pdf">FIA 技术规则</a> · <a href="https://www.fia.com/system/files/decision-document/2026_austrian_grand_prix_-_power_unit_information.pdf">奥地利站PU文件</a></p>'
    body+=table([{'实验':r['experiment'],'计算提交':r['source_commit'],'结果manifest SHA256':r['manifest_sha256']} for r in numbers],['实验','计算提交','结果manifest SHA256'])
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F1 ERS Lab v7</title><style>body{font:16px/1.6 Arial,sans-serif;max-width:1280px;margin:35px auto;padding:0 24px;color:#213142}h1,h2{line-height:1.3}.bad{border-left:5px solid #b4433d;padding:12px;background:#fff1ef}table{border-collapse:collapse;width:100%;font-size:13px;overflow-wrap:anywhere}th,td{border-bottom:1px solid #d5dde2;text-align:left;padding:8px}th{background:#eff4f6}a{color:#165e9b}</style>'+body+'</html>'
    output.mkdir(parents=True)
    (output/'F1_ERS_Lab_v7.html').write_text(page,encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return facts


if __name__=='__main__':
    print(json.dumps(build(Path(sys.argv[1]),Path(sys.argv[2])),ensure_ascii=False))
