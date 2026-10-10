"""Append a separately frozen R7 research result to the post-compute report."""
import argparse,gzip,json,html,sys
from pathlib import Path

def read(p):
    if p.suffix=='.gz':
        with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(p.read_text(encoding='utf-8'))
def fmt(v,d=6):return '—' if v is None else f'{v:.{d}f}'
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'

def main(root):
    root=Path(root).resolve();out=root/'results/austria_2026_v6';folder=out/'R7'
    e=read(folder/'experiment.json');verify=read(folder/'verification.json')
    if not verify['passed']:raise ValueError('Do not append unverified R7 as a completed comparison')
    selected=e['selected_margin_s'];candidate=next(c for c in e['candidate_summaries'] if c['margin_s']==selected)
    result=read(folder/e['selected_result_path']);summary=result['summary']
    cal=read(folder/e['selected_calibration_path']);dev=read(folder/'development_score.json')
    numeric=read(folder/'numerical_audit.json')
    sys.path.insert(0,str(root/'.cache/reporting-deps'))
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    trace=result['internal_trace'];s=trace['s_m']
    figure=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.06,
        subplot_titles=('R7 carried lap 6 speed','Executed CU-K DC power','Ideal energy: cold warm-up is scored separately'))
    figure.add_trace(go.Scatter(x=s,y=np.asarray(trace['speed_mps'])*3.6,name='R7 warm speed'),row=1,col=1)
    figure.add_trace(go.Scatter(x=np.asarray(s)[1:],y=trace['dc_kw'],name='R7 actual DC'),row=2,col=1)
    figure.add_trace(go.Scatter(x=s,y=trace['energy_mj'],name='R7 ideal energy'),row=3,col=1)
    figure.update_layout(height=900,template='plotly_white',hovermode='x unified',legend={'orientation':'h','y':1.08})
    figure.update_yaxes(title_text='km/h',row=1,col=1);figure.update_yaxes(title_text='kW DC',row=2,col=1);figure.update_yaxes(title_text='MJ',row=3,col=1)
    figure.update_xaxes(title_text='Fixed route distance (m)',row=3,col=1)
    block='<h2 id="r7">R7 独立日志驱动修正</h2>'
    block+=f'<p>新增已知几何前视保护修复 BC2027 raw 的轮端需求与降功率冲突。获选全局裕量 {selected:g}秒，六圈末两圈均时 {candidate["last_two_mean_time_s"]:.6f}秒。暖态第5/6圈通过附加 HOLD+LIMITED 代理一致性检查；冷态六圈完整强门槛为 {candidate["cold_strong_proxy_six_laps"]}。它是受控暖机后的有限稳态研究候选，不能称为完整冷态或真实FIA认证。</p>'
    block+=table(['裕量秒','六圈原子集','冷态全强门槛','暖态5/6强门槛','末两圈均时秒'],[[c['margin_s'],c['six_original_subset_passed'],c['cold_strong_proxy_six_laps'],c['steady_strong_proxy_laps5_6'],fmt(c['last_two_mean_time_s'])] for c in e['candidate_summaries']])
    block+=figure.to_html(full_html=False,include_plotlyjs=False,config={'displaylogo':False,'responsive':True})
    block+=table(['圈','时间秒','圈初 MJ','圈末 MJ','净 MJ','原周期子集','全period代理一致性'],[[i+1,fmt(s['formal_lap_time_s']),fmt(s['energy_start_mj'],9),fmt(s['energy_end_mj'],9),fmt(s['energy_end_mj']-s['energy_start_mj'],10),s['feasible'],candidate['fullperiod_proxy_gates'][i]] for i,s in enumerate(candidate['lap_summaries'])])
    block+=table(['评分数据','整体 RMSE km/h','S1 / S2 / S3 RMSE','最大圈时误差 %','最大分段误差秒'],[[name,fmt(v['aggregate']['mean_speed_rmse_kmh']),'/'.join(fmt(x) for x in v['aggregate']['sector_speed_rmse_kmh']),fmt(v['aggregate']['max_abs_lap_error_pct']),fmt(v['aggregate']['max_abs_sector_error_s'])] for name,v in [('校准3–6',cal),('历史开发7/9',dev)]])
    block+=table(['比较','相邻细化','圈时差秒','速度RMSE差 km/h','通过'],[[name,i+1,fmt(x['lap_time_difference_s']),fmt(x['speed_rmse_kmh']),x['passed']] for name,rows in numeric['comparisons'].items() for i,x in enumerate(rows)])
    energy=cal['metrics'][0]['sector_energy']
    block+=table(['R7段','部署 MJ','回收 MJ','净 MJ','账本残差 MJ'],[[x['sector'],fmt(x['deploy_mj']),fmt(x['harvest_mj']),fmt(x['net_energy_mj']),fmt(x['balance_residual_mj'],12)] for x in energy])
    block+='<p>新层只预先限制请求，没有截断已接受DC或重置状态。每个候选从2MJ冷态开始真正衔接六圈；速度残差不参与裕量选择，选定后历史开发数据评分一次。本轮没有增加SAC episode、模型参数或读取FP2。质量和ICE敏感性沿用R0–R6报告；真实损耗未知，理想4→4MJ并不证明真实SOC可持续。</p>'
    block+=f'<p>独立源码提交 {e["source_commit"]}；精确轨迹/动作/终态/额外代理/分段复验 {verify["passed"]}；正式放行 {e["gates"]["formal_release"]}。原精度门槛仍失败；完整FIA、真实SOC及新会话验收待完成。浏览器视觉验收未实测。</p>'
    page=(out/'report.html').read_text(encoding='utf-8')
    page=page.replace('<h1>V6 可持续充放电控制研究</h1>','<h1>V6 可持续充放电控制研究</h1><p class="warn">现已完成 R0–R7 八轮。新增稳态工程门槛候选见 <a href="#r7">R7</a>；下方70.6725秒为原声明子集基准，它未通过附加代理一致性门槛。</p>',1)
    page=page.rsplit('</html>',1)[0]+block+'</html>'
    (out/'report.html').write_text(page,encoding='utf-8')
    combined=read(out/'summary.json')
    combined['R7']={'source_commit':e['source_commit'],'selected_margin_s':selected,'steady_time_s':candidate['last_two_mean_time_s'],'calibration':cal['aggregate'],'development':dev['aggregate'],'gates':e['gates'],'exact_verification':verify['passed'],'new_SAC_episodes':0}
    (out/'summary.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# R7 追加验收结果','',f'完成 R0–R7 八轮。原声明子集基准70.6725秒未通过附加代理一致性门槛。几何前视保护修复原BC2027 raw失败；获选裕量{selected:g}秒，六圈末两圈均时 **{candidate["last_two_mean_time_s"]:.6f}秒**。','',f'原子集六圈通过：{e["gates"]["original_subset_six_laps"]}；暖态第5/6圈附加代理门槛：{e["gates"]["steady_strong_proxy_laps5_6"]}；冷态六圈完整强门槛：{e["gates"]["cold_strong_proxy_six_laps"]}；数值验收：{e["gates"]["numerical_convergence"]}；正式放行：False。','', '| 数据 | 整体 RMSE | S1 RMSE | S2 RMSE | S3 RMSE | 最大圈时误差% | 最大分段误差秒 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in [('校准3–6',cal),('历史开发7/9',dev)]:
        a=v['aggregate'];lines.append('| '+name+' | '+' | '.join(fmt(x) for x in [a['mean_speed_rmse_kmh'],*a['sector_speed_rmse_kmh'],a['max_abs_lap_error_pct'],a['max_abs_sector_error_s']])+' |')
    lines+=['','理想DC等效储能在暖态闭合，速度与完整状态真实跨圈携带。真实SOC损耗、完整FIA ECU验证、新会话与真实精度仍待完成。冷态代理不一致是额外工程门槛失败，不能解释为FIA违规或丢弃冷圈。','',f'独立源码提交 `{e["source_commit"]}`。`tools/run_r7.py --verify results/austria_2026_v6/R7` 精确复验已通过；原主实验与随机训练清单保持不变。新增SAC训练0集；全部180集结果仍是没有超过BC。','', '完整六圈表、分段能量收支、失败和数值对照见 report.html 与 R7/。获选范围仅限三个已冻结裕量的稳态研究族，未证明全局最优。浏览器视觉验收未实测。']
    md='\n'.join(lines)+'\n'
    (root/'RESULTS_V6_R7.md').write_text(md,encoding='utf-8');(folder/'results.md').write_text(md,encoding='utf-8')
    print(json.dumps(combined['R7'],ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
