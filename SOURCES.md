# 数据来源与审计范围

本项目使用公开数据构建可运行、可验证的二维原型。真实圈时、分段时间和遥测直接保留来源；ERS-K 功率、回收、SOC 以及仿真 G 值是模型输出，不是车队公开的实际电能数据。

## 2026 奥地利正式比赛

- [OpenF1 API](https://openf1.org/)：session_key `11315`，meeting_key `1288`，2026-06-28 13:00 UTC 的 Spielberg Race；完整 session、drivers、laps、race_control、weather、pit、stints 均保存在 `data/raw/openf1/`。
- 车手 George Russell，车号 `63`。第 3、4、5、6 圈用于校准，第 7、9 圈用于独立整圈验证，按时间划分，不把同圈采样点随机拆分。第 8 圈缺少第一分段时间，未补造，因此排除。
- 这些圈均为 MEDIUM 干胎、无降雨，排除了发车/暖胎、进出维修区、黄旗/红旗/SC/VSC 区间、删除圈和明显圈时异常。仅此清理不能证明没有交通影响、驾驶差异、Overtake 或车队策略变化。
- `car_data` 和 `location` 原始时间窗覆盖第 3 至 9 圈。下载节流为至少 2.2 秒/请求，低于免费 30 请求/分钟。后续默认从磁盘读取。
- 速度原始精度为整数 km/h，采样约 3.7 Hz。空间和时间插值只用于对齐，不增加测量精度。throttle/brake 中出现 `104` 异常码；原始数据保留，处理值按有效邻近采样插值，brake 仅作布尔状态，修复数量写入审计。
- OpenF1 是非官方社区数据，不是 FIA 车队机密遥测。其站点声明非商业教育/研究使用及 CC BY-NC-SA 4.0；F1 标识和原始数据权利由相应权利人保留。见 [OpenF1 使用说明](https://openf1.org/#access-support)。

## 赛道几何

[TUMFTM racetrack-database](https://github.com/TUMFTM/racetrack-database)，固定提交 `e59595d1f3573b30d1ded6a08984935b957688e0`。中心线来自平滑的 OpenStreetMap GPS 点，宽度来自卫星图像，固定路线由**最小曲率**优化生成，不能称为最短圈时路线。

- 原中心线长度约 4315.45 m，固定最小曲率路线约 4284.75 m；CSV 的六位小数是存储格式，不代表微米级实测精度。
- 来源自身提示几何精度因地区差异而不同；此历史地图未经 2026 FIA 测绘核验。宽度是卫星估计值，不能用来证明精确压线或真实赛车轨迹。
- 采用前四个校准圈的位置和积分速度拟合二维旋转、尺度与闭环相位，把路线 `s=0` 对齐控制线。位置数据只用于粗略配准及进度，不作为高精度 GPS 或实际赛车线。配准参数、误差和长度差异保存在 `data/manifest.json`。
- 等距闭环约 10 m 间隔，末端不重复；曲率由周期五点差分计算。二维仿真忽略奥地利赛道高程和坡度。
- 原始和派生 TUM 数据按仓库 LGPL-3.0 许可保存；完整文本在 `third_party/LICENSE_TUM.txt`。源 CSV 及 README 保留在 `data/raw/tum/`。

## 赛事当时有效的 FIA 规则

采用 2026-06-28 赛事所适用的历史版本，不以 10 月最新版本替换：

1. [技术规则 Section C，Issue 19，2026-06-25](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_19_-_2026-06-25.pdf)：C5.2、C5.12。已人工渲染校对第 65、66 页，保留完整 PDF。
2. [体育规则 Section B，Issue 07，2026-06-25](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_07_-_2026-06-25.pdf)：B7.1、B7.2。
3. [奥地利 Doc 5：Power Unit Information，2026-06-25](https://www.fia.com/system/files/decision-document/2026_austrian_grand_prix_-_power_unit_information.pdf)：已人工渲染核验第二页的赛事特定能量上限、速度功率曲线与区段。
4. [奥地利 Doc 7：Circuit Map 等文件](https://www.fia.com/system/files/decision-document/2026_austrian_grand_prix_-_competition_notes_-_circuit_map_pit_lane_drawing_emergency_exits_map_and_red_zone.pdf)：赛道图 Version 3，2026-06-22；已人工渲染核验第二页。
5. [奥地利 Doc 6：Competition Notes](https://www.fia.com/system/files/decision-document/2026_austrian_grand_prix_-_race_directors_competition_notes.pdf)：确认计时/完赛使用控制线等赛事说明。

关闭 Overtake 的奥地利 Race 每圈回收上限为 **8.0 MJ**，不是通用 8.5 MJ。ERS-K 绝对功率上限 350 kW 指 **DC 电功率**，不能直接冒充轮端机械功率。4 MJ 是 ES 允许使用的充电状态窗口，不是公开的真实电池总容量。赛车 ICE 最大机械功率和效率属于模型假设；法规给出燃料能量流限制。

Doc 5 写中心线 `4.236 km`，Doc 7 写 `4.326 km`，后者的三个分段长度之和也为 `4.326 km`。项目采用 Doc 7 作 FIA 距离参考并保留这个官方文件冲突。FIA 区段距离通过 `s_fia = s_route × 4326 / route_length` 显式映射到历史固定路线，这是近似换算，不是已测绘的计时环坐标。

## 主动空气动力与规则未知项

赛道图明确公布 Normal Grip 直道模式激活点：T10 后 110 m、T1 后 110 m、T3 后 90 m、T8 后 10 m；Low Grip 对应 160、170、160、60 m。几何中的弯道位置和退出点依据历史路线及仅校准圈的制动状态估计。官方精确退出坐标和真实驾驶员翼面状态未取得；2026 OpenF1 的 `drs` 值为 null，不能把它当成 2026 主动翼面真值。

FIA-F1-DOC-058 的 Standard ECU pending/limited/Boost 状态机、DOC-034 车辆基础参数、DOC-096 车队效率映射和 DOC-111 低抓地曲线未公开取得。公开功率下降规则包含低速、换挡、负驾驶需求和速度上限等例外；模型的近似状态机必须单独声明。项目不能声称完整 FIA 合规或真实车队 ERS 算法复现。

## 可复现清单

`data/manifest.json` 为每个必需原始文件记录实际 URL、下载时间、字节数、SHA256，另记录处理数据哈希、六个圈的清理和覆盖指标、控制线配准误差。`config/rules_austria_2026.json` 记录每个规则值的版本、日期、来源及未验证项。原始与处理数据都随本地项目保存；`prepare_data(..., refresh=False)` 可离线重建，网络不可用或数据门槛不达标时抛出明确错误，不替换为合成真实圈。
