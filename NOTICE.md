# 第三方资料与许可范围

自有实现与自写文档按根 MIT LICENSE 提供。此许可不覆盖 FIA 文档、F1 标识、第三方原始数据或基于这些数据的派生资料。

- FIA：完整规则 PDF、官方赛道图及渲染页面未进入公开快照；来源与版本见 SOURCES.md，散列见 docs/PUBLICATION_SOURCE.json。适用[官方网站使用条款](https://www.fia.com/file/83930/download)，不授予全文再分发权。
- OpenF1：来源为 [openf1.org](https://openf1.org/)，仅用于教育、个人学习、研究和非商业分析。官网声明 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，原数据权利由相应权利人保留。公开快照不分发原始/派生遥测数组；有关归一化、筛选、插值和分析修改见 SOURCES.md 及各阶段说明。相关数据派生成果不能被根源码许可改为商业数据授权。
- TUMFTM racetrack-database：固定版本 e59595d1f3573b30d1ded6a08984935b957688e0，LGPL-3.0。原 CSV 与派生几何缓存未进入公开快照；历史 LGPL 文本保留在 third_party/LICENSE_TUM.txt。若自行取得并再分发其资料，应同时保留相应 GPLv3/LGPLv3 文本及来源说明。
- NumPy、SciPy、Numba、llvmlite、pdfplumber、Plotly 等依赖沿用各自许可；本仓库不分发其 wheel。

项目不是 F1、FIA 或车队的官方产品。实测速度/圈时与模拟速度/SOC/功率的区分及模型未验证项见 SOURCES.md 与 v7 摘要。
