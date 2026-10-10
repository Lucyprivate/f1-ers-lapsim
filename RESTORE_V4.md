# 从完整历史包恢复 v4 制品

已发布的 `F1_ERS_Lab_v4.zip`、正式源码提交 `ea70476`、结果提交 `0ad8824` 和 `v4-0.4.0` 标签保持冻结。后续恢复工具提交不改变模型和结果；`F1_ERS_Lab_v4_with_restore.bundle` 包含该工具及全部此前历史。

在有 Git 和 Python 3.12 的离线机器上，先将 ZIP、历史包和交付清单放到同一目录，再执行：

```powershell
git -c core.autocrlf=false clone --branch improve/v4-dynamics ./F1_ERS_Lab_v4_with_restore.bundle ./f1-ers-lab
Set-Location ./f1-ers-lab
git config --local core.autocrlf false
git config --local user.name 'F1 ERS Lab'
git config --local user.email 'f1ers@local.invalid'
git remote remove origin
python tools/restore_v4.py --root . --archive ../F1_ERS_Lab_v4.zip
python -m venv .venv
.venv/Scripts/python.exe -m pip install --no-index --find-links vendor/wheels -r requirements-lock.txt
./run.ps1 -Mode test-v4
./run.ps1 -Mode verify-v4
```

恢复工具仅使用 Python 标准库，安装数值依赖前即可运行。它固定校验 ZIP 的 SHA256：

`e4c3d065593b2727e86d92e8a8106127092a628c5de546fca1b1b64bc5195d3a`

随后预检全部成员的路径、链接、大小、散列和 Git 源码字节。仅恢复忽略的 `vendor/wheels`、`results/austria_2026_v4` 和本交付的 FP2 未访问状态文件。源码只校验，不覆盖；已有同散列制品保持不动，已有不同制品拒绝。越界路径、Windows 路径别名、链接和晚圈数组均拒绝。所有预检通过后才写入新制品；第二次恢复应为零新增文件。

该入口不访问或重建仓库外的测试访问账本，不产生新测试机会。离线 `verify-v4` 复现校准及开发结果，只读取已有最终测试凭证；本轮新会话仍为“验收待完成”。原基线 v3 恢复方法见 `README_V4.md`。
