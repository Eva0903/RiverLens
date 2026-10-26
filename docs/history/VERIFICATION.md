# v4 验证记录

本项目已在随包提供的四份原始 Excel 上完成自动验证。

## 单元测试

新增测试覆盖：验证集基线选择不读取测试目标、残差门控可以精确回退朴素预测，以及所有 LSTM 候选窗口/变量组使用完全相同的样本时间支持。

交付前已在随包原始数据上运行全部 13 项 `unittest`，结果为 `OK`；同时完成 `compileall` 语法检查，并确认 validation-only 与正式配置的选择协议哈希一致。

运行命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 端到端烟雾测试

使用 DO、单种子和 Ridge 的 `configs/smoke.json`，五个入口全部完成，覆盖时间顺序与随机诊断、站点消融、共同支持集窗口比较、四种气象组合、两折折内预处理 rolling-origin 和三目标站轮换。

验证输出包含指标、技能分数、预测、漂移诊断、bootstrap、环境清单、执行状态和图件。100 次烟雾 bootstrap 的最小双侧 p 值为 `2/101 ≈ 0.0198`，不会出现伪精确的零值。

## GPU 验证边界

当前交付容器没有安装 PyTorch，因此神经网络代码在这里完成了语法检查和非 Torch 协议检查，但没有宣称完成 LSTM/CUDA 运行验证。用户的 NVIDIA/CUDA 组合必须先运行 `configs/lstm_upgrade_smoke.json`；该检查仅三 epoch，不代表论文性能。v4 在进程启动前设置 cuBLAS workspace并启用严格确定性；若本机环境没有 CUDA 版 Torch，`device=auto` 会回退 CPU。

## 正式结果边界

烟雾测试只证明流程可运行，不是论文结果。V3 已完成的 01 可作为历史基准保留；V4 的 06 会写入独立结果目录。旧的 `metrics.csv`、`training_history.csv` 和完成记录不能复制到 V4 的 06。
