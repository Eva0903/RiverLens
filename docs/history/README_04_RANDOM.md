# 04 气象增量实验：随机窗口划分与固定训练预算

将本包的 experiments、src、configs、tests 合并到现有 Jinjiang_LSTM_Upgrade_V4 项目根目录。全部为新增文件，原 04_meteorology.py 和共享训练函数保持原样。本包依赖此前已安装并完成 03 的 baseline_random.py、baseline_within_split.py 及项目原有依赖。

## 固定设置

- 训练上限 512 epoch，patience 100，五个种子 42、2026、3407、7、99。
- 使用与 03 相同的两次 sklearn train_test_split 算法，80%/10%/10%，split_seed=42。窗口内的时间顺序不变。
- 固定使用 03 主实验 4–72 小时平均验证 MAE 最小的窗口：DO 84 步即14天；NH3N 78步即13天；CODMN 48步即8天；TP 66步即11天。这些是最佳验证窗口，不是2%容差内最短窗口。
- 本轮预测6步，即4–24小时。上述窗口从03迁移使用，不声称是6输出模型重新优化后的最优窗口。
- 四组输入为三站水质水文、加历史气象、加降雨滞后、加气象与降雨滞后。每个指标先按完整输入筛共同样本，再统一划分，其他组仅抽取相应列。所有输入均不晚于发布时刻；不使用未来实测气象。
- 缩放器仅拟合训练窗口，验证集用于早停。模型按验证MSE恢复最佳权重。四组统一固定预算，不追加训练预算敏感性实验。SHAP暂时关闭。
- 正式任务数120，其中80次LSTM训练和40个naive任务。512为上限，并不要求每次都训练512轮，也不保证所有训练充分收敛。

## 执行

在项目根目录打开PowerShell：

```powershell
$gpuPython = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PWD "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

& $gpuPython experiments/04_meteorology_random.py --config configs/meteorology_random_gpu.json --audit-only

& $gpuPython experiments/04_meteorology_random.py --config configs/meteorology_random_gpu.json
```

中断后使用相同正式命令续跑。已完成任务跳过，未完成任务重新训练。不要更改配置后复用旧结果目录，也不要加 --overwrite。本入口不支持 --overwrite。

若只想验证新增入口的GPU连接，可在正式运行前选择独立的两轮检查配置：

```powershell
& $gpuPython experiments/04_meteorology_random.py --config configs/meteorology_random_smoke_gpu.json
```

正式结果位于 `results/meteorology_random_gpu/04_meteorology`。两轮检查位于另一个目录，不参与正式统计。

## 当前原始数据审计参考

| 指标 | 共同窗口 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|---:|
| DO | 817 | 653 | 82 | 82 |
| NH3N | 877 | 701 | 88 | 88 |
| CODMN | 1285 | 1028 | 128 | 129 |
| TP | 1015 | 812 | 101 | 102 |

四组输入在每个指标内共用相同时间和集合归属。由于气象完整性筛选、窗口长度和输出长度发生变化，不要求与03有相同样本数或逐窗口集合归属。

## 结果验收

上传结果目录内CSV和JSON即可。重点文件：run_status.json、run_config.json、input_groups.csv、sample_manifest.csv、sample_support_audit.csv、metrics.csv、metrics_mean_sd.csv、training_history.csv、model_diagnostics.csv、paired_seed_horizon.csv、meteorology_gain_summary.csv。需要重新核算预测时再附predictions.csv。

model_diagnostics.csv记录触顶情况，作为预算限制描述，不自动启动追加训练。meteorology_gain_summary.csv给出相对水质基线的气象增益和种子方向。置信区间仅针对固定随机划分及已拟合模型，各项对照未做多重比较校正，应作为探索性结果解释。

随机窗口评价不提供时间独立的未来泛化证据。后续结论限定为固定预算、当前候选窗口和随机窗口留出条件下的气象增量价值。

## 本包验证

5项测试通过，包括共同样本筛选、随机划分复现、输入早于目标时间、按种子和时效配对、断点续跑及训练预算记录；四指标真实数据审计通过。另用真实DO数据、单种子Ridge验证完整运行流程。当前环境未执行GPU/LSTM训练。

```powershell
& $gpuPython -m unittest discover -s tests -p test_meteorology_random.py -v
```
