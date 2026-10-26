# 01 Baseline：随机窗口留出评价

这是基于当前V4项目新增的一套独立评价入口。保留已经完成的时间顺序01和06结果，不替换现有01入口或runner。

## 复制文件

将补丁中的下列文件，按相对路径复制到你的 `Jinjiang_LSTM_Upgrade_V4` 项目：

| 路径 | 操作 |
|---|---|
| `experiments/01_baseline_random.py` | 新增，随机版01入口 |
| `src/jinjiang/baseline_random.py` | 新增，构窗、随机划分、审计及统计 |
| `configs/baseline_random_gpu.json` | 新增，八方法正式配置 |
| `configs/baseline_random_smoke_gpu.json` | 新增，DO两轮GPU检查 |
| `tests/test_baseline_random.py` | 新增，验证划分、共同样本与续跑 |

不需要覆盖已有文件。补丁基于你已成功运行严格内部滑窗01的项目，复用其中现有的模型训练函数和MASE函数。
`reference_cpu/` 仅为当前环境的参考结果；不要将它复制进正式结果目录用于跳过训练。

## 固定设计

1. 先在完整4小时序列上构建滑动窗口，每个窗口内时间保持正序，步长1；再对有效窗口随机划分。
2. 恢复上传旧Python脚本的两次 `sklearn.model_selection.train_test_split`：先取80%训练，剩余20%再各分一半为验证和测试。两次 `random_state=42`。
3. 所有模型共用同一有效窗口池、相同随机索引和五个训练种子。数据划分只随机一次，不按模型或训练种子重新抽取。
4. 历史窗口仍为当前01的18步（3天），直接输出未来6步（4/8/12/16/20/24小时）；没有改成旧脚本的42步。
5. 输入仍为A/B/C三站26列，不含气象；不是旧脚本17列输入的完整复现，也不是纯上游预测。
6. 复用V4短缺口 `causal_ffill` 规则，在连续时间序列上处理特征，再筛选窗口。目标保持原始观测，不对缺失目标插补后评分。此版本不存在预先定义的原始时间训练/验证/测试段。
7. 特征和目标StandardScaler只从随机训练窗口估计；验证集只用于神经网络早停；测试不参与选epoch。
8. 普通LSTM仍为2层hidden64、dropout0.1、batch32、Adam lr0.001、MSE、max_epochs256、patience15；其余模型结构与原01相同。没有新增超参数或架构搜索。
9. 模型：Persistence、24h Seasonal naive、Ridge、ANN、CNN、RNN、LSTM、Transformer。种子42/2026/3407/7/99。

四指标×八方法×五种子＝160任务，其中100次神经网络训练。两种naive和Ridge是确定性方法，重复种子的结果相同，不代表五次独立数据划分。

## GPU运行

项目根目录PowerShell：

```powershell
$gpuPython = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PWD "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

先检查共同样本和划分，不训练：

```powershell
& $gpuPython experiments/01_baseline_random.py --config configs/baseline_random_gpu.json --audit-only
```

若原始数据与本次上传一致，四指标都应显示：

```text
common_windows=3029, train=2423, validation=303, test=303
```

推荐先做两轮GPU连通检查：

```powershell
& $gpuPython experiments/01_baseline_random.py --config configs/baseline_random_smoke_gpu.json
```

此配置只包含DO、seed42、两种naive及一个2轮LSTM，结果保存在 `results/baseline_random_smoke_gpu/01_baseline`，不作为论文性能结果。

正式运行全部八方法：

```powershell
& $gpuPython experiments/01_baseline_random.py --config configs/baseline_random_gpu.json
```

正式输出：`results/baseline_random_gpu/01_baseline`。完整运行不要加 `--model-override lstm`。
中断后重复同一条命令即可续跑。不要并行启动两个写入同一目录的进程。新入口没有覆盖旧结果的开关。
如果原始Excel内容发生改变，新入口会拒绝继续混用旧任务，应换新结果目录。
更改配置后，也必须使用新结果目录。

## 验收

- `run_status.json`：正式应为complete、completed_tasks160、expected_tasks160、失败0；检查selected_models是否齐全。
- `metrics.csv`：960行；`metrics_mean_sd.csv`：192行。按指标、六个时效报告RMSE、MAE、R²及五种子均值和标准差。
- `sample_manifest.csv`：每个窗口的原始起止时间及随机集合归属。303个测试窗口在所有方法间相同。
- `sample_support_audit.csv`：窗口编号互斥、真实目标时间重叠率、输入列及集合签名；不伪称时间独立。
- `training_history.csv`、`model_ranking.csv`、`skill_scores.csv`、`predictions.csv`和基准图均保留。
- `paired_block_bootstrap.csv`：固定随机划分、固定已拟合模型条件下的配对日历时间块区间。按实际24小时块重采样，保留没有测试样本的时间块；它不是对独立未来时期泛化的验证，也未做块长敏感性和多重比较校正。
- `skill_scores.csv`中best_naive是测试期较优naive的描述性参照，不是部署时可从测试集挑选的基线。

## 评价范围

这套设计评价的是 **random-window holdout performance**。它保留窗口内的时间顺序，但训练、验证、测试的原始时间片段会交错和重叠。
本次数据中，测试目标覆盖1417个不同时间点，其中1403个也被训练窗口用作目标，即约99.01%。窗口ID互不重复不意味着原始目标时点独立。
海报可以报告这套协议下的模型和窗口差异，同时应明确随机窗口划分；不能将其称为时间独立的未来预测可靠性检验。已完成的时间顺序结果单独保留，不拼接两套分数。

可复用英文方法说明：

Samples were constructed using chronologically ordered 18-step input windows and six-step output windows. The resulting valid windows were randomly allocated to training (80%), validation (10%), and test (10%) sets using a fixed split seed. All benchmark models shared identical window assignments, and scaling parameters were estimated from training samples only. This evaluation measures performance under a random-window holdout protocol; temporal overlap between partitions was documented, and chronological evaluation results were retained separately.

## 当前已计算的参考值

当前环境没有PyTorch，未运行ANN/CNN/RNN/LSTM/Transformer，也未在你的电脑上自动训练。
仅在CPU上真实计算了四指标的Persistence、Seasonal naive、Ridge，共60个输出任务，360行指标。
`reference_cpu/`提供原始指标和汇总，均值表可复算；这些不是完整八模型排名。
六时效R²算术均值如下，不是合并预测后重算的R²：

| 指标 | Persistence | Seasonal naive | Ridge |
|---|---:|---:|---:|
| DO | 0.62665 | 0.84884 | 0.86012 |
| NH₃-N | 0.29897 | -0.13205 | 0.34781 |
| CODMn | 0.48867 | 0.37913 | 0.43221 |
| TP | 0.56392 | 0.40541 | 0.59469 |

原有29项测试加新增7项测试，共36项通过；最终新增测试又复验通过。新增测试覆盖旧sklearn划分复现、窗口内顺序、共同样本、真实时间重叠审计、训练集标准化、目标观测缺失处理、日历块区间和部分运行/续跑/原始数据变更保护。
神经网络调度测试采用mock，不表示GPU训练已执行。
