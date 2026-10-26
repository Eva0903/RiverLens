# V4：02站点信息消融，随机窗口留出版本

本补丁用于已经安装随机版01的 `Jinjiang_LSTM_Upgrade_V4`。它直接替换02入口，不修改01、runner.py、models.py、原始数据或旧结果。此次交付的是可运行代码，尚未生成GPU LSTM正式结果。

## 本次要回答的问题

在完全相同的样本、18步历史和未来6步输出下，比较不同站点输入的模拟表现。主比较固定为 **C站多变量 → A+B+C三站多变量**，检验下游自身历史可用时，上游变量的额外价值。

| 组名 | 输入 | 当前数据维数 |
|---|---|---:|
| local_target | C站当前预测指标的历史 | 1 |
| local_multivariate | C站全部水质变量的历史 | 9 |
| station_A | A站变量的历史 | 8 |
| station_B | B站变量的历史 | 9 |
| upstream_AB | A+B站变量的历史 | 17 |
| all_stations | A+B+C站变量的历史 | 26 |

六组均使用普通LSTM，并增加Persistence、24 h Seasonal naive、全站Ridge三种参照。上游输入组的模型输入不含C站变量；两种naive对照仍使用C站目标历史，单独标明。各组隐藏层规模和训练预算相同，输入维数不同，因此参数量不完全相同，实际参数量会输出。

## 安装：先备份，再合并

先停止正在运行的02进程，在**当前项目根目录**执行：

```powershell
$backupDir = Join-Path $PWD ("backups\02_before_random_" + (Get-Date -Format yyyyMMdd_HHmmss))
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Copy-Item "experiments\02_station_ablation.py" $backupDir
```

解压本包，进入最外层 `Jinjiang_V4_02_Random_Station_Patch`，将里面的 `experiments`、`src`、`configs`、`tests` 四个文件夹复制到现有项目根目录，选择**合并文件夹**。仅 `experiments/02_station_ablation.py` 是对旧代码的替换，其余是新增文件；若此前已经安装过本补丁，也先备份同名文件再替换。

| 文件 | 操作 |
|---|---|
| experiments/02_station_ablation.py | 替换 |
| src/jinjiang/station_ablation_random.py | 新增 |
| configs/station_ablation_random_gpu.json | 新增，正式配置 |
| configs/station_ablation_random_smoke_gpu.json | 新增，2轮GPU连通检查 |
| tests/test_station_ablation_random.py | 新增 |

本补丁依赖随机版01已安装的 `src/jinjiang/baseline_random.py`、`baseline_within_split.py` 及其配套runner。不要只复制02入口；如提示这些模块不存在，先安装之前的01随机版补丁。本补丁不要求替换runner.py或workspace文件。

## 固定实验协议

- 原始序列4小时采样；**全部指标均固定18步历史（3天）**，直接输出未来6步（4—24小时）。02不会读取 `target_default_windows` 来改变窗口。
- 先在连续序列上构建全站共同有效窗口，再用两次 `train_test_split(random_state=42)` 按80%/10%/10%随机分配窗口，复用01的具体算法。
- 每个指标只构窗和划分一次。六组从同一个张量切列，目标、简单基线、样本时间、训练/验证/测试索引完全相同。C-only不能额外使用全站组因缺失值而放弃的日期。
- 当前上传数据四指标都是共同窗口3029个，训练2423、验证303、测试303。与上传01逐窗口清单对照，12116条指标-窗口记录的时间与集合归属完全一致。
- 沿用01的因果前向填补规则。目标评分只用原始观测；标准化参数只从训练窗口估计。
- 普通LSTM：2层、hidden_size=64、dropout=0.1、batch_size=32、学习率0.001、最多256轮、patience=15；按最低验证损失恢复权重。其余细节复用现有models.py，不使用06的残差模型。
- 种子42、2026、3407、7、99。正式共 **180个任务**：120次LSTM训练、20个Ridge任务、40个naive任务。确定性基线按种子重复记录仅为方便配对，不是五份独立证据。
- 三站LSTM和三种参照会在02目录重新计算，便于形成完整的02共同样本对照；不会把01的完成标记复制过来跳过任务。

## 检查、运行、续跑

在VS Code终端中确认当前位置为项目根目录：

```powershell
$gpuPython = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PWD "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# 1. 样本审计：不训练任何模型
& $gpuPython experiments/02_station_ablation.py --config configs/station_ablation_random_gpu.json --audit-only

# 2. GPU连通检查：DO、1种子，六组LSTM各训练2轮，独立目录
& $gpuPython experiments/02_station_ablation.py --config configs/station_ablation_random_smoke_gpu.json

# 3. 正式：四指标、五种子
& $gpuPython experiments/02_station_ablation.py --config configs/station_ablation_random_gpu.json
```

审计四指标都应打印：

```text
common_windows=3029, train=2423, validation=303, test=303
```

数量只针对本次原始数据；若不同，先检查数据是否变化。GPU检查应完成9个任务，其中6次LSTM。正式配置目录是 `results/station_ablation_random_gpu/02_station_ablation`；检查目录是 `results/station_ablation_random_smoke_gpu/02_station_ablation`。2轮结果仅用于检查流程。

按一次Ctrl+C可以中断。待终端提示符返回后关机；再次执行同一正式命令即可续跑。已完整写盘的任务显示 `skip:`；中断时尚未完成的任务从头训练。保持配置和目录不变。本入口不提供 `--overwrite` 或 `--model-override`，避免混用实验；也不要继续使用旧的 `--config configs/paper.json`。

源码、原始数据或配置变化时会拒绝混入已有任务；确需变更，使用新的results_dir。不要在同一结果目录启动两个进程。

可选本机测试：

```powershell
& $gpuPython -m unittest discover -s tests -p "test_station_ablation_random.py" -v
```

## 跑完先看什么

1. `run_status.json`：正式应为 `run_status=complete`、`completed_tasks=expected_tasks=180`、失败0个；`expected_lstm_tasks=120`。
2. `sample_support_audit.csv` 和 `input_groups.csv`：每组共同样本、索引哈希和实际输入列。
3. `metrics_mean_sd.csv` 和 `figures/station_ablation_R2.png`：四指标、六时效、各输入组的五种子表现。正常正式指标明细1080行，五种子汇总216行。
4. `upstream_gain_summary.csv`：主比较C多变量→三站多变量，按六时效等权平均的MAE降低；**正值表示三站输入误差更小，负值表示更大**，同时列出同向种子数和条件区间。
5. `paired_seed_horizon.csv`：逐种子、逐时效的MAE降低、降低百分比、RMSE降低和R²增量。
6. `paired_block_bootstrap.csv`：主比较和补充比较的逐时效及六时效均值。`horizon_hours=0`代表六时效平均，不是0小时预测。未齐全部配对种子时标记partial，不以不配对的种子比较。

下一次上传：`run_status.json`、`run_config.json`、`metrics_mean_sd.csv`、`paired_seed_horizon.csv`、`upstream_gain_summary.csv`、`paired_block_bootstrap.csv`、`sample_support_audit.csv`、`input_groups.csv`以及站点消融图。若训练异常，再补 `training_history.csv`。通常先不需要完整predictions。

## 如何解释

主比较支持的是：**在此固定随机窗口留出评价、全站共同可用样本和相同训练规则下，加入上游变量是否改善模拟表现**。A/B单站和A+B组用于展示空间信息组合的差异，不把不同输入维数的网络称为参数量完全相同。

随机窗口内部按时间排列，但不同集合会共享原始时间点，审计明确记录这一事实。五种子反映固定划分下的训练随机性，不是五次独立划分。区间按真实日历时间的24小时块重采样，先平均配对种子、保留无测试窗口的空白时段；区间仍以当前划分和已拟合模型为条件，不能修正随机划分的评价偏差，也不能替代独立未来时间验证。补充对照不做多重比较校正，不据单个最好时效宣称普遍改善。

最终结果允许“上游对部分指标有帮助，也可能对部分指标无增益或产生负收益”。后续03再独立研究历史窗口，不在02中同时按测试结果选择窗口。
