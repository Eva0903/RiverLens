# V4：03历史窗口 × 4—72小时预测时效

适用于已运行随机版01/02的 `Jinjiang_LSTM_Upgrade_V4`。本补丁替换03入口，增加独立模块、三份配置和测试。共享的 `models.py`、`data.py`、`runner.py` 保持当前版本；新增模块自行构造72小时共同样本，复用现有普通LSTM训练。尚未生成GPU LSTM正式结果。

## 两层实验

| 项目 | 主实验 main | 长窗口补充 long |
|---|---|---|
| 原始输入 | A/B/C三站26列历史变量 | 相同 |
| 目标 | C站DO、NH3N、CODMN、TP | 相同 |
| 输入步数 | 3、6、12、18、…、84 | 18、42、84、126、168 |
| 输入天数 | 0.5，以及1—14天，共15组 | 3、7、14、21、28天，共5组 |
| 共同历史支持 | 最长84步（14天） | 最长168步（28天） |
| 输出 | 一次直接输出18步，逐4小时覆盖4—72小时 | 相同 |
| 模型/种子 | 普通LSTM，42/2026/3407/7/99 | 相同 |
| LSTM训练数 | 300 | 100 |
| naive任务数 | 40 | 40 |
| 正式任务总数 | **340** | **140** |

两个层次分别建立共同池和划分，不能把main的14天分数与long的28天分数直接相减。long重新训练3、7、14天锚点，使14/21/28天能在同一批样本上比较。

本次原始数据四指标的审计结果一致：

| 层次 | 共同窗口 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|---:|
| main | 1487 | 1189 | 149 | 149 |
| long | 712 | 569 | 71 | 72 |

03与01/02的样本池、输出头长度和训练上限不同，因此不能直接将相同3天历史的分数视为同配置复现。样本减少是最长历史、缺失输入和18步完整观测标签的共同约束，不是截取某段更好模拟的数据。

## 安装：先备份，再合并

先停止旧03进程。在当前项目根目录运行：

```powershell
$backupDir = Join-Path $PWD ("backups\03_before_random_" + (Get-Date -Format yyyyMMdd_HHmmss))
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Copy-Item "experiments\03_window_horizon.py" $backupDir
```

解压本包，进入最外层 `Jinjiang_V4_03_Window_Horizon_Patch`，把里面的 `experiments`、`src`、`configs`、`tests` 合并到现有项目根目录，选择替换03入口。

| 文件 | 操作 |
|---|---|
| experiments/03_window_horizon.py | 替换 |
| src/jinjiang/window_horizon_random.py | 新增 |
| configs/window_horizon_random_main_gpu.json | 新增，主实验 |
| configs/window_horizon_random_long_gpu.json | 新增，长窗口补充 |
| configs/window_horizon_random_smoke_gpu.json | 新增，GPU检查 |
| tests/test_window_horizon_random.py | 新增 |

必须安装过01随机版的 `baseline_random.py`、`baseline_within_split.py` 及配套runner。不要只复制03入口。无需修改VS Code workspace文件。

## 实验协议

1. 每个指标/层次只构造一次最长历史支持池。要求三站全部输入历史有限、未来18步目标均为原始观测，并有发布时刻之前的最后6步目标历史。每个较短窗口从同一张量取末尾相应步数，不重新筛选日期。
2. 对共同窗口执行与01/02相同的两次 `train_test_split(random_state=42)`，比例80/10/10。层次内所有窗口、模型和种子共用索引，窗口内部保持时间顺序。
3. 输入短缺口仍按当前因果前向填补规则处理；评分标签不使用填补值。标准化参数只从训练窗口估计，验证集用于早停。
4. 普通LSTM保留2层、hidden_size64、dropout0.1、batch_size32、学习率0.001、标准化MSE损失；03统一提高为**最多512轮，patience30**。最终恢复验证损失最低的权重，不直接使用最后一轮。
5. 一次训练从同一个选定权重推断验证和测试，不为获取验证预测额外训练第二个模型。全部18个输出共同训练；不是为每个时效单独优化一个模型。
6. Persistence向18个时效重复最后一个可用目标值。Seasonal naive向前重复发布前最后24小时的6步模式：第7—12步重复第1—6步，第13—18步再重复一次，不使用未来真实观测。
7. **0.5天LSTM实际输入只有3步**；naive仍可使用发布前24小时历史作为强参照。较长共同支持只是共同选样条件，不会把额外历史送进短窗口LSTM。
8. 两种naive每层/指标/种子各计算一次，记录 `lookback_steps=0`（仅为共享基线标记，不是零历史模型）。计算各窗口skill时按相同指标、种子、时效匹配，不重复训练naive。确定性基线跨种子重复记录不代表独立证据。

## 在当前项目终端依次执行

```powershell
$gpuPython = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PWD "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# 1. 两层审计：不训练
& $gpuPython experiments/03_window_horizon.py --config configs/window_horizon_random_main_gpu.json --audit-only
& $gpuPython experiments/03_window_horizon.py --config configs/window_horizon_random_long_gpu.json --audit-only

# 2. GPU检查：DO、seed42，0.5/3/14天输入，18步输出，每组2轮
& $gpuPython experiments/03_window_horizon.py --config configs/window_horizon_random_smoke_gpu.json

# 3. 正式主实验
& $gpuPython experiments/03_window_horizon.py --config configs/window_horizon_random_main_gpu.json

# 4. 正式长窗口补充
& $gpuPython experiments/03_window_horizon.py --config configs/window_horizon_random_long_gpu.json
```

GPU检查完成5个任务，其中3次LSTM，2个naive任务；检查结果不能作为性能结论。可先跑完main并核对完整性，再运行long。无需依据main测试表现重新选择long的输入、窗口或训练规则。

输出：

- 主实验：`results/window_horizon_random_main_gpu/03_window_horizon`
- 长窗口：`results/window_horizon_random_long_gpu/03_window_horizon`
- GPU检查：`results/window_horizon_random_smoke_gpu/03_window_horizon`

本入口不接受 `--overwrite`、`--model-override`，不使用旧的 `paper.json`。保持配置、数据和源码不变即可断点续跑；Ctrl+C后等终端提示符返回，再关机。已完整任务自动skip，未完成任务从头训练。验证指标写入不完整的任务也会重新运行。改变源码、原始数据或配置时，须使用新的results_dir，不能混入旧任务。

可选本机测试：

```powershell
& $gpuPython -m unittest discover -s tests -p "test_window_horizon_random.py" -v
```

## 窗口选择与“趋稳”的判读

`validation_window_selection.csv` 在**某指标全部窗口和种子完成后**才发布该指标的选窗结果；未齐时在 `window_selection_status.csv` 标记pending。不会根据已完成的一小部分模型临时报告最佳窗口。

每个指标分别报告：

- `horizon_hours=0`：18个时效等权平均验证MAE，用于选择一个整体窗口。
- `horizon_hours=4…72`：该时效的平均验证MAE，用于展示时效之间的窗口需求差异。
- `best_lookback_days`：平均验证MAE最小的窗口；并列取较短者。
- `shortest_near_optimal_days`：验证MAE不超过最小值1.02倍的最短窗口。
- `near_optimal_windows_steps`：全部达到上述容差的候选，不假定它们一定连续。

2%是预设的实用容差，不是显著性检验或置信区间。种子波动另在验证汇总中报告。若28天仍更好，应报告“尚未观察到平台”，而非强行称为收敛。窗口曲线与训练损失收敛是两个不同概念。

`test_metrics_for_validation_selected_windows.csv` 只描述验证集已经选出的窗口，不在测试集重新排名。整体结果中的R2是18个时效R2的算术平均；逐时效RMSE先计算再平均，均非把全部预测混合计算。

## 完成检查和主要输出

| 文件 | 用途 |
|---|---|
| run_status.json | main应340任务、300次LSTM；long应140任务、100次LSTM；失败0个 |
| sample_manifest.csv | 最长历史共同支持清单，input_start_time为支持池起点 |
| sample_support_audit.csv | 各集合数量、原始时间重叠、支持时间哈希 |
| window_input_audit.csv | 各实际窗口的起始时间范围、列数与共同索引哈希 |
| metrics.csv / metrics_mean_sd.csv | 测试指标明细及种子均值/标准差 |
| validation_metrics.csv / validation_metrics_mean_sd.csv | 验证指标，供选窗使用 |
| window_horizon_full_grid.csv | 测试性能面及相对最佳naive的MAE skill |
| skill_scores.csv | 每窗口、种子、时效的基线技能 |
| validation_window_summary.csv | 验证误差、种子波动、相对最小误差及2%容差标记 |
| validation_window_selection.csv | 验证最优窗口及较短近优窗口 |
| test_metrics_for_validation_selected_windows.csv | 验证选定窗口的测试表现 |
| model_diagnostics.csv | 最佳epoch、实际训练轮数、是否触及上限及最佳epoch是否在上限最后5% |
| training_history.csv | 逐epoch损失，用于检查训练是否充分 |
| figures/ | 每指标R2和MAE skill的窗口×时效热力图，共8张；负R2不裁掉 |

完整main应有6120行测试指标、6120行验证指标和1224行五种子汇总；long分别为2520、2520、504行。naive仅一份，所以不再按每窗口重复计算总行数。

如果一批长窗口经常在512轮附近取得最佳验证损失，需要另做训练预算敏感性检查；不能直接把较差分数解释成长期历史无用。诊断只作标记，不会暗中改变不同窗口的训练预算。

下一次上传每层的 `run_status.json`、`metrics_mean_sd.csv`、`window_horizon_full_grid.csv`、`validation_window_summary.csv`、`validation_window_selection.csv`、`test_metrics_for_validation_selected_windows.csv`、`model_diagnostics.csv`、`sample_support_audit.csv` 和R2热力图。通常先不需要完整predictions；该文件为复算保留，体积可能较大。

## 评价范围

这是固定随机窗口留出实验。五种子反映固定划分下的训练随机性；相邻窗口可能共享输入及标签原始时间点，较长输出块也会增加标签重叠。性能面、近优窗口和远时效分数均不能独立证明物理记忆长度、污染物旅行时间或独立未来预测边界。main/long为两批共同支持上的敏感性分析，不是独立外部验证。

## 本补丁验证范围

交付环境没有PyTorch，GPU LSTM训练未执行。真实数据两层审计、DO的两层Ridge/naive CPU流程、72小时基线因果性、同权重验证/测试预测拆分、训练集缩放、验证选窗、2%较短窗口、缺失标签处理、Ctrl+C和断点续跑均已检查。受控替身用于验证神经网络接口和续跑，不能视为真实LSTM性能结果；本包不包含模拟GPU结果。
