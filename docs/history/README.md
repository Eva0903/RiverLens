# 锦江水质预测实验框架 v4：验证集选择的残差 LSTM

这是可直接在 VS Code 中运行的完整升级项目。v4 保留 v3 的因果预处理、时间顺序拆分和强朴素基线，同时将论文主模型明确升级为 LSTM：它学习相对 Persistence/24 h Seasonal naive 的残差，并用验证集门控决定每个提前量应采用多少 LSTM 修正。

## 先不要覆盖 v2

建议把本压缩包解压成新目录 `Jinjiang_LSTM_Upgrade_V4`，保留 V3 项目和已经完成的 `results/paper/01_baseline` 作为历史记录。V4 的 06 是独立实验，不需要重跑或删除 V3 的 01，也不能把旧的完成记录复制进 06。

## Windows / VS Code 快速开始

1. 打开 `Jinjiang_LSTM_Upgrade_V4.code-workspace`。
2. 推荐直接复用已经能调用 CUDA 的 V3 虚拟环境，避免再次下载 PyTorch/CUDA。在 VS Code 的“Python: Select Interpreter”中选择：

```text
C:\path\to\previous-project\.venv\Scripts\python.exe
```

也可以在 PowerShell 记录该解释器：

```powershell
$v3Python = "C:\path\to\previous-project\.venv\Scripts\python.exe"
& $v3Python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

只有确实希望创建全新隔离环境时，才运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

3. 先运行单元测试和 Ridge 烟雾测试。若复用 V3 环境，将下方 `python` 换成 `& $v3Python`：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe experiments\run_all.py --config configs\smoke.json --model-override ridge --overwrite
```

5. 若有 NVIDIA GPU，再运行两 epoch CUDA/LSTM 检查：

```powershell
.\.venv\Scripts\python.exe experiments\02_station_ablation.py --config configs\smoke_cuda.json --model-override lstm --max-tasks 3 --overwrite
```

`--max-tasks 3` 会依次完成 Persistence、24 h Seasonal naive 和第一个 LSTM，真正检查 CUDA 反向传播。若机器没有 CUDA，请继续使用 CPU 或把配置中的 `device` 改为 `auto`。

6. 再运行 v4 残差 LSTM 的短烟雾测试：

```powershell
python experiments\06_lstm_upgrade.py --config configs\lstm_upgrade_smoke.json --overwrite
```

正常结束时会显示 `Finished: ...\results\smoke_lstm_v4\06_lstm_upgrade`。

## 推荐的 LSTM 主线

先只用训练集和验证集完成 64 个候选方案的筛选，不读取测试目标：

```powershell
python experiments\06_lstm_upgrade.py --config configs\lstm_upgrade_select.json --overwrite
```

检查 `run_status.json` 中 `test_set_used` 必须为 `false`。候选输入包括本地目标、三站同指标、过程聚焦变量和全站变量；不同目标使用预先给定的窗口网格。候选采用相同时间样本，按六个提前量的平均验证集 MAE skill 选择。

确认流程正常后，正式执行五种子版本：

```powershell
python experiments\06_lstm_upgrade.py --config configs\lstm_upgrade_paper.json --overwrite
```

正式配置会校验并自动复用上一步的 validation-only 候选缓存（若缓存不存在才自行重跑筛选），再将所选 LSTM 在训练集+验证集上重训，最后只评估一次测试段。请不要根据正式测试结果修改候选窗口、变量、门控或超参数；否则需要声明为探索性分析或准备新的外部测试集。

## 正式实验

按以下顺序运行；第一次启动某个实验时可以用 `--overwrite` 清除该实验目录，续跑时不要加：

```powershell
.\.venv\Scripts\python.exe experiments\01_baseline.py --config configs\paper.json --overwrite
.\.venv\Scripts\python.exe experiments\02_station_ablation.py --config configs\paper.json
.\.venv\Scripts\python.exe experiments\03_window_horizon.py --config configs\paper.json
.\.venv\Scripts\python.exe experiments\04_meteorology.py --config configs\paper.json
.\.venv\Scripts\python.exe experiments\05_robustness.py --config configs\paper.json
```

建议先完成并审查 01，再决定是否继续训练大量 LSTM。按 `Ctrl+C` 中止是安全的；重新执行同一条命令会依据 `completed_tasks.txt` 跳过完整任务。若在写文件的瞬间中断，启动时会按 `task_id` 清理该任务的残行并重新运行。不要在续跑时使用 `--overwrite`。

## v3 的关键升级

- 主分析采用 `causal_ffill`：验证和测试输入只使用当时及更早的观测，不利用未来值插补过去。
- 同时报告 Persistence 与 24 h Seasonal naive；模型必须超过最佳朴素基线才称为有技能。
- 新增 MASE、测试集标准差归一化 RMSE、相对两种朴素基线的 skill score。
- 所有窗口比较固定为 84 步共同时间支持集，再裁剪到各自窗口，确保公平。
- rolling-origin 每折按该折边界重新处理缺失值和拟合缩放器。
- 输出训练/验证/测试分布漂移表，负 R² 不再从图中裁掉。
- moving-block bootstrap 使用有限重复的 add-one 校正，避免报告 `p=0`。
- CUDA 严格确定性配置，Transformer 强制数学 attention；环境清单记录软件和设备。
- 每个任务使用配置指纹和唯一 `task_id`，支持崩溃后修复式续跑并记录失败任务；跨进程锁会阻止两个终端同时写同一实验。

## v4 的 LSTM 能力升级

- 预测“观测值减去验证集所选朴素预测”的残差，降低非平稳水质水平变化带来的训练难度。
- 每个提前量独立选择 Persistence 或 24 h Seasonal naive，并在验证集上确定 0–1 的残差门控；门控为 0 时严格回退到朴素基线。
- 使用小容量单层 LSTM、Huber loss、AdamW、学习率衰减与梯度裁剪，降低异常值和过拟合影响。
- 在原始变量之外追加一阶差分、日周期和年周期特征，全部只由预测发布时刻及其历史计算。
- 候选比较使用最长窗口与变量并集生成的共同样本；选型锁定后只按入选变量重建最终样本，避免无关缺失值浪费训练数据。
- 早停只用于选择 epoch；最终模型用训练集+验证集重训固定 epoch 数，提高有效训练样本量。

## 输出目录

01–05 结果位于 `results/paper/<experiment>/`；06 的筛选和正式结果分别位于 `results/lstm_upgrade_selection/06_lstm_upgrade/` 与 `results/lstm_upgrade_paper/06_lstm_upgrade/`。核心文件包括：

- `metrics.csv`：逐任务、逐提前量的 R²、RMSE、MAE、sMAPE、MAPE、MASE、nRMSE；
- `metrics_mean_sd.csv`：单层列名的均值与标准差；
- `skill_scores.csv`：相对 Persistence、Seasonal naive 与二者最佳值的 MAE skill；
- `predictions.csv`：真实时间、真值、预测和误差；
- `distribution_shift.csv`：训练/验证/测试的均值、标准差、范围和缺失诊断；
- `paired_block_bootstrap.csv`：成对移动块 bootstrap；
- `environment.json`、`run_config.json`、`run_status.json`：可复现性与执行状态；
- `completed_tasks.txt`、`failed_tasks.json`、`errors.log`：续跑与错误诊断；
- `figures/`：自动检查图。

01 另有 `model_ranking.csv` 和 `split_optimism_summary.csv`；03 另有 `forecastability_full_grid.csv` 与 `forecastability_best_windows.csv`。

06 另有 `candidate_validation_metrics.csv`、`candidate_validation_summary.csv`、`selected_configs.json` 和 `model_diagnostics.csv`。正式 LSTM 仍需同时报告与两种朴素基线的 skill 和成对块 bootstrap；门控可以降低风险，但不保证每个独立测试时期都优于朴素法。

## 结果解释底线

- 正式结论只用时间顺序拆分和 rolling-origin；随机窗口拆分仅量化乐观偏差。
- `R² < 0` 表示模型不如测试集均值预测，不应删除或截断。
- “可预测”要求同时满足 `R² > 0` 且 `MAE skill vs best naive > 0`；阈值可在配置中提高。
- 站点轮换仅是同一网络内验证，不代表跨流域外部泛化。
- SHAP 描述模型依赖，不证明因果作用。

更完整的方法定义见 `EXPERIMENT_DESIGN.md`，从 V3 迁移见 `UPGRADE_FROM_V3.md`，验证记录见 `VERIFICATION.md`。
