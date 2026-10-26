# 新版06：同骨干残差修正与验证集门控

## 实验要回答的问题

03固定了候选集内按验证表现选择的窗口，04检验了气象输入的增量，05检验了这一增量随目标站点变化的重复性。本轮在C站、相同完整输入和共同样本下，检验残差学习能否超过普通LSTM，以及门控后的修正是否超过朴素预测。

**只新增20次LSTM训练。** 四指标乘五种子；每次训练同时产生未门控与门控两套预测，不重复训练。普通LSTM复用04已完成的20次完整输入任务；三种朴素预测均无需训练。

旧 `06_lstm_upgrade.py` 使用了不同的时间划分、窗口候选、差分/日历特征、骨干与优化流程，还包括重新拟合。其结果不能作为本轮普通LSTM的直接配对对照。本包使用独立入口与目录，保留旧实验和结果。本轮是残差与门控的控制比较，不复现旧版完整升级包。

## 固定设计

| 项目 | 本轮设置 |
|---|---|
| 目标站点 | C，主研究站点；本轮不再扩展A/B训练 |
| 指标 | DO、NH3N、CODMN、TP，四项均保留 |
| 输入 | `all_stations_weather_rain_lags`，当前三站历史水质水文＋气象＋降雨历史，共39列 |
| 窗口 | DO84步/14天；NH3N78步/13天；CODMN48步/8天；TP66步/11天 |
| 预测 | 六步，4、8、12、16、20、24小时 |
| 样本与划分 | 完整输入筛选同一批共同样本；沿用04的两次sklearn随机划分，80/10/10，split_seed=42 |
| 骨干 | 与04相同的64维、2层LSTM及线性输出头；dropout0.1；当前神经网络参数数均为60,550 |
| 优化 | 同一共享 `fit_torch_predict` 函数，Adam、标准化MSE、batch32、学习率0.001、CUDA严格确定性 |
| 预算 | 每次最多512轮，patience100；种子42、2026、3407、7、99 |
| 缩放与选择 | 缩放器仅拟合训练集；验证集早停并恢复最佳权重；朴素基线与门控仅用验证集选择 |
| 固定排除项 | 不搜索新窗口或架构，不添加差分/日历通道，不进行train+validation重训，不追加预算检查，不运行SHAP或已暂停的降雨时滞分析 |

窗口直接迁移自03主实验18时效平均验证MAE的选择结果，不声称为残差模型重新优化的最优窗口。输入只使用发布时刻之前的观测，不使用未来实测气象。

## 对照组和计算过程

| 输出名称 | 作用 | 新增神经训练 |
|---|---|---:|
| `ordinary_lstm` | 直接预测水质的普通LSTM，从04完整输入组复用 | 0 |
| `persistence` | 将发布时刻的目标值持续到各时效 | 0 |
| `seasonal_naive` | 使用目标时刻24小时前已观测的同一指标 | 0 |
| `validation_selected_naive` | 每个时效按验证MAE在上述两种朴素预测中选择，随后锁定 | 0 |
| `residual_lstm` | 相同LSTM学习目标值减所选朴素预测的误差，再加回朴素预测 | 四指标×五种子，共20次 |
| `gated_residual_lstm` | 对同一次训练产生的误差修正乘以验证集选择的系数，再加回朴素预测 | 0，复用上一行的训练 |

公式为 `预测值 = 验证集选定的朴素预测 + α × LSTM预测的残差`。

未门控组固定α=1。门控组每个时效独立从 `0, 0.05, …, 1` 的21个固定值中按验证MAE选择；并列时取较小α。α=0恰好回退到选定的朴素预测，α=1等于未门控修正。六个门控系数为额外的验证集校准量，不改变神经网络参数数量。

共享训练器只在训练索引拟合输入和残差缩放器、更新权重，在验证索引执行早停。为获得最佳权重的验证预测，新入口将验证和测试索引拼接到**推理输出列表**，训练及早停的索引保持不变。基线选择、缩放、早停和门控选择均不使用测试标签。最终分别评分原始测试索引。

门控保证包含“不修正”的验证集候选，不保证测试集一定改善。验证表现也不能充当独立检验。

## 主要评价与解释规则

主要比较为每项指标的六时效等权MAE：

1. `gated_vs_lstm`：门控残差相对普通LSTM，有没有额外收益。
2. `gated_vs_selected_naive`：门控残差相对验证集选定的朴素基线，有没有实际增量。

同时检查五个种子的方向、RMSE变化、R²变化和分时效结果。4小时和24小时等单时效结果为辅助描述，不能只选改善时效代表整体。次要比较为 `residual_vs_lstm`、`residual_vs_selected_naive`、`gating_increment`，用于区分残差策略与门控的贡献。

`MAE_reduction_percent`按配对种子、六时效等权后的组均值计算：`100 × (1 − 新方法MAE / 对照MAE)`。正值表示改善。`RMSE_reduction`是逐时效RMSE之差的等权平均，`R2_gain`是逐时效R²增量的等权平均。各指标分别报告，避免跨量纲平均。

若仅超过普通LSTM、未超过所选朴素预测，应写“改善了普通LSTM，但未确立相对简单基线的增量技能”。若门控大多为零，应写“验证选择倾向于朴素预测”，不把回退称为神经网络学到了更强规律。若两项主要比较均改善且种子方向一致，可报告固定协议下残差门控的增量价值；不据此推断因果机制。

日历块区间沿用04：4小时分辨率下的6步日历块、2000次重采样、种子2026。先平均配对种子损失，再按预测起点的日历块重采样，保留空日历块。区间只描述固定划分和已拟合模型下的不确定性；多项指标/时效/对照未做多重比较校正，应作为探索性结果。

本轮是已查看01—05结果之后提出的方法比较，同一随机留出测试集已被用于前序分析。因此属于探索性后续比较，不是全新独立的确认性测试，也不提供时间独立未来泛化证据。训练上限触顶时，结论继续限定为固定预算下表现，不称已充分收敛。

## 安装位置

把本文件夹**里面的内容**合并到现有项目根目录：

`C:\path\to\jinjiang-water-quality`

| 包内文件 | 项目位置 |
|---|---|
| `src/jinjiang/residual_random.py` | `src/jinjiang/` |
| `experiments/06_residual_random.py` | `experiments/` |
| `configs/residual_random_gpu.json` | `configs/` |
| `configs/residual_random_smoke_gpu.json` | `configs/` |
| `tests/test_residual_random.py` | `tests/` |
| `run_06_residual_random.ps1` | 项目根目录，与src和experiments同级 |
| 本说明及PATCH_MANIFEST | 项目根目录，供查阅 |

全部为新增文件，保留01—05、旧06和现有结果。依赖当前已运行05的共享代码，尤其是 `station_rotation_random.py` 和04运行时的 `models.py`。无需重新安装依赖，不替换共享训练模块。

## 运行顺序

在**合并后的项目根目录**打开PowerShell。默认使用 `.venv_gpu\Scripts\python.exe`：

```powershell
# 1. 核验04来源与四指标共同样本，不训练
powershell -ExecutionPolicy Bypass -File .\run_06_residual_random.ps1 -Mode audit

# 2. 单独目录运行C站DO、种子42、2轮、1次LSTM拟合
powershell -ExecutionPolicy Bypass -File .\run_06_residual_random.ps1 -Mode smoke

# 3. 正式20次训练；中断后再次执行此命令续跑
powershell -ExecutionPolicy Bypass -File .\run_06_residual_random.ps1 -Mode run
```

若GPU Python路径不同，三个命令都加 `-Python "完整路径\python.exe"`。

直接Python入口等价：

```powershell
$gpuPython = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PWD "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
& $gpuPython experiments/06_residual_random.py --config configs/residual_random_gpu.json --audit-only
& $gpuPython experiments/06_residual_random.py --config configs/residual_random_smoke_gpu.json
& $gpuPython experiments/06_residual_random.py --config configs/residual_random_gpu.json
```

支持 `--max-tasks N` 限制本次新增拟合次数，以及 `--continue-on-error`。无覆盖选项。正式配置与冒烟配置不能共用输出目录。冒烟不复用正式普通LSTM、不参与正式比较，仅检查GPU和新入口；其状态为 `smoke_complete`。

## 来源核验、续跑与任务数

默认从 `results/meteorology_random_gpu/04_meteorology` 读取普通LSTM。若位置不同，开始前修改正式配置中的 `reference_04_dir`。保留完整04目录，包括训练历史、预测、样本清单和指纹。

复用现有05中的C站严格核验器，检查04完整120任务状态、配置、数据和共享源码指纹、两组40个LSTM任务的完整性、特征与集合内索引顺序、所有样本时间、测试标签及MAE/RMSE一致性。核验后只把完整气象输入组20个普通LSTM用于06。初始日志中的 `[05] Verified C reference` 来自该共享核验器，表示检查通过，不表示运行了05训练。

相同输入的普通LSTM和残差LSTM必须具有完全相同的神经参数数；不匹配时停止该任务。04文件只读，来源校验记录在 `reference_04_provenance.json`。

20个拟合各有一个 `tasks/C_指标_seed种子/` 目录。每次拟合完成后写出五类预测、训练历史和门控记录，最后原子写入 `commit.json`。续跑时检查文件哈希和完整性，仅跳过完整任务；残缺、损坏或无提交标记的任务重新训练。一个任务中断后从该次训练开头重跑，不是逐epoch恢复权重。

不要手动删除本地 `tasks` 文件夹，它是续跑依据。汇总文件可从完整任务重建。配置、数据、源码或04来源改变时会拒绝混入同一目录，需使用新的结果目录。

## 输出及验收

正式输出：`results/residual_random_gpu/06_residual_random`。

冒烟输出：`results/residual_random_smoke_gpu/06_residual_random`。

| 文件 | 作用 |
|---|---|
| `run_status.json` | 正式完成：20次新增拟合、20次普通LSTM复用、0未解决失败、`complete` |
| `residual_gain_summary.csv` | 四指标×五对照共20行；优先看两个primary对照，共8行 |
| `paired_seed_horizon.csv` | 各种子、各时效的配对MAE、RMSE、R²变化 |
| `paired_block_bootstrap.csv` | 日历块探索性区间；horizon_hours=0为六时效等权结果 |
| `gate_diagnostics.csv` | 每个指标、种子和时效的所选朴素基线、α及验证MAE，正式120行 |
| `model_diagnostics.csv` | 20次新增神经拟合的最佳轮、实际轮数、触顶情况和参数数 |
| `metrics.csv` | 五类新输出×20拟合×六时效，共600行；朴素预测未经过神经训练 |
| `combined_metrics.csv` | 加上普通LSTM的120行，共720行、六种方法 |
| `metrics_mean_sd.csv` | 各方法按时效的种子均值与标准差 |
| `predictions.csv`、`combined_predictions.csv` | 新输出与合并普通LSTM的逐样本预测 |
| `training_history.csv` | 20次残差拟合的训练历史，未门控和门控不会重复计为两次训练 |
| `sample_manifest.csv`、`sample_support_audit.csv`、`input_groups.csv` | 共同样本和输入列核验 |
| 配置、来源与指纹JSON | 复现与来源追踪 |

三类朴素预测相同目标/时效下不依赖神经种子，其记录按种子重复仅为配对，不代表五次独立朴素模型训练。门控与未门控同属一次残差拟合，不是40次新增训练。`parameter_count`仅记录神经参数，朴素预测为0。

仅审计时，合并指标中可能只有复用的普通LSTM，收益表为空；部分训练时只汇总完整配对种子，标为 `partial_paired_seeds`。以整体状态文件确认是否全部完成。

完成后可上传整个06正式结果目录。若需缩小上传体积，可省略上传 `tasks`，但必须在本机保留以便续跑；CSV和JSON汇总用于结果审核。

## 本包验证

5项自动测试通过，覆盖：固定20次拟合与旧配置拒绝；每次只调用一次共享训练器；验证集与测试集标签隔离；门控为零时精确回退；20次完整模拟续跑及未提交/残缺文件重跑；配对结果与任务数；骨干参数不一致时拒绝；配置变更拦截与冒烟隔离。

真实数据审计已通过，四指标样本数与04完全一致：DO817、NH3N877、CODMN1285、TP1015。20个完整输入普通LSTM任务可复用，读取校验未修改04文件。另以真实四指标数据，用CPU Ridge代替神经训练器完成8次内部数据通路检查，确认更改测试标签不改变预测或门控；这些验证不写入正式LSTM结果。当前环境未执行GPU/LSTM训练，需在你的GPU环境先运行2轮冒烟。

```powershell
$gpuPython = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PWD "src"
& $gpuPython -m unittest discover -s tests -p test_residual_random.py -v
```
