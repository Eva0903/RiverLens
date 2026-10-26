# 01 Baseline：先按时间分集合，再分别滑动窗口

本补丁只为01启用新协议。没有在你的电脑上训练模型，也没有生成新的神经网络性能结果。

## 1. 应当复制哪些文件

关闭正在运行的实验进程。在项目根目录先备份下面两个现有文件，再把补丁中同名目录的内容合并进去。

| 文件（相对于项目根目录） | 操作 | 作用 |
|---|---|---|
| `experiments/01_baseline.py` | 替换 | 新版01入口；默认新配置；支持只做样本审计 |
| `src/jinjiang/runner.py` | 替换 | 仅01的新协议启用严格内部构窗；沿用模型训练和统计输出接口 |
| `src/jinjiang/baseline_within_split.py` | 新增，必须 | 分集合构窗、完整窗口边界断言、样本清单、MASE换算 |
| `configs/baseline_within_split_gpu.json` | 新增，必须 | 正式GPU配置 |
| `configs/baseline_within_split_smoke_gpu.json` | 新增，推荐 | 单DO、单种子、2轮GPU连通检查 |
| `tests/test_baseline_within_split.py` | 新增，推荐 | 新协议边界和断点续跑测试 |
| `README_01_WITHIN_SPLIT.md` | 新增，说明 | 本文件 |
| `VERIFICATION_01_WITHIN_SPLIT.md` | 新增，说明 | 已做验证、尚未做的验证 |
| `reference_audit/` | 可保留作参考 | 由本次上传数据生成的样本边界审计，不是新模型性能结果 |

不要用整个补丁目录替换原项目目录；合并 `experiments`、`src`、`configs`、`tests` 即可。
不需要覆盖 `data.py`、`models.py`、`evaluation.py`，不需要覆盖任何原始数据或旧结果。
02—07的实验逻辑、模型结构和配置文件未在此补丁中重设计；此补丁也不表示它们已经符合新01协议。

在PowerShell项目根目录备份两个替换文件：

```powershell
$backupDir = Join-Path $PWD ("backup_before_within_split_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $backupDir | Out-Null
Copy-Item .\experiments\01_baseline.py (Join-Path $backupDir "01_baseline.py")
Copy-Item .\src\jinjiang\runner.py (Join-Path $backupDir "runner.py")
```

## 2. 固定实验设计

- 划分单位：完整、按时间排序的4小时原始网格，先按70%/15%/15%划为训练、验证、测试。切分前不删除缺失行，不打乱日期。
- 预处理：继续使用现有站点读取和时间对齐；各集合独立执行既有短缺口 `causal_ffill`，不跨集合填补。此补丁不重新设计时间对齐。
- 构窗：在每个集合内部独立运行步长1的滑动窗口。18步输入（3天），6步直接多输出（4/8/12/16/20/24小时）。
- 每条样本的18步输入、6步目标、Persistence和24h Seasonal naive所需历史均位于同一集合。集合开头不足18步历史的预测不进入该集合评价。
- 标签：保持原始观测值；缺失目标不通过插补进入评价。
- 输入：A/B/C三站全部可用水质/水文变量，当前数据共26列，不含气象。包含下游C历史，因此不是纯A/B上游预测任务。
- 共同样本：每个指标的八种方法共享同一个有效窗口集合。即使naive本来可以预测更多时点，也只在共同窗口上比较。
- 标准化：沿用现有模型实现，只用训练窗口拟合特征、目标的StandardScaler，再变换验证和测试。
- 模型：Persistence、24h Seasonal naive、Ridge、ANN、CNN、RNN、普通LSTM、Transformer，结构与现有01相同。
- 普通LSTM：2层、hidden64、dropout0.1；batch32、Adam lr0.001、MSE、最多256轮、patience15，按验证损失选最佳权重。此01不进行train+validation重训。
- 种子：42/2026/3407/7/99。固定配置比较，不额外进行模型或窗口搜索。
- 不再运行 `random_window_diagnostic`，不产生随机切分乐观程度表。

正式任务总数：4指标×8方法×5种子＝160任务。其中神经网络训练100次；Ridge20次；naive输出40任务（naive在不同种子下相同，不是独立随机重复）。

## 3. 运行顺序

在现有 `Jinjiang_LSTM_Upgrade_V4` 项目根目录打开PowerShell，使用已经验证成功的GPU环境。不需要重新安装PyTorch。

```powershell
$gpuPython = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PWD "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

### 第一步：只做样本审计，不训练

```powershell
& $gpuPython experiments/01_baseline.py --config configs/baseline_within_split_gpu.json --audit-only
```

成功时四指标应显示 `train=2234, validation=295, test=484`（仅当数据文件和配置与本次上传一致）。
打开 `results/baseline_within_split_gpu/01_baseline/sample_support_audit.csv`：

- 共12行（4指标×3集合）；`full_window_contained` 全部为True；
- `cross_partition_overlap` 全部为False；
- `training_MASE_scale` 为有限正数；若训练目标不变，DO约0.603655；
- `run_status.json` 此时应显示 `run_status=audit_only`、完成任务0，不代表训练完成。

### 第二步：推荐先做GPU连通检查

```powershell
& $gpuPython experiments/01_baseline.py --config configs/baseline_within_split_smoke_gpu.json
```

此检查有3任务：两种naive和一次DO LSTM，LSTM仅训练2轮。输出到单独的 `results/baseline_within_split_smoke_gpu/01_baseline`，不得当作论文结果。重复同命令会跳过已完成任务；需要重做这个小检查时才加 `--overwrite`。

### 第三步：正式运行全部八种方法

```powershell
& $gpuPython experiments/01_baseline.py --config configs/baseline_within_split_gpu.json
```

完整比较不要加 `--model-override lstm`，否则只会运行两种naive和LSTM。
首次正式运行不需要 `--overwrite`。中断后直接重执行同一条命令，自动续跑。
不要并行启动两个写入同一结果目录的进程。新入口会拒绝旧配置，避免误用旧目录。

正式输出目录：`results/baseline_within_split_gpu/01_baseline`。
旧的 `results/baseline_lstm_gpu/01_baseline` 与 `results/lstm_upgrade_paper/06_lstm_upgrade` 保留，不合并新旧指标。

## 4. 输出与验收

- `sample_support_audit.csv`：原始集合边界、候选/有效窗口数、完整窗口隔离检查、特征列和训练MASE分母。
- `sample_manifest.csv`：每个有效样本所属集合、输入起点、预测发布时刻、目标起止时刻。
- `metrics.csv`：完整正式运行应有960行（160任务×6时效）；RMSE、MAE、R²和MASE应有有效值。
- `metrics_mean_sd.csv`：应有192行（4指标×8方法×6时效），报告5种子均值和样本标准差。
- `training_history.csv`：100个神经网络任务的训练记录，逐epoch保留。
- `model_ranking.csv`、`skill_scores.csv`：逐指标、逐时效比较。`best_naive`技能使用测试误差中较好的naive作为描述性参照，不能视为验证集锁定的部署选择。
- `paired_block_bootstrap.csv`：沿用现有配对移动块bootstrap，块长6、2000次。区间考虑相邻保留窗口的相关性；尚未做时间缺口敏感性、块长敏感性或多重比较校正。
- `predictions.csv`、`figures/model_benchmark_R2.png`：完整预测和现有绘图输出。
- `run_status.json`：应为 `run_status=complete`、`completed_tasks=160`、`expected_tasks=160`、`unresolved_failed_tasks=0`。

注意，`Finished`表示当前命令结束；若使用了 `--max-tasks`，仍可能是partial。以run_status为准。

## 5. 本次已核验的样本规模

原始网格共4380步：训练3066、验证657、测试657，严格为70/15/15。
独立滑窗和缺失筛选后，四指标均为训练2234、验证295、测试484，合计3013窗口。
相对于旧正式01的2234/309/484，验证少14个窗口，训练和测试支持集不变。
这说明旧正式01在测试边界附近本来就没有留下需要删除的跨界窗口；不是新协议没有生效。

原始时间比例与最终有效窗口比例不必相同。不能通过删除缺失行后重新拼接时间来强行保持窗口数量70/15/15。
旧正式01按目标边界隔离、允许借用此前可用历史；这与新协议不同，但这种借用本身不等于未来泄漏。
新设计落实严格集合内部构窗，不保证改善RMSE或R²，不得预写模型排名。

## 6. 可复用英文方法说明

The original 4-hourly time grid was partitioned chronologically into training (70%), validation (15%), and test (15%) sets before constructing sliding windows separately within each set. Every sample contained 18 historical steps and six future target steps, with its full temporal support restricted to a single partition. All benchmark methods were evaluated on identical valid samples for each target. Scaling parameters were estimated from training samples only, and neural-network checkpoints were selected using validation loss. Performance was reported separately for each constituent and lead time across five random seeds.
