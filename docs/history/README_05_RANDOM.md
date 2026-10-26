# 05：目标站点轮换实验（可直接合并到当前V4项目）

将本包内的 `src`、`experiments`、`configs`、`tests` 四个目录，以及根目录的PS1脚本合并到当前 `Jinjiang_LSTM_Upgrade_V4` 项目根目录。包内均为新增文件；已有01—04代码和结果保留。入口为 `05_station_rotation_random.py`，不要运行项目旧版 `05_robustness.py`。

依赖当前已完成04的项目，包括 `meteorology_random.py`、`baseline_random.py`、`baseline_within_split.py` 及04运行时的共享训练模块。补丁不会替换这些模块。无需重新安装依赖。

## 本轮任务与固定设置

- 新训练仅A、B两个目标站点，四指标、两组输入、五个种子，共 **80次LSTM训练**。C站复用已完成04中对应的40次LSTM训练，不重训，也不增加naive任务。
- 两组输入：`all_stations`（三站历史水质/水文变量）；`all_stations_weather_rain_lags`（再加气象和显式降雨历史）。当前数据对应26列和39列。
- 窗口直接使用03主实验按18个验证时效平均MAE选出的结果：DO84步/14天、NH3N78步/13天、CODMN48步/8天、TP66步/11天。直接迁移到各站，不重新筛窗口，不声称是各站或6输出模型重新优化的最优窗口。
- 六步预测，4、8、12、16、20、24小时；512轮上限、patience=100；种子42、2026、3407、7、99。LSTM 64维、2层、dropout=0.1、batch=32、Adam学习率0.001，CUDA严格确定性设置与04相同。
- 两次 `sklearn.train_test_split`，80%/10%/10%，split_seed=42，算法与01—04一致。每个站点和指标先按完整39列筛一次共同样本，再分割，两组只选择不同输入列。不同站点的样本量可以不同，不强行将三个站点取交集。
- 标准化只拟合训练数据；验证集用于早停和恢复验证MSE最小的权重；测试集用于评价。所有输入不晚于发布时刻，无未来实测气象。关闭SHAP，不增加预算敏感性或时滞分析。

## 推荐执行方式

在**合并后的项目根目录**打开PowerShell，依次运行。脚本默认使用完成04时的 `.venv_gpu\Scripts\python.exe`。

```powershell
powershell -ExecutionPolicy Bypass -File .\run_05_station_rotation.ps1 -Mode audit
powershell -ExecutionPolicy Bypass -File .\run_05_station_rotation.ps1 -Mode smoke
powershell -ExecutionPolicy Bypass -File .\run_05_station_rotation.ps1 -Mode run
```

`audit`不训练，核对C站来源并输出A/B/C样本审计；`smoke`只运行A站DO两组各2轮，结果独立保存；`run`执行正式80次训练。审计和冒烟通过后执行正式命令。训练中断后再次执行同一条 `-Mode run` 命令即可续跑：跳过已完整写出的任务，当前未完成的一次训练从头重跑。

若GPU环境不在默认位置，三个命令都加 `-Python "你的GPU环境\python.exe"`。

也可直接运行Python入口：

```powershell
$gpuPython = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PWD "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
& $gpuPython experiments/05_station_rotation_random.py --config configs/station_rotation_random_gpu.json --audit-only
& $gpuPython experiments/05_station_rotation_random.py --config configs/station_rotation_random_smoke_gpu.json
& $gpuPython experiments/05_station_rotation_random.py --config configs/station_rotation_random_gpu.json
```

本入口支持 `--max-tasks N` 限制本次新训练次数，和 `--continue-on-error` 记录失败后继续其他任务。不支持覆盖旧结果；不要添加 `--overwrite`。同一结果目录锁定同一配置、数据和代码，修改协议需使用新的结果目录。正式和冒烟结果不会混用。

## C站复用检查

默认读取 `results/meteorology_random_gpu/04_meteorology`。如果你的04目录位置不同，在正式配置的 `reference_04_dir` 中填写实际路径后再开始；相对路径按项目根目录解析。请保留04完整目录，尤其是预测、训练历史、样本清单和指纹文件。

复用前必须通过：04的120任务全部完成、无未解决失败；固定训练与预处理配置一致；四个原始XLSX及04共享源码指纹一致；完整输入特征列一致；每个C站指标的所有样本时间、随机集合归属、集合内索引顺序、测试标签一致；对应40个LSTM任务具有六个时效的完整指标、预测和训练历史。还会从预测复核MAE、RMSE及逐样本绝对误差。CSV中的float32值按原精度回读核对。

检查失败会停止并指出具体原因，不会改写04结果。不要通过删除检查或随意替换指纹来绕过。`reference_04_provenance.json`记录来源目录及来源文件SHA256，续跑时继续核对。

## 当前原始数据审计参考

| 目标站点 | 指标 | 总窗口 | 训练 | 验证 | 测试 |
|---|---|---:|---:|---:|---:|
| A | DO | 801 | 640 | 80 | 81 |
| A | NH3N | 864 | 691 | 86 | 87 |
| A | CODMN | 1237 | 989 | 124 | 124 |
| A | TP | 998 | 798 | 100 | 100 |
| B | DO | 822 | 657 | 82 | 83 |
| B | NH3N | 881 | 704 | 88 | 89 |
| B | CODMN | 1276 | 1020 | 128 | 128 |
| B | TP | 1009 | 807 | 101 | 101 |
| C（复用） | DO | 817 | 653 | 82 | 82 |
| C（复用） | NH3N | 877 | 701 | 88 | 88 |
| C（复用） | CODMN | 1285 | 1028 | 128 | 129 |
| C（复用） | TP | 1015 | 812 | 101 | 102 |

## 结果与验收

正式输出：`results/station_rotation_random_gpu/05_robustness`。冒烟输出：`results/station_rotation_random_smoke_gpu/05_robustness`。

| 文件 | 内容 |
|---|---|
| `run_status.json` | 新训练完成数、C复用数、整体状态；正式完成应为80新任务、40复用任务、`complete` |
| `metrics.csv`、`predictions.csv`、`training_history.csv` | 仅A/B新增训练；正式480行指标，80个任务 |
| `combined_metrics.csv`、`combined_predictions.csv` | 合并A/B新增和C复用；正式720行指标，120个任务 |
| `metrics_mean_sd.csv` | 按站点、指标、输入组、时效汇总种子均值和标准差 |
| `station_rotation_gain_summary.csv` | 三站四指标共12行主结果：六时效等权MAE、MAE降幅、五种子方向、RMSE减少与R²增量 |
| `paired_seed_horizon.csv` | 同站点、同指标、同种子、同时效的两组输入配对比较 |
| `paired_block_bootstrap.csv` | 按站点给出逐时效及六时效等权MAE减少的日历块探索性区间 |
| `model_diagnostics.csv`、`combined_model_diagnostics.csv` | A/B及三站训练轮数、最佳轮、是否触顶 |
| `sample_manifest.csv`、`sample_support_audit.csv`、`input_groups.csv` | 三站样本、划分及特征列审计 |
| `reference_04_provenance.json`、`data_fingerprints.json`、`source_fingerprints.json` | 复用与复现来源 |

合并文件的 `result_origin` 区分 `new_05` 与 `reused_04`，原始 `task_id`、来源实验与协议标识保留。04文件保持原样。仅审计或部分运行时，合并文件中可能只有已验证的C站或已完成的A/B任务；以整体 `run_status.json` 为准。部分配对明确标为 `partial_paired_seeds`，不会当作五种子结论。

主结果的 `MAE_reduction_percent` = 100×(1−气象组MAE/水质组MAE)，先在每种子的六个时效等权平均，再在配对种子间平均后取比值；正值表示改善。另保留各个种子降幅的均值、标准差与改善种子数。RMSE为逐时效RMSE之差的等权平均，不是先合并预测再计算的RMSE。区间针对固定随机划分和已拟合模型，按预测起点的日历块重采样，未做多重比较校正，不表示未来泛化或划分不确定性。

## 结论范围

本轮回答“气象增益是否随目标站点改变而保持方向”，每个目标站点分别训练模型。它不是将C站训练好的模型直接迁移到未见站点。A/B作为预测目标时仍使用三站历史信息，其中可能包括空间上位于下游的站点，因此不称为纯上游输入的业务预报。随机重叠窗口留出不提供时间独立的未来预测证据。三个站点样本支持不同，以各站内配对增益和种子一致性比较，不用跨站绝对误差差异代替气象收益。

## 本包已完成的验证

6项自动测试通过：80任务固定协议、站点目标标签及共同样本、两次随机划分精确复现、C来源配置不匹配拒绝、冒烟隔离、完整80任务模拟续跑、残缺任务重跑及指纹变更拦截（多项检查合并在单个测试中）。真实数据审计覆盖A/B/C全部12个站点指标组合；已成功核验C站40个任务及240行指标，读取和校验过程未修改04来源文件。另用真实A/B站DO数据、两组输入，共4次CPU Ridge拟合验证数据与缩放通路；这些仅作内部验证，不写入正式LSTM结果。当前环境未执行GPU/LSTM训练。

```powershell
$gpuPython = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PWD "src"
& $gpuPython -m unittest discover -s tests -p test_station_rotation_random.py -v
```
