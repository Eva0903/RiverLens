# 当前01—07运行手册

以下命令在仓库根目录执行。已有GPU环境时只需指向原解释器，或在其已激活的PowerShell中运行。不要把`.venv_gpu`复制进仓库。

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
$env:PYTHONPATH = Join-Path $PWD "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

统一入口会自行设置源码搜索路径。直接执行`experiments/`中的文件时，应先设置上述`PYTHONPATH`，或安装`python -m pip install -e .`。

## 01—05

```powershell
python run_study.py --experiment 01 --mode audit
python run_study.py --experiment 01 --mode smoke
python run_study.py --experiment 01 --mode run
```

将编号替换为02、03、04、05即可调用各自对应配置。`audit`不训练，`smoke`只做小规模软件通路检查，`run`执行正式预算。正式运行可能训练大量模型。不要因为上传代码而重复运行。

| 实验 | 输出根目录 | 关键依赖 |
|---|---|---|
| 01 | `results/baseline_random_gpu/01_baseline` | 本地观测 |
| 02 | `results/station_ablation_random_gpu/02_station_ablation` | 本地观测 |
| 03主网格 | `results/window_horizon_random_main_gpu/03_window_horizon` | 本地观测 |
| 04 | `results/meteorology_random_gpu/04_meteorology` | 本地观测；沿用03选窗配置 |
| 05 | `results/station_rotation_random_gpu/05_robustness` | 已完成04，核验并复用C站两类输入的40次训练结果 |

01/02固定18步历史，输出六个时效。03主网格以84步为共同样本支持，比较15个输入窗口、输出18步；不要将不同实验绝对分数直接串接比较。04/05的窗口为DO84、NH3N78、CODMN48、TP66步。

## 03预算与早停诊断

从已完成结果的`run_config.json`恢复了1024轮参照配置。它是诊断的依赖，不替换512轮主网格：

```powershell
python experiments\03_window_horizon.py --config configs\window_horizon_random_budget1024_gpu.json --audit-only
# 缺少参照结果、且确实需要重建时才运行下条正式训练
python experiments\03_window_horizon.py --config configs\window_horizon_random_budget1024_gpu.json

python experiments\03_training_diagnostics.py --config configs\window_training_cap_gpu.json --audit-only
python experiments\03_training_diagnostics.py --config configs\window_training_cap_gpu.json
python experiments\03_training_diagnostics.py --config configs\window_training_patience_gpu.json
```

两项诊断分别比较7个预先列出的任务，输出2048轮上限/早停检查及cap合并网格。历史结果整理中DO/NH3N采用512轮主网格，CODMN/TP使用cap合并网格；这不是四指标统一重跑2048轮。长窗口配置`window_horizon_random_long_gpu.json`为可选扩展，不代表本包已验收其研究结果。

## 06解释与权重依赖

```powershell
python run_study.py --experiment 06 --mode audit
python run_study.py --experiment 06 --mode overview
python run_study.py --experiment 06 --mode run
```

`overview`汇总已完成04/05的条件收益；`run`使用已有检查点进行置换分析。当前06入口包含原预测一致性的精度修复，保留原阈值，不调用此前失败的梯度解释通路。

完整解释包括C站四指标以及A/B站CODMn，共6个固定模型；海报主要展示C站四指标。每个模型仅使用训练随机初始化编号42的检查点。每组/时段置换10次。

**06不是拿到代码就能从零直接执行的实验。** 它需要04/05完整本地结果及对应模型权重。早期04/05不保存检查点；缺少时，默认列出缺失项，不自动训练。若已有原始训练记录且明确需要恢复，可显式执行：

```powershell
# 此模式会按原最佳训练轮数重放模型，每次最多恢复1个
python run_study.py --experiment 06 --mode recover --max-replays 1
```

不要在有可用权重时重复恢复。你的现有06权重应直接保留。主要输出为`results/interpretation_random/06_interpretation_random`。

## 07分组Kernel SHAP

```powershell
python run_study.py --experiment 07 --mode audit
python run_study.py --experiment 07 --mode smoke
python run_study.py --experiment 07 --mode run
```

07读取已完成06目录中的C站四项指标检查点，并核验其原始04预测。使用三套训练集参考样本，每套64个窗口；31个完整历史解释单元覆盖39列输入，每个待解释窗口采用512个联盟采样预算，同时输出六个时效。整个步骤不计算梯度、不训练新模型。

结果写入`results/shap_kernel_standalone/07_kernel_shap`。中断后使用同一条命令续算。加和一致性检查不是精确Shapley值或采样收敛的证明。

## 继续使用你现有的本地结果

1. 保留原V4项目和原GPU环境；本包是另一个代码目录。
2. 本地数据、04/05结果、06权重等若需用于新目录，可自行复制到相同相对路径；Git会忽略它们。
3. 原结果有源码、配置、数据和内容指纹约束。当前模型与核心模块已同上传结果中的源码记录核对，但文件在Windows上的换行转换仍可能影响字节哈希。
4. 如遇指纹不一致，保留旧项目和报错，先比较源码及换行。不要编辑旧指纹文件、删除核验记录或放宽阈值来继续。
5. 新配置、实验预算或源代码确实变化时，使用新的输出目录，避免混入旧结果。

上传GitHub只需要代码包；完成原实验的本地目录不必发生任何改动。
