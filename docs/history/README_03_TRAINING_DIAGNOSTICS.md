# 03训练充分性：7项上限检查与7项早停检查

适用于已完成随机版03，以及`window_horizon_random_budget1024_gpu`的当前V4项目。
本包增加独立入口和模块，不替换现有03或共享训练代码。训练函数、共同样本与划分均复用原实现。

## 安装

解压后，进入`Jinjiang_V4_03_Training_Diagnostics_Patch`，将`experiments`、`src`、`configs`、`tests`合并到项目根目录。
包内同名文件属于本次新增诊断文件；若已经安装本补丁，请先结束对应诊断进程再更新。

新增文件：

- `experiments/03_training_diagnostics.py`
- `src/jinjiang/window_training_diagnostics.py`
- `configs/window_training_cap_gpu.json`
- `configs/window_training_patience_gpu.json`
- `tests/test_window_training_diagnostics.py`

必须保留本地原结果目录：

```text
results/window_horizon_random_budget1024_gpu/03_window_horizon
```

程序会读取其中的配置、状态、数据与源代码指纹、样本清单、训练记录、模型诊断和指标。原文件只读。
如果你移动了原结果，只修改两个新配置中的`baseline_results_dir`，应指向包含`run_config.json`的那一级目录。
原结果不完整、数据/代码改变、样本或集合归属不一致时，程序会停止。不要绕过这些检查。

## 两项检查

| 检查 | 训练上限 | patience | 训练数 | 结果目录 |
|---|---:|---:|---:|---|
| cap | 2048 | 30 | 7 | `results/window_horizon_diagnostic_cap_gpu/03_window_horizon` |
| patience | 2048 | 100 | 7 | `results/window_horizon_diagnostic_patience_gpu/03_window_horizon` |

没有额外的naive训练，不展开全部窗口和种子的组合。每个任务从相同种子重新训练。
当前共享训练函数不保存可严格续接的完整优化器与随机状态，因此本入口不声称从第1025轮继续训练。

cap任务：CODMN 4/8/12天的seed99、14天的seed2026；TP 5/6天的seed42、11天的seed2026。

patience任务：CODMN 8天的seed42/3407/2026；TP 11天的seed99/42；TP 9天的seed3407/2026。
四个较高验证误差任务和三个同窗口对照的身份记录在`task_plan.csv`。

## 运行命令（PowerShell，项目根目录）

```powershell
$gpuPython = Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $PWD "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# 1. 两份审计都不训练；CODMN/TP应仍为1487个窗口，1189/149/149。
& $gpuPython experiments/03_training_diagnostics.py --config configs/window_training_cap_gpu.json --audit-only
& $gpuPython experiments/03_training_diagnostics.py --config configs/window_training_patience_gpu.json --audit-only

# 2. 新入口的连通检查：一个任务、两轮，写入独立的_smoke目录。
& $gpuPython experiments/03_training_diagnostics.py --config configs/window_training_cap_gpu.json --smoke

# 3. 检查一：七个触顶任务。
& $gpuPython experiments/03_training_diagnostics.py --config configs/window_training_cap_gpu.json

# 4. 检查二：七个patience诊断任务。
& $gpuPython experiments/03_training_diagnostics.py --config configs/window_training_patience_gpu.json
```

可用同样的`--smoke`检查patience配置。每个smoke只有一个任务，不能作为正式结果。
按一次Ctrl+C可中止；再次执行相同正式命令将跳过已完成任务，重新开始未完成任务。
不要改原配置后复用同一输出目录。两个检查顺序运行即可。

## 配对检查与结果

每次拟合后，程序先比较新旧训练共同epoch的训练损失和验证损失。
允许CSV浮点往返的极小差异：`rtol=1e-6, atol=1e-8`。
不匹配的任务不会被标记为完成，错误写入`errors.log`，请保留日志排查。
这可以发现代码、设备环境或随机轨迹不一致，不能用来证明额外时段独立预测能力。

每个检查输出：

- `run_status.json`：正式每项应完成7个任务，失败0个。
- `task_plan.csv`：实际任务及诊断角色。
- `paired_task_summary.csv`：相同任务全部18时效等权平均的验证/测试MAE、RMSE、R2与训练轮数比较。
- `paired_horizon_comparison.csv`：逐时效配对比较。
- `model_diagnostics.csv`：触顶、最佳epoch、patience以及共同训练轨迹检查。
- `training_history.csv`、`metrics.csv`、`validation_metrics.csv`、`predictions.csv`。
- `sample_manifest.csv`、`sample_support_audit.csv`以及数据、原代码和本诊断模块的指纹。

`validation_MAE_reduction_percent`为正表示验证误差降低。判断训练规则优先使用验证结果。
测试指标只作配对描述，不用于选择patience、轮数或窗口。

### cap完整网格合并

七项正式cap全部完成且轨迹核对通过后，自动生成`cap_merged_full_grid`。
该子目录复用143个按patience30自然早停的LSTM、20个naive，并替换七个触顶任务，形成统一2048轮上限规则的完整比较。
程序会核查所有原自然早停任务均已完成30轮等待；任何一个不满足时不会复用。

子目录含重新汇总的指标、验证最优窗口、2%容差最短窗口及已选窗口的测试报告。
每行保留原来源的task_id/protocol_hash，通过`lineage`与`effective_max_epochs`说明来源和有效预算；源文件不修改。
重新统一汇总时使用共同subexperiment名称，原名称保存在`source_subexperiment`。
完整逐样本预测和训练记录仍分别保存在原1024结果和本cap结果中，不额外复制大型文件。

### patience仅作诊断

patience检查不生成完整窗口排名，也不会把挑选出的7个任务混入正式网格。
如果验证结果支持调整patience，再对对应指标统一运行15窗口×5种子。
当前入口不会根据测试指标自动决定后续实验。

## 上传检查

分别将两个正式结果目录中的CSV、JSON打包上传，保留cap的`cap_merged_full_grid`子目录。
逐样本`predictions.csv`可以暂不上传；原1024结果已提供过，无需再次上传。

## 验证范围

包内测试不需要torch，通过受控的训练返回值检查任务筛选、原训练轨迹复现、断点续跑、样本/源码漂移阻断和完整网格合并。
可在项目中运行：

```powershell
& $gpuPython -m unittest discover -s tests -p test_window_training_diagnostics.py -v
```

交付环境没有PyTorch，未执行真实GPU训练。实际GPU连接与共同epoch复现请通过上述两轮smoke确认。
