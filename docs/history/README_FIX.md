# 06缓存推理精度修复

## 这次错误发生在哪里

报错位于 `explain()` 中重新加载缓存后的预测核验，而不是 `recover()` 的训练输出核验。按原代码的执行顺序，NH3N已经完成365轮恢复、通过恢复输出核验，并写出模型权重、缩放参数和元数据，随后才出现当前错误。

**保留现有checkpoints和attributions目录，不要删除NH3N权重，不必先重训365轮。**

## 原因与修复

共享训练器调用当前环境 `StandardScaler.transform()` 和 `inverse_transform()`；旧缓存加载器用手写的 `float32数组 -= float64均值` 等表达式。新版本scikit-learn在运算前将统计参数转换为输入数组的数据类型，这与手写表达式的浮点舍入顺序不同。输入缩放差异进入神经网络后，小浓度预测容易超出原核验阈值。已在回归用例中复现该差异。

补丁从已有NPZ统计参数恢复StandardScaler，只调用transform/inverse_transform，不重新拟合缩放器。另恢复原共同样本索引产生的N/特征/时间存储布局，并按照原训练器的顺序：整批测试输入送入设备、分batch推理、拼接输出、统一逆缩放。

**原核验阈值仍为rtol=1e-5、atol=1e-7，没有放宽，没有删除核验。** 若仍不一致，停止归因，记录实际最大误差、超限数量和最差位置。

补丁通过独立新入口启用修复，不修改原 `interpretation_random.py`、`models.py` 或旧源码指纹JSON。新增加载器源码、依赖版本和处理版本单独记录在 `inference_precision_fix.json`。因此现有模型的来源核验及缓存提交标记可以继续使用，无需清空结果目录。

已完成的DO归因保留并跳过重新计算；新计算记录会带有 `inference_implementation` 列，合并表区分旧DO结果与修复后推理结果，避免隐藏处理来源。

## 合并位置

把本文件夹内的内容合并到当前项目根目录：

`C:\path\to\jinjiang-water-quality`

| 新增文件 | 放置位置 |
|---|---|
| `src/jinjiang/interpretation_precision_fix.py` | 项目 `src/jinjiang/` |
| `experiments/06_interpretation_precision_fix.py` | 项目 `experiments/` |
| `run_06_interpretation_fix.ps1` | 项目根目录，与src、experiments同级 |
| `tests/test_interpretation_precision_fix.py` | 项目 `tests/` |

全部为新增文件。原06解释代码包和 `configs/interpretation_random.json` 仍需保留。

## 当前只做这一条：读取已有缓存，不训练

```powershell
powershell -ExecutionPolicy Bypass -File .\run_06_interpretation_fix.ps1 -Mode explain
```

它会核验已有DO、NH3N缓存，读取并跳过已完成DO的归因，尝试为NH3N完成置换归因。没有其他权重时列出缺失项，不会训练TP或CODMN。

预期看到类似：

```text
[precision check] C_NH3N_seed42 cached_inference: passed=True, max_abs=..., outside_tolerance=0
[interpretation] C_NH3N_seed42: complete
```

通过基准预测核验后还需要计算置换，因此两行之间可能等待一段时间。不要同时运行另一个写入相同结果目录的进程。

可选：先在现有GPU Python环境跑回归测试：

```powershell
$env:PYTHONPATH=Join-Path $PWD "src"
python -m unittest discover -s tests -p test_interpretation_precision_fix.py -v
```

其中一项微型神经检查仅在CPU训练两轮，不是重跑正式NH3N模型。

## 后续需要恢复下一模型时

使用修复入口：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_06_interpretation_fix.ps1 -Mode recover -MaxReplays 1
```

这条命令才允许实际训练，每次最多恢复一个缺失模型。已有DO、NH3N权重核验有效时会跳过恢复，接下来才是TP。不要先用这条命令解决当前缓存核验问题；先使用上面的explain命令。

默认GPU Python路径为 `.venv_gpu\Scripts\python.exe`，不同时加 `-Python "完整路径\python.exe"`。

## 诊断输出与验证范围

修复后沿用 `results/interpretation_random/06_interpretation_random`。新增 `inference_precision_checks` 内按模型保存核验JSON，记录：原阈值、最大绝对差、超限数量、最差位置、预测与参考值及恢复/缓存推理阶段。若explain仍报错，提供其中的 `C_NH3N_seed42_cached_inference.json` 即可继续定位，先保留所有权重。

6项回归测试通过，覆盖原缩放API精确一致、原测试张量布局、输入不被修改、旧精度错误复现、缓存模式不训练、严格拒绝不匹配、诊断输出和动态补丁退出后恢复。另1项微型CPU Torch权重加载核验因当前环境未安装Torch而跳过。已核对本项目原共享训练器的缩放和推理代码。当前没有取得此次NH3N权重文件，未在其GPU上复核；本补丁会在你的实际环境继续执行原严格核验，未通过就不会生成归因。

参考：[StandardScaler官方实现](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_data.py)。
