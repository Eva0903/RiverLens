# 07：原模型直接预测的分组Kernel SHAP，独立版

## 为什么换方法

用户提供的梯度诊断中，差分步长缩小后导数估计不收敛，部分样本发生符号变化。例如4h、direction0、第3个样本，中心差分依次为0.0138447、0.0145018、-0.0195920。这不能证明解析梯度错误，也不足以锁定具体CUDA精度设置，但说明该差分诊断当前不可靠。

因此本包换用**直接调用原模型预测的Kernel SHAP**，不再采用前两个07入口的Expected Gradients或自定义反向适配器。本版不调用梯度、不启用训练模式、不切换原生LSTM、不修改原预测阈值、不重新训练。

这是解释方法和解释单元的明确调整，不是关闭失败检查。原模型预测仍必须与04保存结果在`rtol=1e-5, atol=1e-7`下吻合。

## 合并与运行

本独立版已包含`shap_kernel_common.py`，不需要旧07的`shap_supplement.py`或两个梯度修复模块。仍需保留现有01—06项目源码、原始数据、04/06结果与06精度修复模块。

把本文件夹**内部内容**合并到项目根目录：

```text
C:\path\to\jinjiang-water-quality
```

`run_07_shap_kernel.ps1`放在与`src`、`experiments`同级的位置。遇到同名文件选择替换；新模块`shap_kernel_common.py`必须进入`src/jinjiang/`。保留此前错误报告和输出目录。独立版使用新的结果目录，避免旧协议指纹冲突。

```powershell
# 检查原权重、数据、样本与原预测；不求梯度，不计算SHAP
powershell -ExecutionPolicy Bypass -File .\run_07_shap_kernel.ps1 -Mode audit

# 独立冒烟：C站DO、1个测试窗口、4个训练背景、64个联盟采样；六时效同时输出
powershell -ExecutionPolicy Bypass -File .\run_07_shap_kernel.ps1 -Mode smoke

# 正式归因；中断后相同命令续算
powershell -ExecutionPolicy Bypass -File .\run_07_shap_kernel.ps1 -Mode run
```

你的环境已有Torch及SHAP 0.51.0，通常无需安装依赖。不应重新安装GPU Torch。默认仍使用原CUDA环境；没有CUDA时直接报错，不静默改用CPU。

正式结果独立保存在：

```text
results/shap_kernel_standalone/07_kernel_shap
```

如果希望先只计算一个新的“测试窗口×背景”组合：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_07_shap_kernel.ps1 -Mode run -MaxNewOrigins 1
```

重复该命令会跳过已验证缓存，继续下一个缺失组合；去掉限制可继续完成全部。每个组合一次输出六个时效，不训练模型。Kernel SHAP需要大量模型预测调用，可能耗时较长，程序逐窗口打印进度并逐窗口保存。

## 解释对象与完整历史分组

模型仍是C站DO/NH3N/CODMn/TP的种子42完整输入模型，窗口保持84/78/48/66步，输入覆盖全部39列。

| 解释单元 | 数量 | 替换规则 |
|---|---:|---|
| A/B/C站水质水文变量 | 26 | 每个变量整个历史序列一起替换 |
| 气压、湿度、气温 | 3 | 每个变量整个历史序列一起替换 |
| 降雨及其历史衍生变量 | 1 | 原7列、整个历史一起替换 |
| 风速及风向正余弦 | 1 | 原3列、整个历史一起替换 |
| 合计 | 31 | 39个输入列各出现一次，无遗漏或重复 |

降雨滞后/累计列及风向编码不拆开独立替换，避免破坏同一组内部的编码联系。变量轨迹在组内完整保留；不同组之间的相关关系仍会被背景替换打破，不能把这种解释当作物理干预。

**本版是31个完整历史解释单元的组Shapley归因，不是原39列逐历史步Expected Gradients，也不是把旧SHAP值相加。** 两种方法的数值不可直接混合比较。07现在回答变量/变量组的预测方向；历史时段敏感性继续由06置换结果回答，本包不输出虚构的逐时段SHAP。

## 背景与Kernel SHAP设置

- 各模型从原训练集不放回抽取64个完整窗口，背景种子2026、3407、99。
- 解释原测试窗口全部：DO82、NH3N88、CODMn129、TP102。
- 4模型×3背景＝12个背景任务，每个任务同时生成六时效，共72组结果视图。
- 每个待解释窗口使用512个Kernel SHAP联盟采样预算，不是512次训练。实际独立联盟数保存在缓存中。
- Kernel的稀疏特征选择显式关闭`l1_reg=0.0`，不会沿用新版SHAP默认仅选择10个变量的设置。
- Kernel回归设计必须满秩，输出必须有限，并核验所有组贡献加背景预测等于原预测。
- 加和约束来自Kernel SHAP的加权回归，**加和通过不证明估计已收敛或结果是精确Shapley值**。这是有限联盟采样的近似解释。三个背景重复主要反映背景选择敏感性，不是完整的联盟采样误差、训练种子误差或置信区间。

定义的合作博弈为：

```text
v(S) = 对当前训练背景逐窗口替换缺失组后，原模型预测的平均值
空集合 = 当前训练背景上的实际原模型平均预测
全集合 = 已经核验的当前测试窗口的实际原模型预测
```

全集合预测直接复用已经计算并核验的原模型输出缓存。不存在输出偏置补偿、跨内核拼接梯度或把预测强行改成参考数值。

## 输出与判读

| 文件 | 内容 |
|---|---|
| `run_status.json` | audit_complete、smoke_only、partial或complete |
| `case_audit.csv` | 原权重与原预测核验、样本数、解释单元数 |
| `C_*_players.json` | 每个解释单元对应的全部输入列及颜色变量 |
| `C_*_inference_check.json` | 原预测一致性检查，标记gradient_calls=0 |
| `feature_by_background.csv` | 各背景、各时效、各解释单元的结果 |
| `feature_summary.csv` | 三个背景汇总的mean_abs_net_SHAP、mean_signed_SHAP与background_draw_sd |
| `group_summary.csv` | 按站点和气象家族汇总的有符号组贡献 |
| `tasks/*/local_contributions.csv` | 各样本在六个时效上的有符号贡献 |
| `tasks/*/origin_*.npz` | 31×6 SHAP值、基线/预测、联盟掩码、联盟输出、核权重与回归秩 |
| `figures/*importance.png` | 各指标4h和24h全局重要性 |
| `figures/*direction.png` | 各指标4h和24h有符号贡献分布 |

正SHAP表示模型输出高于所选背景，负值表示低于背景。它不表示预测误差、不表示新增变量收益，也不是改变环境变量的因果效果。

方向图颜色：普通变量采用整段历史均值；降雨组采用原始4h降雨序列的历史均值；风组采用风速序列的历史均值。颜色只展示关联，不能用一个均值代表整段多变量历史。

正式汇总应具有3个背景重复，且12个背景任务均完整。局部限量运行只保存已经完整的背景任务汇总，未完成背景的原始窗口缓存会保留。完整状态不包括尚未完成的任务。

每个窗口的缓存有协议/数据/权重和内容哈希。发生中断直接重新执行相同命令。若进程被强制终止并残留新07目录的`.run.lock`，确认进程已结束后仅移除该锁。不要删除原权重、04/05/06结果或修改旧协议记录。

## 验证状态

本次CPU环境下8项自动测试通过，包含：分组完整覆盖、相关编码和历史轨迹共同替换、已知线性函数SHAP、与穷举非线性组Shapley结果对照、不使用默认前10变量筛选、协议约束、旧07模块完全不可导入时的独立导入回归检查、禁止梯度调用情况下的真实SHAP及续算验收。续算测试确认权重文件不变。

另使用四个上传的真实C站权重，在合成输入上完成无梯度Kernel SHAP检查。详见`VALIDATION_KERNEL.md`。未在用户本机真实原始数据和CUDA上执行；需依次运行audit和smoke。

```powershell
$env:PYTHONPATH = Join-Path $PWD "src"
python -m unittest discover -s tests -p test_shap_kernel_supplement.py -v
```

Kernel SHAP官方文档：https://shap.readthedocs.io/en/stable/generated/shap.KernelExplainer.html
