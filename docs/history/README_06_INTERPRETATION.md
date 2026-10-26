# 06：气象收益条件与模型变量、历史时段依赖

本包承接01—05，分析已确认的气象收益，不运行已取消的残差模型升级。全部新增文件；原共享训练代码、04和05结果保留。

## 两个不同层面的结果

1. **零训练收益分层**：直接用05全部三站、四指标、五种子的成对预测，描述气象收益在温度、湿度、气压、风和降雨条件下怎样变化。已随包提供实际计算结果。这些是条件关联，不是变量重要性或某个时间段的因果效应。
2. **具体模型的置换归因**：恢复/加载模型后，联合置换一组变量或历史片段，计算测试MAE增加多少。它描述固定模型对输入的依赖，不证明真实水环境过程由该变量控制，也不提供物理降雨径流时滞。

04/05代码当时没有保存模型权重，上传结果只有预测与训练历史，因此不能只从这些CSV算出真实模型归因。本包**默认不训练**。缺少权重时，explain模式列出缺失项并记录 `waiting_for_checkpoints`，不会自动重训，不会用代理模型结果冒充LSTM归因。

## 范围与预算

| 项目 | 设置 |
|---|---|
| 零训练部分 | A/B/C三站×DO/NH3N/CODMN/TP，使用全部五种子及六个预测时效 |
| 首批模型归因 | C站四指标＋A/B站CODMN，共6个模型；固定种子42 |
| 模型与输入 | 已完成04/05的普通LSTM，三站水质水文＋气象＋降雨历史完整39列 |
| 窗口 | DO84、NH3N78、CODMN48、TP66步，沿用03选择，不重新优化 |
| 权重恢复 | 只有显式recover命令才会发生；按各原模型已记录的最佳轮数重放；不新增参数搜索或预算敏感性实验 |
| 原最佳轮数 | C/DO509、C/NH3N365、C/TP475、C/CODMN512、A/CODMN483、B/CODMN273，六个合计2617轮 |
| 每次恢复上限 | 默认1个模型，已完成的模型和归因自动跳过；可以分次执行 |
| 置换 | 固定10次，置换随机种子2026；同一次置换的同一供体窗口映射用于所有变量/片段 |

种子42取自原声明种子序列的首项，不按测试表现挑选。它不代表五种子归因稳定性。尤其C站CODMN：种子42的气象MAE收益约48.1%，五种子合并收益只有0.2%。A站CODMN种子42约14.1%，B站约−3.4%。因此单模型归因图**不能直接解释05的五种子平均站点差异**；跨站总体差异继续依据05和本包零训练的五种子汇总。每张归因表保留单种子与五种子表现及作用范围，避免混用。

如果将来需要稳健的跨种子归因，应另立扩展方案，明确恢复量；本包不自动扩展为30个模型，也不声称已完成五种子重要性分析。

## 零训练分层的计算

每个站点与指标继续使用05共同测试样本。每个种子先在六时效等权平均绝对误差，再在该条件下的共同预测起点和五个种子间平均，计算完整气象组相对水质组的MAE降幅。

历史段相对发布时刻定义为 `[0,24)`、`[24,72)`、`[72,168)` 和168小时及更早的输入行。最新输入行年龄为0小时；目标首个值发生在发布4小时后。各指标窗口长度不同，最后一段实际长度随窗口变化。

温度、湿度、气压和风速取段内平均，分层阈值为该站点、指标训练窗口的1/3和2/3分位数；阈值不根据测试表现调整。降雨取原始 `W_RAIN_4H` 在段内求和，累计量大于0为wet，否则为dry，不重复加上派生降雨列。测试起点少于15的分层标为 `small_stratum`，仅作提示，不自动删去。天气组之间和不同历史段之间存在相关性，分层不控制混杂，不作多重筛选后显著性宣称。

`gain_by_weather_condition.csv`保留全部672行分层结果，不只保留表现突出的条件。预测窗口重叠、站点支持不同、分层样本数不同；不将站点之间的绝对误差差异解释为气象效应。

## 变量组与历史段置换

| 组别 | 处理 |
|---|---|
| A、B、C站历史变量 | 各站变量整组置换，用于与气象组比较模型依赖 |
| 降雨 | 当前4小时雨量、四个显式滞后、12/24小时累计雨量，共7列联合置换 |
| 气温、湿度、气压 | 各自一列；与水温区分，水温属于站点组 |
| 风 | 最大风速、风向sin/cos共3列联合置换 |
| 全气象 | 13列气象及降雨历史联合置换 |
| 历史段 | 对每个历史段的全气象进行置换 |
| 变量×历史段 | 对五个气象变量组分别在四个历史段内置换 |

在测试起点之间交换完整组的历史序列，组内列使用同一供体窗口，不按单个时间点独立打乱。全窗口降雨组保持供体窗口内部的历史配置。分段置换会破坏跨段连续性和部分跨段派生量关系；跨变量相关性也可能被破坏。因此结果是模型扰动敏感性，不是物理可行的气象干预。相关变量的重要性可以被其他输入替代，不接近零也不等于具有独立因果贡献。

主指标 `MAE_increase = 置换后MAE − 原模型MAE`，正值表示模型依赖该组提供的信息，负值说明扰动后误差反而较低。记录各时效及六时效等权平均，辅以RMSE增加。每组保留置换重复标准差，它只衡量置换随机性，不是种子、训练、划分或未来泛化的不确定性。

各组有交叠，全气象包含多个组，历史段长度和变量数不同，不能把组重要性加总到100%，不能仅按最长段的重要性推断最优记忆长度。降雨特征自身包含滞后，输入行的年龄也不等于原始降雨观测年龄，两者更不能直接等同于径流进入河道的物理时滞。

方法依据：

- [scikit-learn置换重要性说明](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [PyTorch模型前向检查钩子](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html)

## 原模型恢复与一致性检查

复用原共享 `fit_torch_predict`，只将训练循环上限设置为已保存的最佳轮数；保持原输入、样本、两次随机划分、种子、骨干、优化、batch、缩放和早停规则。使用临时只读前向检查钩子取得最终恢复最佳权重后的模型引用，不修改输入或输出，finally中移除钩子。训练代码本身不改写。

恢复预测必须逐样本复现原测试预测，最佳轮和神经参数数也必须一致，否则阻止归因。原预测比较容差为 `rtol=1e-5, atol=1e-7`。记录最大差值、来源任务、数据/源码指纹、模型与缩放器哈希。每次加载缓存后再次核对原预测。缓存保存权重张量和NPZ缩放参数；加载权重使用 `weights_only=True`。

使用不同硬件或软件版本可能无法复现。检查失败时保留错误，不放宽检查后将另一套模型冒充原模型。

恢复是**实际重新训练**，不是从CSV瞬间还原权重。虽然它不搜索新模型，也仍然消耗GPU时间。默认overview、audit、explain均不会发生恢复。只有用户主动执行recover或显式添加 `--replay-missing` 才会训练。

## 安装与运行

将本文件夹内的src、experiments、configs、tests以及PS1脚本合并到项目根目录：

`C:\path\to\jinjiang-water-quality`

`run_06_interpretation.ps1`与src、experiments同级。入口为 `06_interpretation_random.py`；已取消的 `06_residual_random.py` 不用运行。依赖已成功完成05的代码和原04、05结果，无需安装SHAP。

在项目根目录打开PowerShell：

```powershell
# 核验六个案例的样本、原任务及权重缺失情况；不训练
powershell -ExecutionPolicy Bypass -File .\run_06_interpretation.ps1 -Mode audit

# 五种子收益条件分层；不训练。此部分实际结果已随包提供。
powershell -ExecutionPolicy Bypass -File .\run_06_interpretation.ps1 -Mode overview

# 只解释已经存在于本包缓存中的权重；缺失时列出缺失项，不训练
powershell -ExecutionPolicy Bypass -File .\run_06_interpretation.ps1 -Mode explain
```

**以下命令会训练，属于可选步骤**。每次最多恢复1个缺失模型，然后生成其归因；再次执行会保留已完成项，恢复下一个缺失模型：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_06_interpretation.ps1 -Mode recover -MaxReplays 1
```

默认GPU Python为 `.venv_gpu\Scripts\python.exe`，路径不同时加 `-Python "完整路径\python.exe"`。如需一次恢复全部六个模型，可显式设置 `-MaxReplays 6`，本包不会自动这样执行。

也可直接运行：

```powershell
$gpuPython=Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH=Join-Path $PWD "src"
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
& $gpuPython experiments/06_interpretation_random.py --config configs/interpretation_random.json --mode overview
```

Python入口支持 `--mode audit|overview|explain`；只有explain允许 `--replay-missing`；`--max-replays`默认1，`--max-cases`可限制归因案例数。无覆盖选项，数据、源码、来源或配置改变时需新的结果目录。

## 输出

位于 `results/interpretation_random/06_interpretation_random`。

| 文件 | 含义 |
|---|---|
| `case_audit.csv` | 六个模型是否有缓存、原最佳轮数、所选种子收益与五种子收益 |
| `five_seed_gain_overview.csv` | 三站四指标的12行五种子总体收益 |
| `gain_by_weather_condition.csv` | 五气象条件×历史段分层收益，672行 |
| `importance_summary.csv` | 已完成的具体模型归因，区分变量组、时段、变量×时段 |
| `attributions/*/permutation_raw.csv` | 各次置换和各时效的原始结果 |
| `checkpoints/*/` | 恢复后的权重、缩放参数与来源，后续不必重复恢复 |
| `run_status.json` | 当前模式、实际恢复数、完成和缺失案例；零训练完成不代表模型归因完成 |
| 配置/来源/指纹JSON | 样本、代码、数据与结果来源核对 |

归因按模型原子提交：CSV及其哈希标记完整时跳过重算。权重及缩放文件写完后才提交元数据。保留本地checkpoints与attributions目录才能续跑；中断中的一次恢复从头开始，不做逐epoch恢复。

## 随包结果与验证范围

`reference_outputs`内是当前05上传数据已经算出的零训练结果，含12行总体、672行天气条件分层、6行案例审计和简短解读。没有SHAP值、模型权重或真实LSTM置换重要性，不能把这些分层表称为变量归因。

8项自动检查中7项通过；1项微型CPU Torch“原训练—捕获—存权重—加载—复现预测”检查因当前环境未安装Torch而跳过。其余覆盖分组、年龄轴、不修改输入、固定随机数、训练集分层阈值、默认零训练、显式恢复上限与归因续跑、原预测不一致时阻止归因。已完成三站四指标真实数据零训练流程，12项总体收益与05官方汇总一致到0.001个百分点内；源训练函数被替换为一旦调用就报错的保护，确认该流程没有训练。

当前环境未执行GPU权重恢复及真实LSTM归因。建议先在现有GPU Python运行测试，微型Torch检查仅使用CPU两轮；然后按需恢复第一个原模型。

```powershell
$gpuPython=Join-Path $PWD ".venv_gpu\Scripts\python.exe"
$env:PYTHONPATH=Join-Path $PWD "src"
& $gpuPython -m unittest discover -s tests -p test_interpretation_random.py -v
```
