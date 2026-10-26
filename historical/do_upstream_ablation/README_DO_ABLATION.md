# V4 DO上游增量信息消融：独立运行补丁

适用项目：用户于2026-10-16提供的 `Jinjiang_LSTM_Upgrade_V4.zip`。
目的：在下游DO历史可用的条件下，比较C-only与A/B/C三站DO历史，检验上游的额外预测价值。此实验不是纯上游预测，也不用于证明残差LSTM优于普通LSTM。

## 安装

先保留原始V4压缩包。将本补丁压缩包内的 `experiments`、`configs`、`src`、`tests` 和说明文件复制到现有V4项目根目录，合并同名文件夹。

新增入口和配置不会替换旧文件。仅 `src/jinjiang/models.py` 是对原文件的更新：在残差LSTM函数中加入可选的 `locked_baseline_names` 关键字参数。旧01–06调用不传此参数，仍使用原有的验证集选基线流程。本补丁保留其余原函数代码。

压缩包不包含原始数据或旧结果，也不包含模拟训练结果；继续使用现有项目的 `data/raw`。

## 按顺序运行

在V4项目根目录、原有Python环境中执行。

### 1. 只检查共同样本，不训练

```powershell
python experiments/07_do_upstream_ablation.py --audit-only
```

上传数据的已核验计数应为训练2417、验证375、测试549。若原始数据、预处理或协议变化，计数可能变化，请检查审计文件，不要为匹配数字调整样本。

### 2. 先运行seed=42的两组

```powershell
python experiments/07_do_upstream_ablation.py --seeds 42
```

这里使用正式96轮上限和12轮早停，不是2轮冒烟试跑。只用于确认正式运行可完成；不要根据seed=42的测试表现修改模型或选择输入。

### 3. 续跑其余种子并汇总

```powershell
python experiments/07_do_upstream_ablation.py
```

已有完整任务自动跳过。默认种子42、2026、3407、7、99，每种子两组，共10个任务；每个任务还包含验证选轮数后在训练+验证集上的重训。与V4原实现一致，不要将此处10个任务误认为只有10次网络初始化。

若原先使用 `$v2Python`，将上述命令中的 `python` 替换为 `& $v2Python` 即可。

可选：先做两轮的流程试跑（独立输出目录，不能作为性能证据）：

```powershell
python experiments/07_do_upstream_ablation.py --config configs/do_upstream_ablation_smoke.json
```

## 固定实验协议

- 目标C站DO，4小时采样，历史6步，直接输出未来6步。
- 两组原始输入分别为C_DO与A_DO/B_DO/C_DO；均增加自身输入的一阶差分和4个日历特征，因此输入维度分别为6和10。
- 只调用一次 `make_sequences`：先对两组原始列的并集过滤缺失、缺失目标和跨分割预测块，再从同一个序列张量切列。两组没有各自重新选样本的步骤。
- 共同池以未来6步观测目标均有效为条件；标签沿用原始观测值，不把填补目标作为评分标签。
- 时间划分及分割内因果前向填补继承V4。训练/验证/测试预测标签块互不重叠；验证和测试输入允许使用其发布时刻之前已获得的历史观测，这属于滚动发布预测，不是未来信息。
- 共用下游历史构造的简单基线。4/8/12/16/20/24小时分别固定为 persistence/seasonal_naive/seasonal_naive/seasonal_naive/seasonal_naive/persistence。这是V4已选DO方案，不会在测试集上重新挑基线。
- 两组均为1层32隐藏单元、dropout0.1、Huber损失、同一训练预算。各自在相同验证集上早停和校准门控；规则相同，不要求门控数值和最佳轮数相同。
- 隐藏层规模固定，输入增加会使参数量增加，不能声称两组参数量完全相同。实际参数量保存于各任务diagnostics.json。
- 不进行窗口或特征候选筛选，不读取旧64候选结果作为新消融结果。

## 输出位置与读法

正式目录：`results/do_upstream_ablation/07_do_upstream_ablation/`。

| 文件 | 用途 |
|---|---|
| common_samples.csv | 唯一样本清单：sample_id、分割、发布/目标时间、六步观测目标 |
| sample_audit.json | 三段计数、边界、原始输入列、增广维度及两组共同样本哈希 |
| protocol.json | 固定配置、代码/数据指纹和依赖版本 |
| tasks/组名__seed种子/ | 每任务预测、指标、历史、诊断和完成标记 |
| metrics.csv、metrics_mean_sd.csv | 三种简单基线与两组模型的逐时效指标及种子均值/标准差 |
| predictions.csv | 仅包含已经完成两组的配对种子 |
| paired_seed_horizon.csv | 同种子同预测时效的MAE/RMSE降低、R²增加和MAE技能增益 |
| paired_seed_summary.csv | 每种子六时效平均MAE技能增益 |
| paired_time_block_ci.csv | 按实际4小时网格进行时间块bootstrap的逐时效MAE降低区间 |
| primary_endpoint.json | 主终点及决策状态；未跑齐种子时仅为pending_all_seeds |
| run_status.json | complete或partial以及已完成的配对种子 |

正的reduction/gain表示加入上游有帮助。技能增益使用同一锁定基线：

`平均技能增益 = mean_h[(MAE_C-only,h − MAE_ABC,h) / MAE_locked_naive,h]`。

主终点等权平均六时效，先平均随机种子损失，再按真实时间网格抽取连续块；缺失时段以NaN保留，避免将相隔很远的有效样本硬拼为相邻4小时。默认块长6步、重复2000次。时间块区间不是独立时间外部验证，也不覆盖所有训练随机性；种子离散程度另报。

五种子完整运行后，主终点正向且95%时间块区间下限>0、至少4/5种子同向，才给出当前单时间分割支持上游增益的标记。逐时效结果仍须完整展示，不能据测试结果挑选时效重新宣称全时效提升。此条件是预设的研究判据，不是独立外部验证或多重比较校正。

## 续跑与保护

- 不提供 `--overwrite`，不会删除旧01–06结果。
- 结果父目录限制为项目根目录下 `results/do_upstream_ablation` 或以该名称开头的新目录；不能指向 `results/lstm_upgrade_paper`。
- 配置、源码或数据变化时拒绝混用已有任务；请在配置中改用新的 `results/do_upstream_ablation_...` 目录。
- 安装PyTorch可在仅做样本审计之后进行；一旦有训练任务完成，续跑会检查依赖版本。
- 每任务先写入 `.incomplete` 临时目录，完整写出后才发布完成标记；再次运行会重做未完成任务。汇总仅包含两组都完成的种子。
- 同一结果目录不要同时启动多个进程。
- 如需中断后继续，保持配置和代码不变，再运行同一命令即可。

## 验证边界

本次已使用上传的真实数据验证共同样本，并运行包含原数据管线测试及新增消融测试的unittest检查。当前交付环境没有安装PyTorch，因此未执行真实LSTM训练、GPU兼容性和五种子正式实验。新增集成测试使用受控预测替身，验证锁定基线传递、配对统计方向、两阶段续跑和旧目录保护；不代表模型效果。

本地复验：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
