# Jinjiang Water-Quality Forecasting

**Diagnosing Indicator-Specific Temporal Dependence in Upstream-Informed Deep Learning for Urban River Water Quality Forecasting**

城市河流水质预测中的指标差异化信息需求：上游观测、历史配置与气象输入。

本仓库整理当前01—07研究代码。预测指标为DO、NH₃-N、CODMn和TP；主研究对象为下游C站，05补充实验分别以A、B站为预测目标。时间步长为4小时，主要预测时效为4—24小时，03另外考察至72小时。

**仅发布代码、配置和说明，不提供原始观测或可还原逐时观测的实验输出。** 数据不可公开，因此下载代码不能直接复现论文数值；单元测试使用程序生成的合成数据。研究仍处于海报及论文整理阶段。

## 研究主线

先比较模型及上游信息增量，再选择各指标的历史配置，检验气象输入的指标与站点差异，最后分别用置换和SHAP分析误差敏感性与预测归因。

| 实验 | 研究问题 | 当前入口 | 主配置 |
|---|---|---|---|
| 01 | 哪个模型适合作为统一诊断基准？ | `01_baseline_random.py` | `baseline_random_gpu.json` |
| 02 | 本站历史之外，上游观测是否仍有预测增量？ | `02_station_ablation.py` | `station_ablation_random_gpu.json` |
| 03 | 各指标需要怎样的历史配置？ | `03_window_horizon.py` | `window_horizon_random_main_gpu.json` |
| 04 | 不同指标如何响应气象输入及其组合？ | `04_meteorology_random.py` | `meteorology_random_gpu.json` |
| 05 | 气象增益在不同目标站点是否一致？ | `05_station_rotation_random.py` | `station_rotation_random_gpu.json` |
| 06 | 误差对哪些变量组和历史时段敏感？ | `06_interpretation_precision_fix.py` | `interpretation_random.json` |
| 07 | 各变量组对模型预测的归因贡献有何差异？ | `07_shap_kernel.py` | `shap_kernel.json` |

所有入口位于`experiments/`，配置位于`configs/`。统一选择入口为`run_study.py`。

## 必须保留的评价边界

当前01—05主线使用**重叠窗口的80/10/10随机划分**。不同集合可能共享原始输入或目标时间点；结果不能作为时间独立未来预测、跨流域泛化或实际预警能力的证明。五次不同随机初始化训练反映固定划分下的训练变异。

03的近优窗口是验证集配置，不能直接解释为物理记忆长度。05为每个目标站点重新训练，输入仍包含三站历史，不是模型直接迁移。06—07分析固定的一次训练结果；07的三套训练集参考样本检验背景选择敏感性，不是三次独立模型训练。归因结果不构成环境过程的因果证据。

历史时间顺序划分与残差LSTM代码仍保留，但其结果不能直接拼接到当前主线。见[历史入口说明](docs/HISTORICAL_CODE.md)。

## 安装和无数据检查

建议Python 3.11或3.12，源码最低声明为3.10。CPU本次检查环境与验证范围见[验证记录](docs/VALIDATION.md)。CUDA实验应优先复用已经验证过的原GPU环境，避免替换已有PyTorch。

在项目根目录的PowerShell中：

```powershell
# 新建环境时使用；已有GPU环境可跳过这两行
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -e .
python scripts\audit_public_release.py
python -m unittest discover -s tests -v
python run_study.py --experiment 01 --mode run --dry-run
```

`--dry-run`仅显示命令，不访问数据，也不训练模型。单元测试将合成数据放入临时目录并自动清理；不需要研究观测文件。若PowerShell不允许激活，可直接使用`.\.venv\Scripts\python.exe`代替`python`。

## 使用本地授权数据

四份Excel由你在本地放入`data/raw/`。文件名、列名和时间规则见[数据接口](docs/DATA_SCHEMA.md)。整个数据目录及实验输出默认由`.gitignore`排除。

```powershell
# 先审计、再小规模检查、最后正式运行；每次只选一个实验
python run_study.py --experiment 01 --mode audit
python run_study.py --experiment 01 --mode smoke
python run_study.py --experiment 01 --mode run
```

02—05可用相同格式；03预算诊断、06权重恢复及07依赖顺序见[运行手册](docs/RUNNING.md)。**已有结果无需为了上传GitHub重新训练。** 保留原项目及其环境；本代码包应解压到一个新的独立目录。

## 目录

| 路径 | 内容 |
|---|---|
| `src/jinjiang/` | 数据处理、模型、评价、实验和归因实现 |
| `experiments/` | 当前及历史实验命令入口 |
| `configs/` | 主实验、小规模检查和训练预算诊断配置 |
| `tests/` | 合成数据及受控测试，不含真实观测 |
| `scripts/` | 发布文件检查、合成数据生成 |
| `docs/` | 研究设计、结果摘要、运行及GitHub教程 |
| `docs/history/` | 既往版本说明，供追溯，不作为当前操作顺序 |
| `historical/do_upstream_ablation/` | 隔离保存的旧DO专门消融补丁 |

## 进一步说明

- [研究设计与结果摘要](docs/STUDY_AND_RESULTS.md)
- [逐实验运行手册](docs/RUNNING.md)
- [Windows建立GitHub仓库教程](docs/GITHUB_GUIDE_ZH.md)
- [数据与发布范围](docs/DATA_POLICY.md)
- [源码来源与版本对应](docs/PROVENANCE.md)

本包没有预设开源许可证。首次上传建议设为Private；作者确认代码权属与共享方式后，再决定公开及许可证。参考[GitHub关于无许可证仓库的说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。
