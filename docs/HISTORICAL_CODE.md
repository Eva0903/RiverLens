# 当前主线与历史入口

本仓库保留项目演变所需源码，但编号重复不代表同一实验。

| 历史入口/配置 | 定位 |
|---|---|
| `experiments/01_baseline.py`、`baseline_within_split_*.json` | 时间段内部窗口的旧基准协议；当前01采用`01_baseline_random.py` |
| `experiments/04_meteorology.py`、`05_robustness.py`、`paper.json` | 早期通用runner协议，不是当前随机版04/05 |
| `experiments/06_lstm_upgrade.py`、`lstm_upgrade_*.json` | 时间顺序划分下的残差LSTM项目 |
| `experiments/06_residual_random.py`、`residual_random_*.json` | 随机协议下的残差模型补充，未作为当前06解释分析 |
| `experiments/run_all.py` | 早期通用五实验入口；不要用它启动当前01—07 |
| `historical/do_upstream_ablation/` | 旧DO专门消融补丁，未确认正式结果；隔离归档，不合并到当前环境 |

旧DO补丁自带的`models.py`增加了锁定朴素基线的接口。当前01—07的已完成结果记录对应主项目原版`models.py`。为了保持其源码指纹和已存结果一致，本仓库将旧DO补丁隔离保存，未覆盖当前模型实现。若未来研究该补充实验，应在单独副本中按其原说明安装和验收。

`docs/history/`是既往说明的快照，其中的“本包”“尚未执行”“合并补丁”等描述属于写作当时，不表示此次发行包需要再次安装这些补丁。当前使用顺序以根README和`docs/RUNNING.md`为准。

此前两个梯度SHAP入口未纳入当前发行包。当前07采用无需梯度的分组Kernel SHAP；不要混用旧Expected Gradients结果。
