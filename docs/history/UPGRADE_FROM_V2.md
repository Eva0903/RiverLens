# 从 v2 升级到 v3

## 推荐方式：并行保留

1. 保留原目录 `Jinjiang_revise_v2`，不要删除其中的 `results/paper/01_baseline`。
2. 将本包解压为相邻的新目录 `Jinjiang_LeakageSafe_V3`。
3. 在新目录运行安装、单元测试、Ridge 烟雾测试和 CUDA/LSTM 检查。
4. 从 v3 的 01 baseline 重新开始正式实验。

这是最安全的方式，也便于在论文修订中追溯为什么结果发生变化。

## 必须覆盖当前目录时

先复制整个 `results` 到项目外。然后用本包内容覆盖当前目录，并删除或重命名旧的 `results/paper`。不要保留旧 `completed_tasks.txt`，因为 v3 的数据处理和任务定义已经变化。覆盖后重新运行：

```powershell
python -m unittest discover -s tests -v
python experiments\run_all.py --config configs\smoke.json --model-override ridge --overwrite
python experiments\01_baseline.py --config configs\paper.json --overwrite
```

## 为什么不能直接续接旧结果

v3 改用因果填充；加入 24 h Seasonal naive、MASE 和技能指标；窗口实验改成统一 84 步支持集；rolling-origin 改为折内预处理；任务记录改为唯一 `task_id`。因此 v2 与 v3 的样本、基线、统计量和完成状态不是同一实验协议。

## 与旧命令的关系

五个入口名称保持不变，所以熟悉的命令仍然可用。区别是：第一次建立 v3 正式结果时使用一次 `--overwrite`，随后中断续跑时去掉该参数。看到 `skip completed` 表示续跑机制正常。
