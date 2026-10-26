# 数据与发布范围

本发行包只包含文本源码、配置、人工编写的文档及少量汇总性能数值。所有水质和气象原始Excel均未复制到仓库。没有模型权重、样本数组、逐样本预测、逐样本SHAP、训练结果包或已执行Notebook。

`predictions.csv`中含`y_true`，即使它位于`results/`，也属于不应公开的观测输出。本次同样排除了方向图数据中的历史观测均值、样本缓存、原始数据指纹清单，以及嵌有这些数值的演示文稿/图表源文件。你之前的PPT和完整结果应继续本地保存。

`docs/STUDY_AND_RESULTS.md`仅整理汇总指标与研究解释，不提供任何逐时浓度。测试夹具由代码生成，不取自研究数据，不能用作研究结果。

## 上传前检查

```powershell
python scripts\audit_public_release.py
```

若使用Git命令并已执行`git add .`，再检查实际暂存内容：

```powershell
python scripts\audit_public_release.py --staged
git diff --cached --name-only
```

检查器会拦截常见数据/二进制文件、私有输出目录和部分密钥模式，且不打印疑似密钥值。它是启发式检查，不能取代人工查看文件列表。`.gitignore`只影响尚未追踪的文件，不能移除已经提交的内容。不要使用`git add -f`绕过数据排除规则。

原项目和新仓库分开保存即可；不必删除本地数据、结果或环境。即使仓库设为Private，也继续排除上述文件。
