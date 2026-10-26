# 从 V3 升级到 V4

## 保留已经完成的 01

V4 的新增入口是 `experiments/06_lstm_upgrade.py`。它不会修改 V3 的 01–05，也不要求重新运行已经完成的正式 01。建议将本包解压为相邻目录 `Jinjiang_LSTM_Upgrade_V4`，旧结果继续保存在 V3 中，V4 只生成新的 06 结果。

## 复用现有 CUDA 环境

不要为了 V4 再安装一份 CUDA PyTorch。可直接使用 V3 中已经验证过的解释器：

```powershell
$v3Python = "C:\path\to\previous-project\.venv\Scripts\python.exe"
& $v3Python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
& $v3Python experiments\06_lstm_upgrade.py --config configs\lstm_upgrade_smoke.json --overwrite
```

`06_lstm_upgrade.py` 会自动把当前 V4 的 `src` 放入模块搜索路径，因此调用的是 V3 环境里的第三方依赖，但运行的是 V4 项目代码和 V4 数据。

## 推荐运行顺序

```powershell
# 1. 三 epoch，验证完整代码和 GPU
& $v3Python experiments\06_lstm_upgrade.py --config configs\lstm_upgrade_smoke.json --overwrite

# 2. 仅在训练/验证段筛选，不读取测试目标
& $v3Python experiments\06_lstm_upgrade.py --config configs\lstm_upgrade_select.json --overwrite

# 3. 正式五种子测试；自动校验并复用第 2 步候选结果
& $v3Python experiments\06_lstm_upgrade.py --config configs\lstm_upgrade_paper.json --overwrite
```

运行中可按 `Ctrl+C`。恢复时重新执行同一条命令但去掉 `--overwrite`；程序会依据 `completed_candidates.txt` 和 `completed_final_tasks.txt` 跳过完整任务。不要把 V3 的 `completed_tasks.txt` 复制到 V4。
