# DO固定两组消融：交付验证记录

验证日期：2026-10-16。

- 真实数据共同样本：训练2417、验证375、测试549。
- C-only与A/B/C两组从同一个并集序列切列，增广输入维度6/10。
- 样本审计中每组cohort SHA256相同；完整审计见本文件末尾。
- 18项unittest通过，包括原有13项与新增5项。新增流程集成使用受控预测替身，不代表实际训练。
- Python编译检查通过。
- 已逐字节核对原压缩包内41个结果文件，全部保持不变。
- 此环境没有PyTorch，未进行真实神经网络训练或GPU检查；正式性能仍待用户本地运行。

以下是交付前的审计快照，仅供核对，不是需要续跑的训练输出。实际运行会在独立结果目录重新生成。

```json
{
  "shared_cohort_sha256": "e6298b1208271dd73ef01373de11b97767f54bed21b1faad0ab7a921d049b6a0",
  "n_common_samples": 3341,
  "per_arm_cohort_sha256": {
    "local_target": "e6298b1208271dd73ef01373de11b97767f54bed21b1faad0ab7a921d049b6a0",
    "cross_station_target": "e6298b1208271dd73ef01373de11b97767f54bed21b1faad0ab7a921d049b6a0"
  },
  "split_boundaries": {
    "train_end": "2023-05-26 20:00:00",
    "validation_end": "2023-09-13 08:00:00",
    "test_end": "2023-12-31 20:00:00"
  },
  "raw_columns": {
    "local_target": [
      "C_DO"
    ],
    "cross_station_target": [
      "A_DO",
      "B_DO",
      "C_DO"
    ]
  },
  "augmented_feature_count": {
    "local_target": 6,
    "cross_station_target": 10
  },
  "locked_baseline_names": [
    "persistence",
    "seasonal_naive",
    "seasonal_naive",
    "seasonal_naive",
    "seasonal_naive",
    "persistence"
  ],
  "first_differences": true,
  "calendar_features": true,
  "splits": {
    "train": {
      "n": 2417,
      "first_issue": "2022-01-01 20:00:00",
      "last_issue": "2023-05-25 20:00:00",
      "first_target": "2022-01-02 00:00:00",
      "last_target": "2023-05-26 20:00:00"
    },
    "validation": {
      "n": 375,
      "first_issue": "2023-05-27 12:00:00",
      "last_issue": "2023-09-09 08:00:00",
      "first_target": "2023-05-27 16:00:00",
      "last_target": "2023-09-10 08:00:00"
    },
    "test": {
      "n": 549,
      "first_issue": "2023-09-16 16:00:00",
      "last_issue": "2023-12-30 08:00:00",
      "first_target": "2023-09-16 20:00:00",
      "last_target": "2023-12-31 08:00:00"
    }
  },
  "assertion": "Both arms slice the same union sequence. No per-arm row filtering.",
  "scope": "Downstream-history-conditioned upstream information ablation; not pure-upstream prediction."
}
```
