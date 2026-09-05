# E25 leave-one-material-out gauge-invariant prior 测试

状态：`LOMO_LATENT_PRIOR_TRANSFER_FAILED_KEEP_FIVE_LABELS`

## 问题

周报 2026-08-19 下一阶段表：检验五个 coarse-q EPC 标签的 gauge-invariant prior 能否跨材料迁移；未通过则保持每体系五点预算。

## 协议

只用已存储的顶角数组（graphene E17、VSe₂ E18+E19、TaS₂ E21、NbS₂ E24），不新增 DFT 或 q 标签。测试方案（含门槛和 μ 网格）先写入 `lomo_test_spec.json` 再计算。三组模型：

- **B1**：部署基线——五个标签、SVD rank r_m、四阶 Chebyshev latent（在五个标签上精确插值）。
- **B2(k)**：减标签基线——k 个等距标签、k−1 阶 Chebyshev 精确插值，不用先验。
- **P(k)**：先验正则——k 个标签、四阶 Chebyshev，系数 ridge 收向其余三个材料五标签拟合的平均归一化 latent 系数（LOMO）。两个变体：unit（先验单位范数）与 scaled（先验幅度匹配到本材料标签拟合系数范数）。每个变体取 μ 网格（10⁻⁶–10¹）上 holdout 最优值。

判据：G1 复现（B1 holdout 相对顶角 RMSE ≤ 0.02）；G2 迁移（每个材料 P(r_m) ≤ max(0.02, 1.5×B1) 且 P(r_m) ≤ B2(r_m)）。

## 结果（holdout 相对顶角 RMSE）

| 材料 | rank | B1（五标签） | B2(k=rank) | P unit | P scaled |
|---|---:|---:|---:|---:|---:|
| graphene | 3 | 0.00045 | 0.00084 | 2.881 | 0.269 |
| 1T-VSe₂ | 4 | 0.00480 | 0.01013 | 1.747 | 0.054 |
| 2H-TaS₂ | 4 | 0.00043 | 0.00102 | 1.687 | 0.044 |
| 2H-NbS₂ | 4 | 0.00054 | 0.00123 | 1.696 | 0.034 |

- G1 通过：B1 复现已发表门槛（graphene 与 E17 一致；VSe₂ 0.0048，优于 E19 rank-4 修复报告的 0.011；TaS₂/NbS₂ 与 E21/E24 同量级）。
- G2 四个材料全部失败：即使取最优 μ 的 scaled 先验，P 也是 B2 的 28–320 倍。

## 机制

k 个标签配四阶 Chebyshev 时设计矩阵有一维零空间；ridge 对任意 μ>0 都把零空间方向的系数完全交给先验（与 μ 大小无关）。该方向是振荡的 T₄ 型基函数，跨材料的 latent 形状差异（graphene K 点 A′ 与 TMD Σ 线模式物理不同）通过它直接进入插值曲线，造成百分之几十到几百的相对误差。先验幅度匹配只能把误差从 ~200% 压到 ~3–27%，仍远劣于直接降阶。

## 结论与适用范围

1. 本测试否决的是"其余材料的平均归一化 latent Chebyshev 系数"这一先验族；按周报预设分支，**保持每体系五个 coarse-q EPC 标签的预算**。
2. 附带观察：把标签从 5 减到 rank 个（B2），顶角 RMSE 只升约 2 倍；第五个标签的价值主要在补足四阶形状自由度。
3. 本阴性结果不排除其他形式的先验（例如以规范不变响应曲线而非 latent 系数为目标、或以电子结构描述符为条件的生成式先验）。零样本跨材料生成模型仍是开放问题。
4. 全部计算在存储数组上完成（本地 CPU，约一分钟）；未运行任何新 DFT/EPW。

## 产物

- 测试方案：`lomo_test_spec.json`
- 汇总：`lomo_summary.json`
- 脚本：`scripts/smearing_kink/analyze_lomo_gauge_invariant_prior.py`
- 文档：`docs/GRAPHENE_LOMO_PRIOR_RESULT_2026-09-05.md`
