# Graphene leave-one-material-out prior 测试结果（2026-09-05）

对应周报 2026-08-19 下一阶段表中的 "leave-one-material-out gauge-invariant prior" 一项。

## 结论

LOMO latent 先验**未通过**迁移门槛，四个材料（graphene、1T-VSe₂、2H-TaS₂、2H-NbS₂）全部失败。按事先设定的分支，保持每体系五个 coarse-q EPC 标签的预算，不减少新体系的标签开销。

主要数值（holdout 相对顶角 RMSE）：

| 材料 | rank | B1（五标签） | B2(k=rank) | 先验 unit | 先验 scaled |
|---|---:|---:|---:|---:|---:|
| graphene | 3 | 0.00045 | 0.00084 | 2.881 | 0.269 |
| 1T-VSe₂ | 4 | 0.00480 | 0.01013 | 1.747 | 0.054 |
| 2H-TaS₂ | 4 | 0.00043 | 0.00102 | 1.687 | 0.044 |
| 2H-NbS₂ | 4 | 0.00054 | 0.00123 | 1.696 | 0.034 |

即使取 μ 网格上最优的尺度匹配先验，误差也是"直接降阶到 k−1 次 Chebyshev"的 28–320 倍。失败机制：k 个标签配四阶 Chebyshev 存在一维零空间，ridge 把该方向系数完全交给先验，而跨材料 latent 形状差异（K 点 A′ 与 TMD Σ 线模式）经振荡的 T₄ 型基函数直接进入插值曲线。

## 对后续工作的含义

- 五标签预算保持不变；"冻结后的新体系最小验证"仍按每体系五个 coarse-q EPC 标签规划（4–12 V100 GPU-h）。
- 阴性结果限定于"平均归一化 latent Chebyshev 系数 + ridge"这一先验族；以规范不变响应曲线为目标、或以电子结构描述符为条件的生成式先验未被本测试覆盖，零样本跨材料生成模型仍是开放问题。
- 附带观察：标签从 5 减到 rank 个，顶角 RMSE 只升约 2 倍；第五个标签的价值主要在补足四阶形状自由度。若未来需要压缩预算，B2 降阶比注入跨材料先验更安全。

详细协议、门槛与逐材料数值见
[`results/graphene_physics_temperature/post_p4_feasibility/E25_lomo_gauge_invariant_prior/RESULT.md`](../results/graphene_physics_temperature/post_p4_feasibility/E25_lomo_gauge_invariant_prior/RESULT.md)。
