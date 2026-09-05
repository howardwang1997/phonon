# Graphene K cusp A0 可行性结果

**状态：**`passed_feasibility`  
**范围：**已有数据上的五点 mode-projected q-space 验证；尚不是完整 Hermitian dense-q 曲线。

## 数值结果

| T (K) | 背景 | direct cusp depth | q6 cusp depth | q-space cusp depth | kink 相对误差 | K 频率相对 static DFPT 偏移 |
|---:|---|---:|---:|---:|---:|---:|
| 300 | L0 | 2.344 | 0.111 | 2.072 | 11.62% | +1.294 cm⁻¹ |
| 300 | Q0 | 2.344 | 0.117 | 2.090 | 10.82% | -5.942 cm⁻¹ |
| 450 | L0 | 1.548 | 0.109 | 1.549 | 0.04% | -10.539 cm⁻¹ |
| 450 | Q0 | 1.548 | 0.115 | 1.572 | 1.50% | -23.458 cm⁻¹ |
| 600 | L0 | 1.158 | 0.106 | 1.205 | 4.10% | -19.750 cm⁻¹ |
| 600 | Q0 | 1.158 | 0.225 | 1.337 | 15.46% | -60.840 cm⁻¹ |

## 阶段性结论

- direct DFPT 在三个 `smearing/degauss (Ry)` 条件下均给出正的 K 点局部下凹。
- 有限 q6/实空间电子算子只留下很浅的圆滑下凹；把同一电子响应直接放回 q-space 后，L0 和 Q0 的 cusp depth 与 slope jump 均恢复到 direct DFPT 的量级。
- 所有六个 T×背景组合的 kink 相对误差都低于 20%，A0 shape gate 通过。
- 450/600 K 的绝对频率与静态晶格 DFPT 有明显偏移，主要来自 L0/Q0 的晶格温度重整化；该偏移不是本 A0 的电子 cusp 误差，不能用静态 DFPT 门槛直接判定。
- 下一步需要生成与 720×720 fine-k 网格严格可公度的 29 点 Hermitian EPW 修正，验证连续曲线、模式混合、time-reversal 和 K-star 对称性。

## 图

- `graphene_k_cusp_a0_absolute.png`：绝对频率。
- `graphene_k_cusp_a0_centered.png`：各曲线减去自身 K 点频率后的 cusp 形状。

## 适用范围

A0 使用 EPW 已通过静态门槛的最高模标量自能修正。它证明了 q-space 表示可以恢复 cusp，但尚未证明完整 6×6 Hermitian 修正在任意 dense q 点都满足矩阵级重放和对称性门槛。
