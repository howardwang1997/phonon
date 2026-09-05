# Graphene K cusp B0 dense-q 结果

**状态：**`passed_quantitative_dense`  
**范围：**29 个与 720×720 fine-k 网格严格可公度的 q 点；MLIP L0/Q0 背景加 K-$A_1'$ Hermitian rank-one EPW 修正。

## 指标

| T (K) | 背景 | direct cusp depth | q6 cusp depth | dense cusp depth | kink 相对误差 | K 频率偏移 |
|---:|---|---:|---:|---:|---:|---:|
| 300 | L0 | 2.351 | 0.111 | 2.072 | 11.88% | +1.296 cm⁻¹ |
| 300 | Q0 | 2.351 | 0.117 | 2.090 | 11.09% | -5.939 cm⁻¹ |
| 450 | L0 | 1.554 | 0.110 | 1.549 | 0.30% | -10.537 cm⁻¹ |
| 450 | Q0 | 1.554 | 0.116 | 1.572 | 1.16% | -23.456 cm⁻¹ |
| 600 | L0 | 1.162 | 0.106 | 1.205 | 3.74% | -19.749 cm⁻¹ |
| 600 | Q0 | 1.162 | 0.225 | 1.337 | 15.07% | -60.838 cm⁻¹ |

## 结论与限制

- dense q-space 结果是否通过，以 cusp depth、slope jump、矩阵 Hermiticity、time-reversal、K-star 和五个旧 pilot 点重放共同判断。
- 绝对频率相对 static-lattice DFPT 的偏移仍包含 lattice-temperature renormalization，不作为电子修正失败的判据。
- 本阶段的 6×6 矩阵是目标 $A_1'$ 模的 Hermitian rank-one 修正；没有给其他五个声子模加入未经验证的电子自能混合。
