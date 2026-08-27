# Graphene 固定电子 smearing 下的有限晶格温度 full-EPC 结果

状态：`matrix_assembly_passed_finite_temperature_reference_pending`

## 计算定义

300 K 与 450 K 使用同一个 `smearing/degauss = 0.0019000869 Ry`。先从已经收敛的 SSCHA 自由能 Hessian 中精确扣除旧的 q6 电子算符，再在每个 q 点把当前 full-EPC 响应部分作为 A' 模的 Hermitian projector 加回。E42 用于静态绝对频率对齐的公共常数不属于 q6 算符，因此不重复加入。两条曲线之间只改变晶格温度背景。

## 主要数值

下表的 q 距离使用 `|q-K|/(2π/a)`，与旧 E47 晶格温度图一致；E42 内部 path coordinate 满足 `q distance = 2d/3`。

| lattice temperature | 方法 | K 频率 | KG, q=0.003 上升 | KM, q=0.003 上升 | KG, q=0.025 上升 | KM, q=0.025 上升 |
|---:|---|---:|---:|---:|---:|---:|
| 300 K | full-EPC LR | 1281.522 | 0.672 | 0.683 | 16.732 | 17.474 |
| 300 K | 旧 q6 | 1282.100 | 0.034 | 0.034 | 2.314 | 2.354 |
| 450 K | full-EPC LR | 1263.173 | 0.673 | 0.685 | 16.396 | 17.217 |
| 450 K | 旧 q6 | 1263.759 | 0.025 | 0.025 | 1.698 | 1.792 |

full-EPC 结果的 K 点 450−300 K 位移为 `-18.349 cm^-1`。

## 数值检查

- 静态 E42 曲线的矩阵级重放最大误差：`9.095e-13 cm^-1`。
- SSCHA 已保存 K 点频率重放最大误差：`4.547e-13 cm^-1`。
- SSCHA 已保存近 K 曲线的插值重放最大误差：`1.289e-02 cm^-1`。
- full-EPC 总矩阵 Hermiticity 残差：`0.000e+00`。
- time-reversal 频率残差：`0.000e+00 cm^-1`。
- K-star 频率 spread：`1.397e-07 cm^-1`。
- 241 点扩展响应相对独立 nk720 full-EPC 29 点的最大等效频率差：`0.550 cm^-1`。

## 适用范围

矩阵组合、双重计数消除、分支跟踪和对称性检查已经通过。现有 finite-smearing 静态晶格 DFPT 对 full-EPC 模型给出约 3.06 cm^-1 的绝对 RMSE 和 0.93 cm^-1 的 K-referenced shape RMSE；有限晶格温度曲线目前没有独立的有限温 DFPT/自洽 EPC 基准。因此这里可以确认计算链已经实现，不能把它写成有限晶格温度绝对精度已经由独立第一性原理数据验证。
