# Graphene R2AT 全路径 ZA Γ 点下探诊断（2026-09-05）

## 1. 问题

R2AT 验收（`R2AT_paired_full_epc_acceptance_20260827`）在全 M–Γ–K–M 路径的 Fourier 插值谱上记录了 Γ 附近的 ZA 下探：验收摘要给出 `interpolated_full_band_min_frequency_cm-1_not_gated = -2.99 / -10.59 / -6.40 cm⁻¹`（300/450/600 K），并把它们标注为 Fourier 插值残差；稳定性门槛以 commensurate 超胞谱（≥ −1 cm⁻¹）另行定义并通过。本诊断验证该归因，并排除另外两个候选原因。

## 2. 方法

脚本 [`diagnose_r2at_za_asr.py`](../scripts/smearing_kink/diagnose_r2at_za_asr.py) 直接读取 S0 三个温度的 `formal_T{300,450,600}/result.npz` 中的 `free_energy_fc2_eV_A2`，用 SHA256 校验过的 T300 q6 operator（`114fb9a3…475c3`，本地副本 `data/graphene_r2c_eval/operators/T300_operator.npz`）重建 6×6 phonopy 超胞（原子位置最大偏差 3.8×10⁻¹⁵ Å），然后：

1. 检查力常数平移不变性（声学和规则，ASR）：`Σ_j Φ_ij` 的最大绝对值与刚体平移块本征值；
2. 检查 commensurate 谱与最小 commensurate 环 `q=(1/6,0)`、`q=(1/6,1/6)` 的 ZA 频率；
3. 在 Γ→K 方向做 200 点加密扫描，定位非公度 q 上的 ZA 最低值；
4. 用 phonopy 平移+置换对称化后重复上述检查。

单位说明：phonopy `run_qpoints` 返回 THz；乘 `33.35641` 得 cm⁻¹。换算后与存储曲线逐点一致（例如 Γ→K 上 t=1.0 的 ZA：存储 513.569 cm⁻¹，重算 513.5 cm⁻¹）。

## 3. 结果

| lattice temperature (K) | 存储曲线 ZA 最低 | 加密扫描 ZA 最低（ΓK 分数位置） | commensurate 谱最低 | ZA @ (1/6,0) | Γ 点 ZA | ASR 最大漂移 |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | −5.096 | −5.097（t=0.070） | −0.000 | 74.30 | 0.000 | 1.08×10⁻¹³ eV/Å² |
| 450 | −12.274 | −12.274（t=0.104） | −0.000 | 71.21 | 0.000 | 1.17×10⁻¹³ eV/Å² |
| 600 | −8.732 | −8.741（t=0.097） | −0.000 | 67.13 | 0.000 | 1.28×10⁻¹³ eV/Å² |

验收摘要中较浅的 −2.99/−10.59/−6.40 来自其 541 点重建路径的采样位置（path index 168/198/195），同一分支（branch 0）、同一现象。

三个候选原因的判定：

- **ASR 违反：排除。** 三温度 `Σ_j Φ_ij` 最大绝对值在 10⁻¹³ eV/Å² 量级，Γ 点 ZA 严格为 0；phonopy 对称化后路径最低值不变（T300 重算 −0.153 THz → 对称化后相同）。
- **物理负弯曲刚度：排除。** commensurate 谱最低值为 −0.000，最小 commensurate 环上的 ZA 为 +67.1 到 +74.3 cm⁻¹（随温度升高而软化，方向合理）。若自由能 Hessian 的 ZA 二次曲率为负，commensurate 点本身应为负。
- **Fourier 插值振铃：确认。** 负井只出现在 Γ 与第一 commensurate 环之间的非公度 q 上（ΓK 分数距离 t≈0.07–0.10），深度 −5 到 −12 cm⁻¹；插值在 commensurate 节点上精确再现正值（t=0.5 处插值 ZA 与节点值 196–197 cm⁻¹ 逐位一致）。6×6 超胞对第一环以内的 ZA 形状本身不含信息，trig 多项式在节点之间过冲为负。下探深度随温度非单调（450 K 最深），也与数值伪影而非物理趋势一致。

## 4. 处理建议

- 验收结论不受影响：稳定性门槛以 commensurate 超胞谱定义，三个温度均通过（无低于 −1 cm⁻¹ 的模）。
- 图层面如需消除下探，可将近 Γ 的 ZA 分支替换为 commensurate 锚定的二次外推（`ω²_ZA = c₂ q²`，`c₂` 由 `(1/6,0)` 环拟合，为正），或在图注保持现状（验收图已注明插值残差）。修改已冻结的验收图不属于本诊断范围。
- 根本缓解需要更大超胞（例如 8×8 会让第一 commensurate 环更靠近 Γ），代价是重新蒸馏与 SSCHA。

## 5. E44/E46 |d|>0.10 回摆检查

- E44 零展宽最终 dense curve（`zero_dense_curve.csv`，|d|≤0.15）：KG 方向各分量与总频在 0.10–0.15 全部单调；KM 方向总频单调（`long_range_correction_cm-2` 分量有一步幅度可忽略的回落，不改变总频）。
- E46 四联图自带的 monotonicity audit：显示范围（|d|≤0.10）非单调序列 0 个；计算诊断范围（到 |d|=0.15）非单调序列 11 个，正式图截断到 |d|≤0.10 处理正确。

两项均无需进一步修复。

## 6. 产物

- 诊断脚本：[`scripts/smearing_kink/diagnose_r2at_za_asr.py`](../scripts/smearing_kink/diagnose_r2at_za_asr.py)
- 数值输出：[`za_asr_diagnostic.json`](../results/graphene_physics_temperature/post_p4_feasibility/R2R_multipolar_background/R2AT_paired_full_epc_acceptance_20260827/za_asr_diagnostic.json)
- 输入：S0 `formal_T{300,450,600}/result.npz`、q6 operator（SHA256 `114fb9a3…475c3`）
