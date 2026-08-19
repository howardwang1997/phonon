# Graphene 长程模型结构调整记录（2026-08-18）

## 研究决定

当前不把增加 DFPT k 网格或扩大 DFT 训练集作为模型开发主线。更密网格可以降低参考值不确定性，但不能修复长程项把 K 点 cusp 过度平滑的问题。短程 MLIP 暂时固定，新计算只在模型冻结后用于少量独立测试。

前期用于判别结构的两标量基模型写成

\[
D(q,s)=D_{\mathrm{MLIP}}^{\mathrm{SR}}(q)
+U_{A'}(q)\left[c_K(s)+a_1(s)f_1(q,s)+a_2(s)f_2(q,s)\right]U_{A'}^\dagger(q),
\]

其中 `s` 直接使用 `smearing/degauss (Ry)`。这套表达用于确认矩阵 rank 和标量 q-rank，不再作为最终物理核。完整 EPC 重放通过后，现阶段主模型改为

\[
D(q,s)=D_{\mathrm{MLIP}}^{\mathrm{SR}}(q)
+U_{A'}(q)\left[c_K(s)+\alpha\,\Delta\Pi_{A'}(q,s;s_{\mathrm{ref}})\right]U_{A'}^\dagger(q),
\]

其中 `c_K` 单独控制 K 点锚值，`ΔΠ_A′` 由 Wannier 能带、化学势、Fermi–Dirac 占据和 A′ EPC 顶角直接求和。有限展宽重放给出的全局幅度比例为 `0.9917`，因此 `α` 只允许作为接近 1 的校准量，不再用自由 crossover 拟合 cusp。

为了迁移到新体系，A′/目标软模的有效 EPC 顶角进一步写成

\[
g_\nu(k,q)\approx\sum_{r=1}^{R\le 4} z_r(q)B_r(k),
\qquad z_r(q)=\mathrm{Chebyshev}_4(q),
\]

其中材料可以自动选择 `R=1…4`，但不允许通过增加 dense-q 标签无限扩大 rank；在当前一维 q 窗口内，五个 coarse-q 模式投影 EPC 标签确定 `B_r` 和四阶 q-latent。smearing 只进入显式 band-sum 的 Fermi–Dirac 占据，不进入短程 MLIP，也不为每个 smearing 重新训练顶角。K/K′ 时间反演和 C3/C6 点群矩阵表示已经实现；任意二维偏移的 q-star 几何可由 canonical wedge 精确生成，尚缺的是 canonical wedge 内物理 EPC 顶角的二维生成模型。

## 已完成的无新增 DFT 实验

### 1. K-star 矩阵低秩检查

使用已有 300/450/600 K EPW Cartesian 算子，检查 K 和 K′ 的完整 6×6 修正矩阵。

- A′ rank-one 的最小 Frobenius capture：`99.9833%`。
- 修正落在 bare A′ 顶模上的最小比例：`99.9833%`。
- 三个 smearing 之间主投影器最小重叠：`0.9999999999999996`。
- 温度差分算子的 rank-one capture：至少 `99.9712%`。
- Hermitian bare-mode basis 中的对角比例：`1.0`。

结果支持继续使用 A′ rank-one 矩阵修正。目前没有数据支持扩大为 rank-two phonon projector，也没有理由先增加短程 MLIP 容量。

结果：`results/graphene_physics_temperature/post_p4_feasibility/E3_epw_kstar_low_rank/`

### 2. 零/有限 smearing 联合物理核

旧的单尺度 `logcosh` 核把 K 锚值、cusp 振幅和热圆化宽度耦合在一起。新候选将这三项分开，并满足：

- `smearing/degauss = 0` 时具有有限的单侧线性斜率；
- 有限 smearing 时在 K 点附近为二次圆化，离开热宽度后恢复近线性；
- 解析的 `r²` 背景与非解析电子响应分开；
- 同一套参数用于零展宽和三个有限展宽条件。

当前开发集结果：

- 选择模型：`rounded_two_scale_curvature`。
- 平衡 LOSO RMSE：`1.097 cm⁻¹`。
- finite-smearing direct-DFPT 最大 K kink 相对误差：`13.39%`。
- finite-smearing K 中心块最大 MAE：`0.373 cm⁻¹`。
- zero-smearing `d=0.003` cusp 深度误差：KG `17.17%`，KM `17.66%`。
- zero-smearing 内外斜率比误差：最大 `13.19%`。
- dynamical-matrix Hermiticity 误差：`0.0`。
- rank-one 频率重放最大误差：`6.82×10⁻13 cm⁻¹`。

单尺度加解析 `r²` 曲率仍将零展宽 `d=0.003` 深度低估约 `47%`。在当前参考数据范围内，第二个标量 crossover 不能只用普通曲率替代。

联合核读取生产 `k18/q9 ex1_pifroz` Wannier Hamiltonian 的两侧速度 `5.461/5.440 eV·Å`，使用平均值 `5.450 eV·Å`。它不再复用早期 `k12/q6 pi45` 候选的速度。

结果和模型资产：`results/graphene_physics_temperature/post_p4_feasibility/E4_joint_zero_finite_rank1/`

### 3. 参考值敏感性

零展宽 `d=0.003` 的现有 cusp depth 随电子 k 网格变化：

| k 网格 | KG depth (cm⁻¹) | KM depth (cm⁻¹) |
|---:|---:|---:|
| 192 | 6.225 | 6.252 |
| 240 | 5.407 | 5.437 |
| 288 | 4.827 | 4.856 |

同一结构在 k192、k240、k288 参考下得到的最大零展宽深度误差分别为 `33.4%`、`25.3%`、`17.7%`。因此当前通过 20% 门槛依赖 k288 参考，窄尺度仍停在现有 q 分辨率下限 `0.003`，不能解释为已精确测得的材料常数。

这一限制不改变矩阵 rank-one 结论，但阻止当前联合核直接冻结。

### 4. finite-smearing 标量低秩压缩

把每条 dense EPW 曲线减去自身 K 锚值后，对 300/450/600 K 的标量修正做 SVD：

- 第一标量 q 基累计解释 `99.9212%`。
- 两个标量 q 基累计解释 `99.99977%`。
- 一个基的最大 cusp 深度误差为 `21.67%`。
- 两个基的全曲线频率 RMSE 为 `0.0194 cm⁻¹`，最大频率误差为 `0.0449 cm⁻¹`。
- 两个基的最大 cusp 深度相对误差为 `2.08%`。
- 第三奇异值为 `335.9 cm⁻²`，小于现有 EPW 左右两侧非对称量。

这说明需要区分两种“rank”：矩阵仍是一个 A′ projector；它的标量 q/smearing 响应需要两个共享基。

固定两个 q 基后，每次用另外两个 smearing 对 K 锚值和两个系数作仿射拟合：

- 留出整条曲线最大 RMSE：`0.182 cm⁻¹`。
- 最大点误差：`0.295 cm⁻¹`。
- K 锚值最大误差：`0.062 cm⁻¹`。
- cusp 深度最大误差：`9.30%`。

这一结果支持在已覆盖的 smearing 区间内只预测三个标量系数，不为每个新 smearing 重算或重新训练整条声子谱。

结果和两基资产：`results/graphene_physics_temperature/post_p4_feasibility/E5_epw_scalar_low_rank/`

## 与理论约束的关系

Piscanec 等对 graphene/graphite 的解析结果表明，Γ-E2g 和 K-A′1 是主要 Kohn anomaly 通道，零温 kink 的斜率与 EPC 平方成正比。因此零展宽模型必须显式保留非解析斜率，不能依靠局域 MLIP 或平滑实空间截断自动产生 cusp。

- S. Piscanec et al., *Phys. Rev. Lett.* **93**, 185503 (2004): https://doi.org/10.1103/PhysRevLett.93.185503
- arXiv: https://arxiv.org/abs/cond-mat/0407164

`rounded` 坐标仍可作为低成本代理和消融基线，但完整 EPC band-sum 已经给出更直接的物理响应，不需要继续增加经验多项式或 crossover。

### 5. 生产 Wannier 能带几何

使用 `k18/q9 ex1_pifroz` Hamiltonian 检查 K 附近的电子结构：

- 两个方向在小 q 外推到相同的线性速度，约 `5.451 eV·Å`；
- 到 `d=0.06` 时，KG/KM 的三角翘曲分量达到 `29.38 meV`，约为平均激发能的 `5.29%`；
- 解析的曲率和方向差异可以由 Wannier Hamiltonian 固定，不需要用声子数据重新拟合。

结果：`results/graphene_physics_temperature/post_p4_feasibility/E6_wannier_k_geometry/`

### 6. 完整 EPC band-sum 重放

从 V100-B 通过 Tailscale 取回已有的 `graphene.epmatwp`，没有启动新计算。文件大小为 `11,985,792` 字节，SHA-256 为 `3e1b7b8325ed8ab60468995290fe03f066b5e394b14af01a214acb1a16de7a4b`。按实际 EPW 7.3.1 源码解析为 `2×2×343×6×91` 个 complex128：两个 Wannier 轨道、343 个电子 WS 向量、6 个 Cartesian 声子分量和 91 个声子 WS 向量。

解析校验：

- `decay.epmate/epmatp` 的实空间长度最大误差分别为 `1.73×10⁻8 Å` 和 `8.71×10⁻9 Å`；
- EPC 最大值重放误差低于 `5×10⁻11 Ry`；
- 裸 A′ 频率相对 EPW 文本输出最大误差 `0.0403 cm⁻¹`。

在与生产 EPW 相同的 `720×720` fine-k 网格上，直接计算 `Π(T)-Π(T_high)`，其中 `T_high=0.020000 Ry`：

- 不做幅度重拟合的频率 RMSE：`0.2126 cm⁻¹`；
- 最大点误差：`0.2912 cm⁻¹`；
- 300/450/600 K 对应 smearing 的 cusp 深度误差：`2.90% / 3.92% / 5.07%`；
- 最优全局幅度比例：`0.99167`。

这一步通过了有限展宽物理核的重放门槛。结果：`results/graphene_physics_temperature/post_p4_feasibility/E9_epc_bandsum_replay_nk720/`

### 7. 零展宽积分与 k 网格参考

零温阶跃占据在全 BZ 均匀网格上收敛很慢。完整 EPC 的 `d=0.003` 深度随 `nk=720→1440` 从约 `0.72` 增到 `1.31 cm⁻¹`，说明不能继续把全局均匀加密当作生产算法。改用“粗全 BZ + K 谷局部补丁”后，结果在补丁 `n=120→360` 已稳定；将数值分母宽度从 `0.005 eV` 降到 `0.0002 eV` 后得到：

- KG：`1.905 cm⁻¹`；
- KM：`1.911 cm⁻¹`。

上述数值来自 midpoint 补丁。随后增加独立的三角形质心积分，并把三角补丁从 `n=180→360→720` 加密。对 `1/n_patch` 外推后，推荐值更新为：

- KG：`1.965 ± 0.058 cm⁻¹`；
- KM：`1.976 ± 0.064 cm⁻¹`。

这里的不确定度将最后一次分辨率变化、补丁半宽、粗网格、`η` 扫描、midpoint/triangle 差异和外推形式差异线性相加，属于偏保守的数值积分误差。补丁半宽 `0.03–0.05`、粗网格 `144–288` 和 `η=0.00005–0.0002 eV` 的检查均通过。

现有 optimized-tetrahedron DFPT 的 k192/k240/k288 深度并未收敛。对三点作主导 `1/N_k` 误差外推：

- KG 极限 `2.046 cm⁻¹`，最大拟合残差 `0.0135 cm⁻¹`；
- KM 极限 `2.079 cm⁻¹`，最大拟合残差 `0.0151 cm⁻¹`；
- 三角积分外推后的完整 EPC 相对上述极限的误差分别为 `3.94%` 和 `4.92%`。

结果表明 k288 的 `4.827/4.856 cm⁻¹` 仍显著高估收敛 cusp depth；此前基于 k288 得到的“模型低估”不能作为模型结构失败的证据。这里只验证了 `d=0.003` 深度，外侧斜率仍缺少 k 网格收敛参考。

结果：

- `results/graphene_physics_temperature/post_p4_feasibility/E11_epc_zero_adaptive_c288_p360_eta0p0002/`
- `results/graphene_physics_temperature/post_p4_feasibility/E12_zero_k_extrapolated_epc/`
- `results/graphene_physics_temperature/post_p4_feasibility/E13_epc_zero_triangle_c288_p720_hw0p04_eta0p00005/`
- `results/graphene_physics_temperature/post_p4_feasibility/E14_zero_quadrature_convergence/`

### 8. 平滑零展宽曲线与 EPC 顶角压缩

用 `144×144` 粗全 BZ、`360×360` 三角形质心局部补丁，在 K 两侧各生成 61 个 q 点。121 个 q 点的本地 CPU wall time 为 `28.3 s`。所得 A′ 频率从 K 向 KG/KM 两侧严格单调增加，没有插值锯齿；`d=0.003` 深度分别为 `1.932/1.940 cm⁻¹`，均落在上一节冻结的积分误差区间内。图中的实线由冻结的短程 MLIP 背景和完整 EPC 长程响应共同得到。

结果：`results/graphene_physics_temperature/post_p4_feasibility/E15_epc_zero_dense_curve/`

将 K 邻域相位对齐后的 A′ 有效 EPC 顶角写成 q 与复数实空间顶角之间的矩阵，得到：

- KG/KM 窗口中 rank 1 已解释 `99.9646%`，rank 3 解释 `99.9999988%`；
- rank-3 因子只需 `4479` 个复数，约为原始完整 `epmatwp` 参数量的 `0.60%`；
- 在有限 smearing 的 29 点生产路径上，rank 3 解释 `99.9999866%`；
- 把 rank-3 顶角重新送入 `720×720` band-sum 后，相对完整 EPC 的频率 RMSE 为 `0.00042 cm⁻¹`，最大点误差 `0.00088 cm⁻¹`，最大 cusp 深度误差 `0.0073%`；
- 只用 5 个 coarse-q EPC 标签学习 rank-3 基底和系数，再预测其余 24 个 q 点时，频率 RMSE 为 `0.00116 cm⁻¹`，最大点误差 `0.00212 cm⁻¹`，最大 cusp 深度误差 `0.0365%`。

最后一项是在 graphene 内固定 5 点后的 q-holdout，不是跨体系盲测。它支持把新体系的训练目标从“整条 dense 声子谱”改成“少量 coarse-q A′ EPC 顶角 + rank-3 生成表示”，但不能直接证明所有新体系都只需 5 个标签。

结果：

- `results/graphene_physics_temperature/post_p4_feasibility/E16_epc_vertex_low_rank/`
- `results/graphene_physics_temperature/post_p4_feasibility/E17_epc_vertex_rank_response_nk720/`

### 9. 跨材料盲测与模型修复

远端已有完整 EPW/Wannier 资产，因此这一阶段没有新增 DFT。通过 Tailscale 取回 `1T-VSe2`、`2H-TaS2` 和 `2H-NbS2` 的 `epmatwp`、Hamiltonian、`epwdata.fmt` 和晶体元数据；每个体系约 `45 MB`（包括 EPW 输出），文件哈希记录在 `E18_cross_system_assets/ASSET_MANIFEST.json`。

第一次在 VSe₂ 上固定使用 graphene 的 rank-3 + natural-cubic 表示，并在 41 个 q 点中只用 `[0,10,20,30,40]` 五点训练：

- rank-3 全数据 capture：`99.9798%`；
- 36 点 holdout 顶角 RMSE：`2.043%`，略高于事先确定的 `2%` 门槛；
- 最大顶角误差：`3.43%`；
- 相邻 q 的模态重叠最低 `0.99903`，说明是连续的软模旋转，不是错误跳带。

因此 VSe₂ 的原始跨材料测试记为未通过。失败后只调整模型，不增加 q 标签：允许材料自适应 rank 上限从 3 增到 4，并把 latent 的 q 依赖从自然三次样条改为五点四阶 Chebyshev 表示。修复结果：

- VSe₂ rank-4 顶角 holdout RMSE 从 `1.097%`（cubic）降到 `0.480%`（Chebyshev）；
- 传播到 `smearing/degauss = 0.005/0.010/0.015 Ry` 后，相对完整 EPC 的频率 RMSE 从 `0.184` 降到 `0.0257 cm⁻¹`；
- 最大频率误差 `0.0539 cm⁻¹`，窗口深度最大误差 `0.33%`。

随后在尚未检查顶角结构前，固定 rank 4、五点位置、Chebyshev 表示和验收门槛，在 2H-TaS₂ 上做第二次跨材料测试：

- rank-4 capture：`99.999991%`；
- 36 点 holdout 顶角 RMSE：`0.0432%`，最大误差 `0.0663%`；
- 相邻模态重叠最低 `0.99996`；
- 三个 smearing 上相对完整 EPC 的频率 RMSE：`0.00666 cm⁻¹`，最大误差 `0.0201 cm⁻¹`；
- 窗口深度最大误差 `0.111%`。

保持同一套参数和门槛，又在事先未检查的 2H-NbS₂ 上做第三次材料测试：

- rank-4 capture：`99.999987%`；
- 36 点 holdout 顶角 RMSE：`0.0535%`，最大误差 `0.0820%`；
- 相邻模态重叠最低 `0.99993`；
- 三个 smearing 上相对完整 EPC 的频率 RMSE：`0.00162 cm⁻¹`，最大误差 `0.00477 cm⁻¹`；
- 窗口深度最大误差 `0.00354%`。

结果支持的可复用结构更新为“单一声子模态或连续模态子空间 + 材料自适应复数 EPC rank（上限 4）+ 解析 Chebyshev q-latent + 显式 Fermi–Dirac band-sum”。smearing 不需要单独 EPC 训练数据；同一个 Hamiltonian 和 EPC 表示可直接计算不同 `smearing/degauss (Ry)`。目前验证的是压缩表示相对完整 EPC 的一致性，尚未验证 VSe₂/TaS₂ 的短程 MLIP 背景与绝对收敛 DFT 声子谱。

结果：

- `results/graphene_physics_temperature/post_p4_feasibility/E18_VSe2_rank3_five_q_blind/`
- `results/graphene_physics_temperature/post_p4_feasibility/E19_VSe2_rank_repair/`
- `results/graphene_physics_temperature/post_p4_feasibility/E20_VSe2_rank_response/`
- `results/graphene_physics_temperature/post_p4_feasibility/E21_TaS2_rank4_five_q_blind/`
- `results/graphene_physics_temperature/post_p4_feasibility/E22_TaS2_rank_response/`
- `results/graphene_physics_temperature/post_p4_feasibility/E24_NbS2_rank4_five_q_blind/`
- `results/graphene_physics_temperature/post_p4_feasibility/E25_NbS2_rank_response/`

### 10. K/K′ 时间反演适配与 MLIP+LR 矩阵接口

在 q6 Cartesian 算子上，K 与 K′ 原始长程矩阵严格满足 `D(-q)=D(q)*`；裸矩阵的最大绝对差为 `6.3×10⁻14`，属于数值舍入。新的适配层只保存 K 侧无相位歧义的 Hermitian 投影器，K′ 由复共轭生成。三个 smearing 共用一个投影器：

- 最小 Frobenius capture：`99.9833%`；
- 生成矩阵的时间反演误差：`0.0`；
- Hermiticity 误差：`0.0`；
- K/K′ A′ 频率最大误差：`2.3×10⁻13 cm⁻¹`。

这里的六个几何 K 角在模倒格矢后归并为 K/K′ 两个 valley。适配层已能处理 K 附近任意 `q↔-q` 配对；任意二维偏移的 C3/C6 旋转还需要加入 Cartesian 旋转、原子置换和 Bloch gauge。

随后把零展宽 121 点曲线重新通过统一矩阵接口计算：短程项读取冻结的 MLIP-derived force constants，长程项以 `Δλ(q)|e_A′(q)><e_A′(q)|` 加到动力学矩阵上。结果逐点复现实线：

- 短程频率回放最大误差：`0.0 cm⁻¹`；
- MLIP+LR 频率回放最大误差：`6.8×10⁻13 cm⁻¹`；
- 投影器幂等误差：`2.8×10⁻16`；
- 修正泄漏到正交子空间的最大值：`5.7×10⁻17`。

因此当前图中的实线确实是 `D_MLIP^SR + ΔD_EPC^LR` 的 Hermitian 矩阵本征值，不是对频率曲线事后平滑。通用接口位于 `src/phonon_accel/long_range.py`，同时包含五 q Chebyshev 顶角适配器。

结果：

- `results/graphene_physics_temperature/post_p4_feasibility/E23_kstar_equivariant_adapter/`
- `results/graphene_physics_temperature/post_p4_feasibility/E26_mlip_lr_matrix_interface/`

进一步加入 spglib 的 24 个空间群操作、Cartesian 位移旋转、原子置换和 Bloch cell phase 后，三个 smearing 的 K/K′ 长程矩阵在全部操作下相对误差最大为 `5.6×10⁻16`，共享投影器的重复轨道误差为 `1.9×10⁻16`，时间反演误差为 `2.8×10⁻17`。一个偏离 K 的一般二维点生成 12 个不同的面内 q-star 成员。这里必须使用 EPW q6 的 cell-gauge 原子坐标 `[0,0,0]` 和 `[1/3,2/3,0]`；误用 MLIP 超胞的等价原点/原子约定会产生数量级为 1 的假矩阵误差。

结果：`results/graphene_physics_temperature/post_p4_feasibility/E28_kstar_point_group_adapter/`

### 11. 当前可冻结的结论

把 graphene 内部 q-holdout、VSe₂ 开发失败/修复、TaS₂ 和 NbS₂ 的独立测试分开汇总后：

- 修复后的表示在独立材料上通过 `2/2`；
- 两个独立体系中最差顶角 holdout RMSE 为 `0.0535%`；
- 最差频率 RMSE 为 `0.00666 cm⁻¹`，最大点误差 `0.0201 cm⁻¹`；
- 最差窗口形状误差为 `0.111%`。

据此冻结“五个模式投影 coarse-q EPC 标签 + 材料自适应 rank≤4 + 四阶 Chebyshev q-latent + 显式 band-sum”的低数据适配层。当前不能宣称已经训练出跨材料生成模型：五个标签仍在学习材料特定的 `B_r(k)`，而不是由预训练模型零样本生成。下一项模型工作是定义 gauge-invariant 的生成输入和输出；在这之前不启动 dense DFPT。

汇总：`results/graphene_physics_temperature/post_p4_feasibility/E27_low_data_epc_transfer_summary/`

## 下一阶段（更新后）

### A. 零温积分冻结

1. 已完成 midpoint 与二维 triangle-centroid 的独立积分比较。
2. 已完成 patch 尺寸、patch 分辨率、粗网格和 `η→0` 扫描，`d=0.003` 数值误差区间已冻结。
3. 已生成 K 两侧平滑的 A′ dense-q 曲线；rank-one 标量矩阵更新按构造保持 Hermiticity。
4. K/K′ 时间反演和无相位投影器约定已经完成。
5. 任意二维偏移的 C3/C6 q-star 旋转表示已经完成；下一步生成 canonical wedge 内的物理顶角。外侧斜率仍只与收敛参考比较，当前 k192 外侧点保留为开发数据，不作为冻结门槛。

这一阶段继续只使用已有 Wannier/EPW/DFPT 数据和本地 CPU，不启动新的 V100 DFPT 队列。

### B. 冻结条件

- zero-smearing 两个方向 `d=0.003` 深度相对明确写出的 k-grid 外推模型低于 10%；
- finite-smearing 完整 EPC 重放的 cusp 深度误差低于 10%，频率 RMSE 低于 `0.3 cm⁻¹`；
- K 中心块 MAE 低于 `1 cm⁻¹`；
- Hermiticity 误差低于 `1×10⁻10`；
- rank-one matrix replay 误差低于 `1×10⁻6 cm⁻¹`；
- 不再保留落在搜索边界上的经验窄尺度；q/smearing 形状由 band-sum 给定；
- 模型、测试点和验收标准在独立测试前固定。

### C. 新体系的数据策略

新体系不采集完整 q 网格训练谱。预计输入分为：

- 常规结构/力数据：用于短程 MLIP；
- Wannier/生成模型提供的低能 Hamiltonian：K-star、`v_F`、化学势、费米面几何和对称性；
- A′ EPC 顶角的低秩表示：优先从预训练生成模型迁移，再用少量 coarse-k/coarse-q EPC 标签微调；
- 少量模式投影矩阵：用于校准 K 锚值和接近 1 的整体 EPC 幅度；
- 冻结后少量未见 q/smearing 矩阵：只用于独立测试。

`epmatwp` 的有效顶角压缩、频率级重放和跨材料测试已经完成。graphene 使用 rank 3；跨材料模型使用材料自适应 rank，当前上限为 4。五个 coarse-q 标签已在 graphene、VSe₂ 修复以及独立 TaS₂/NbS₂ 测试中覆盖当前 q 窗口，但它们仍是完整的模式投影 EPC 标签，不是普通能量/力数据。

K/K′ 时间反演、C3/C6 空间群表示、第三材料测试和矩阵级组合接口已经完成。下一项模型实验是为 rank≤4 生成器规定 gauge-invariant 输入，例如局部能带能量、费米面速度/曲率、模式不可约表示、质量归一化极化向量的投影不变量和 q-star 标识；输出先预测 rank、奇异值谱和归一化 q-latent prior，材料特定 Wannier-gauge 基底仍由五个标签校正。只有模型和验收标准冻结后，才为完全没有 EPC 资产的新体系计算固定的 5 个 coarse-q 标签。目前不启动 dense DFPT 扫描。

### D. 接下来所需时间与算力

| 阶段 | 计算资源 | 预计用时 | 可得到的结论 |
|---|---|---:|---|
| K/K′ time-reversal 封装 | 已完成；本地 CPU 运行低于 10 s | 共用投影器通过，A′ 误差低于 `3×10⁻13 cm⁻¹` |
| 第三个已有资产材料 holdout | 已完成；45 MB Tailscale 传输，本地分析约 5 s | NbS₂ 顶角与频率 gate 均通过 |
| MLIP 背景与长程项组合接口 | 已完成；本地 CPU 约 2 s | 121 点矩阵级曲线与原结果逐点一致 |
| 二维 C3/C6 等变扩展 | 已完成；本地 CPU 约 2 s | 24 个操作的矩阵误差低于 `6×10⁻16`，一般二维点生成 12-member orbit |
| gauge-invariant 生成数据契约与归档提取 | 本地 CPU；优先使用已有 NbSe₂/TaSe₂/TiSe₂ 资产 | 4–8 h | 哪些量可跨材料预训练，哪些量必须由五点微调确定 |
| leave-one-material-out latent prior | 本地 CPU/GPU；不做新 DFT | 4–12 h | 生成 prior 是否能进一步把五点减少到 3 点；未通过则保持五点预算 |
| 冻结后最小独立验证 | 现有资产优先；缺失时才用 V100 | 预计 5 个 coarse-q 标签，约 4–12 V100 GPU-h | 检查真正未见体系；不做 dense-q 扫描 |

当前可给出的数据预算是每个新体系五个模式投影 EPC 标签，并且同一组标签覆盖所有 smearing。能否进一步降到三个标签，必须由下一轮 leave-one-material-out 结果决定，不能现在预先承诺。

生成 prior 的输入、gauge 边界、文件结构、loss、验收门槛和算力预算见 `docs/LOW_DATA_EPC_GENERATOR_CONTRACT_2026-08-18.md`。
