# 低数据 EPC 生成与微调接口（v1，2026-08-18）

## 目标与适用范围

现阶段冻结的可部署对象不是一条声子频率曲线，而是

\[
D(q,s)=D_{\mathrm{MLIP}}^{\mathrm{SR}}(q)
+\Delta D_{\mathrm{EPC}}^{\mathrm{LR}}(q,s),
\]

其中短程项来自 MLIP force constants，长程项由低秩模式投影 EPC 顶角、Wannier Hamiltonian 和显式 Fermi–Dirac band-sum 得到。`s` 直接使用 `smearing/degauss (Ry)`。同一套 Hamiltonian/EPC 参数覆盖零展宽和有限展宽，不按 smearing 重训模型。

当前数据支持冻结“五个 coarse-q 标签 + 材料自适应 rank≤4 + 四阶 Chebyshev q-latent”的材料适配层。它在 TaS₂、NbS₂ 两个独立体系上通过，但还不是跨材料预训练生成器。v1 数据契约的目的，是把可以跨材料预训练的 gauge-invariant 信息与必须在新材料上少量微调的 Wannier-gauge 信息分开。

## 模型分层

### 1. 短程 MLIP

输入为平衡结构或有限晶格温度构型；输出力和短程 force constants。训练数据仍使用常规能量、力和应力标签。短程 MLIP 不接收 electronic smearing，也不负责产生 Kohn anomaly 的非解析 cusp。

### 2. Gauge-invariant 生成 prior

生成 prior 只预测下列量：

- 有效顶角 rank 的概率，`R=1…4`；
- 归一化奇异值谱；
- canonical q wedge 内归一化 q-latent 的均值与不确定度；
- 需要的标签位置或主动学习优先级；
- 对应响应是否超出训练分布。

它不直接预测一个固定 Wannier gauge 下的完整复数 `epmatwp`。不同材料的 Wannier 子空间可以发生任意 unitary rotation，直接比较原始复数分量会把 gauge 差异误当成材料差异。

### 3. 五标签材料适配层

新材料在 canonical q wedge 内计算五个模式投影 EPC 标签。默认位置为局部坐标

\[
x=(-1,-0.5,0,0.5,1)x_{\max},
\]

对应当前测试中的索引 `[0,10,20,30,40]`。五个复数顶角经材料内部相位对齐后做 SVD，选择满足 capture 门槛的最小 rank，最大为 4；五点 latent 用四阶 Chebyshev 表示。生成 prior 只提供正则化和不确定度，五个真实标签负责确定材料特定的 `B_r(k)`。

### 4. 显式电子响应与对称性

适配后的顶角进入显式 band-sum。零展宽使用局部 K-valley 自适应积分；有限展宽使用 Fermi–Dirac 占据。canonical wedge 的结果通过空间群表示扩展到完整 q-star：

\[
\Delta D(Rq)=U_R(q)\Delta D(q)U_R^\dagger(q),\qquad
\Delta D(-q)=\Delta D(q)^*.
\]

当前 graphene 实现已经包含 Cartesian 旋转、原子置换和 Bloch cell phase；矩阵相对误差低于 `6×10⁻16`。

## 每个体系的输入记录

### 必需的 gauge-invariant 输入

- 晶格矩阵、分数坐标、元素和原子质量；
- 空间群操作、目标 q-star、little group 和目标模态不可约表示；
- 化学势相对的局部能带本征值；
- 费米面速度、曲率、谷间连接向量和必要的轨道投影不变量；
- 裸声子本征值和相位无关的模态投影器；
- 相邻 q 点模态投影器的 principal-angle/overlap；
- 五个标签的 q 坐标、训练/holdout 标识和来源哈希。

### 材料内部 gauge-covariant 输入

- Wannier Hamiltonian `H(R)`；
- 五个 q 点的模式投影复数 EPC 顶角；
- Wannier/Wigner-Seitz 向量顺序、简并度和 Fourier gauge；
- 质量归一化极化向量；
- 将 q-star 操作映射到当前原子/Wannier 约定所需的 representation metadata。

这些数据只能在同一材料和同一 gauge 内直接比较。跨材料 loss 必须使用奇异值、Gram kernel、band-sum 响应或经过显式 Procrustes/unitary 对齐的量。

## 建议的 v1 文件结构

每个材料目录只需保存：

```text
material_id/
  manifest.json
  structure.npz
  symmetry.npz
  electronic_invariants.npz
  phonon_projectors.npz
  five_q_vertex_labels.npz
  hamiltonian_hr.dat
  validation_spec.json
```

`five_q_vertex_labels.npz` 至少包含：

- `qpoints_reduced: (5,3)`；
- `coordinate: (5,)`；
- `vertex: (5, nbnd*nbnd*nrr_k)` complex；
- `mode_projector: (5,3N,3N)` complex；
- `mode_frequency_cm1: (5,)`；
- `training_mask` 与原始文件 SHA-256。

对当前 11-band、157 个电子 WS 向量的 TMD，五个复数模式投影标签约为 `5×11²×157` complex128，即约 `1.52 MB`。部署时不需要保存 `35.6 MB` 的完整 Cartesian `epmatwp`。graphene 的五标签顶角约为 `0.11 MB`。

## 训练和验收

### 数据划分

- VSe₂ 只作为模型修复开发体系；
- graphene 是体系内 q-holdout；
- TaS₂、NbS₂ 已经用作独立测试，不能再次当作未见测试；
- 下一轮采用 leave-one-material-out，并按结构族记录 1T/2H、元素族和费米面拓扑；
- 最终测试材料及其五个标签位置、模型 checkpoint 和门槛必须在读取标签前固定。

### Loss

- rank/capture loss；
- 归一化奇异值谱 loss；
- gauge-invariant q–q Gram kernel loss；
- canonical wedge latent reconstruction loss；
- 频率级 band-sum response loss；
- K/K′ 时间反演、C3/C6 covariance 和 Hermiticity loss；
- uncertainty calibration，用于决定是否保持五点或触发额外标签。

### 固定门槛

- 相邻模态 overlap `>0.99`；
- 所选 rank capture `>0.9999`；
- 36 点 holdout 顶角相对 RMSE `<2%`，最大误差 `<5%`；
- 相对完整 EPC 的频率 RMSE `<0.05 cm⁻¹`，最大误差 `<0.15 cm⁻¹`；
- 窗口深度相对误差 `<2%`；
- time-reversal、point-group 和 Hermiticity 矩阵误差 `<1×10⁻10`。

只有 leave-one-material-out 结果表明三个标签在所有未见体系上仍满足同一门槛，才允许把默认预算从五点降到三点。

## 算力、数据和时间

| 阶段 | 数据/算力 | 预计墙钟 | 输出结论 |
|---|---|---:|---|
| v1 extractor 与 schema 验证 | 本地 CPU；现有 VSe₂/TaS₂/NbS₂/graphene | 2–4 h | 所有文件是否可重建五标签适配器和 band-sum |
| 扩充既有归档 | 通过 Tailscale 读取 NbSe₂、TaSe₂、TiSe₂ 等已有 EPW 资产；每体系约 45 MB | 2–6 h | 至少 5–8 个体系的结构族覆盖和 gauge 元数据是否完整 |
| leave-one-material-out prior | 本地 CPU 或单卡 GPU；压缩后数据通常低于 20 MB | 4–12 h | prior 是否改善五点误差；三个标签是否可行 |
| 冻结后现有资产测试 | 本地 CPU | 每体系数秒至数分钟 | 模型结构和数据预算是否再次通过 |
| 真正新体系的最小标签 | 单台 V100；5 个 coarse-q DFPT/EPC 标签 | 约 4–12 V100 GPU-h，先用 pilot 校准 | 对完全未见体系的最终结论 |

当前不运行 dense-q DFPT，也不为不同 smearing 分别生成标签。若五点适配器失败，优先检查 rank、二维 q-latent、模态子空间和 gauge；只有明确定位为标签覆盖不足时，才增加一个主动学习点。

## 当前结论

五点材料适配层已经可以冻结并接入 `D_MLIP^SR + ΔD_EPC^LR`。下一阶段应训练和验证 gauge-invariant prior，而不是继续加密 DFT 网格。生成 prior 的成功标准是减少新材料适配误差或可靠识别失败体系；它不必取代五个标签，也不能通过隐藏的 dense EPC 数据获得性能。
