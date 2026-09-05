# Graphene K 点 Kohn cusp：两种方法复现计划

**计划日期：**2026-08-10  
**研究目标：**用两条独立路线定量复现 graphene K 点最高光学支 $A_1'$ 的 Kohn-anomaly cusp：

1. primitive-cell DFT/DFPT 直接计算；
2. 本项目方法，即 MLIP 给出有限晶格温度背景，再用 q-space 电子响应修正恢复 cusp。

本轮的验收重点是 K 点附近的非解析折点。仅得到稳定结构、无虚频或一条平滑色散曲线，不算完成。

## 1. 时间结论

按交付强度分为三个层级：

| 交付层级 | 能回答的问题 | 新计算 | 从可启动时算的墙钟时间 |
|---|---|---|---:|
| A. 快速可行性复现 | 两种方法在已有离散 q 点上是否都产生方向正确、量级合理的 K cusp | 不新增 DFT；复用三温度 DFPT、EPW 静态结果和 L0/Q0 结果 | **6–10 h，可立即做** |
| B. 三温度连续曲线 | 300/450/600 K 下，MLIP+q-space 的连续 cusp 是否定量接近 direct DFPT | 每温度 29 个与 720×720 fine-k 网格严格可公度的 dense-q EPW 后处理点；不新增 SCF/NSCF/dvscf | **V100-B/EPW 数据恢复可达后 6–10 h** |
| C. 近零展宽尖锐 cusp | 在更接近 `degauss -> 0` 时，两种方法能否复现更尖的 DFT cusp，并排除 k/q 网格伪影 | k192 direct DFPT 增加约 8–10 个 K 邻域 q 点，并做固定 `degauss` 的 k 网格复核 | **两台 V100 恢复后 24–36 h；含重试按 36–48 h 预留** |

因此：

- 第一张两种方法的可行性对比图预计 **当天 6–10 h** 可以得到，不需要等待 GPU；
- 可以支撑阶段性定量结论的三温度连续图，在 EPW 原始插值文件可访问后还需 **6–10 h**；
- 若论文结论需要强调“尖锐、接近零展宽的 cusp”，总周期由 direct DFPT 决定，从两台 V100 恢复在线起需要约 **1–1.5 天**，稳妥排期为 **1.5–2 天**。

当前两台 V100 的 Tailscale 地址均连接超时，以上 GPU 时间均从机器恢复可达后开始计时。RTX 2060 当前空闲，可以承担已有结果分析和绘图，但不适合替代 V100 跑 Quantum ESPRESSO 的高密度 DFPT。

## 2. 已有数据说明了什么

### 2.1 direct DFPT 已经有三条有限展宽参考线

已有 300、450、600 K 对应的 direct DFPT K 邻域数据，每条含 11 个 K 邻域 q 点：

| 配对温度标签 | `smearing/degauss (Ry)` | k 网格 | 历史单机耗时 |
|---:|---:|---:|---:|
| 300 K | 0.0019000869 | 144×144×1 | 约 40.7 h |
| 450 K | 0.00285013035 | 144×144×1 | 约 46.3 h |
| 600 K | 0.0038001738 | 120×120×1 | 约 31.0 h |

这些数据已经能画出有限 `degauss` 下的 DFT cusp，不需要重算整条线。历史耗时较长，是因为每个 q 点都是独立 DFPT；本轮首先复用已有结果。

已有近零展宽扫描还给出：

| `smearing/degauss (Ry)` | k 网格 | K 中心相对两侧的局部下凹深度 |
|---:|---:|---:|
| 0.0006333623 | 192×192×1 | 9.91 cm⁻¹ |
| 0.0012667246 | 144×144×1 | 7.78 cm⁻¹ |
| 0.0019000869 | 120×120×1 | 6.23 cm⁻¹ |

下凹随 `degauss` 减小而加深，方向符合 Kohn anomaly 的物理预期。当前最低展宽只有 K 左、中心、右三个点，足以看到信号，但不足以证明尖点形状和斜率跳变已经对 q 网格收敛，所以需要层级 C 的加密计算。

### 2.2 项目方法的电子修正已经通过离散点门槛

现有 EPW q-space 电子算子在 300/450/600 K 的静态 K 区门槛均通过：

| 温度 | K line MAE | K 点绝对误差 | kink 相对误差 |
|---:|---:|---:|---:|
| 300 K | 5.105 cm⁻¹ | 4.814 cm⁻¹ | 8.76% |
| 450 K | 5.120 cm⁻¹ | 5.053 cm⁻¹ | 3.28% |
| 600 K | 5.109 cm⁻¹ | 5.097 cm⁻¹ | 7.93% |

这说明 q-space 电子响应已经在少量测试点上抓住了 cusp 的幅度和斜率变化。当前完整声子谱在 K 点仍然偏平滑，原因是电子修正曾被表示在有限 q6 网格上，再 Fourier 变换为有限范围实空间算子；这个步骤会截断产生 Kohn cusp 所需的长程振荡力常数。

本轮不再用共享 Dirac 拟合作为主结果。现有共享参数模型在 450/600 K 的留一展宽测试中，K kink 误差约为 25%，没有通过 20% 门槛；它只保留为消融对照。

## 3. 两种复现方法

### 方法一：primitive-cell direct DFT/DFPT

在固定结构、固定 `smearing/degauss (Ry)` 和收敛 k 网格下，直接沿

\[
\mathbf q=t\mathbf K,\qquad t\approx 1
\]

计算动力学矩阵。K 点附近对最高光学支使用本征矢重叠追踪，不能简单地每个 q 点都取最高频率，以免发生支交换。

输出包括：

- 300/450/600 K 对应展宽的三条 finite-smearing 参考线；
- 最低 `degauss=0.0006333623 Ry` 的加密 K 邻域线；
- K 中心频率、局部 cusp depth、左右斜率和 slope jump；
- 固定 `degauss` 下的 k 网格和 q 点密度收敛结果。

这里的 300/450/600 K 是用于配对物理电子展宽的温度标签。direct DFPT 本身没有包含晶格热涨落；图表中必须把 `smearing/degauss (Ry)` 与 lattice temperature 分开说明。

### 方法二：MLIP + q-space 电子修正

MLIP 负责有限晶格温度的平滑短程背景：classical TDEP 给出 L0，quantum SSCHA/SCPH 给出 Q0。电子 Kohn 项由 finite-smearing EPW/Wannier 响应在目标 q 点直接计算。最终动力学矩阵写成

\[
D_{L0/Q0}^{\mathrm{dense}}(\mathbf q,T)=
D_{L0/Q0}^{\mathrm{finite}}(\mathbf q,T)
-D_{\mathrm{el}}^{q6\rightarrow R}(\mathbf q,T)
+D_{\mathrm{el}}^{\mathrm{EPW,dense}}(\mathbf q,T).
\]

第一项是已经得到的有限温 TDEP 或 SSCHA/SCPH 背景；第二项移除其中已经加过、但被 q6/实空间截断而变平滑的电子项；第三项在 dense q 点直接加回电子响应。这样既避免双计数，也不再把非解析 cusp 强行压回有限范围实空间力常数。

实现时需要：

1. 为 K 点 $A_1'$ 模构造 Hermitian 模投影；
2. 用本征矢重叠连续追踪该模；
3. 在共同 q 点验证 q6 电子项“减去再加回”能重放原结果；
4. 检查 Hermiticity、time-reversal 和 K-star 对称性；
5. L0 与 Q0 分别出图，不用 classical 结果代替 quantum 结果。

## 4. 统一验收标准

两种方法使用完全相同的 q 坐标、模式追踪、拟合窗口和指标。建议冻结以下标准后再生成最终图：

### 4.1 数值收敛

- direct DFPT 在最低 `degauss` 的 K 中心五点，k 网格复核后的 MAE ≤ 1 cm⁻¹；
- cusp depth 必须为正，左右拟合窗口改变一个 q 间隔后，slope jump 变化不超过 10%；
- q-space 矩阵 Hermiticity 残差 ≤ $10^{-10}$；
- 共同 q 点上，电子算子减去再加回后的顶支重放误差 ≤ 1 cm⁻¹；
- time-reversal 和等价 K 点的频率离散不得超过既定数值收敛误差。

### 4.2 项目方法相对 direct DFPT

- K 邻域 line MAE < 10 cm⁻¹；
- K 中心绝对误差 < 15 cm⁻¹；
- cusp depth 和 slope jump 的相对误差均 < 20%；
- 300/450/600 K 三个条件使用同一电子响应构造，不允许每个温度单独换一套经验 cusp 参数；
- 随 `smearing/degauss (Ry)` 增大，cusp 应连续变钝，不允许出现由模式交换造成的非物理跳点。

通过层级 B 后，可以形成“本项目方法在有限晶格温度背景上定量恢复 Kohn cusp”的阶段性结论。层级 C 通过后，才适合进一步声明该方法能够追踪接近零展宽时的尖锐极限。

## 5. 实验步骤与排期

### A0：复用现有数据的快速可行性图

**耗时：6–10 h；本地或 RTX 2060，GPU 非必需。**

1. 审计三温度 DFPT CSV、EPW 离散点、L0/Q0 动力学矩阵及单位；约 1–2 h。
2. 写统一的模式追踪和 cusp 指标程序；约 2–3 h。
3. 画 direct DFPT 三温度 K 区放大图；约 1 h。
4. 用现有 EPW 离散点在 L0/Q0 背景上做 q-space 替换，画项目方法的初步离散点图；约 2–3 h。
5. 自动生成指标表和输入哈希；约 1 h。

**输出：**一张两种方法的离散点对比图、一张指标表、一份是否进入 dense-q 的判断。该阶段能验证信号和实现方向，不能代替连续 dense-q 曲线或独立的新 DFT 验证。

### B0：三温度 dense-q 连续 cusp

**前置条件：**保存 EPW coarse-grid/Wannier 插值文件的机器恢复可达。  
**耗时：6–10 h。**

1. 固定 K 两侧对称的 29 点 q 列表，取 (t=m/240,\ m=226,\ldots,254)。这是 `nkf=720` 下覆盖现有 DFPT 参考区间的全部等间距可公度点；约 0.5 h。
2. 对 300/450/600 K 对应的三个 `degauss (Ry)` 跑 EPW dense-q 后处理；历史速度约 53 s/q，两个温度并行、第三个接续，预计约 1–2 h。
3. 分别与 L0 classical TDEP、Q0 quantum SSCHA/SCPH 背景组合；约 2–4 h。
4. 计算共同指标、对称性检查并画最终 K 区对比图；约 2 h。

**输出：**300/450/600 K 三个面板，每个面板同时给出 direct DFPT、MLIP+q-space(L0) 和 MLIP+q-space(Q0)；附 cusp depth、slope jump、line MAE 与 K 点误差表。

### C0：近零展宽 direct DFPT 加密与项目方法复核

**前置条件：**两台 V100 均恢复可达。  
**耗时：24–36 h；含排队和重试按 36–48 h 预留。**

1. 在 `degauss=0.0006333623 Ry, k=192×192×1` 下增加 8–10 个 K 邻域 q 点；历史每点约 3–5 h，两台 V100 对称拆分，约 14–20 h 墙钟。
2. 在固定最低 `degauss` 下对 K 中心及左右关键点做 k 网格复核；约 8–12 h，可与末批 q 点部分并行。
3. 用相同 q 点运行 dense EPW 电子项，并与 L0/Q0 背景组合；约 1–2 h。
4. 做 `degauss -> 0` 的 $\omega^2$ 线性/二次外推对照、窗口稳定性和最终绘图；约 2–3 h。

外推只作为辅助诊断。最终结论应以直接算到的最低有限 `degauss (Ry)` 为主，不能把外推值写成 `smearing=0.00` 的直接 DFT 数据。

## 6. 三台机器的分工

| 机器 | 本计划中的任务 | 说明 |
|---|---|---|
| 本地 / RTX 2060 | 数据审计、模式追踪、矩阵组合、指标和绘图 | 当前可立即开始；主要是 CPU/内存任务，RTX GPU 可保持空闲 |
| V100-A | C0 中 K 左侧/一半新增 direct DFPT q 点，随后做部分 k 网格复核 | 恢复 Tailscale 可达后启动 |
| V100-B | B0 dense-q EPW；C0 中 K 右侧/另一半 direct DFPT q 点 | 优先使用原 EPW 插值文件所在机器，避免复制大体积中间文件 |

RTX 2060 当前约有 72 GiB 空闲空间，不建议把整套大型 EPW 中间文件镜像过去；只同步 q 列表、最终动力学矩阵、频率和验收产物。

本轮暂不释放 375/525 K 的新 holdout DFPT 队列。两项各约 67–77 V100 box-h，会直接延迟 cusp 的近零展宽加密。应先完成层级 B，再决定是否用 375/525 K 做未见温度验证；若随后两台 V100 并行运行这两个 holdout，预计还需约 3–4 天墙钟。

## 7. 决策点

### A0 通过

若两种方法在已有点上都得到正的 cusp depth，且项目方法保持现有 `<20%` kink 误差，则立即进入 B0。

### A0 不通过

- 若静态 EPW 离散点通过，而与 L0/Q0 组合后 cusp 消失，优先检查 q6 电子项是否重复加减、模式投影和本征矢追踪；
- 若静态 EPW 离散点本身不能重放已有门槛结果，先停止新增 DFT，修复数据版本或实现；
- 不通过时不扩大经验 Dirac 拟合，也不靠加密平滑插值制造尖点。

### B0 通过

形成三温度阶段性结论和可用于周报/论文草图的主图。此时可以判断“有限 `degauss`、有限 lattice temperature 下能否复现 cusp”。

### 是否进入 C0

如果研究表述只要求有限温度定量复现，B0 已足够。若要与文献中的尖锐 Kohn anomaly 或 `degauss -> 0` 极限比较，必须进入 C0。

## 8. 计划完成后的结论边界

层级 B 成功时可以写：本项目方法通过 MLIP 的有限晶格温度背景与显式 q-space 电子响应组合，在 300/450/600 K 对应条件下恢复了 direct DFPT 的 K 点 cusp，并满足预先固定的 K-line、K 点和 slope-jump 误差门槛。

在层级 C 完成前不能写：已经直接得到 `smearing=0.00` 的 DFT 结果，或已经证明严格零温非解析极限。有限 k 网格和有限展宽会把数学上的尖点变成可分辨但仍有宽度的 cusp；本计划验证的是随网格加密和展宽降低而收敛的数值复现。

## 9. 直接复用的输入与脚本

- 300 K direct DFPT：`results/graphene_physical_fd_dfpt/campaigns/FD300_LINE/graphene_FD300_LINE_dfpt.csv`
- 450 K direct DFPT：`results/graphene_physics_temperature/post_p4_feasibility/source_p4_450/dfpt/FD450_LINE/graphene_FD450_LINE_dfpt.csv`
- 600 K direct DFPT：`results/graphene_physical_fd_dfpt/campaigns/FD600_LINE/graphene_FD600_LINE_dfpt.csv`
- 近零展宽扫描：`results/graphene_physical_fd_dfpt/campaigns/FD0_SCAN/graphene_FD0_SCAN_dfpt.csv`
- L0/Q0 有限晶格温度结果：`results/graphene_physics_temperature/post_p4_feasibility/S0_unified_short/`
- EPW 三温度静态门槛：`results/graphene_physics_temperature/post_p4_feasibility/E0_epw_matched/k18_q9_ex1_pifroz/`
- dense-q EPW 启动脚本：`scripts/v100/run_graphene_e0_multitemp_spectral_gate.sh`
- 近零展宽外推诊断：`scripts/smearing_kink/analyze_graphene_fd0_extrapolation.py`

## 10. 执行记录

### A0：已完成并通过

2026-08-10 已用现有数据完成五点 mode-projected q-space 验证，结果目录为：

```text
results/graphene_kohn_cusp_two_methods/A0_existing_data/
```

direct DFPT 的 cusp depth 为 1.158–2.344 cm⁻¹；有限 q6/实空间算子只保留 0.106–0.225 cm⁻¹ 的浅下凹；改为直接 q-space 投影后，L0/Q0 六个组合得到 1.205–2.090 cm⁻¹。六个组合的 kink 相对误差为 0.04%–15.46%，均通过 20% 门槛。

输入重放检查同时通过：operator Hermiticity 残差为 0，保存的 L0/Q0 K 点重放最大误差为 $2.27\times10^{-13}$ cm⁻¹，连续模式追踪的最小相邻本征矢重叠为 0.9980。

### B0：已完成并通过

第一次使用普通等间距 61 点时，EPW 在产生自能结果前以 `k+q does not fall on k-grid` 停止。原因是任意 q 点与 `nkf=720` 不可公度。失败目录保留为诊断记录，没有覆盖原 EPW restart。

修正版采用 $q=m/720$、$t=m/240$、$m=226,\ldots,254$，即现有 DFPT 参考区间内全部 29 个等间距可公度点。300/600 K 于 2026-08-10 18:52 HKT 启动并行计算，450 K 自动接续；三组于 19:43 HKT 全部结束，19:47 HKT 完成矩阵和曲线验收。

B0 状态为 `passed_quantitative_dense`。direct DFPT 的 cusp depth 为 1.162–2.351 cm⁻¹，有限 q6 算子只有 0.106–0.225 cm⁻¹，29 点 q-space 投影得到 1.205–2.090 cm⁻¹；六个 L0/Q0 组合的 kink 相对误差为 0.30%–15.07%。Hermiticity 和 time-reversal 频率残差为 0，rank-one 标量重放最大误差为 $6.82\times10^{-13}$ cm⁻¹，K-star 最大频率散布为 $4.36\times10^{-8}$ cm⁻¹，最小相邻模式重叠为 0.9998。结果表明，当前 q-space 修正能够在 300、450 和 600 K 的两种有限晶格温度背景上定量保留 K 点 cusp。

### C0：q 点密度与跨机器验收已完成

V100-A 于 2026-08-11 22:26 HKT 完成 K 左侧五个新 q 点和重复 K 点；V100-B 于 21:03 HKT 完成右侧五个新 q 点。本地监控于 22:31 HKT 自动同步两端 CSV、manifest 和日志，并将既有 $t=0.985,1.000,1.015$ 与十个新增点合并为 13 点 K 线。

C0 状态为 `passed_q_density`。两台 V100 独立计算的 K 点六个模最大差为 0；六个对称窗口的 cusp depth 从 2.311 增至 14.886 cm⁻¹，全部为正；左右最大差为 0.497 cm⁻¹，K 是全线最低点。绝对值 cusp 拟合的 RMSE 为 0.162 cm⁻¹，slope jump 为 1385.15 cm⁻¹/t。结果表明，在 `degauss=0.0006333623 Ry, k=192×192×1` 下，direct DFPT 已定量复现 K 点 cusp，且 q 点密度和跨机器一致性通过验收。

C0 尚有一个数值收敛子项：在固定最低 degauss 下复核 K 中心关键点的 k 网格，使相对 k192 的 MAE 不超过 1 cm⁻¹。在该子项完成前，当前结果应写为“最低已直接计算的有限展宽”，不能写成 `smearing=0.00` 或严格零温非解析极限。

### C1：固定 degauss 的 k 网格复核正在运行

2026-08-11 23:02 HKT，两台 V100 同时启动 `degauss=0.0006333623 Ry` 下的 k168/k216 中心五点复核。为平衡墙钟时间，V100-A 计算 k168 的 $t=0.993,0.997,1.000$ 和 k216 的 $t=1.003,1.007$；V100-B 计算其余对称点。每个 k 网格最终均有 $t=0.993,0.997,1.000,1.003,1.007$ 五点，并与 k192 的相同点比较。

两台机器均已完成输入检查并进入 k168 SCF，8 个 MPI rank 正常占用物理核。按 C0 实测耗时和 k 点数平方缩放，预计在 2026-08-12 21:00–23:30 HKT 完成，保留约 3 小时波动。完成后本地监控会自动同步四组结果，并检查 k216 相对 k192 的顶支 MAE 是否不超过 1 cm⁻¹。
