# Graphene 零/有限 smearing 声子谱与可迁移长程微调实验计划

**状态更新时间：** 2026-08-15 16:30 NZST  
**当前算力：** V100-A、V100-B 通过 Tailscale 地址连接  
**第一阶段范围：** graphene K 点附近 A' 支，不要求先完成整条声学谱  
**电子积分记法：** 零 smearing 使用 `occupations='tetrahedra_opt'`；有限条件直接记录 `smearing/degauss (Ry)`，不换算为电子温度

**运行更新（2026-08-16 17:38 NZST）：** S4a 的 k192→k240 cusp-depth 变化为 15.13%/14.98%，超过 10% 门槛。k288 已于 17:34:15/17:34:28 NZST 在 V100-B/A 启动，两边均已进入实际 k288 SCF；随后每台依次计算 K 和对应方向的 `d=0.003`。自动监控已启动，预计 8 月 17 日约 09:00 完成，未重试时保守截止为 11:30。

**重启恢复（2026-08-17 01:10 NZST）：** 两台实例重启后通过 Tailscale 恢复连接。A 的 SCF 完整，K 点保留了 QE recovery 检查点；B 的 SCF 和 K 点均完整。已清理两台机器各 29 GiB、且已本地归档审计的 k240 `_ph0` 可重算响应 scratch。A 于 01:09:13 从 K 点检查点续算，B 于 01:09:06 复用 K 并重启 K→M `d=0.003`。修订后的中央完成时间为 8 月 17 日约 14:00，未再次中断时保守截止为 18:00。

## 1. 当前结论和完成时间

现在运行的是 S4a：在 k240 网格上重算零 smearing 的两个极近 K 点：

- V100-A：K→Γ，`d=0.003`；
- V100-B：K→M，`d=0.003`；
- 两边均复用已经审计过的 k240 SCF，只运行一个一般 q 点的 `ph.x`；
- 任务于 2026-08-15 14:40:12/14:40:23 NZST 启动；
- 2026-08-15 16:30 NZST 检查时，两边 `ph.x` 已运行约 1 h 50 min，CPU 占用约 105–106%，没有退出或停滞。

同一 k240 设置下，先前 K→Γ `d=0.007` 实测耗时 8 h 06 min，K→M `d=0.007` 实测耗时 8 h 39 min。据此估计：

| 项目 | 预计时间（NZST） |
|---|---:|
| V100-A 完成 | 8 月 15 日约 22:47 |
| V100-B 完成 | 8 月 15 日约 23:19 |
| 自动同步、矩阵审计、k192/k240 对比和诊断图 | 23:20–23:45 |
| 无重试时的保守截止 | 8 月 16 日 00:30 |

本轮完成标志和结果文件为：

- `results/graphene_k_cusp_nosmear/monitor_d003_kgrid/DONE`；
- `results/graphene_k_cusp_nosmear/S4a_d003_kgrid/d003_kgrid_summary.json`；
- `results/graphene_k_cusp_nosmear/S4a_d003_kgrid/d003_kgrid_diagnostic.png`。

本轮只回答一个问题：k192 的 `d=0.003` cusp 深度是否已经收敛。它不会直接给出一个新的、通过盲测的 MLIP+长程微调声子谱。

## 2. S4a 完成后的分支

判定标准已经写入分析脚本，结果出来后不修改：

- k192→k240 的 A' 频率变化不超过 `1 cm^-1`；
- cusp depth 相对变化不超过 `10%`；
- 模式重叠不低于 `0.95`。

### 分支 A：k192 已收敛

若两方向都通过，说明 S4 盲测在 `d=0.003` 的约 60% cusp-depth 误差主要来自 v1 线性 q-space 模型把极近 K 变化过度平滑，不是 k 网格伪影。随后执行：

1. 不再增加普通 MLIP 训练构型；
2. 把已经揭盲的 S4 六点转入 development，不能再用于独立验证；
3. 冻结连续、有限斜率的长程响应形式；
4. 在完整动力学矩阵上拟合投影修正，不直接拟合绘图曲线；
5. 在计算新标签前固定新的零 smearing 测试点和门槛；
6. 用新 DFPT 点做一次独立盲测，再生成最终对比图。

### 分支 B：k192 未收敛

若任一方向未通过，不重拟合模型，先计算 k288 的 K、K→Γ `d=0.003` 和 K→M `d=0.003`。两台 V100 并行预计需要 14–18 h 墙钟；若在 8 月 16 日 00:30 前释放，预计 8 月 16 日 15:00–19:00 NZST 得到 k 网格结论。只有 k240/k288 通过相同门槛后，才进入模型修改。

## 3. Graphene 阶段的执行顺序

### G0：完成当前 S4a

- **数据：** 两个 k240、`d=0.003` 完整动力学矩阵；
- **剩余时间：** 约 7 h，包括同步和分析；
- **结论：** 区分模型形状误差与 k 网格误差；
- **图：** 只画 k192/k240 的两方向 cusp-depth 诊断。

### G1：零 smearing 长程模型 v2 冻结

主模型写成

\[
D(\mathbf q,s)=D_{\mathrm{SR}}^{\mathrm{MLIP}}(\mathbf q)
+P_{A'}(\mathbf q)\,\Delta\lambda(\mathbf q,s)\,P_{A'}^\dagger(\mathbf q).
\]

零 smearing 的 `d→0` 极限要求频率连续、单边斜率有限。初始采用 rank-one A' 投影；只有完整矩阵残差或模式重叠表明一个投影不足时，才启用 rank-two 消融。绘图时在冻结公式上密集取点得到平滑曲线，DFPT 仍只以离散标记显示，不用样条曲线替代物理模型。

- **输入：** 现有 pilot、development 和已揭盲 S4 数据；
- **新 DFT：** 0；
- **本地计算：** 4–6 h；
- **产物：** 模型参数、代码哈希、数据清单、新盲测点及门槛的冻结 manifest。

### G2：新的零 smearing 独立测试

建议在模型冻结前固定以下六点：

\[
d\in\{0.005,0.013,0.021\},\qquad \mathrm{K\to\Gamma/K\to M}.
\]

这组点同时覆盖极近 K、过渡区和外侧线性区，并且没有参与 v1 或 v2 的选择。

- **新标签：** 6 个 k192 完整动力学矩阵；
- **算力：** 约 30–36 V100 box-h；
- **两台 V100 墙钟：** 15–18 h；
- **后处理：** 1–2 h；
- **结论：** v2 是否真正复现零 smearing cusp，而不只是回归已经揭盲的 S4 点。

零 smearing 验收标准：

- A' 支 MAE `≤2 cm^-1`，最大误差 `≤4 cm^-1`；
- 每个测试距离的 cusp depth 相对误差 `≤15%`；
- 两侧斜率和 slope jump 相对误差 `≤15%`；
- 最小模式重叠 `≥0.95`；
- Hermiticity 误差 `≤1e-10`；
- 结果必须来自完整矩阵重建。

### G3：有限 smearing 共享模型

现有开发数据已经包含三条 K 邻域 DFPT 线：

| `smearing/degauss (Ry)` | k 网格 | K 邻域点数 | 用途 |
|---:|---:|---:|---|
| 0.0019000869 | 144 | 11 | development |
| 0.00285013035 | 144 | 11 | development |
| 0.0038001738 | 120 | 11 | development |

已有共享 Dirac 原型虽然全线 MAE 较小，但在 0.00285013035 和 0.0038001738 Ry 的 K kink 相对误差分别约 23% 和 28%，没有通过既定门槛。因此下一步不能为每个 smearing 单独拟合一条曲线，需要把零 smearing 的非解析极限与有限 smearing 的连续 rounding 写进同一个响应核。

有限 smearing 模型开发与 G2 的远端 DFPT 可以并行进行：

- **新 DFT：** 0；
- **本地开发与矩阵重放：** 6–10 h；
- **必须固定：** 共享参数、smearing 依赖、模式投影、外推范围和未见 smearing 验收点。

### G4：未见 smearing 的独立测试

第一轮固定一个未参与拟合的条件：

\[
s_{\mathrm{test}}=0.003325152075\ \mathrm{Ry}.
\]

只计算 K 和两方向的 `d={0.005,0.013,0.021}`，共 7 个完整动力学矩阵。

- **算力：** 约 18–24 V100 box-h；
- **两台 V100 墙钟：** 10–13 h，包括新 SCF；
- **结论：** 一个共享模型能否在未见 smearing 上同时预测 K 点频率、cusp 深度和 rounding 宽度；
- **确认实验：** 第一轮通过后，再以 `0.002375108625 Ry` 做同规模第二次确认；第一轮失败时不生成第二组标签，先修改电子响应形式。

有限 smearing 的主要验收标准与 G2 相同，另加：

- rounding 宽度或等价 crossover 尺度相对误差 `≤15%`；
- 未见 smearing 的误差不能通过重新拟合体系参数降低；
- 相对于不含电子响应核的 MLIP 基线，cusp-depth 误差至少降低 50%。

### G5：最终图和 graphene 阶段结论

最终图只展示 K 邻域 A' 支：

- direct DFPT 离散点；
- frozen short-range MLIP；
- MLIP + frozen long-range adapter 的平滑实线；
- 零 smearing 和有限 `smearing/degauss (Ry)` 分面显示；
- legend 放在坐标轴外，不遮挡 K 附近数据；
- development 与独立 test 使用不同标记。

后处理预计 2–3 h。

## 4. 什么时候能得到哪一级结论

以下日期假定两台 V100 持续可用、没有失败重试，并且 G2/G4 一次通过。通过与否不能预先保证，但到对应时间应能得到明确的通过/失败结论。

| 结论 | 快速路径 | 触发 k288 时 |
|---|---:|---:|
| k192 的极近 K 点是否收敛 | 8 月 15 日 23:45 前 | 8 月 16 日 15:00–19:00 |
| 零 smearing v2 新盲测结论 | 8 月 16 日晚至 8 月 17 日凌晨 | 8 月 17 日晚 |
| 一个未见有限 smearing 的结论 | 8 月 17 日中午至晚间 | 8 月 18 日中午至晚间 |
| graphene 零/有限 smearing 最终对比图 | 8 月 17 日晚 | 8 月 18 日晚 |

如果 G2 或 G4 未通过，对应日期仍会得到失败定位和下一项实验，而不是继续画一条看起来更平滑的曲线。一次失败后的修模和第二轮独立测试另预留 1–2 天。

## 5. 为什么现在不增加大量 MLIP 数据

当前 E1 交叉 smearing 力残差门槛已经通过：扣除长程 Kohn 算子后，30 个任务、6480 个力分量的残差 RMSE 为 `0.632 meV/Å`，最大绝对残差为 `2.320 meV/Å`。这说明现阶段主要限制来自 q-space 电子响应的形状和收敛，而不是短程 MLIP 缺少数百个热构型。

因此 graphene 阶段只增加：

- 6 个零 smearing 独立 q 点；
- 7 个未见有限 smearing q 点；
- 必要时再增加 7 个第二未见 smearing 确认点。

不新增完整 DFT-MD 轨迹，也不为每个 smearing 重训一个 MLIP。

## 6. Graphene 通过后的迁移实验

可复用方法分为两个模块：

1. 共享短程 MLIP，加少量体系级 harmonic/LoRA 适配；
2. 共享物理长程响应核，加少量材料参数和异常模式投影。

第一轮材料集合按物理覆盖度选择，而不是在 graphene 上继续加深：

- graphene：Dirac Kohn anomaly；
- h-BN：有带隙的负对照，长程电子异常幅度应接近零；
- NbSe2：2H 金属/EPC/CDW 类；
- TiSe2：1T 软模或不同电子结构类；
- NbS2 或 TaSe2：整体系留出测试，不参与共享参数训练。

每个源体系的初始预算：

- 10–15 个短程 FC-distillation 构型，优先复用公开或仓库数据；
- 约 12–15 个 q–smearing 完整动力学矩阵；
- 一个未见 smearing 和若干未见 q 点作为锁定测试。

新体系先做零样本预测，再按 `0/4/8/12/20` 个长程标签绘制学习曲线；短程适配最多增加 15 个构型。若总标注超过 25 个仍不能通过门槛，该体系记为超出当前方法的少样本迁移范围，不继续堆数据直到通过。

迁移阶段的算力和时间目标为：

| 项目 | 目标预算 |
|---|---:|
| 单个新体系新增标注 | 不超过 25 个 |
| 单个新体系 DFT/DFPT | 35–60 V100 box-h |
| 两台 V100 墙钟 | 1–3 天 |
| 相对同密度直接 DFPT | 成本不超过三分之一 |
| 多体系共享模型与 leave-one-material-out | 1–2 周 |

这一阶段的核心结论不是“模型能画出曲线”，而是：在事先固定的新体系和新 smearing 上，是否能以不超过 25 个新标签达到直接 DFPT 的 Kohn-anomaly 精度，并实现至少约 3 倍的标注算力节省。

## 7. 立即执行清单

1. 保持当前 S4a 两条 `ph.x` 运行，不改输入、不重启；
2. 等待自动监控写入 `DONE`，读取 k192/k240 判定和诊断图；
3. 通过时立即冻结 v2 和六个新的零 smearing 测试点；未通过时释放 k288；
4. 两台 V100 运行 G2 时，本地并行完成 G3 共享 smearing 响应核；
5. G2 通过后释放 G4 的未见 smearing 标签；
6. 只在 G2 和 G4 都通过后生成正式 MLIP、MLIP+长程微调和 direct DFPT 对比图；
7. graphene 闭环完成后再启动多体系迁移，不提前消耗大规模训练数据。

## 8. 相关记录

- `docs/GRAPHENE_K_CUSP_NOSMEAR_REPO_METHOD_PLAN_2026-08-12.md`：零 smearing S1–S4 的原始方案和运行记录；
- `results/graphene_k_cusp_nosmear/S4_holdout/holdout_summary.json`：v1 盲测失败的定量结果；
- `results/graphene_k_cusp_nosmear/S4b_v2_development/`：揭盲后的 v2 开发性试验，不能作为独立验证；
- `results/graphene_physics_temperature/post_p4_feasibility/E0_shared_dirac/shared_dirac_fit.json`：现有有限 smearing 共享模型及失败门槛；
- `results/graphene_physics_temperature/post_p4_feasibility/E1_cross_degauss/analysis/e1_residual_gate.json`：短程局域 smearing 项的数据需求判断；
- `docs/CAMPAIGN_FINDINGS.md`：已有多材料数据效率结果；
- `docs/NCS_ROADMAP.md`：电子通道和可迁移方法的中长期路线。
