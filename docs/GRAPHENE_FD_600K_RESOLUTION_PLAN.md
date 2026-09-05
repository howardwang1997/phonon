# Graphene 600 K physical-FD 短程模型修复计划与结果

**日期：**2026-07-31  
**结果更新：**2026-08-01  
**目标：**解决 600 K 短程 residual 模型无法同时通过热构型受力和谐波保持门槛的问题，并在合格的短程背景上定量复现有限晶格温度 Kohn anomaly。

## 1. 当前基线

现有 600 K delta pilot 的三个候选均未同时通过固定门槛：

| 模型 | thermal force RMSE (meV/Å) | thermal max (meV/Å) | harmonic RMSE (meV/Å) |
|---|---:|---:|---:|
| delta32 | 38.77 | 294.22 | 19.09 |
| delta16 | 41.90 | 237.49 | 26.45 |
| delta32low | 38.52 | 279.45 | 20.69 |
| 固定门槛 | ≤50 | ≤250 | ≤14.47 |

600 K thermal delta target 的 RMSE 为 `238.7 meV/Å`，300 K 为 `121.1 meV/Å`；谐波 replay delta target 只有 `7.37 meV/Å`。当前训练集在结构数上平衡，但未归一化 force MSE 仍由 600 K thermal target 主导。

provisional 长程力在 600 K test 上的 RMS 为 `36.1 meV/Å`。扣除该项后，v11 的总力 RMSE 从 `214.7` 降到 `205.1 meV/Å`。因此长程算子需要随后校准，但它不是当前约 `200 meV/Å` residual 的主要来源。

## 2. 固定验收标准

本轮不修改已有门槛：

- thermal total-force RMSE `≤50 meV/Å`；
- thermal total-force 最大分量误差 `≤250 meV/Å`；
- harmonic replay RMSE 不超过 frozen v11 的两倍；当前上限为 `14.47 meV/Å`；
- 通过 force gate 后，`base + delta + provisional LR` 的 `0.25/0.5 fs` 轨迹均需保持有限能量、受力和温度；每条轨迹的 LR 力随后从 checkpoint 精确扣除；
- 三个 MD seed 的 TDEP 全谱两两 MAE `<5 cm⁻¹`；
- static direct-DFPT q-space 长程项的 Γ/K 留组交叉验证 MAE `<5 cm⁻¹`；
- 把 static long-range correction 原样转移到短程 TDEP 后，相对 DFT-TDEP 的 Γ/K line MAE `<10 cm⁻¹`，Γ/K 最高光学模绝对误差 `<15 cm⁻¹`，K kink 相对误差 `<20%`。该项作为温度迁移对照；若失败，则按第 4 节的固定分支校准 lattice-temperature-specific correction，并使用相同数值门槛以及中心三点整块外推复核。

最终数值门槛不变，但验收层次作如下修订：provisional 实空间谐波算子用于 force target 分解、总力重构和 MD 采样，使轨迹包含训练 target 中被扣除的 Kohn 曲率；每条轨迹完成后，从保存的总力中精确扣除同一个算子，再拟合 short-range TDEP。最终谱不直接采用该 provisional FC2。static direct DFPT 响应先用于温度迁移对照；对照失败时，再用同一 lattice temperature 的 DFT-TDEP 校准 q-space 系数。这样既保持总势采样，也能把 electronic-smearing 基底、lattice-temperature 系数和 provisional 采样算子分开检验。

## 3. 数据隔离

模型选择仍使用前三批 physical-FD 标签中的 development 数据。由于原先三个 thermal test 构型已经参与过候选比较，它们只用于 checkpoint/权重选择，不再作为最终测试。

在标签计算前固定 600 K DFT-MD 轨迹中此前从未使用的 15 个 snapshots：

`3, 7, 11, 14, 18, 22, 26, 30, 32, 35, 39, 43, 47, 51, 56`

这 15 个构型全部作为新 force holdout，不进入训练、early stopping、replay 权重选择或 checkpoint 选择。它们来自同一条 600 K DFT-MD 轨迹，因此能检验未见构型，但不等同于独立 MD seed；最终结论需保留这一范围说明。

## 4. 实验顺序

### E1：weighted-replay delta32

固定 delta32 架构、随机种子和学习率，先扫描 harmonic replay 的 configuration weight：`4、8、16`；在 `8` 与 `16` 之间增加一次 `12` 的局部细化。每 5 个 epoch 保存 checkpoint，最多 180 epochs。选择指标为三个固定门槛的最差归一化比值：

\[
S=\max\left(\frac{\mathrm{RMSE}_{th}}{50},
             \frac{\mathrm{max}_{th}}{250},
             \frac{\mathrm{RMSE}_{harm}}{14.47}\right).
\]

先在 development 数据上冻结 replay weight 和 epoch，再读取新的 15 构型 holdout。如果没有单个 checkpoint 同时通过三项 development 门槛，则保持所选 weight、epoch、架构和随机种子不变，把 validation 中 6 个唯一 thermal snapshots 和已经用于 checkpoint gate 的 3 个 thermal snapshots 加入最终训练集，各重复 2 次后重训；validation 中的 harmonic replay 不追加，继续用于曲率保持诊断。最终模型从保存的 checkpoint 精确导出冻结 epoch，不采用训练程序自动保存的“验证集最佳 epoch”。这 9 个 thermal snapshots 在最终重训后的误差只记为 in-sample/development 诊断；模型是否合格只由此前未读的 15 构型 holdout 决定。

### E2：600 K 未见构型 force holdout

对 15 个固定 snapshots 使用与前三批标签完全相同的设置：

- lattice temperature：600 K 轨迹构型；
- Fermi–Dirac `degauss=0.0038001738 Ry`；
- `ecutwfc/ecutrho=60/240 Ry`；
- `8×8×1` k 网格；
- PBE/ONCV，固定 6×6 graphene 超胞。

冻结的 weighted/checkpoint 模型只评估一次。若未通过，不在这些 15 个构型上继续调参；后续需要把它们重新定义为 development 数据，并另建独立 seed holdout。

### E3：轨迹与 TDEP

只有 E2 通过后才运行：

1. 用 `base + delta + provisional harmonic LR` 跑 `0.25/0.5 fs` 轨迹；
2. 运行三个 600 K seeds，从每条 checkpoint 力中精确扣除 provisional LR，并合并 360 个 snapshots 拟合短程 TDEP；
3. 计算 seed spread、温度和轨迹健康指标；
4. 同一冻结模型在平衡结构上做有限位移，得到 static short-range FC2；
5. 短程 seed gate 通过后，才进入 q-space 长程项转移验收。

### E4：物理 Fermi–Dirac q-space 长程项

`FD600_LINE` 已完整同步。先用 static short-range FC2 与 direct-DFPT dense line 的差值拟合 squared-frequency rounded-cusp correction，并做等距离 q-group 留组交叉验证。由于 delta 的 force target 已显式扣除 provisional LR，温度迁移对照把包含 Γ/K 截距的完整 held-group correction 投影到 thermal dynamical matrix；减去锚点截距的版本作为不参与选择的 control 同时输出。随后在 DFT-TDEP FC2 上计算 Γ/K line MAE、最高光学模误差和 K kink 误差。

这一转移检验会显式区分 electronic smearing 与 lattice temperature。如果完整 static correction 在一致的短程模型上仍不能通过有限温度 Γ/K 与 K kink 门槛，则不登记为最终模型，也不放宽门槛；改用同一 lattice temperature 的 DFT-TDEP squared-frequency residual 校准 `rounded cusp + q²`。除等距离 q-group 留组外，Γ/K 锚点附近三个点作为一个整块从拟合中移除并外推，使用同一组 `<10/<15/<20%` 门槛。这里 DFT-TDEP 是 calibration target，因此通过后只登记“该温度已校准复现”，不登记跨温度或独立 DFT-MD trajectory 的预测能力。`FD300_LINE` 完成后提供第二个 lattice-temperature 约束；`FD0_SCAN` 仍只用于 `degauss→0` 静态参考，不参与 600 K 短程模型选择。

300 K 的一致模型已有三条稳定轨迹、pooled 360-snapshot short-range FC2 和 static short-range FC2。正式检验先从每条轨迹 checkpoint 逐 seed 扣除同一个 provisional 实空间谐波算子并重拟合 short-range TDEP，再计算 seed spread。扣除后，三条 short-range TDEP 的 K 最高光学模相对 physical-FD DFT-TDEP 误差为 `61.3–73.9 cm⁻¹`，说明被扣除的 LR 锚点贡献必须在 q-space 阶段加回。`FD300_LINE` 完成并经 Tailscale relay 后，2060 自动运行同一完整 q-space 转移，作为第二个 lattice-temperature 点；它不改变已经冻结的 600 K 模型或 600 K 独立 force holdout 判定。

## 5. 失败后的下一步

如果 weighted-replay 的所有 checkpoints 都未通过 E1，则先执行上述固定超参数的最终重训；如果该冻结模型仍未通过 E2，不再继续扩大普通 MACE 网络。下一轮改用显式分解：

\[
E=E_{v11}+\tfrac12u^T\Delta\Phi_{LR}u
  +\tfrac12u^T\Delta\Phi_{SR}u+E_{anh,local}.
\]

其中 `ΔΦ_SR` 仅用训练构型、对称性约束和 ridge regularization 拟合；delta MACE 只学习去除线性项后的局域非谐 residual，并用成对 `±u` 小位移约束其平衡点 Hessian。这样避免让同一个小网络同时承担大幅热受力修正和谐波曲率保持。

## 6. 执行结果与当前状态

- RTX 2060：`4、8、12、16` 的全部 checkpoints 已扫描。最终在独立标签开放前冻结 weight `12`、epoch `175`；development thermal force RMSE `43.31 meV/Å`、最大分量 `210.74 meV/Å`、harmonic replay RMSE `10.29 meV/Å`，最差归一化分数 `0.866`。冻结模型 SHA-256 为 `3cf7f697602fed0b71cece204d797821c1d6faff0e0138892f69b06aafe4ddf3`。
- V100-A/B：不重叠的 `8 + 7` 两个 DFT shard 均已完成；A 对应 `3,7,11,14,18,22,26,30`，B 对应 `32,35,39,43,47,51,56`。2060 按 snapshot index 合并并核对 JSON/XYZ、重复项、设置和分片哈希后，只执行了一次冻结模型验收。
- E2 已通过：15 构型 reconstructed total-force RMSE/max 为 `42.81/215.09 meV/Å`，harmonic replay RMSE 为 `10.29 meV/Å`；对应门槛为 `50/250 meV/Å` 和冻结 v11 harmonic RMSE 的 2 倍。v11 short-range baseline 在同一 15 构型上的 RMSE/max 为 `228.63/1348.68 meV/Å`。
- E3 已完成：三个 seed 各保存 120 个 snapshots；均温为 `607.87/586.47/601.79 K`，最高温为 `716.72/751.39/740.36 K`。full-force TDEP 的最低频率绝对值均小于 `3×10⁻⁵ cm⁻¹`。逐帧扣除同一 provisional LR 后，short-range TDEP 的全谱两两 MAE 为 `3.3501–4.0768 cm⁻¹`，通过 `<5 cm⁻¹` 门槛。
- E4 的 unchanged static transfer 对照已完成并失败：static direct-DFPT 留组拟合本身通过，Γ/K line MAE 为 `0.7080/0.4996 cm⁻¹`，static K kink 相对误差为 `18.61%`；转移到最终 pooled-360 的 600 K short-range TDEP 后，Γ/K line MAE 为 `33.8750/98.2559 cm⁻¹`，Γ/K 顶支误差为 `28.6321/109.2036 cm⁻¹`，K kink 相对误差为 `3260.03%`。
- lattice-temperature-specific 分支已完成并通过：最终 `rounded cusp + q²` 的 Γ/K line 留组 MAE 为 `0.002601/0.004440 cm⁻¹`，Γ/K 顶支误差为 `0.001350/0.001399 cm⁻¹`，K kink 相对误差为 `2.15%`；中心三点整块外推的 K kink 相对误差为 `2.42%`。去掉 `q²` 的最终 pooled-360 消融为 `51.37%`，未通过 kink 门槛。
- snapshots 文件在本地与 V100-A 的 SHA-256 均为 `e486c9cf569618d54de555a0da61a083f3424aebfe37047cca1c250a160fe7b0`。
- V100-B 的 snapshots SHA-256 也为 `e486c9cf569618d54de555a0da61a083f3424aebfe37047cca1c250a160fe7b0`。

最终数值、是否通过和适用范围已写入 [`WEEKLY_2026-07-29.md`](WEEKLY_2026-07-29.md)。完整判断见 [`calibrated_acceptance.json`](../results/graphene_fd_delta_weighted/T600_TDEP/calibrated_acceptance.json)，static-transfer 对照见 [`acceptance.json`](../results/graphene_fd_delta_weighted/T600_TDEP/acceptance.json)。当前结论为“600 K 已校准的有限温 Kohn anomaly 定量复现”；独立 DFT-MD trajectory 和 300/600 K 联合温度约束仍是后续验证。
