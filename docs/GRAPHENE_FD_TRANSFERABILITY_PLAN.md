# Graphene physical-FD 可迁移预测模型实验计划

**日期：**2026-08-01  
**当前起点：**300 K 和 600 K 已分别完成有限温校准；尚未完成跨温度、统一模型和独立轨迹预测验证。  
**最终目标：**冻结一个短程模型栈和一个温度依赖的 q-space 长程项，在未见 lattice temperature、未见 DFT 标签和独立 DFT-MD 轨迹上定量预测有限晶格温度 Kohn anomaly。

## 1. 本轮解决的问题和结论范围

本轮先限定在 physical-FD 路径：

\[
d(T)=0.0019000869\frac{T}{300}\ {\rm Ry},
\]

其中 `T` 是 lattice temperature，`d(T)` 是同时使用的 Fermi–Dirac smearing/degauss。主要 holdout 取 `450 K`，对应 `0.00285013035 Ry`。因此本轮检验的是 `300–600 K` 区间内、沿上述路径的温度插值，不把 lattice temperature 与 electronic smearing 的作用分开，也不检验区间外外推。

本轮 q-space 验收范围是现有 19 点 Γ/K dense line、最高光学支和 K kink。全路径 TDEP 差异继续报告，但在 q-space correction 扩展到更多方向并设置独立 q holdout 前，不宣称整个 Brillouin zone 的长程修正已经可迁移。

结论按下表逐级登记，低一级通过不能替代高一级：

| 等级 | 必须通过的实验 | 可以使用的结论 |
|---|---|---|
| C0 | 300/600 K 分别校准 | 两个温度点的已校准定量复现 |
| C1 | 一个冻结预测栈在未见 450 K 前瞻 holdout 上通过 | physical-FD 路径上 `300–600 K` 的插值预测通过 |
| C2 | C1，并在 450 K 独立 physical-FD DFT-MD 轨迹上通过 | 对独立轨迹有效的 bounded interpolation model |
| C3 | C2，并在未见的区间外温度上通过 | 在实际测试区间内的 bounded extrapolation model |
| C4 | 交叉改变 lattice temperature 和 degauss，并留出一个组合 | 可分别依赖 lattice temperature 与 electronic smearing 的二维模型 |

本计划的关闭条件是 C2。若最新周报截稿时只完成 C1，应明确写成“450 K 前瞻插值通过，独立 DFT-MD 轨迹仍在计算”，不把它写成已经完成 C2。

## 2. 当前证据和主要缺口

| 项目 | 300 K | 600 K |
|---|---:|---:|
| short-range seed 两两全谱 MAE | `1.98–2.71 cm⁻¹` | `3.35–4.08 cm⁻¹` |
| static correction 原样迁移后的 Γ/K line MAE | `39.09/19.27 cm⁻¹` | `33.88/98.26 cm⁻¹` |
| static transfer 的 K kink 相对误差 | `3018.84%` | `3260.03%` |
| 同温度 `rounded cusp + q²` 的 Γ/K 留组 MAE | `0.00491/0.02411 cm⁻¹` | `0.00260/0.00444 cm⁻¹` |
| 同温度校准后的 K kink 相对误差 | `4.90%` | `2.15%` |
| short model SHA-256 | `6560f7bb...34337666` | `3cf7f697...afe4ddf3` |

300 K 最终 JSON 目前只在 RTX 2060 端完成，P0 要先同步到本仓库并复核完整哈希。600 K 本地结果见 [`calibrated_acceptance.json`](../results/graphene_fd_delta_weighted/T600_TDEP/calibrated_acceptance.json)。

当前结果还不是可迁移预测模型，原因有五项：

1. 300 K 和 600 K 的 q-space 系数都用各自 DFT-TDEP 拟合，DFT-TDEP 是 calibration target；同温度 q-group 留组只检验 q 点插值。
2. 两个温度使用不同 short model。系数随温度的变化同时包含真实温度效应和 short-range background 改变，不能直接拟合温度规律。
3. 没有未参与任何拟合的第三个温度。两个端点只能确定一条直线，不能验证这条直线。
4. 现有 physical-FD force labels 来自同一批 legacy DFT-MD 构型的重新标注；600 K 的 15 构型 holdout 也仍来自同一轨迹。三条 TDEP seed 是 MLIP 轨迹，不是新的 DFT-MD 轨迹。
5. 当前没有 DFT-TDEP 的 block-bootstrap 误差区间。远小于 `1 cm⁻¹` 的 q-space 拟合误差不能替代对有限样本不确定度的估计。

## 3. 冻结的预测模型

### 3.1 统一短程模型

优先使用一个温度无显式输入的 joint delta MACE：

\[
E_{\rm SR}=E_{v11}+E_{\Delta,\rm joint}.
\]

训练只使用 300/600 K 已有 development labels 和 harmonic replay；450 K 的构型与标签均不得进入训练、early stopping、checkpoint 选择或 loss-weight 选择。

训练规则在运行前固定：

- 保持当前 `delta32` 架构、`seed=83`、`lr=0.001`、最多 180 epochs、每 5 epochs 保存 checkpoint；
- 300 K 和 600 K thermal lane 各占 thermal loss 的一半，避免 600 K 较大的 force amplitude 主导损失；
- harmonic replay weight 首轮固定为 `12`，只在所有 checkpoint 均未通过 endpoint development gate 时运行事先列出的 `16`；
- checkpoint 分数取 300 K thermal、600 K thermal 和 harmonic retention 各门槛归一化后的最大值；
- 450 K 标签开放前冻结模型文件、checkpoint、训练数据清单和 SHA-256。

原 600 K 的 15 构型 holdout 继续排除在 joint training 之外。它对原 600 K 模型是一次性 holdout，但其结果现在已经可见，因此对新 joint model 只能作为 regression diagnostic，不能重新登记为盲测。

在训练 joint model 前，先用现有 300/600 K 模型做 `model × temperature` 交叉受力矩阵。若某一个现有模型已经同时通过两个 endpoint gate，可直接冻结该模型并跳过 joint training。若 joint model 在 endpoint development 数据上仍不能同时通过，停止 450 K 标签任务，改用第 8 节的温度条件化失败分支。

### 3.2 温度依赖 q-space 长程项

Γ、K 两个区域都固定使用下式，不在 450 K 改变基底：

\[
\Delta\lambda_r(x,T)=c_{r0}(T)
+c_{r1}(T)\left[\sqrt{x^2+(4d(T))^2}-4d(T)\right]
+c_{r2}(T)x^2,
\]

其中 `r∈{Γ,K}`，`x` 是距 Γ 或 K 的线坐标，`Δλ` 是最高光学模 squared-frequency correction。`q²` 项保留；600 K rounded-only 消融的 K kink 相对误差为 `51.37%`，不再把它作为可选择模型。

统一 short model 冻结后，必须重新生成该同一模型在 300/600 K 的三 seed short-range TDEP，并重新拟合两端的六个 `c_rj`。随后只使用端点定义线性插值：

\[
c_{rj}(T)=\frac{600-T}{300}c_{rj}(300)
+\frac{T-300}{300}c_{rj}(600).
\]

两点不足以比较线性、二次或其他温度函数，因此本轮不做函数形式选择。450 K 只评价上述已冻结直线，不能重新拟合任何系数、宽度或 `q²` 权重。

### 3.3 450 K 计算前的冻结清单

`freeze_manifest.json` 至少记录：

- v11、delta model、checkpoint、300/600 K provisional sampling operator 的路径和 SHA-256；
- joint training/validation 文件、endpoint TDEP、端点 q-space acceptance JSON 的 SHA-256；
- 六条线性系数函数和 `d(T)` 公式；
- 450 K 的三个 MD seeds、时间步、平衡步数、snapshot indices 和 DFT shard 分配；
- Γ/K 的 19 个 q points、分支 projector、全部数值门槛和 bootstrap 规则；
- 当前 Git HEAD，以及本计划涉及脚本的逐文件 SHA-256。

冻结清单写入并核对后，才允许启动 450 K DFT force labels、DFPT 或 DFT-MD。原始 450 K 标签不能用于 checkpoint 选择；失败后也不能在同一 450 K 数据上调参并继续称其为 holdout。

## 4. 实验顺序

### P0：同步和溯源复核

1. 通过 Tailscale 从 RTX 2060 同步 300 K 的 `acceptance.json`、预测 CSV、图片、模型、checkpoint 和 gate JSON。
2. 生成 `results/graphene_fd_transferability/source_inventory.json`，登记 300/600 K 数据、模型、operator、TDEP、DFPT line 和全部哈希。
3. 确认 300 K 最终结果的 scope 仍为 development calibration，并复核本地周报已把过期的 `FD300_LINE 17/19` 状态更新为 `19/19`。

P0 只同步与核对，不改变任何已有验收结果。

### P1：现有模型的交叉温度诊断

复用 [`evaluate_graphene_fd_delta_model.py`](../scripts/smearing_kink/evaluate_graphene_fd_delta_model.py)，对固定 v11 加下列两个 delta model，计算 300/600 K development thermal labels 和同一 harmonic replay：

| predictor | 300 K labels | 600 K labels | harmonic replay |
|---|---:|---:|---:|
| 300 K delta | 必做 | 必做 | 必做 |
| 600 K weighted delta | 必做 | 必做 | 必做 |

门槛保持为 thermal force RMSE `≤50 meV/Å`、最大分量 `≤250 meV/Å`、harmonic RMSE 不超过 frozen v11 的两倍。结果写入 `cross_temperature_force_matrix.json`。这一阶段用于决定能否直接复用一个模型，不作为最终 holdout。

### P2：joint short model 与端点重算

若 P1 没有单模型通过两个温度：

1. 合并 300/600 K residual 数据，保留温度、degauss、轨迹和 snapshot provenance；
2. 按第 3.1 节训练 joint delta，并只用 endpoint development 数据选 checkpoint；
3. 在 300/600 K 各跑三个 MLIP seeds，每 seed 120 snapshots；provisional LR 用于采样，随后从 checkpoint 力中逐帧精确扣除；
4. 两个温度都要求 short-range TDEP seed 两两全谱 MAE `<5 cm⁻¹`；
5. 用同一 joint model 重建 300/600 K static short FC2 和 pooled-360 short TDEP。

同一 frozen delta model SHA-256 必须出现在两个温度的 static FC2 manifest、TDEP manifest 和最终 temperature-law manifest 中。

### P3：只用 300/600 K 固定温度函数

1. 在统一 short background 上分别拟合 300/600 K 的 `rounded cusp + q²` 六个系数；
2. 确认两端仍通过 Γ/K line MAE `<10 cm⁻¹`、Γ/K 顶支误差 `<15 cm⁻¹`、K kink 相对误差 `<20%` 和 projector replay `<1e-5 cm⁻¹`；
3. 写出 `temperature_law.json`，随后生成 450 K 的预测系数和最终 `freeze_manifest.json`；
4. 在任何 450 K DFT 结果回传前，把冻结 manifest 复制到两个 V100 和 RTX 2060，并核对哈希一致。

### P4：450 K 前瞻温度 holdout

固定设置：lattice temperature `450 K`，Fermi–Dirac smearing/degauss `0.00285013035 Ry`，6×6×1 graphene 超胞，PBE/ONCV，`ecutwfc/ecutrho=60/240 Ry`，force labels 使用 `8×8×1` k 网格。

实验分四部分：

1. 用同一冻结 predictor 和冻结的 450 K provisional sampling operator 运行三个 MLIP seeds；seed 0/2 使用 `0.5 fs`，seed 1 使用等物理时长的 `0.25 fs`，每条保存 120 snapshots。
2. 每个 seed 固定选择 indices `3,9,15,...,117` 共 20 个构型，合计 60 个 450 K DFT labels。15 构型 force primary subset 固定为每个 seed 的 `3,27,51,75,99`。
3. 用全部 60 个未见 DFT labels 拟合 holdout DFT-TDEP；先输出冻结预测，再读取 target 并计算误差。该 DFT-TDEP 只用于评价，不回写 450 K 系数。
4. 增加 `FD450_CONV` 和 `FD450_LINE`。先检查 `k=120→144`，若 Γ/K 顶支最大变化 `<1 cm⁻¹`，使用 `k=144` 的相同 19 点 dense line；该结果用于检验中间 degauss 的静态电子响应，不参与有限温系数拟合。

P4 通过后只能登记 C1。由于 60 个构型来自冻结 MLIP 的 on-policy 轨迹，它们是未见温度和未见 DFT 标签，但还不是独立 DFT-MD trajectory。

### P5：450 K 独立 physical-FD DFT-MD holdout

现有 [`dft_md_tdep.py`](../scripts/smearing_kink/dft_md_tdep.py) 把 smearing 固定为 `cold`，且以温度直接作为随机种子。运行 P5 前必须增加并测试：

- `--smearing fd|cold`，本实验固定 `fd`；
- 独立的 `--seed`；
- `smearing/degauss (Ry)`、seed、QE 设置和输入结构的 manifest；
- checkpoint 恢复后对设置与结构哈希的强制核对。

随后在任一 V100 上从平衡结构启动新的 450 K physical-FD DFT-MD：`degauss=0.00285013035 Ry`、`8×8×1` k 网格、`60/240 Ry`、`dt=0.5 fs`、至少 500 个 equilibration steps、60 个 snapshots、stride 20、`seed=45001`。前三个已保存构型另做 `k=8→12` force 检查；要求构型 RMSE `<2 meV/Å`、最大分量差 `<5 meV/Å`。若未通过，提高生产 k 网格并从同一初始结构重新运行，不能把不同 k 网格产生的轨迹片段混合。

冻结 predictor 在这条轨迹上做三项一次性评价：

1. DFT-MD 构型的 off-policy force RMSE/max；
2. 独立 DFT-MD TDEP 与冻结模型自行采样所得谱的全谱、Γ/K line、顶支和 K kink 差异；
3. 同一批 DFT-MD 构型上，DFT force TDEP 与冻结 predictor force TDEP 的 paired comparison。

P5 的 off-policy force 使用同一组 `50/250 meV/Å` 门槛；独立 DFT-MD TDEP 的 Γ/K line、顶支和 K kink 使用与 P4 相同的 `<10/<15/<20%` 门槛。全路径 MAE 作为范围诊断报告，不用于替代 Γ/K 主验收。

若 60 snapshots 的 bootstrap 区间过宽，只允许按事先固定的增量追加到 90、120 snapshots，不改变模型、温度函数或门槛。P5 通过后达到 C2。

### P6：外推与二维依赖

P5 通过后再做，不与本轮 450 K 模型选择混合：

- 区间外 holdout：优先 `750 K`、`degauss=0.00475021725 Ry`。它通过后只在 `300–750 K` 已实测范围内登记 bounded extrapolation；
- lattice temperature/degauss 解耦：至少计算 `(300 K,0.0019000869 Ry)`、`(300 K,0.0038001738 Ry)`、`(600 K,0.0019000869 Ry)`、`(600 K,0.0038001738 Ry)`，并额外留出一个中间组合。完成前不把本轮一维模型描述为两个变量可独立迁移。

## 5. 统一验收标准

所有门槛在 450 K 标签计算前写入冻结清单。

下列 q-space 门槛分别应用于 P4 on-policy target 和 P5 独立 DFT-MD target；不能用其中一项通过替代另一项。

| 层级 | 指标 | 门槛 |
|---|---|---:|
| short force | 450 K primary subset total-force RMSE | `≤50 meV/Å` |
| short force | 450 K primary subset最大分量误差 | `≤250 meV/Å` |
| independent trajectory | P5 off-policy force RMSE/max | `≤50/250 meV/Å` |
| harmonic | joint model harmonic replay RMSE | `≤2×` frozen v11 |
| sampling | 三 seed short-TDEP 两两全谱 MAE | `<5 cm⁻¹` |
| sampling | 各 seed mean temperature 相对目标偏差 | `≤5%` |
| q-space | Γ/K 顶支 line MAE | 各 `<10 cm⁻¹` |
| q-space | Γ/K 高对称点顶支绝对误差 | 各 `<15 cm⁻¹` |
| q-space | K kink 相对误差 | `<20%` |
| q-space | 中心三点整块评价 | 同一组 `<10/<15/<20%` 门槛 |
| 数值重放 | matrix projector 最大差 | `<1e-5 cm⁻¹` |

不确定度采用按 trajectory/时间块分组的 bootstrap，默认 1000 次重采样；block 长度由冻结规则 `ceil(2τ_int)` 决定，并至少为 5 个相邻 snapshots。固定门槛先用于 point estimate；只有相关指标的 95% bootstrap 上界也低于门槛时才登记“通过”。若点估计通过但区间跨过门槛，状态记为“样本不足”，追加 snapshots，不放宽门槛。

## 6. 代码改造与复用清单

| 工作 | 处理方式 |
|---|---|
| 多温度 force 评价 | 复用 `evaluate_graphene_fd_delta_model.py`，增加聚合 JSON |
| joint residual 数据 | 新增任意温度 lane 合并脚本；保留每个结构的来源和 operator SHA-256 |
| joint delta 训练 | 基于 `run_graphene_fd_delta_weighted_600.sh` 泛化为 300/600 balanced loss |
| arbitrary-T provisional operator | 泛化 `prepare_graphene_fd_residual_forces.py`，不再只循环 300/600 K |
| 三 seed TDEP | 复用 `td_phonon_friedel.py` 和 `recompute_graphene_tdep_from_checkpoints.py` |
| 温度系数与冻结预测 | 新增 `fit_graphene_fd_temperature_law.py` 和只读的 holdout predictor |
| 450 K DFPT | 给 `run_graphene_physical_fd_dfpt.sh` 增加 `FD450_CONV/FD450_LINE` |
| 450 K DFT labels | 泛化现有 thermal holdout shard/merge 流程到任意温度与 seed |
| 独立 DFT-MD | 给 `dft_md_tdep.py` 增加 `fd`、独立 seed 和 provenance 校验 |
| TDEP 不确定度 | 新增按 trajectory/block 重采样的 bootstrap 脚本 |
| 总验收 | 新增 `evaluate_graphene_fd_transferability.py`，只读取冻结预测和 holdout target |

所有 Python 命令使用项目 Conda 环境，例如 `conda run -n phonon python ...`。远程提交、同步和部署统一走 Tailscale 地址或 MagicDNS。

## 7. 机器分工和产物

| 机器 | 任务 |
|---|---|
| RTX 2060 | P0 同步、P1 交叉评价、P2 joint training、三 seed MLIP-TDEP、温度函数、bootstrap 和总验收 |
| V100-A | 450 K DFT label shard A、`FD450_CONV/LINE` 或独立 DFT-MD |
| V100-B | 450 K DFT label shard B、独立 DFT-MD 或 k-grid 复核 |
| 本地 | 哈希复核、文档和结果归档；不运行替代远端正式结果的低精度计算 |

统一结果根目录：

```text
results/graphene_fd_transferability/
  source_inventory.json
  short_model/cross_temperature_force_matrix.json
  short_model/freeze_manifest.json
  endpoints/T300/acceptance.json
  endpoints/T600/acceptance.json
  temperature_law.json
  freeze_manifest.json
  T450/on_policy/force_acceptance.json
  T450/on_policy/tdep_bootstrap.json
  T450/on_policy/prediction_acceptance.json
  T450/dftmd/trajectory_manifest.json
  T450/dftmd/acceptance.json
  transferability_acceptance.json
```

`transferability_acceptance.json` 必须分别给出 `passes_C1`、`passes_C2`、失败指标、置信区间和允许的结论文字，不能用一个总布尔值掩盖不同证据层级。

## 8. 失败分支

1. **P1 已有单模型跨温度通过：**直接冻结该模型；joint training 只作不参与选择的对照。
2. **P2 joint model 不能同时通过端点：**不启动 450 K DFT。改用守恒的温度条件化组合
   \(E_\Delta(T)=(1-w)E_{\Delta,300}+wE_{\Delta,600}\)，或采用显式 `ΔΦ_SR + local anharmonic residual` 分解。候选必须在 450 K 标签开放前选定。
3. **450 K force gate 失败：**判定为 short model 失败，不拟合或修改 450 K q-space 系数。450 K 转为 development；下一次最终验证改用事先未计算的 `375 K` 或 `525 K`。
4. **force 通过但 seed spread/温度失败：**先延长轨迹或修正 thermostat、时间步；模型和 q-space law 保持冻结。
5. **force 与采样通过，但 450 K q-space 失败：**两端线性系数规律不足。允许把 450 K 加入下一版温度函数训练，但必须把另一个未见温度作为新 holdout，不能复用 450 K 宣布成功。
6. **P4 通过、P5 失败：**结论停在 C1；检查训练轨迹覆盖、off-policy force error 和 DFT-MD/TDEP sampling bias。
7. **point estimate 通过但 bootstrap 上界失败：**只追加样本，不改门槛或模型。
8. **插值通过、750 K 失败：**模型范围限定为 `300–600 K` interpolation，不宣称外推能力。

## 9. 最新周报的收口写法

周报按实际最高证据等级写结论：

- P4、P5 均通过：`一个在 300/600 K 固定的 short model + temperature-law q-space correction，在未见 450 K 和独立 physical-FD DFT-MD 轨迹上通过固定受力、谱线与 K kink 门槛；当前适用范围为 300–600 K physical-FD 路径内插值。`
- 只有 P4 通过：`未见 450 K 的前瞻插值通过；独立 DFT-MD 轨迹尚未完成，因此暂不登记 trajectory-transferable model。`
- 任一关键门槛失败：直接报告失败数值、对应层级和下一未见温度，不把 450 K 重新拟合结果计为 holdout。

该写法把已校准复现、未见温度预测和独立轨迹验证分开，结论强度与实际数据一致。
