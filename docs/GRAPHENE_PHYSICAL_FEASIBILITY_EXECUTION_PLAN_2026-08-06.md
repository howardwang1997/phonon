# Graphene 物理分解可行性实验执行计划

**制定时间：**2026-08-06  
**执行状态：**A0、D0、E0-S、E0-P 可行性检查均已完成；E0-P2 k18/q9 稳定 pz 子空间、300/450/600 K 完整 q6 和三温度守恒算符均已通过；E1 已释放到 V100-A/B，等待两卡上现有训练结束后自动起跑  
**适用数据：**300/450/600 K development；375/525 K validation_locked 继续保持未见  
**目标：**先用最小计算量判断“统一短程 MLIP + 电子物理响应公式 + 晶格温度系综”是否值得进入新的 DFT 标签与未见温度验证。

## 1. 本轮需要回答的问题

本轮可行性实验只回答四个问题：

1. P4/C1 的大受力离群点是否来自结构映射、单位、PBC、长程算子重复加减等实现错误；
2. 三个开发 `degauss (Ry)` 的 Γ/K 静态响应能否由一套共享参数的有限展宽 Dirac/Mermin 公式描述；
3. 物理公式能否在动力学矩阵层面重放，并进一步构造可用于受力扣除的守恒算子；
4. 若前述条件成立，扣除电子长程力后是否仍需要局域 Mermin 自由能项。

本轮不把 450 K 的开发回放重新登记为 C1，也不生成或读取 375/525 K target。新的温度泛化结论必须留给后续固定模型后的验证。

## 2. 固定输入和数据边界

### 2.1 P4/C1 开发数据

450 K 已打开的失败 holdout 已同步到独立目录：

```text
results/graphene_physics_temperature/post_p4_feasibility/source_p4_450/
```

其中包括：

- 60 个 DFT force 标签及 seed/snapshot 标识；
- frozen short、long-range 和 total force 预测；
- static FC2、DFT-TDEP、bootstrap 与总验收；
- 450 K static DFPT line 和 k 网格收敛结果。

300/600 K static DFPT line 使用：

```text
results/graphene_physical_fd_dfpt/campaigns/FD300_LINE/graphene_FD300_LINE_dfpt.csv
results/graphene_physical_fd_dfpt/campaigns/FD600_LINE/graphene_FD600_LINE_dfpt.csv
```

固定宽展宽参考背景使用：

```text
results/vq_kink6_fd/graphene_sc6_dg0.040_phonopy.yaml
```

### 2.2 条件划分

| 数据域 | lattice temperature | `smearing/degauss (Ry)` | 用途 |
|---|---:|---:|---|
| development | 300 K | 0.0019000869 | 归因、公式拟合、消融 |
| development | 450 K | 0.00285013035 | 已打开失败 holdout，仅作开发与 leave-one-smearing-out 检查 |
| development | 600 K | 0.0038001738 | 归因、公式拟合、消融 |
| validation_locked | 375 K | 0.002375108625 | 固定后一次性验证，当前不得生成 target |
| validation_locked | 525 K | 0.003325152075 | 固定后一次性验证，当前不得生成 target |

## 3. 可行性实验

### A0：失败结果封存和可重放清单

**计算量：**本地 CPU，约 0.5 h。

执行内容：

1. 对同步后的 P4 标签、预测、DFT-TDEP、DFPT 和验收文件计算 SHA-256；
2. 保存远端来源、文件大小、哈希和当前验收状态；
3. 检查同步目录不包含 375/525 K target；
4. 原始目录只读使用，后续诊断输出写入新目录。

输出：

```text
results/graphene_physics_temperature/post_p4_feasibility/A0_archive_manifest.json
```

完成标准：所有必需输入存在、哈希固定，且 `C1=failed`、`C2=not_run` 被保留。

### D0：P4 失败归因

**计算量：**本地/RTX CPU；必要时 V100-A 做批量重放；预计 2–4 h。

#### D0-F：受力归因

对全部 60 个构型输出：

- seed、snapshot、原子、笛卡尔分量级误差；
- DFT 力、short force、long-range force、total force 的模长与方向关系；
- short-only 和 frozen-total 相对 DFT 的误差变化；
- 用 development target 求得的全局和逐构型最优长程缩放系数
  \(\alpha^*=\arg\min_\alpha\|F_{\rm DFT}-F_{\rm SR}-\alpha F_{\rm LR}\|^2\)；
- 位移幅度、最小原子间距、净力、误差与长程力幅度的相关性；
- `seed0:snapshot099` 和 `seed0:snapshot063` 的独立记录。

这里的 \(\alpha^*\) 只用于失败归因，不进入最终模型。若 \(\alpha^*\) 明显偏离 1，或去掉长程项反而系统性降低误差，说明现有经验算子的幅度或力映射需要重构。

#### D0-Q：谱线归因

分别比较：

- 450 K static frozen prediction 与 direct DFPT；
- 450 K finite-lattice prediction 与 DFT-TDEP；
- Γ/K line、顶支和 K kink；
- point estimate 与 bootstrap 区间。

输出：

```text
results/graphene_physics_temperature/post_p4_feasibility/D0_diagnostics/
  force_component_records.csv
  force_structure_summary.csv
  force_diagnostics.json
  qspace_residuals.csv
  qspace_diagnostics.json
  force_error_diagnostics.png
  qspace_residuals.png
```

停止条件：发现标签/几何哈希不一致、单位错误、原子映射错误、PBC 错误或长程项重复加减时，先修实现；保留原始失败记录，不继续 E1。

### E0-S：共享电子响应公式的谱线可行性

**计算量：**本地/RTX CPU，预计 2–8 h；不增加 DFT 标签。

使用宽展宽参考背景 \(s_{\rm ref}=0.04\,\mathrm{Ry}\)，对 Γ 和 K 附近的电子自由能曲率采用有限展宽 Dirac 核：

\[
\Phi(x,s;v_F)=2s\log\left[2\cosh\left(\frac{\hbar v_F |\Delta q(x)|}{2s}\right)\right],
\]

\[
\Delta\lambda_r(x,s)=A_r\left[\Phi(x,s;v_F)-\Phi(x,s_{\rm ref};v_F)\right],
\qquad r\in\{\Gamma,K\}.
\]

实际拟合的平方频率写为

\[
\lambda_r(x,s)=\lambda_{\rm bg}(x)
+B_{r,0}+B_{r,2}x^2+\Delta\lambda_r(x,s).
\]

其中 \(B_{r,0}+B_{r,2}x^2\) 是三个展宽共享、与 `degauss` 无关的解析短程重整化，用于吸收宽展宽参考背景和当前 DFPT 线之间的解析差异；展宽依赖只能来自 \(\Phi\)。固定 \(\mu=0\)，Γ/K 共享一个 \(v_F\)，只允许两个区域具有不同的耦合幅度 \(A_\Gamma,A_K\)。`degauss (Ry)` 只进入 Fermi–Dirac/Mermin 核，不为每个展宽分别拟合参数。

对照组：

1. 宽展宽背景，不加 Kohn 修正；
2. 三展宽共享的解析短程项，但不含电子核；
3. 当前 `rounded cusp + q²` 经验模型在 450 K 已打开数据上的结果；
4. 共享参数 Dirac/Mermin 公式。

评价分两层：

- development fit：三个展宽同时拟合；
- leave-one-smearing-out：每次只使用两个展宽拟合，共享参数预测第三个展宽。450 K 已经打开，因此该检查只是开发稳定性检查，不恢复其 holdout 身份。

E0-S gate：

- 每个被留出的 `degauss` 上 Γ/K line MAE 均 `<10 cm⁻¹`；
- Γ/K 高对称点顶支绝对误差均 `<15 cm⁻¹`；
- K kink 相对误差 `<20%`；
- 共享 \(v_F\) 落在事先固定的 `4–8 eV·Å` 范围；
- 共享模型优于不加修正的背景和旧经验插值；
- 不允许为三个 `degauss` 分别拟合无关参数后宣称通过。

输出：

```text
results/graphene_physics_temperature/post_p4_feasibility/E0_shared_dirac/
  shared_dirac_fit.json
  shared_dirac_predictions.csv
  shared_dirac_comparison.png
```

### E0-M：矩阵和实空间算子可行性

**前置：**E0-S 通过。

把同一 Dirac/Mermin 核投影到冻结的 Γ/K 光学模规范，构造 Hermitian q-space 修正；随后在固定 6×6 q 网格上傅里叶变换为实空间 FC2，并执行：

- Hermitian、pair symmetry 和 acoustic sum rule 检查；
- q-space → real-space → q-space 重放；
- Γ/K line 最大重放误差 `<1e-5 cm⁻¹`；
- 力为同一个二次自由能的解析梯度。

若稀疏谱线不足以唯一构造实空间算子，E0-M 不用任意插值补齐，转入 E0-P。

### E0-P：EPW/Wannier pilot，条件执行

现有 V100-B `/data/graphene_epw3` 已完成 Wannier90 和 EPW，但其参数为：

- SCF `degauss=0.02 Ry`；
- coarse `k=12×12×1`、`q=6×6×1`；
- fine `k/q=18×18×1`。

这只能说明管线可运行，不能满足当前低展宽与 10/15 cm⁻¹ 定量门槛。

触发条件：

- E0-S 失败，表明低能 Dirac 公式表达力不足；或
- E0-S 通过但 E0-M 无法构造可重放的守恒力算子。

pilot 先在参考 `degauss=0.00285013035 Ry` 上做一个固定 Γ/K 小 q 集，使用包含 K 点的 k 网格；先检验电子结构、Wannier 和 EPC 插值误差是否低于最终门槛，再决定是否扩展三展宽 band sum。预计 12–24 V100 box-h；若需收敛 band sum，再增加 12–36 box-h。

停止条件：Wannier/EPW 自身的 Γ/K 误差已超过 `10/15 cm⁻¹`，或必须修改不同展宽的电子结构参数才能拟合时，不扩展生产计算。

### E1：交叉 `degauss` force pilot，已释放并等待空闲 GPU

D0 数据检查和 E0-P 三温度守恒算符验收已经完成，E1 于 2026-08-07 01:48--01:49 CST 分别释放到 V100-B/A。构型、任务分片和验收标准保持为此前固定的版本。

- 300/450/600 K 各固定 5 个已有热构型；
- 每个构型计算三个开发 `degauss`；
- 复用 15 个对角标签，新算 30 个 DFT single points；
- V100-A/B 各 15 个，预计 4–5 h 墙钟，后处理 1–2 h。

扣除电子长程力后的局域残差 gate：

- RMSE `≤10 meV/Å`；并且
- 原始差 `≤10 meV/Å`，或残差不超过原始差的 20%。

通过时固定局域 Mermin 项为零；失败时才进入 E2，可积分局域自由能修正。

## 4. 当前机器安排

| 机器 | 第一批任务 | 后续任务 |
|---|---|---|
| 本地 Mac | A0 哈希清单；D0/E0-S 实现与复核 | 固定方案和汇总 |
| RTX 2060 | 保留 P4 原始产物；D0/E0 后处理 | 清理至可用空间 ≥100 GiB 后才运行 TDEP/SSCHA |
| V100-A | D0 必要的模型推理重放 | E1 的 15 个新标签 |
| V100-B | 保存现有 EPW3 管线；E0-P 条件触发 | E1 另 15 个标签；随后 S0 训练 |

不以 GPU 满载为目标。D0/E0-S 是 CPU/数据分析 gate；在它们完成前启动大批 DFT 或训练不会缩短关键路径。

## 5. 时间线和结论强度

| 从启动起 | 里程碑 | 可以支持的结论 |
|---:|---|---|
| 0.5 h | A0 完成 | 失败结果和数据边界已固定 |
| 2–4 h | D0 完成 | 明确主要失败来自实现、短程覆盖还是电子算子 |
| 4–12 h | E0-S 第一版 | 判断共享 Dirac/Mermin 公式在开发谱线上的表达力 |
| 12–24 h | E0-M 或 E0-P 启动 | 判断能否形成守恒、可出力的电子算子 |
| 1–2 d | D0+E0 开发 go/no-go | 判断物理分解路线是否值得进入 E1/S0 |
| E0 后 5–7 h | E1 | 判断是否需要局域 Mermin 自由能 |
| 约 12–16 d | 新 375/525 K 最小验证 | 才能判断未见温度的定量泛化 |

开发阶段只有在以下条件同时满足时给出“物理分解可行”的结论：

1. D0 没有未解决的数据或实现错误；
2. E0-S 的 leave-one-smearing-out gate 通过；
3. E0-M 或 E0-P 能构造守恒、可重放的力算子；
4. 后续 E1 表明局域项可忽略，或 E2 能以可积分自由能形式闭合。

即使全部通过，结论范围仍限定为 300/450/600 K development。375/525 K 固定后验证完成之前，不写成“已定量复现未见有限晶格温度 Kohn anomaly”。

## 6. 执行记录（2026-08-06）

### 6.1 已完成

1. A0 已封存 29 个关键文件，共 3,145,017 byte；450 K 原失败状态保留为 `C1=failed`、`C2=not_run`，目录中没有 375/525 K target。
2. D0 的 60 个构型数据完整，几何哈希、结构 key 和 72 原子映射均一致；长程算子受力重放最大差为 0。当前没有证据表明 P4 失败来自单位、PBC、原子映射或长程力重复加减。
3. 60 个构型的 frozen-total force RMSE 为 `53.10 meV/Å`，short-only 为 `67.38 meV/Å`；长程项改善 59/60 个构型。全局最优长程缩放为 1.2866，但 RMSE 只能降到 `52.25 meV/Å`，说明简单重缩放不足以解决受力误差，主要问题仍在短程模型和高受力构型覆盖。
4. E0-S 的共享 Dirac/Mermin 公式显著优于不含电子核的对照：leave-one-smearing-out 总 RMSE 分别为 `0.737` 和 `2.206 cm⁻¹`。但 K kink 相对误差在 450 K 和 600 K 分别为 25.4% 和 25.7%，超过事先固定的 20% 门槛，因此 E0-S 判为未通过。
5. E0-S 未通过后按计划触发 E0-P。V100-B 已用原有 `degauss=0.020 Ry` Wannier/EPW 表示完成 450 K、9 个 Γ/K 点、360×360 fine-k 的重启试验；实际墙钟 91 s。EPW 回显目标 Fermi–Dirac 温度为 `0.038778 eV`，对应 450 K，并以 `0.020 Ry` 静态自能作为参考项。
6. 360×360 动态自能修正在 Γ 区转移后的顶支 MAE 约 `7 cm⁻¹`；K 区 MAE 为 `20.8–22.5 cm⁻¹`，K kink 相对误差为 `3.51–4.09`。该结果没有通过 10/15 cm⁻¹ 与 K kink 门槛。
7. 720×720 动态复核的有效墙钟为 6 min 1 s。K 区 off-shell kink 相对误差降到 42.1%，说明 360×360 的动态 kink 尚未收敛；K 区 MAE 仍为 `20.83 cm⁻¹`，不能通过门槛。
8. 720×720 的零频静态 `specfun_ph` 导出墙钟为 6 min 11 s。把静态 \(\Pi(q,0,450\,K)-\Pi(q,0,0.020\,Ry)\) 转移到收敛的 `0.020 Ry` 背景后，Γ line MAE 为 `5.36 cm⁻¹`、Γ 点误差为 `2.12 cm⁻¹`；K 点误差为 `13.97 cm⁻¹`、K kink 相对误差为 12.5%，均通过对应门槛，但 K line MAE 为 `13.78 cm⁻¹`，仍高于 `10 cm⁻¹`。因此 E0-P 当前接近门槛但尚未通过。
9. 静态收敛检查已完成。360→720 的顶支最大点差为 `0.247 cm⁻¹`；720×720 下数值展宽 `0.010→0.005 eV` 的最大点差为 `0.340 cm⁻¹`，均低于沿用的 `1 cm⁻¹` 点级收敛标准。最终 720×720、`0.005 eV` 结果为：Γ line MAE `5.48 cm⁻¹`、Γ 点误差 `2.16 cm⁻¹`；K line MAE `13.89 cm⁻¹`、K 点误差 `14.05 cm⁻¹`、K kink 相对误差 10.0%。K line 是唯一未通过项，而且对 fine-k 和数值展宽已经收敛。

### 6.2 当前决策和下一步边界

- 旧 EPW3 的 fine-k 和数值展宽已经收敛，但静态 K line gate 未通过。按事先确定的停止条件，不扩展这套表示到 300/600 K，也不在其上生成完整 q 网格。
- 旧 EPW3 与目标 DFPT 使用相同赝势、面内晶格常数和 `60/240 Ry` 截断；它使用 coarse `k=12`、`q=6` 和约 `29.5 Å` 真空层，目标 DFPT 真空层为 `15 Å`。下一轮若继续 EPW，应从目标晶胞重新生成 matched-cell Wannier/EPC，并显式做 coarse q 收敛，不能复用现有 EPC 后只增加 fine-k。
- `degaussw (eV)` 是 EPW 能量分母的数值展宽，与由晶格温度公式得到的 `smearing/degauss (Ry)` 分开记录。本轮已经验证后者可通过 450 K Fermi–Dirac 占据进入静态 band sum；尚未验证的是足够准确、可转为守恒力的 EPC 矩阵。
- E1 继续锁定。释放条件仍是新的 E0-S/E0-P 方案同时通过频谱 gate，并给出完整 Hermitian q 网格、q-space/real-space 重放和解析受力。
- 截至本记录，两台 V100 没有运行任务，V100-B 的 systemd 服务保留为已完成、幂等的恢复入口。RTX 2060 没有 GPU 计算，但有一个 7 月 30 日启动的旧后处理等待循环，每 300 s 记录一次 `wave3 failed; waiting for long-range-subtracted residual model`；它不是本轮任务，也没有可执行的模型子进程，因此不计为有效实验。是否终止该旧等待器单独处理。
- E1 继续锁定，不生成新的交叉 `degauss` DFT force 标签，也不读取 375/525 K target。

## 7. E0-P2：目标晶胞 EPW 表示（2026-08-06 启动）

旧 EPW3 的 fine-k 和数值展宽已经收敛，但其约 29.5 Å 真空与直接 DFPT 的 15 Å 目标晶胞不一致，且 coarse `k=12`、`q=6` 尚未做表示层收敛。因此下一轮不再加密旧表示的 fine-k，而是保持同一 C ONCV 赝势、`60/240 Ry` 截断和 `degauss=0.020 Ry` 参考背景，重新生成 15 Å 晶胞的 Wannier/EPC 表示。

并行安排如下：

| 机器 | 工作目录 | 当前任务 | 早停条件 |
|---|---|---|---|
| V100-A | `/data/graphene_e0_epw_matched/k12_q6` | 完整 SCF→q6 DFPT→k12 NSCF→Wannier/EPW→450 K 九点静态 gate | Wannier/EPW 不能重放，或 K line 仍明显高于门槛 |
| V100-B | `/data/graphene_e0_epw_matched/k18_q9` | 并行完成 SCF→q9 DFPT→k18 NSCF；先不做昂贵 EPW 转换 | V100-A 表示层失败且不能通过输入修复 |
| RTX 2060 | 不新增 DFT/EPW | 后续结果同步与分析 | 不占用 GPU |

V100-A 的九点 gate 使用 `720×720` fine-k 和 `0.005 eV` 数值展宽；`degaussw (eV)` 仍只作为 EPW 分母的数值展宽。lattice temperature 固定为 450 K，电子响应相对于 `smearing/degauss=0.020 Ry` 参考背景按零频自能差引入。

决策顺序：

1. k12/q6 matched-cell gate 全部通过，或 line/高对称点通过而仅 K kink 失败时，对 k18/q9 执行 EPW 转换和同一九点 gate；后一种情况只用于判定 coarse-q 收敛，不释放后续生产计算；
2. 两个 coarse 表示的九点顶支最大点差需 `<1 cm⁻¹`，且最终 Γ/K line、高对称点和 K kink 同时满足 E0 门槛；
3. 通过后才生成完整 Hermitian q 网格、实空间算子并释放 E1；
4. 若 matched-cell k12/q6 与旧 EPW3 结果基本相同且 K line 仍失败，则不把 300/600 K 或完整 q 网格排入队列。

### 7.1 首轮结果与输入修正

k12/q6 的 SCF、DFPT、NSCF、Wannier/EPC 和 450 K 九点 gate 均正常结束，但该 gate 不能用于判断 15 Å 晶胞的物理效果。Wannier Hamiltonian 诊断显示：旧 EPW3 的 K 点两带 gap 为 `7.0e-5 eV`，Dirac 中点距 Fermi 能 `0.0113 eV`，两侧 (v_F) 为 `4.51/4.60 eV·Å`；首轮 matched-cell 表示的 K 点 gap 却为 `7.14 eV`，Dirac 中点偏离 Fermi 能 `-7.21 eV`。对应 EPW DOS 从旧表示的 `0.102273` 变为 `0.000000 states/spin/eV/cell`，电子自能差接近零。

这说明首轮 `dis_win_max=4 eV` 没有稳定选出 graphene 的 π/π* 子空间。原始失败目录 `/data/graphene_e0_epw_matched/k12_q6` 保留不变。修正版在独立目录 `/data/graphene_e0_epw_matched/k12_q6_pi45` 中显式设置 `bands_skipped='exclude_bands = 1-3, 6-14'`，只保留 DFT 第 4/5 条 π/π* 带；先检查 K gap、Dirac–Fermi 对齐和 (v_F)，通过后才解释九点频谱并决定是否释放 k18/q9 EPW 转换。

修正版于 2026-08-06 17:04（V100-A 本地时间）启动并复用已完成的 coarse 数据。Wannier 阶段生成后的即时检查为：K 点 gap `1.35e-5 eV`，Dirac 中点距 Fermi 能 `0.01147 eV`，两侧 (v_F) 为 `6.20/6.26 eV·Å`。这三个量均通过输入完整性检查；EPC 转换和 450 K 九点频谱 gate 随后正常完成，结果见 7.2。

### 7.2 显式 π 带 k12/q6 频谱结果

修正版于 17:16 正常完成，零重启、零运行错误。转移后的 Γ line MAE、Γ 点误差、K line MAE 和 K 点误差依次为 `5.85`、`4.65`、`6.93` 和 `8.41 cm⁻¹`，均通过 `10/15 cm⁻¹` 门槛。K kink 相对误差为 `58.2%`，高于 `20%` 门槛，是唯一失败项。

因此不扩展到 300/600 K、不生成完整 q 网格、不释放 E1。由于 k18/q9 的 coarse 数据已经完成，增加显式 π 带 EPW 转换只用于回答 q6 插值是否压低了 K kink；若 q9 仍不通过，则停止当前 EPW 表示路线并回到电子响应模型本身。

### 7.3 k18/q9 电子子空间门控与正式任务

直接沿用固定 DFT 第 4/5 带的 k18/q9 任务在 Wannier Hamiltonian 生成后主动早停。该表示的 K 点 gap 为 `1.87e-5 eV`、Dirac 中点距 Fermi 能 `0.01836 eV`，但两侧 (v_F) 达到 `10.57/10.62 eV·Å`，且 gauge-invariant spread `Omega I=16.17 Å²`。这不是 q9 频谱结果，也不能用于判断 q 网格收敛；它表明固定能带编号在更密 k 网格上不能连续跟踪与 σ/真空带交叉的 π/π* 流形。

随后只运行 Wannier 阶段，每个候选约需 2--3 分钟：

1. 排除第 1--3 带并在 Fermi 能附近设置 `[-0.5,+0.5] eV` frozen window，虽然 K 附近速度恢复，但 Γ 点没有保留占据 π 支，排除；
2. 只排除最低第 1 带、保持同一 frozen window 且不截断外窗时，K 点 gap 为 `8.08e-6 eV`、Dirac 中点距 Fermi 能 `0.01835 eV`、两侧 (v_F) 为 `5.46/5.44 eV·Å`。两个 Wannier 中心分别落在两个 C 原子上，每个 spread 约 `0.876 Å²`，总 spread 为 `1.752 Å²`，`Omega I=1.744 Å²`；该候选通过；
3. 在候选 2 上增加 `dis_win_max=4 eV` 后，两侧 (v_F) 降为 `1.63/3.82 eV·Å`，排除。

EPW 文档说明 `bands_skipped` 决定 Wannier 化排除的能带，而 frozen window 用于固定其中必须保留的态；EPW 自 v5.3 起不再用 `dis_win_max` 决定 coarse 电子-声子顶角的 band manifold。因此正式任务采用候选 2：`bands_skipped='exclude_bands = 1'`，frozen window 为 `[-2.2187,-1.2187] eV`（即本次 SCF Fermi 能附近 ±0.5 eV），不设置外窗上限。输入规则见 [EPW Inputs](https://docs.epw-code.org/Inputs/Inputs.html)，版本行为见 [EPW Releases](https://docs.epw-code.org/Releases.html)。

完整任务已于 2026-08-06 21:52（V100-B 本地时间）在独立目录 `/data/graphene_e0_epw_matched/k18_q9_ex1_pifroz` 启动，复用 `/data/graphene_e0_epw_matched/k18_q9` 已完成的 SCF、q9 DFPT 和 k18 NSCF，只运行 Wannier/EPC seed 与 450 K 九点 gate。第一个完整 q 星结束时已处理 7/81 个 full-grid q；扣除约 2 分钟 Wannier 初始化后，平均约需 93 秒/q。按该累计吞吐率外推，EPC seed 加九点 gate 的总墙钟预计为 2.2--2.4 小时，对应 V100-B 本地时间约 2026-08-07 00:05--00:20 完成。该任务只用于判断 q9 是否显著修复 K kink；在 K kink `<20%` 且其他四项继续通过前，不释放 300/600 K、完整 q 网格或 E1。

### 7.4 后续实验队列与自动门控（2026-08-06 23:47 CST）

q18/q9 正式任务已处理到 75/81 个 full-grid q，服务仍正常运行。按当前约 93--96 秒/q 的吞吐率和后续九点 gate 的既有耗时估计，预计在 2026-08-07 00:05--00:15（V100-B 本地时间）完成。后续步骤已经按依赖关系部署，不需要人工盯住完成时刻：

1. `DONE` 出现后，V100-B 自动运行九点后处理，重新计算五项固定指标：Γ/K line MAE `<10 cm⁻¹`、Γ/K 高对称点绝对误差 `<15 cm⁻¹`、K kink 相对误差 `<20%`。判定和原始摘要哈希写入 `spectral_gate_decision.json`。
2. 任一指标失败时只写 `SPECTRAL_GATE_FAIL` 并停止，不生成完整 q 网格，也不触发 E1。五项全部通过时写 `SPECTRAL_GATE_PASS`，随后自动启动 450 K、完整 6×6 q 网格的静态 EPW 导出；fine-k 固定为 720×720，`degaussw=0.005 eV` 只作为能量分母数值展宽。电子/EPC 表示仍来自收敛检查中的 k18/q9；输出 q6 是因为后续 72 原子 E1 构型具有 6×6 周期性。
3. 完整 q 网格阶段检查 36 个 q 点、216 条零频模记录、权重、450 K 温度、数值展宽以及 q/−q 配对。通过后只写 `FULL_Q_DATA_READY`。这个标记表示数据完整，不表示已经得到可出力的算符，也不会释放 E1。
4. Cartesian Hermitian 电子修正算符需要在 300/450/600 K 三个开发温度分别构造完整的 ΔD(q)，满足 `ΔD(−q)=ΔD(q)†`，执行点群对称和 acoustic sum rule 检查，再做 q-space → real-space → q-space 重放及解析力/有限差分梯度一致性检查。单温度通过时只写不释放 E1 的温度标记；三温度全部通过并复核哈希、参考几何和 `degauss=k_BT/Ry` 后，聚合程序才原子创建 `OPERATOR_GATE_PASS`。
5. 两台 V100 已各自启用 E1 等待器。V100-B 在本机验证聚合判定和三个算符后释放 lane B；V100-A 每两分钟通过 Tailscale 地址检查 V100-B，复制后重新验证算符哈希、温度公式、pair symmetry 和 acoustic sum rule，再释放 lane A。当前聚合标记不存在，因此两条 E1 服务保持停止。每条 lane 有 15 个标签，任务支持逐点原子写入和断点续跑。

E1 输入已冻结为 15 个 72 原子开发构型：300、450、600 K 各 5 个。每个构型只计算另外两个温度公式对应的 `smearing/degauss (Ry)`，形成 30 个非对角标签：

| 对应温度 | `smearing/degauss (Ry)` | lane A | lane B |
|---:|---:|---:|---:|
| 300 K | 0.0019000869 | 5 | 5 |
| 450 K | 0.00285013035 | 5 | 5 |
| 600 K | 0.0038001738 | 5 | 5 |

表中数值为便于阅读的舍入值。E1 执行清单和 DFT 输入实际按 `degauss=k_BT/Ry` 计算双精度值：300/450/600 K 分别为 `0.0019000869380739254`、`0.0028501304071108886` 和 `0.003800173876147851 Ry`；已有对角标签保留生成时的历史舍入值。每台 V100 共 15 个任务，DFT 参数固定为 8×8×1 k 网格、`60/240 Ry` 截断和 Fermi–Dirac 占据。任务清单和构型文件的构型哈希为 `6bae3fda25bed3e38a8c176e9b4468b1bd993739d754300aaf47a1120cb6380b`。375/525 K 验证集没有被读取，也不在清单中。

三台机器的后续职责如下：

| 机器 | 已排队任务 | 启动条件 |
|---|---|---|
| V100-B | 450 K q6 → 300/600 K 九点 gate → 若通过则 300/600 K q6 → 三温度算符 → E1 lane B | 全链自动；E1 需三温度聚合 `OPERATOR_GATE_PASS` |
| V100-A | 定时拉取并复核三温度算符；随后 E1 lane A；两条 lane 完成后拉取 B 并自动计算 cross-degauss 残差 | 远端聚合 gate 通过且本机复核通过；残差汇总需 A/B 均完成 |
| RTX 2060 | 保留给 E1 失败后才需要的 E2 局域自由能模型或后续汇总作图 | 当前没有能缩短关键路径的 GPU 任务 |

E1 后先判断同一构型改变 `degauss` 后的力差能否由已验收电子算符解释。残差 RMSE `≤10 meV/Å`，并且原始差 `≤10 meV/Å` 或残差不超过原始差的 20% 时，固定局域 Mermin 项为零；未通过才进入 E2。这个判断仍只覆盖 300/450/600 K development，不改变 375/525 K 的固定验证边界。

### 7.5 q18/q9 正式结果与 q6 算符路线（2026-08-07）

q18/q9 正式任务已正常完成。450 K 九点频谱的五项固定指标为：

| 指标 | 结果 | 门槛 | 判定 |
|---|---:|---:|---|
| Γ line MAE | `5.607 cm⁻¹` | `<10 cm⁻¹` | 通过 |
| Γ 点顶支绝对误差 | `3.463 cm⁻¹` | `<15 cm⁻¹` | 通过 |
| K line MAE | `5.120 cm⁻¹` | `<10 cm⁻¹` | 通过 |
| K 点顶支绝对误差 | `5.053 cm⁻¹` | `<15 cm⁻¹` | 通过 |
| K kink 相对误差 | `3.283%` | `<20%` | 通过 |

与 k12/q6 表示的 `58.2%` K kink 误差相比，q18/q9 将唯一失败项降到 `3.283%`，因此可以支持“450 K、固定九点上的电子响应表示可行”这一阶段性结论。它还不能单独支持可出力算符、三温度一致性或未见温度泛化结论。

E1 构型为 72 原子的 6×6 超胞，允许直接作用于这些构型的实空间算符必须来自与 6×6 周期相容的 q6 网格。这里保留 q18/q9 作为电子与 EPC 插值表示，输出静态自能时改用完整 q6 查询网格。q6 共 36 个 q 点，每个温度应有 216 条零频模记录。

在正式算符构建前已完成 q6 Fourier gauge 审计。以 primitive-cell gauge 进行 q→real-space→q 重放时，q 坐标最大误差为 `4.86×10⁻¹⁰`，超胞 Hessian 的 Hermitian 误差为 `5.94×10⁻¹⁷`，虚部和 acoustic-sum-rule 行和分别为 `0` 和 `6.66×10⁻¹⁶`，矩阵重放误差为 `5.24×10⁻¹⁶`，24 个空间群操作下的最大误差为 `3.21×10⁻⁹`。另外两种 basis-phase 约定产生约 `0.14–0.15` 的虚部和 `0.556–0.587` 的点群误差，已排除。`q2r.x`/`matdyn.x` 的 Γ 点 acoustic sum rule 也已用 `asr='crystal'` 修正到数值零；这套 matdyn 数据用于独立相位和 ASR 诊断，不作为 EPW 自能的模基底。

450 K 完整 q6 于 01:07 CST 通过：36 个 q 点和 216 条零频模记录齐全，q/−q 裸频差为 0，q/−q 自能差最大为 `2.25×10⁻⁵ meV`，低于 `10⁻³ meV` 门槛。随后 300/600 K 九点 gate 并行完成，结果如下：

| lattice temperature | Γ line MAE | Γ 点误差 | K line MAE | K 点误差 | K kink 相对误差 |
|---:|---:|---:|---:|---:|---:|
| 300 K | `5.700 cm⁻¹` | `3.743 cm⁻¹` | `5.105 cm⁻¹` | `4.814 cm⁻¹` | `8.762%` |
| 450 K | `5.607 cm⁻¹` | `3.463 cm⁻¹` | `5.120 cm⁻¹` | `5.053 cm⁻¹` | `3.283%` |
| 600 K | `5.414 cm⁻¹` | `2.849 cm⁻¹` | `5.109 cm⁻¹` | `5.097 cm⁻¹` | `7.930%` |

三温度的五项频谱指标全部通过。这支持“同一 q18/q9 电子/EPC 表示在 300/450/600 K development 上保持定量频谱可行”，但 E1 释放仍需等待三套完整 q6 算符的聚合验收。

首次 Cartesian 试构建还识别出一个表示差异：EPW 默认 `lifc=.false.`，使用 `epwdata.fmt` 中保存的 91 个 Wigner–Seitz 实空间动力学矩阵；`q2r/matdyn` 使用另一套 IFC/ASR 插值。后者与 EPW 输出裸频的最大差为 `70.86 cm⁻¹`，因此没有放宽 `0.5 cm⁻¹` 同源门槛，也没有混用 matdyn 本征矢。当前实现按 EPW 7.3.1 的 `dynwan2bloch` 源码逐式重建 q6 矩阵：91 个 WS 向量的权重和为 81，`decay.dynmat` 幅度重放误差为 `1.08×10⁻¹⁹ Ry`，q/−q 矩阵误差为 `6.27×10⁻¹⁴`，重建频率与 `specfun_sup.phon` 裸频在文本打印精度内最大差 `0.040 cm⁻¹`。

基于该同源矩阵的 T450 Cartesian 算符已通过全部有效 gate。对称投影引起的最大频率改动为 `0.017 cm⁻¹`；q→real-space→q 的矩阵误差为 `4.40×10⁻²²`，非零频模和顶支频率重放误差分别为 `9.09×10⁻¹³` 和 `6.82×10⁻¹³ cm⁻¹`；force pair 误差为 0，ASR 行和为 `2.22×10⁻¹⁶ eV/Å²`，解析梯度有限差分误差为 `1.40×10⁻¹³ eV/Å`。接近零的 Γ 声学本征值在开平方后有 `1.23×10⁻⁵ cm⁻¹` 的平台舍入量，因此保留为全模诊断；固定的顶支重放门槛和矩阵/ASR 门槛均正常通过。

300/600 K 两套完整 q6 于 01:47 CST 并行完成。每套均有 36 个 q 点和 216 条零频模记录，q/−q 裸频差为 0；300/600 K 的 q/−q 自能差分别为 `2.241×10⁻⁵` 和 `2.242×10⁻⁵ meV`，均低于 `10⁻³ meV`。T300/T600 算符随后通过同一组矩阵、对称性、实空间重放和解析梯度门槛。两者的对称投影频率改动分别为 `0.01704` 和 `0.01693 cm⁻¹`，非零频模重放误差分别为 `8.95×10⁻¹³` 和 `1.42×10⁻¹² cm⁻¹`；聚合发布复核中的最大 ASR 行和为 `3.02×10⁻¹⁶ eV/Å²`，三个温度的 force pair 误差均为 0。

三套算符的力常数 Frobenius 范数从 300 K 的 `5.4099 eV/Å²`，依次降为 450 K 的 `5.2002 eV/Å²` 和 600 K 的 `4.9871 eV/Å²`。将算符预先作用到 15 个 E1 构型上，30 个非对角任务的预测温度差力逐构型 RMS 为 `0.216--1.531 meV/Å`，平均 `0.617 meV/Å`，最大单分量为 `2.99 meV/Å`。这个信号明显低于 E1 的 `10 meV/Å` 工程精度门槛。因此 E1 首先检验是否存在不能由长程算符解释、且大到影响当前受力目标的局域温度项；若需要判断亚 meV/Å 级预测相关性，应在 E1 后另做 k 网格和重复计算收敛检查，不能仅凭当前 gate 得出该结论。

450 K q6 首轮 EPW 数值循环于 00:35 CST 完成，但运行中的 shell 文件在部署三温度通用版本时被原地更新，进程读到不完整尾部，未能写出 `DONE`。原始数值输出目录保留；由于第二次启动已覆盖静态输出的开头，不能把残留文件当作完整 gate 输入。systemd 已自动启动幂等重跑，后续部署改为在任务停止后原子替换脚本，避免再次发生同类问题。该事件只增加墙钟，不改变输入参数和验收标准。

### 7.6 当前自动队列、算力和时间估算

E0 关键路径已经完成：300/600 K q6 于 01:47 CST 写出完整性标记，三温度算符聚合于 01:47:58 通过，V100-B 于 01:48:01 完成本机复核并释放 lane B，V100-A 于 01:49:06 经 Tailscale 拉取后独立复核并释放 lane A。A/B 上三套算符的 SHA-256 与聚合判定一致；E1 输入仍只包含 300/450/600 K development，375/525 K 没有被读取。

当前机器状态和后续任务为：

| 机器 | 当前状态 | 已安排的下一步 |
|---|---|---|
| V100-A | 另一项训练自 01:21 CST 起占用约 15.7 GiB 显存，GPU 利用率约 98--100%；E1 lane A 已触发并等待 | GPU 空闲后自动运行 15 个非对角 DFT force 点；A/B 均完成后自动汇总残差 |
| V100-B | q6 和算符任务已结束；另一项训练占用约 15.7 GiB 显存，E1 lane B 已触发并等待 | GPU 空闲后自动运行另外 15 个非对角 DFT force 点 |
| RTX 2060 | GPU 空闲；8 GiB 显存不适合替代当前 72 原子 GPUpw V100 标签任务 | 保留给 E1 失败后才需要的 E2 局域自由能模型或后处理，不在关键路径上重复占卡 |

E1 启动器已经增加共享 GPU 保护：只有 `nvidia-smi` 不再报告其他计算进程，且显存占用不高于 `1024 MiB` 时才启动；每 60 s 复查一次、每 10 min 记录一次等待状态。服务总超时设为无限，已有训练持续较久也不会使队列丢失。每个 DFT 点原子写入独立标签，机器重启或单点失败后可从已有标签继续。

现有训练的结束时间不属于本项目记录，因此不能给出 E1 的绝对起跑时刻。两块 GPU 一旦空闲，E1 会在 60 s 内自动起跑；按每条 lane 15 点的既有估算，完整墙钟约为 4--5 h，A 上的自动拉取、哈希检查和残差判定再需 5--10 min。每条 lane 的前 5 个任务包含 450→300 K 和温差最大的 600→300 K 样本，预计在起跑后约 1.3--1.7 h 可形成阶段性趋势；正式 E1 gate 仍以 30 点全部完成后的结果为准。

目前可以形成两项阶段性结论：同一 q18/q9 电子/EPC 表示在三个开发温度上通过了固定频谱门槛；对应修正可以构造成满足 pair symmetry、ASR、点群对称和解析梯度检查的守恒力算符。是否还需要局域 Mermin 自由能项必须等待 E1，375/525 K 未见温度的定量泛化仍不在当前结论范围内。

本轮新增结果位于：

```text
results/graphene_physics_temperature/post_p4_feasibility/
  A0_archive_manifest.json
  D0_diagnostics/
  E0_shared_dirac/
  E0_epw_pilot/
  E0_epw_matched/
```
