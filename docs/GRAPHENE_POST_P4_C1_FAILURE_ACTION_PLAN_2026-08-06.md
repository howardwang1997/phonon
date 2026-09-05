# Graphene P4/C1 失败后的实验行动计划

**状态时间：**2026-08-06 13:37 NZST / 2026-08-06 09:37 HKT  
**适用范围：**450 K 冻结 holdout 已完成且 C1 未通过之后的开发、重新冻结和下一轮未见验证  
**研究目标：**用统一短程 MLIP 加物理电子长程修正，定量复现有限晶格温度下 graphene 的 Γ/K Kohn anomaly；`smearing/degauss (Ry)` 与 lattice temperature 分别通过电子响应和统计物理公式引入。

本文取代[上一份状态快照](GRAPHENE_EXPERIMENT_STATUS_2026-08-06.md)中的“下一步队列”。上一份快照记录的是 P4 尚未完成时的状态，其学习曲线结果仍然有效，但其中 shard A 和 C1 的运行状态已经过期。

## 1. 当前决定

现在不应直接重跑原 C1，也不应启动 C2。正确顺序是：

1. 封存本次失败结果，将 450 K 从未见 holdout 转入 development；
2. 用现有结果做一次 2–4 h 的失败归因，区分短程受力、电子长程项和有限温采样误差；
3. 优先完成物理电子响应 E0，再补交叉 `degauss` force pilot E1；
4. 用统一物理长程算子重新构造 300/450/600 K residual，训练正式 S0；
5. S0 通过后运行 classical TDEP、quantum SSCHA/SCPH 和必要的电子—晶格交叉项；
6. 在读取任何 375/525 K 标签前，固定公式、模型、测试点、构型索引、代码哈希和门槛；
7. 用 375/525 K 做新的未见验证。450 K 后续拟合结果只能报告为开发结果，不能再次登记为 C1。

最先要做的是失败归因 D0 和电子公式 E0。继续增加当前经验插值模型的训练轮数，不能直接解决已经暴露的静态 Γ 响应和 K kink 问题。

## 2. P4/C1 最终结果

RTX 2060 上的最终总验收文件为：

```text
/home/howardwang/phonon/results/graphene_fd_transferability/transferability_acceptance.json
```

最终状态为 `failed_C1`，`passes_C1=false`，`passes_C2=false`，`C2_status=not_run`。采样、DFPT k 网格收敛和矩阵投影重放通过；force、有限温 q-space 和静态 `degauss` 响应没有全部通过。

| 部分 | 指标 | 结果 | 门槛 | 判断 |
|---|---|---:|---:|---|
| 450 K force，15 构型 primary subset | RMSE | 51.43 meV/Å | ≤50 meV/Å | 失败 |
| 450 K force，15 构型 primary subset | 最大分量 | 611.89 meV/Å | ≤250 meV/Å | 失败 |
| force bootstrap | RMSE 95% 上界 | 59.56 meV/Å | ≤50 meV/Å | 失败 |
| 450 K finite-lattice q-space | Γ line MAE | 16.12 cm⁻¹ | <10 cm⁻¹ | 失败 |
| 450 K finite-lattice q-space | Γ 顶支误差 | 16.12 cm⁻¹ | <15 cm⁻¹ | 失败 |
| 450 K finite-lattice q-space | K line MAE | 6.16 cm⁻¹ | <10 cm⁻¹ | 通过 |
| 450 K finite-lattice q-space | K kink 相对误差 | 11.8% | <20% | point estimate 通过 |
| finite-lattice bootstrap | Γ/K line MAE 95% 上界 | 28.59/22.66 cm⁻¹ | <10/10 cm⁻¹ | 失败 |
| 450 K static `degauss` | Γ line MAE | 53.04 cm⁻¹ | <10 cm⁻¹ | 失败 |
| 450 K static `degauss` | K line MAE | 6.57 cm⁻¹ | <10 cm⁻¹ | 通过 |
| 450 K static `degauss` | K kink 相对误差 | 90.0% | <20% | 失败 |
| 矩阵投影重放 | 最大频率差 | 1.59×10⁻¹² cm⁻¹ | <1×10⁻⁵ cm⁻¹ | 通过 |

force 的 all-60 诊断为 RMSE `53.10 meV/Å`、最大分量 `744.56 meV/Å`。primary subset 的最差点是 `seed0:snapshot099`，最大分量 `611.89 meV/Å`；all-60 的最差点是 `seed0:snapshot063`，最大分量 `744.56 meV/Å`。

这些数值支持三点判断：

- 当前短程模型的平均误差接近门槛，但存在明显离群分量，不能只通过延长训练来假定问题会消失；
- finite-lattice 的 K 区 point estimate 已接近目标，主要 point failure 在 Γ 区，但 bootstrap 表明当前有限温谱仍不稳健；
- static `degauss` 的 Γ 误差和 K kink 误差很大，说明端点线性插值的 `rounded cusp + q²` 不能作为最终物理电子响应公式。

240 epochs 小模型在 300/600 K 代理集上通过 `≤50/250 meV/Å`，只能说明固定的小型 MACE 有足够拟合能力。它没有使用正式物理算子扣除后的三温度数据，也没有预测成功 450 K P4，因此不构成 S0 通过。

## 3. 最终模型中两个温度的引入方式

建议把有效动力学矩阵写成

\[
D_{\mathrm{eff}}(\mathbf q;T_{\mathrm{lat}},s)
=D_{\mathrm{SR}}^{\mathrm{eff}}(\mathbf q;T_{\mathrm{lat}})
+\Delta D_{\mathrm{el}}(\mathbf q;s)
+\Delta D_{\times}(\mathbf q;T_{\mathrm{lat}},s),
\]

其中 \(s\) 为 `smearing/degauss (Ry)`。

### 3.1 `degauss`：通过电子极化公式引入

主方案使用 finite-smearing band sum 或 Wannier 插值的电子极化：

\[
\Pi(\mathbf q;s)=
\sum_{mn\mathbf k}
\frac{f(\epsilon_{m\mathbf k};s)-f(\epsilon_{n,\mathbf k+\mathbf q};s)}
{\epsilon_{m\mathbf k}-\epsilon_{n,\mathbf k+\mathbf q}}
\left|g_{mn}(\mathbf k,\mathbf q)\right|^2,
\]

再由同一投影和模规范构造 \(\Delta D_{\mathrm{el}}\)。`degauss (Ry)` 只进入 Fermi occupation 或对应的 Mermin 电子自由能，不作为任意的模型特征。若 band-sum 数据暂时不足，使用共享 \((v_F,\mu,g_\Gamma,g_K)\) 的 Dirac 极化模型作为低成本备选；不同 `degauss` 不能分别拟合一套无关参数。

### 3.2 lattice temperature：通过系综和自由能曲率引入

短程 MLIP 不接收 lattice temperature。它描述统一势能面，温度效应通过以下计算产生：

- classical MD 系综与 TDEP 有效力常数；
- quantum SSCHA/SCPH 的自由能 Hessian；
- 必要时由耦合采样得到 \(\Delta D_{\times}\)。

因此 lattice temperature 使用 K 表示，并与 `smearing/degauss (Ry)` 保持独立。只有 E1 证明扣除电子 Kohn 项后仍有显著局域 `degauss` 力差时，才加入可积分的局域 Mermin 自由能项。

## 4. 数据边界

P4 已经一次性完成并封存后，数据域调整为：

| 数据域 | 条件 | 允许用途 |
|---|---|---|
| `development` | 300/450/600 K；`degauss` 为 0.0019000869、0.00285013035、0.0038001738 Ry | 归因、公式开发、训练、消融 |
| `validation_locked` | 375/525 K；`degauss` 为 0.002375108625、0.003325152075 Ry | 冻结后一次性验证 |
| `independent_dftmd` | 前序 gate 全部通过后固定 | C2 独立轨迹验证 |

450 K 的 force、DFT-TDEP 和 static DFPT 可以进入下一版开发，但结果说明中必须写明它们来自已经开放的失败 holdout。375/525 K 的 DFT、DFPT 和 target 数据在新 freeze 完成前不得生成或读取。

## 5. 执行阶段

### A0：封存失败结果

**耗时：**0.5–1 h；本地和 RTX 2060，不需要新计算。

工作内容：

- 保存总验收、force 验收、谱线验收、bootstrap、freeze manifest 和 SHA-256；
- 登记 `C1 failed`、`C2 not run`；
- 建立 `development`、`validation_locked` 和 `independent_dftmd` 清单；
- 保留原始 450 K 预测和 target，不覆盖原目录。

完成条件是验收产物可按哈希重放，且 375/525 K 目录中没有 target。

### D0：失败归因

**耗时：**2–4 h；本地负责审计，RTX 2060 或 V100-B 做批量推理；不增加 DFT 标签。

固定输出：

1. 按 seed、snapshot、原子、笛卡尔分量、位移幅度和 DFT 力幅度列出 all-60 force error；
2. 单独检查 `seed0:snapshot099` 和 `seed0:snapshot063` 的结构、原子映射、单位、PBC 和长程力扣除；
3. 分别评价 short-only、long-range-only 和 frozen total force，确定离群误差来自哪一项；
4. 用已经开放的 450 K target 做 development-only oracle decomposition，区分短程 TDEP、经验 q-space law 和统计波动；
5. 对 finite-lattice 与 static 结果分别画 Γ/K residual，不能用其中一类误差代替另一类。

决策规则：

- 若发现标签、单位、结构映射或算子重复加减错误，先修复实现并保留原失败记录；
- 若误差是真实的构型覆盖不足，进入 S0 数据重构，不放宽 `50/250 meV/Å`；
- 若 static Γ/K 主要由电子算子造成，E0 保持最高优先级；根据现有 53 cm⁻¹ 的 Γ 误差和 90% 的 K kink 误差，这是当前最可能的分支。

建议产物目录：

```text
results/graphene_physics_temperature/post_p4_diagnostics/
```

### E0：物理电子响应公式

**耗时：**已有数据足够时 8–16 h；若需要 EPW feasibility pilot，再增加 12–24 V100 box-h；若 pilot 通过但需要收敛 band sum，再增加 12–36 V100 box-h。

使用 300/450/600 K 已有 static DFPT line 开发：

- 主模型：finite-smearing band sum/Wannier 极化；
- 备选：共享参数的 Dirac 极化；
- 消融：当前 `rounded cusp + q²` 端点线性插值。

开发 gate：

- Γ、K line MAE 均 `<10 cm⁻¹`；
- Γ、K 顶支绝对误差均 `<15 cm⁻¹`；
- K kink 相对误差 `<20%`；
- 实空间算子与 q-space 投影重放差 `<1e-5 cm⁻¹`；
- 同一套电子结构和耦合参数覆盖三个开发 `degauss (Ry)`。

如果必须为每个 `degauss` 单独拟合参数，或 Wannier/k 网格误差本身已超过 10/15 cm⁻¹ 门槛，则停止扩展，先修电子模型。

### E1：交叉 `degauss` force pilot

**前置条件：**E0 的力算子可重放，且 D0 没有未解决的数据错误。  
**耗时：**30 个新 DFT single points；两台 V100 各 15 个，约 4–5 h 墙钟；RTX 后处理 1–2 h。

从 300/450/600 K 各固定 5 个已有热构型，在三个开发 `degauss (Ry)` 上补齐标签。已有对角标签复用，每个构型通常新增两个条件。

扣除物理 Kohn 项后，局域 `degauss` 分支的判定为：

- 剩余力差 RMSE `≤10 meV/Å`；并且
- 原始 `degauss` 力差 RMSE `≤10 meV/Å`，或剩余力差不超过原始差的 20%。

通过时冻结局域 Mermin 项为零；未通过时进入 E2，不能把 `degauss` 直接作为非守恒 force-model 特征。

### E2：局域 Mermin 自由能，条件执行

**触发条件：**E1 未通过。  
**耗时：**新增 90 个 DFT single points；两台 V100 平分后约 12–14 h；局域自由能训练和积分检查约 8–20 RTX GPU-h。

每个晶格温度扩展到 20 个固定构型，训练一个可积分的局域 Mermin 自由能修正。只保留一个主模型和一个无局域项消融，不做大规模超参数搜索。

### S0：统一短程 MLIP

**前置条件：**E0 完成，E1 已判定局域项为零或 E2 已完成。

使用 300/450/600 K 的物理长程力扣除标签训练同一个短程势；模型不输入 lattice temperature。起始配置固定为本次代理实验得到的 `batch=1, lr=0.001, 240 epochs`，再在正式数据上重新评价。

验收：

- 未见开发构型 force RMSE/max `≤50/250 meV/Å`；
- harmonic replay 不超过冻结门槛；
- energy–force 一致性和 `degauss` 积分闭合通过；
- Hessian 与长程算子加回后的矩阵重放通过。

240 epochs 代理训练在 V100-B 上只用 80.4 min，但正式数据量更大。第一轮主模型和一个消融预计 3–6 h；算力上仍预留 8–16 V100 GPU-h，或 RTX 2060 上 30–70 GPU-h。

### L0、Q0、X0：晶格温度与交叉项开发

S0 通过后依次执行：

- **L0：**在参考 `degauss` 下运行 300/450/600 K、每温度三 seed 的 classical MD/TDEP，约 90–96 RTX GPU-h；
- **Q0：**先做 450 K SSCHA/SCPH benchmark，再做三温度正式量子计算，约 12–24 RTX GPU-h；
- **X0：**先计算一个固定非对角条件。若交叉项对 Γ/K line 的附加影响 `<2 cm⁻¹` 且不改变顶支/kink gate，其余组合由公式重建；否则补耦合轨迹，预留 30–120 RTX GPU-h。

classical TDEP 和 quantum SSCHA/SCPH 分别报告，不用一种结果替代另一种结论。

### F0/V0/V1/V2：重新冻结与未见验证

进入验证前固定模型、电子公式参数、代码、验证轨迹 seeds、DFT 构型索引和门槛。最低限度的两个对角验证点为：

- `(375 K, 0.002375108625 Ry)`；
- `(525 K, 0.003325152075 Ry)`。

若目标包括二维解耦 X1，再增加交换后的两个非对角组合。验证顺序为：

1. VP：冻结模型生成新的验证轨迹；
2. V0：两台 V100 分别计算两个未见 `degauss` 的 static DFPT；
3. V1：四个组合各 15 个构型的 force screen；
4. V2：两个对角条件各 60 个标签的 DFT-TDEP；
5. 只有这些 gate 通过后才启动 I0 独立 DFT-MD，并登记新的 C2。

## 6. 三台机器的安排

2026-08-06 13:37 NZST 的实时检查结果如下：

| 机器 | GPU | 磁盘 | 当前状态 | 下一项任务 |
|---|---:|---:|---|---|
| V100-A | 0%，5 MiB/32 GiB | `/data` 可用 182 GiB | 空闲 | E1 的 15 个标签；验证阶段负责 375 K 对应 `degauss` |
| V100-B | 0%，5 MiB/32 GiB | `/data` 可用 108 GiB | 空闲 | E0 所需 EPW pilot 或 E1 的另 15 个标签；随后可训练 S0 |
| RTX 2060 | 0%，415 MiB/8 GiB | 根分区可用 72 GiB | 只有旧等待脚本，无计算进程 | D0/E0 后处理；清理存储后运行 L0/Q0/X0 |

第一批实际队列应为：

1. 本地与 RTX 2060 立即做 A0/D0；
2. D0 进行时并行实现 E0 的 band-sum/Dirac 原型；
3. 若 inventory 表明 EPW 输入齐全，V100-B 启动一个固定 \(s_0\) feasibility pilot；否则两台 V100 保持空闲，不生成无助于当前决策的新标签；
4. E0 的算子和 D0 数据审计通过后，两台 V100 立即并行 E1；
5. E1 后按 gate 选择 E2 或 S0。

RTX 根分区只有 72 GiB，低于 L0/Q0 的 100 GiB 启动门槛。A0 归档并核对后，需释放至少约 28 GiB 的可重建缓存、旧 checkpoint 或临时轨迹；清理前不启动新的三 seed 轨迹或 SSCHA population。

## 7. 时间预期

| 目标 | 从现在起的条件性时间 |
|---|---:|
| 完成失败归因，决定主要修复方向 | 2–4 h |
| 得到 E0 电子公式的第一版开发结论 | 8–16 h；需要新 EPW 时约 1–3 天 |
| 完成 E1 并判断是否需要局域 Mermin 项 | E0 后 5–7 h |
| 得到正式 S0 的第一版开发 gate | E1 后约 3–8 h；触发 E2 时增加约 1 天 |
| 得到“物理分解是否可行”的阶段性开发结论 | 最快 1–2 天，触发 EPW/E2 时约 2–4 天 |
| 完成最小 E1+L1 的新 375/525 K 未见验证 | 乐观约 12–16 天；方法或收敛重构时约 2–3 周 |
| 增加二维 X1 | 额外约 3–8 天 |
| 增加一条独立 DFT-MD/C2 | 前序通过后额外 10–14 天 |

“1–2 天”的阶段性结论只说明开发集上的物理公式、交叉力和统一短程模型具有可行性；它不等同于新的未见温度验证。完整 E1+L1+X1+C2 仍按约 4–6 周规划。

## 8. 启动与停止条件

- 不重跑或调参后复用 450 K 宣布 C1；
- 不因 force RMSE 只超出 1.43 meV/Å 就忽略 611.89 meV/Å 的最大分量失败；
- 不因 finite-lattice K point estimate 通过就忽略 Γ point failure 和全部 bootstrap 上界失败；
- E0 未通过，不启动新的统一 S0 训练；
- D0/E0 未完成，不批量生成交叉 `degauss` 标签；
- S0 未通过，不运行 L0/Q0/X0；
- RTX 可用空间未达到 100 GiB，不运行大规模轨迹或 SSCHA；
- freeze 未完成，不生成 375/525 K target；
- V0/V1/V2 未通过，不启动昂贵的独立 DFT-MD。

## 9. 关联文档

- [450 K 条件化失败分支](GRAPHENE_FD_CONDITIONAL_FALLBACK_PLAN.md)
- [原 P4/C1 迁移验证方案](GRAPHENE_FD_TRANSFERABILITY_PLAN.md)
- [`degauss` 与晶格温度物理公式方案](GRAPHENE_PHYSICS_BASED_DEGAUSS_LATTICE_TEMPERATURE_PLAN.md)
- [完整实验与算力预算](GRAPHENE_PHYSICS_TEMPERATURE_EXPERIMENT_COMPUTE_PLAN.md)
- [240 epochs 学习曲线结果](../results/graphene_physics_temperature/benchmarks/mace_learning_curve_240ep_v100b/learning_curve_summary.json)

本计划的近期关闭点是完成 A0、D0、E0、E1 和 S0，并形成一份开发阶段的 go/no-go 记录。下一轮定量结论必须来自冻结后的 375/525 K 未见验证，不能由 450 K 的开发回放替代。
