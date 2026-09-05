# Graphene 固定 smearing 有限温声子谱闭环计划

**状态时间：**2026-08-25 14:48 NZST  
**近期目标：**在相同晶格、相同 lattice temperature 和相同 `smearing/degauss (Ry)` 下，验证并改进当前短程 MLIP，再与冻结的 full-EPC 长程响应组合，定量复现 K 点附近 A' 支的有限温 Kohn anomaly。  
**第一验证条件：**`T_lat = 450 K`，Fermi–Dirac `degauss = 0.0019000869 Ry`，`6×6×1` graphene 超胞。  
**最终输出：**300/450 K 固定 smearing 的 MLIP+长程修正与 matched-ensemble DFT 对比；随后在同一晶格温度背景上画零和多个有限 smearing 的 A' 支。

## 执行状态（2026-08-22）

R0 已完成。当前 `base + S0 delta + T300 q6 operator` 相对 E50 新 fixed-smearing 60 点标签的精确结果为：

| 指标 | 数值 | R1 门槛 | 说明 |
|---|---:|---:|---|
| total-force component RMSE | 23.377 meV/Å | ≤30 | 通过 development 参考线 |
| 最大力分量误差 | 244.517 meV/Å | ≤200 | 未通过 |
| K 点 A' 投影力 RMS | 25.019 meV/Å | ≤15 | 未通过 |
| A' restoring slope 相对误差 | 0.935% | ≤5% | 通过 |
| centered energy RMSE | 9.602 meV/config | ≤19.4 | 通过 |

开发集上的最优长程整体附加缩放为 `0.0985`，但只能消除 `0.151%` 的平方力误差，RMSE 仅从 `23.377` 降至 `23.359 meV/Å`。这不支持先调整长程项幅度；当前优先检验有限温分布上的短程残差。

R1 的 12 个构型已在读取 DFT 结果前冻结，索引为 `94, 65, 87, 272, 168, 277, 216, 99, 152, 20, 6, 203`。A/B 分片分别为：

- V100-A：`168, 65, 20, 99, 203, 94`；
- V100-B：`216, 6, 277, 87, 272, 152`。

两组任务于 `18:06 NZST` 启动，输入哈希已和 freeze manifest 核对，两台机器的 `pw.x` 均已实际占用 GPU。按同参数历史实测，V100-B 预计约 `19:27` 完成，V100-A 预计约 `19:45` 完成；把 SCF 波动和合并验收计入后，计划在 `20:15 NZST` 前给出 R1 结论。若 SCF 出现不收敛，保留已完成标签并从单构型 NPZ 断点续跑。

### 18:57 更新：旧 X0 的 atom-order 接口错误

在首个 DFT 标签返回后核对 SSCHA 二进制单位和 current model 重推理时，发现旧 X0 不能作为 current-S0 自洽分布：

- `python-sscha` 的 `forces_pop1.npy` 单位是 Ry/Å，`energies_pop1.npy` 单位是 Ry；选点文件此前把字段名误写成了 eV 单位。这个常数换算不改变逐列标准化后的选点与 A/B 分片。
- 更关键的是，旧 runner 把 CellConstructor 顺序的构型传给了 phonopy 顺序的参考位置和 q6 Hessian，没有把参考位置与 Hessian 一起换到 CellConstructor 顺序。
- 用历史代码路径重建 12 点保存力，重建误差仅为 `0.01099 meV/Å`；说明问题已经定位到确定的换序接口。
- 正确 q6 力在这 12 点上的 RMS 为 `9.854 meV/Å`，旧接口实际误加的谐波力 RMS 为 `1784.005 meV/Å`。

因此原 R1 的“current-S0 self-consistent overlap”命名作废。这 12 个 DFT 标签继续完成，但只作为旧错误系综的诊断/OOD 数据，不用于宣称 finite-temperature 闭环通过或失败。

修复后的接口已通过独立数值测试：换序前后能量差 `6.5×10^-18 eV`，力最大差 `2.1×10^-17 eV/Å`，有限差分和解析力差 `5.6×10^-13 eV/Å`。新的独立目录 `R1_order_safe_sscha` 已于 `18:54 NZST` 启动 corrected Q0(450/450)，通过后自动运行 corrected X0(450/300)。本轮不改短程模型参数，也不改长程响应的 q 依赖或幅度。

### 20:18 更新：R1 诊断完成，已进入短程修复

修正原子顺序后的 Q0/X0 已完成并通过 runner 自检。固定 `T_lat=450 K` 时，corrected Q0（`degauss=0.0028501304 Ry`）和 corrected X0（`degauss=0.0019000869 Ry`）的 K 点最高 A' 频率分别为 `1270.013` 和 `1266.691 cm^-1`。这只是 q6 采样 Hessian 的顺序修复结果；正式谱仍须减去 q6 并加回 frozen full-EPC response。

两台 V100 的 12 个 DFT 标签于 `19:46:59 NZST` 全部完成。由于选点来自旧的错误 X0，先前在不读取 DFT 力和能量的条件下，用 corrected X0 的 300 个构型冻结了几何支持域：12 点中 8 个为 core、1 个为 edge、3 个在支持域外。因此以下 9 点结果是可用于调整方向和训练的 development diagnostic，不是 finite-temperature holdout：

| 指标 | 支持域 9 点 | 固定门槛 | 结果 |
|---|---:|---:|---|
| total-force component RMSE | 95.237 meV/Å | ≤30 | 未通过 |
| 最大力分量误差 | 1534.396 meV/Å | ≤200 | 未通过 |
| K 点 A' 投影力 RMS | 84.958 meV/Å | ≤15 | 未通过 |
| A' restoring slope 相对误差 | 0.560% | ≤5% | 通过 |
| centered energy RMSE | 30.921 meV/config | ≤19.4 | 未通过 |
| importance-weight ESS/N | 0.371 | ≥0.3 | 通过 |

9 点 force RMSE 的 bootstrap 95% 区间为 `67.064–128.189 meV/Å`，A' 投影力 RMS 区间为 `61.071–105.072 meV/Å`。把当前 q6 长程力额外乘以最优全局系数，只能将 RMSE 从 `95.237` 降到 `94.887 meV/Å`，消除 `0.733%` 的平方力误差；A' 恢复力斜率本身也已经通过。因此本轮判定为：冻结长程核，修复有限温短程 residual。

第一轮短程修复数据已经冻结：

- train/val/test 为 `305/93/26` 个构型；train 相对原 S0 只增加上述 9 个支持域内 DFT 标签；
- 支持域外的 `168, 277, 152` 不进入训练；
- 新增点的目标严格使用 `DFT - frozen base - T300 q6 operator`；
- thermal 能量使用一个全局常数平移后联合训练；harmonic replay 因绝对能量零点不同，能量权重固定为 0，力权重保留；
- lattice temperature 与 `degauss` 都不作为局域 MLIP 输入；q6/full-EPC 参数不变。

两个同架构对照已启动：

1. RTX 2060 上的 `gate_scaled_ef_seed83` 于 `20:08:51 NZST` 启动，使用固定 gate 归一化的 energy+force loss；
2. V100-B 上的 `forces_only_control_seed83` 于 `20:15:59 NZST` 启动，数据完全相同但只拟合力，用来区分新增数据与能量损失的作用。

两者均保持 `2 interactions / 32x0e+32x1o / r_max=5 Å`，最多 240 epoch。预计在 `21:20–22:10 NZST` 完成训练，在 `22:30 NZST` 前完成 9 点 A'、300/450/600 K development 和 harmonic replay 的联合 checkpoint gate。只有同架构 loss/data 修复仍不能降低支持域力误差时，才进入 3-interaction depth 消融；本阶段不增加 DFT 标签。

### 21:47 更新：同架构修复未闭环，depth-3 已启动

同架构的从头训练、标准低学习率微调和聚焦重权微调均已完成。所有模型都使用同一个冻结评估程序重新推理；表中的 9 点是已进入训练的 development support，不能代替新的 holdout。

| 短程候选 | 9 点 force RMSE | 最大力误差 | A' 投影力 RMS | centered energy RMSE | T300/450/600 force RMSE | harmonic RMSE | 最差归一化 gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前 S0 | 95.237 | 1534.399 | 84.959 | 30.920 | 26.295 / 22.566 / 27.343 | 13.932 | 7.672 |
| 标准微调 final | 57.830 | 1064.015 | 46.397 | 16.347 | 19.417 / 17.732 / 21.427 | 18.158 | 5.320 |
| 聚焦微调 final | 61.369 | 1102.850 | 50.188 | 17.632 | 20.334 / 18.669 / 22.653 | 18.123 | 5.514 |
| forces-only 从头训练 final | 57.692 | 941.194 | 45.701 | 27.079 | 23.104 / 21.062 / 24.628 | 16.244 | 4.706 |

力的单位为 meV/Å，能量的单位为 meV/config。四个模型的 A' restoring slope 相对误差均小于 1%，但最大力和 A' 投影误差均远高于 `200` 和 `15 meV/Å` 的门槛。标准微调通过了相对能量和三个 thermal RMSE 门槛，却把 harmonic RMSE 从 `13.932` 增至 `18.158 meV/Å`；聚焦提高 9 点与 replay 权重没有改善这一结果。从头训练的最差 gate 较小，但相对能量仍未通过。冻结的 harmonic 门槛为 foundation base 的 `1.25× = 9.045 meV/Å`，当前 S0 本身也不满足这一项，因此上述候选都不能进入正式 SSCHA 重采样。

三个新模型的最大误差都来自 core 支持点 `sscha_index=99`：标准微调、聚焦微调和 forces-only 从头训练在该点的最大力误差分别为 `1064.015`、`1102.850` 和 `941.194 meV/Å`。因此目前不是少量 edge/OOD 点拖累平均数，depth 消融需要直接检验局域模型能否表达这个支持域内构型。

对 `sscha_index=99` 的原始标签进一步核对后，最大误差固定出现在 atom 48 的 x 分量。该原子的最短 C–C 距离为 `1.27046 Å`，短程 DFT residual 目标为 `2.54269 eV/Å`；当前 S0、标准微调、聚焦微调和 forces-only 从头训练分别只给出 `1.00829`、`1.47867`、`1.43984` 和 `1.60149 eV/Å`。原始 QE 计算在 15 次迭代后收敛，最终 estimated SCF accuracy 为 `7.5×10^-11 Ry`，输出正常结束；DFT 总力向量和接近零，输入位置与标签的 atom 48 顺序一致。现有证据未显示该最大误差来自 SCF 未收敛或又一次原子换序错误，而更像是模型对压缩局域环境的系统性低估。

结果支持进入模型表达能力消融，而不是继续增加相同类型的 DFT 标签，也不支持调整长程项。`depth3_forces_only_seed83` 已于 `21:43:45 NZST` 在 V100-B 启动；相对于 forces-only 从头训练对照，只把 message-passing interactions 从 2 改为 3，保持数据、权重、`32x0e+32x1o`、`r_max=5 Å`、损失、seed 和长程项不变。前 20 epoch 实测每 5 epoch 约 2.5 min，预计 `23:45–00:00 NZST` 完成训练和最终 gate。冻结评估数据与脚本的 SHA256 已在 V100-B 核对，并已启动等待服务；训练完成、GPU 释放后会自动评估 final model。若中间 checkpoint 明确优于 final，再按事先固定的联合 gate 选择 checkpoint，不按单一 validation loss 选择。

若 depth-3 仍未通过，下一步只运行一次 `64x0e+64x1o` capacity 消融。只有 development gate 通过后，才生成新的 corrected-X0 系综和未参与训练的 order-safe matched DFT holdout；在此之前不画新的正式 MLIP+长程有限温声子谱。

### 08-23 05:33 更新：depth-3 改善但未通过，capacity-64 已启动

Depth-3 于 `2026-08-22 23:43:58 NZST` 完成。final model 和事先选定的 epoch 60/120/180/235 checkpoints 均已使用冻结程序评估。final 的支持域 force RMSE、最大力误差和 A' 投影力 RMS 分别为 `47.413`、`859.432` 和 `36.342 meV/Å`；相对能量 RMSE 为 `18.063 meV/config`，T300/450/600 force RMSE 为 `15.233/15.368/16.896 meV/Å`，harmonic replay 为 `12.089 meV/Å`。因此能量与 thermal gate 通过，但支持域的 `30/200/15 meV/Å` 三项和 `9.045 meV/Å` harmonic gate 未通过。

按统一的最差归一化 gate，epoch 180 略优于 final：

| Depth-3 checkpoint | 9 点 force RMSE | 最大力误差 | A' 投影力 RMS | centered energy RMSE | harmonic RMSE | 最差归一化 gate |
|---|---:|---:|---:|---:|---:|---:|
| epoch 180 | 47.821 | 820.243 | 39.781 | 18.232 | 13.902 | 4.101 |
| final | 47.413 | 859.432 | 36.342 | 18.063 | 12.089 | 4.297 |

与 2-interaction forces-only final 相比，depth-3 把支持域 RMSE 从 `57.692` 降到 `47.413 meV/Å`，A' 投影误差从 `45.701` 降到 `36.342 meV/Å`，说明增加 message-passing depth 有实际作用；但主导异常点仍使最大误差保持在 `0.82–0.86 eV/Å`，尚不能进入重采样与新 DFT holdout。

`capacity64_forces_only_seed83` 已于 `2026-08-23 05:31:54 NZST` 在 V100-B 启动。它回到 2 interactions，只把 hidden irreps 从 `32x0e+32x1o` 改为 `64x0e+64x1o`；数据、损失、权重、cutoff、seed 和长程项均不变。前 5 epoch 实测约 `99 s`，按 240 epoch 外加最终评估估算，预计 `06:55–07:10 NZST` 得到 final gate。训练完成后的 final gate 已接入自动等待服务。本分支是当前计划中的最后一个常规 MACE depth/capacity 消融；若仍失败，不继续堆叠相同标签或随意增大网络，而转向显式压缩环境 residual、局域 cutoff/基函数表达和保守 reciprocal-space coupling 的结构诊断。

### 08-23 16:26 更新：capacity-64 未改善，常规结构消融结束

Capacity-64 于 `2026-08-23 06:48:04 NZST` 完成，final 与 epoch 60/120/180/235 checkpoints 均已执行同一套冻结门控。epoch 235 与 final 数值等价，是该分支最优结果：

| 候选 | 9 点 force RMSE | 最大力误差 | A' 投影力 RMS | centered energy RMSE | harmonic RMSE | 最差归一化 gate |
|---|---:|---:|---:|---:|---:|---:|
| depth-3 epoch 180 | 47.821 | 820.243 | 39.781 | 18.232 | 13.902 | 4.101 |
| depth-3 final | 47.413 | 859.432 | 36.342 | 18.063 | 12.089 | 4.297 |
| capacity-64 final/epoch 235 | 59.997 | 937.317 | 54.771 | 29.782 | 10.431 | 4.687 |

Capacity-64 只有 harmonic replay 比 depth-3 更接近门槛，但仍高于 `9.045 meV/Å`；支持域力、A' 投影和相对能量全部退化。T300/450/600 force RMSE 为 `26.741/24.162/29.642 meV/Å`，但 T300 最大力为 `209.319 meV/Å`，也略高于 `200 meV/Å` 门槛。最大支持域误差仍来自 `sscha_index=99`，为 `937.317 meV/Å`。

因此当前数据不支持继续简单增加 hidden channels，也不能据此重调 full-EPC 长程项。常规 loss、重权微调、depth 和 capacity 消融已经完成；其中 depth-3 提供了最明确的改进，但离门槛仍远。下一阶段应先把 `sscha_index=99` 的压缩键残差分解为保守两体径向分量与多体环境分量，并分别检查 radial basis、短距离覆盖和 cutoff 表达。只有结构修复候选在同一 development gate 上通过后，才进入新的 matched holdout 和正式声子谱。

### 08-24 04:21 更新：误差以环境依赖的径向分量为主，单一 pair spline 不适用

已在不增加 DFT 标签的条件下完成原子局域支持域和力残差分解。对 depth-3 final 的 9 个 development 构型，中心原子第一近邻键方向张成的子空间解释了 `87.19%` 的平方力误差；去掉该分量后的 force RMSE、最大力分量和 A' 投影力 RMS 分别为 `16.971`、`137.276` 和 `14.273 meV/Å`，均低于相应的 `30/200/15 meV/Å` 门槛。这说明剩余问题主要沿局域 C–C 键方向，但仅凭方向分解还不能推出它是统一的两体势。

主导异常点 `sscha_index=99` 不能作为明显 OOD 点剔除：其最短键 `1.27046 Å` 位于 corrected-X0 构型最短键分布的第 `4.67` 百分位；最异常 atom 48 的局域 kNN 分位为 `90.33%`，整构型没有原子超过事先固定的 `99%` 局域 OOD 线。该点 `90.12%` 的平方误差属于径向子空间，最短的 atom 12–48 键单独解释 `63.35%`，对应 stretch residual 为 `1339.46 meV/Å`。因此它是当前热分布内必须拟合的压缩环境，不能用几何支持域规则删除。

随后在 depth-3 上叠加了能量守恒的 C–C B-spline pair residual，固定 `1.15–1.75 Å`、12 个三次基函数，并扫描事先限定的 5 个正则强度和 force-only/centered-energy 两种目标。所有 pair 候选都比不加修正更差。原 depth-3 的最差归一化 gate 为 `4.297`；最接近的 pair 候选仍为 `7.053`，其支持域 force RMSE、最大力误差、A' 投影力 RMS、相对能量 RMSE和 harmonic RMSE分别为 `86.691`、`749.842`、`66.512`、`30.470 meV/config` 和 `63.795 meV/Å`。pair 模型的解析力与能量数值差分在测试构型上的绝对差为 `5.3×10^-11`，排除了力符号或能量导数实现错误。

这一结果表明，同一键长在不同角度、邻键压缩和热环境下需要不同修正；径向方向的误差是环境依赖的多体误差。下一步固定为一个小截断、低容量、由能量导出的多体 residual adapter，并与 depth-3 相加。它只学习现有 `target - depth3` 残差，使用 harmonic replay 约束，不修改 base、q6 或 full-EPC，也不增加新 DFT。只有该 adapter 通过当前 development gate 后，才生成新 SSCHA 系综和 order-safe matched DFT holdout。

### 08-24 04:40 更新：多体 residual adapter 已启动

R2D residual 数据已从冻结的 R2C 数据和 depth-3 final 确定生成，train/val/test 为 `305/93/26` 个构型，没有新 DFT 标签。逐构型检查表明 `depth3 + residual target` 对原短程 target 的最大重构误差为 `0 eV` 和 `4.4×10^-16 eV/Å`。train 中 residual force RMSE 为 `16.416 meV/Å`；其中 9 个支持域构型为 `47.413 meV/Å`，harmonic replay 为 `10.061 meV/Å`。

固定 adapter 架构为 `2 interactions / 16x0e+16x1o / r_max=3 Å / 8 radial basis / correlation=3`，使用 forces-only 损失和 `10^-6` weight decay；支持域构型权重为 8，harmonic replay 权重为 16。它作为独立的能量模型从头训练，最终只与冻结 depth-3 相加，不微调 depth-3 权重。2 epoch smoke test 已在 RTX 2060 完成，峰值显存 `636 MiB`，数据和脚本传输哈希与本地一致。

正式 240 epoch 训练于 `2026-08-24 04:40 NZST` 在 RTX 2060 启动，自动门控等待器同时启动。按 smoke test 吞吐率，预计 `06:05–06:30 NZST` 完成训练，并在随后约 5–10 min 内评估 epoch `40/80/120/160/200/235` 与 final。两台 V100 当前经 Tailscale 直连及 RTX 中转均超时，因此本任务使用在线且空闲的 RTX；未通过公网通道传数据或运行实验。

### 08-24 13:56 更新：全数据小 adapter 未通过，已启动目标解耦对照

首个多体 adapter 已完成 240 epoch 和固定 checkpoint 门控，实际训练墙钟 `3994 s`。最优为 epoch 200，但未通过：

| 候选 | 9 点 force RMSE | 最大力误差 | A' 投影力 RMS | centered energy RMSE | harmonic RMSE | 最差归一化 gate |
|---|---:|---:|---:|---:|---:|---:|
| depth-3，无 adapter | 47.413 | 859.432 | 36.342 | 18.063 | 12.089 | 4.297 |
| 小多体 adapter，epoch 200 | 46.914 | 848.534 | 35.992 | 17.895 | 12.246 | 4.243 |

力的单位为 meV/Å，能量为 meV/config。epoch 200 的 T300/T450/T600 force RMSE 为 `15.205/15.358/16.904 meV/Å`，三个 thermal gate 都通过；失败项仍是支持域最大力、A' 投影、支持域 RMSE和 harmonic replay。adapter 在 9 点上的输出 RMS 只有 `0.952 meV/Å`，而目标残差为 `47.413 meV/Å`；方向余弦为 `0.532`。即使事后使用最小二乘幅度 `26.48`，支持域 RMSE 也只能降到 `40.154 meV/Å`。全 305 点上的方向余弦只有 `0.076`，说明不能靠统一放大修复。

为区分架构表达与联合目标压制，随后完成了不用于模型选择的 overfit probe：

- 只拟合 `sscha_index=99` 时，同一架构把该构型 RMSE 从 `114.510` 降到 `53.2 meV/Å`；
- 只拟合 9 个支持点时，RMSE 从 `46.88` 降到 `32.66 meV/Å`，最大误差降至 `255.19 meV/Å`，相对能量 RMSE为 `13.87 meV/config`；
- support9-only 模型在 T300/T450/T600 上仍为 `14.532/16.296/17.818 meV/Å`，但 harmonic replay 外推失稳到 `344.530 meV/Å`，最大分量达到 `16.91 eV/Å`。

这些结果表明，小架构能学习一部分压缩环境方向，但全数据中的低相关 thermal residual 会把修正压到接近零；反过来只拟合支持点又会在 harmonic replay 上失稳。因此下一项固定为 `support residual + harmonic residual` 的目标解耦对照：train/val 为 `81/34` 个构型，其中 gradient 数据只有 9 个支持点和 72 个 harmonic replay，thermal 构型只用于 test/gate；支持与 harmonic 权重均为 8，架构、seed、epoch、base、q6 和 full-EPC 全部不变。

该对照已于 `2026-08-24 13:56 NZST` 在 RTX 2060 启动，训练与自动 checkpoint gate 进程均已实际运行。按上一轮吞吐率缩放，预计 `14:15–14:25 NZST` 得到联合门控结果。本轮仍没有新 DFT；若它也不能同时压低 support 和 harmonic，不再继续调数据权重，而转向带有显式平滑边界/基态锚点的 bond-order 保守修正或重新设计短程基座。

### 08-24 15:10 更新：目标解耦、容量和 bond-order 对照均未通过

`support residual + harmonic residual` 的 16 通道对照完成了 240 epoch 和固定 checkpoint 门控，训练墙钟为 `1080 s`。最优 checkpoint 为 epoch 200。V100-A 同时完成了只把 hidden irreps 从 `16x0e+16x1o` 增至 `32x0e+32x1o` 的容量对照，训练墙钟为 `1788 s`；最优为 final/epoch 235。两项的其他数据、权重、cutoff、interaction 数、seed、base、q6 和 full-EPC 均相同。

V100-B 还完成了一个显式环境依赖、由标量能量严格求导的 bond-order residual。该模型使用 10 个平滑径向基和 7 个一阶邻域环境不变量，共 70 个线性能量基；固定扫描 5 个 ridge 系数。能量有限差分与解析力的绝对差为 `1.25×10^-10 eV/Å`，9 个支持构型的最大净力分量为 `1.09×10^-15 eV/Å`，因此下表差异不是能量—力不一致造成的。

| 候选 | 9 点 force RMSE | 最大力误差 | A' 投影力 RMS | centered energy RMSE | harmonic RMSE | T300/T450/T600 force RMSE | 最差归一化 gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| depth-3，无 adapter | 47.413 | 859.432 | 36.342 | 18.063 | 12.089 | 15.233/15.368/16.896 | 4.297 |
| 16 通道，epoch 200 | 46.272 | 831.797 | 35.334 | 17.840 | 12.596 | 15.206/15.371/16.863 | 4.159 |
| 32 通道，final | 46.270 | 832.689 | 35.333 | 17.836 | 12.512 | 15.202/15.393/16.851 | 4.163 |
| 环境 bond-order，ridge `10^-6` | 46.208 | 826.823 | 35.526 | 16.916 | 12.540 | 15.342/15.402/16.769 | 4.134 |

力的单位为 meV/Å，能量为 meV/config。三个新候选的 thermal gate 均保持通过，但 support RMSE、最大力、A' 投影和 harmonic replay 都没有达到固定门槛。16→32 通道没有可分辨的收益，排除了当前结果主要受 hidden-channel 容量限制的解释。bond-order 候选在训练支持点上只有小幅改善；leave-one-supported-configuration-out 的 RMSE、最大误差、A' RMS 和 centered energy RMSE 分别为 `47.233`、`858.943`、`35.816 meV/Å` 和 `432.238 meV/config`，说明这组线性一阶邻域环境基不能迁移到未参与拟合的压缩环境。

据此停止继续增加相同 adapter 的通道数、调整 support/harmonic 权重或增加同分布 DFT 标签。q6 和 full-EPC 长程项继续冻结，因为 A' 恢复力斜率误差始终小于 `0.1%`，而主要失败仍集中在局域最大力和环境迁移。下一项短程结构实验应增加真正的角向张量通道或允许 depth-3 的末层与 residual 联合调整，并用 harmonic trust region 约束；在该结构通过当前 development gate 前，不生成新的 SSCHA 系综、matched DFT holdout 或正式有限温声子谱。

### 08-24 15:41 更新：两项短程结构实验并行运行

V100-A 已启动独立 residual adapter 的角向张量对照。相对于 16 通道基线，它只把 hidden irreps 从 `16x0e+16x1o` 改为 `16x0e+16x1o+16x2e`；数据仍为 9 个 support residual 加 72 个 harmonic residual，thermal 只用于 development test，其他架构、权重、seed、epoch、base、q6 和 full-EPC 不变。2 epoch smoke test 已确认 MACE 建立了 `max_L=2` 模型并正常完成能量—力训练。完整 240 epoch 于 `15:19 NZST` 在 V100-A 启动。

V100-B 同时运行 depth-3 基座的末块联合微调。为保持训练目标一致，从冻结的 R2C 全短程目标中提取了 9 个 support 和 72 个 harmonic train 构型，val 为 25 个 harmonic 加相同 9 个 support，26 个 thermal 构型仍只用于 development test；没有新 DFT。MACE 0.3.16 的通用 foundation element remapper 无法重建这个三层单元素模型，两个 smoke 在进入优化器前因 `skip_tp` shape 检查失败，失败目录和日志均保留。正式实现改为直接读取已审计的 depth-3 模型，不重建网络，只开放 `interactions.2.*`、`products.2.*` 和 `readouts.2.*`，共 `44,208/118,800` 个参数；模型仍由标量能量求导。1 epoch smoke 的初始 support RMSE/最大误差与既有基线一致到 `0.001 meV/Å`，说明直接读取没有改变模型。

完整末块训练固定为 `160 epoch / lr=5×10^-5 / EMA=0.99 / seed=83`，harmonic 和 support 构型权重均为 8。根据 smoke 中较快的早期变化，在正式训练前固定评估 epoch `1/5/10/20/40/80/120/160` 与 final；V100-B 的训练和自动门控于 `15:40 NZST` 启动。两项实验预计在 `16:05–16:15 NZST` 得到联合开发门控。任一候选未通过时，仍不打开新 SSCHA、matched DFT holdout 或长程项重拟合。

### 08-24 16:20 更新：角向张量无收益，末块调整改善整体误差但未修复 A'

两项结构实验均完成训练和固定 checkpoint 门控。`l=2` adapter 的训练墙钟为 `2225 s`，最优为 epoch 200；depth-3 末块直接微调的训练墙钟为 `2173 s`，最优为 epoch 160/final。两台 V100 的正式训练合计约 `1.22 GPU-h`，另有少量 smoke、传输和门控开销。

| 候选 | 9 点 force RMSE | 最大力误差 | A' 投影力 RMS | centered energy RMSE | harmonic RMSE | T300/T450/T600 force RMSE | 最差归一化 gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| depth-3，无本轮调整 | 47.413 | 859.432 | 36.342 | 18.063 | 12.089 | 15.233/15.368/16.896 | 4.297 |
| `l=2` adapter，epoch 200 | 46.331 | 833.801 | 35.438 | 17.801 | 12.524 | 15.160/15.389/16.871 | 4.169 |
| depth-3 末块，epoch 160 | 31.294 | 513.204 | 33.726 | 13.310 | 13.818 | 20.570/19.703/20.733 | 2.566 |

力的单位为 meV/Å，能量为 meV/config。`l=2` 与不含 `l=2` 的 16/32 通道 adapter 数值几乎重合，因此单独增加二阶角向隐藏通道没有解决当前误差。末块微调明显降低了整体 support RMSE和最大误差，centered energy、ESS、A' 恢复力斜率以及三个 thermal gate 均通过；但 support RMSE 仍略高于 `30 meV/Å`，最大误差仍为门槛的 `2.566` 倍，A' 投影为门槛的 `2.248` 倍，harmonic RMSE 也高于固定的 `9.045 meV/Å` 上限。训练继续到 160 epoch 时，A' RMS 只从 `36.342` 降到 `33.726 meV/Å`，说明末块学到的主要是其他力分量，未对准 K 点 A' 相关残差。

这轮结果不支持继续增加普通角向通道、延长末块训练或放开更多相同层，也不支持修改 q6/full-EPC：A' 恢复力斜率相对误差仍只有 `0.20%`。下一项应改变短程径向分辨率和激活区域，而不是继续扩网络。具体候选是一个能量守恒的短键 expert：只在压缩 C–C 键区间平滑激活，使用非线性环境门控区分相同键长下的邻键和角度环境，并在激活边界令能量和力同时为零；同时保留 harmonic replay 和 leave-one-support-out 验收。该候选通过现有 development gate 前，不增加 DFT、不生成新 SSCHA，也不重拟合长程项。

### 08-24 21:10 更新：压缩短键 expert 与径向对照已启动

根据冻结数据确定了短键 expert 的激活区间。9 个 support 构型的最短键为 `1.270–1.328 Å`，support 键长的第 5 百分位为 `1.344 Å`，正常键长中心约为 `1.42 Å`。模型使用 C² 平滑窗口：`1.15–1.20 Å` 开启，在 `1.32–1.38 Å` 关闭；因此正常键长附近修正严格为零。径向部分使用 `1.20–1.36 Å` 的 12 个高分辨率 Gaussian 基，宽度为 `0.022 Å`。

V100-B 运行非线性环境门控模型，共 `1,188` 个可训练参数。门控输入由中心键两端的邻键长度、端点不对称、键角畸变和长度—角度协方差等旋转、平移和置换不变量组成。V100-A 同时运行只保留 12 个径向系数的对照，用于区分收益来自紧支撑径向分辨率还是环境依赖。两者都从零修正开始，冻结 depth-3 末块模型、q6 和 full-EPC；train 仍为 72 个 harmonic replay 加 9 个 support，26 个 thermal 构型只用于 test/gate，没有新增 DFT。

1 epoch smoke 已完成。零初始化精确复现上一轮末块模型：support RMSE、最大误差、harmonic RMSE 和 centered energy RMSE 分别为 `31.294`、`513.204`、`13.818 meV/Å` 和 `13.310 meV/config`。平移不变性误差为 `2.0×10^-16 eV`，解析力与能量数值差分误差为 `5.8×10^-13 eV/Å`，最大净力分量为 `6.9×10^-17 eV/Å`。正式 400 epoch 任务于约 `21:08 NZST` 在两台空闲 V100 上启动，固定评估 epoch `1/5/10/20/40/80/120/160/240/320/400` 与 final；通过完整 development gate 后才运行九折 leave-one-support-out，不通过则不进入新系综或声子谱。

### 08-24 21:38 更新：紧支撑 expert 有小幅收益，但仍未修复最大误差、A' 和 harmonic

两项紧支撑模型均已完成训练和固定 checkpoint 门控。径向-only 对照墙钟为 `326 s`，最优为 epoch 160；非线性环境门控墙钟为 `1364 s`，最优为 epoch 400/final。

| 候选 | 9 点 force RMSE | 最大力误差 | A' 投影力 RMS | centered energy RMSE | harmonic RMSE | T300/T450/T600 force RMSE | 最差归一化 gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 末块基座，无 expert | 31.294 | 513.204 | 33.726 | 13.310 | 13.818 | 20.570/19.703/20.733 | 2.566 |
| 12 参数径向-only，epoch 160 | 30.690 | 486.000 | 33.268 | 13.331 | 14.484 | 20.577/19.642/20.715 | 2.430 |
| 1,188 参数非线性环境门控，epoch 400 | 29.873 | 467.808 | 30.358 | 12.443 | 14.736 | 20.617/19.616/20.810 | 2.339 |

力的单位为 meV/Å，能量为 meV/config。非线性模型首次使 support 总 RMSE 低于 `30 meV/Å`，centered energy、ESS、A' 恢复力斜率和 thermal gate 也通过；但最大误差仍为门槛的 `2.339` 倍，A' 投影为门槛的 `2.024` 倍，harmonic RMSE 为固定上限的 `1.629` 倍。其 A' 斜率相对误差只有 `0.10%`，仍不支持修改 q6/full-EPC。解析力与能量数值差分误差为 `2.8×10^-11 eV/Å`，最大净力分量为 `1.2×10^-17 eV/Å`；失败不是非保守力或切换边界伪影造成的。

一个理想化覆盖下界显示：若能任意消去与 `r<1.38 Å` 键相邻原子的全部误差，support 和 harmonic RMSE 最低可分别达到 `17.825` 和 `7.142 meV/Å`，因此当前截断在几何覆盖上并非必然失败。实际模型只得到小幅改善，问题更接近手工环境基的表达与跨构型迁移，而不是简单缺少激活原子。两项失败候选不运行九折 LOCO，也不进入新 SSCHA 或声子谱。

下一项改用等变多体表示，同时保持短程聚焦。V100-A 已启动 `r_max=2.0 Å / 24 radial basis / 2 interactions / 16x0e+16x1o / correlation=3` 的 MACE residual expert；对照基线为 `r_max=3.0 Å / 8 radial basis`，其余数据、权重、seed、depth-3、q6 和 full-EPC 均不变。2 epoch smoke 已确认实际模型使用 24 个径向基、平均 3 个近邻并完成能量—力训练。正式 240 epoch 于约 `21:32 NZST` 启动，训练完成后会自动物化 epoch `40/80/120/160/200/235` 与 final 并执行同一门控；仍没有新增 DFT。

V100-B 同时运行相同短 cutoff/高径向分辨率下的 `16x0e+16x1o+16x2e` 对照，只增加 `l=2` 通道，用来检验径向分辨率与角向张量是否存在耦合收益。2 epoch smoke 已确认 `max_L=2`、24 个径向基和 `2.0 Å` cutoff 均实际生效；正式 240 epoch 于约 `21:45 NZST` 启动，并接入相同的自动 checkpoint gate。两台机器的训练及 gate 都在远端后台继续运行；当前尚未把任何中间 validation loss 当作物理结论。

### 08-25 00:10 更新：高径向分辨率改善局部误差，但没有通过联合门控

两项 `r_max=2.0 Å / 24 radial basis` residual expert 均已完成训练和固定 checkpoint gate。`l≤1` 与 `l≤2` 的最优 checkpoint 都是 epoch 200；后者改善更明显，但仍没有通过。

| 高径向 expert | 9 点 force RMSE | 最大力误差 | A' 投影力 RMS | centered energy RMSE | harmonic RMSE | T300/T450/T600 force RMSE | 最差归一化 gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `l≤1`，epoch 200 | 38.379 | 481.126 | 30.352 | 17.259 | 14.478 | 15.363/15.501/16.724 | 2.406 |
| `l≤2`，epoch 200 | 35.287 | 295.011 | 30.244 | 17.101 | 13.548 | 15.469/15.461/17.050 | 2.016 |

力的单位为 meV/Å，能量为 meV/config。相对于 depth-3 无 adapter 的 `47.413/859.432/36.342 meV/Å`，短 cutoff 和高径向基确实降低了整体与极值误差；在这组表示下加入 `l=2` 也对最大误差有作用。但 A' 投影仍约为门槛的两倍，harmonic 也超过 `9.045 meV/Å`。因此这一结果支持保留高径向分辨率作为候选分支，但不足以单独进入 SSCHA 或正式声子谱。

### 08-25 00:37 更新：全局保守混合无可行点，已转向共同下降训练

为检验现有修正能否通过幅度重标定组合起来，固定 depth-3、末块模型、非线性短键 expert 和高径向 MACE expert，扫描

`depth3 + α(lastblock-depth3) + β compact + γ radial`。

扫描范围为 `α=0.0–1.4`、`β=-0.5–2.0`、`γ=-0.5–1.5`，粗网格步长 0.1，并在最优区域用 0.02 细化。所有组合仍由标量能量相加，保持能量守恒；没有新 DFT，也没有修改 q6/full-EPC。

| 径向分支 | 扫描点数 | 最优 `α/β/γ` | force RMSE | 最大误差 | A' RMS | harmonic RMSE | 最差 gate | 全 gate 可行点 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `l≤1` | 16,526 | 0.64/0.96/0.70 | 27.977 | 344.450 | 25.791 | 15.576 | 1.722 | 0 |
| `l≤2` | 14,878 | 0.58/1.00/0.68 | 26.702 | 239.359 | 24.452 | 14.746 | 1.630 | 0 |

最优 `l≤2` 组合已经通过整体 force RMSE、centered energy、ESS、A' 恢复力斜率和全部 thermal gate，但最大误差、A' 投影和 harmonic 仍未通过。更重要的是，整个扫描中 A' 和 harmonic 各自能达到的最低值仍分别为 `21.760` 和 `12.012 meV/Å`；`l≤1` 对应下界为 `21.707` 和 `12.036 meV/Å`。这排除了在当前三个修正方向上继续做全局缩放或线性叠加。

RTX 2060 随后对 depth-3 末块做了零优化步的梯度冲突审计。harmonic 与 support 总力、A' 投影、top-16 力分量损失的梯度余弦分别为 `-0.743/-0.709/-0.752`；后三个 support 目标彼此的余弦为 `0.987–0.999`。因此单独提高 A' 或最大误差损失的权重不会产生新的参数方向，主要矛盾是 harmonic 与 support 的负梯度夹角。

基于这个诊断，两项 gate 归一化的共同下降训练已于 `00:37 NZST` 并行启动：V100-A 只开放最后一个 interaction/product/readout 块，共 `44,208` 个参数；V100-B 开放最后两个块，共 `90,192` 个参数。每一步配对 8 个 harmonic 构型和 1 个 support 构型，分别归一化 harmonic 梯度与由总力、A'、top-16 组成的 support 梯度，再沿两者的等角共同下降方向更新。1 epoch smoke 中，所有步骤对两组训练目标的一阶方向内积均为正，最小值为 `0.267` 和 `0.280`；初始模型与 depth-3 基线一致。正式任务固定为 240 epoch，并已接入 epoch `1/5/10/20/40/80/120/160/200/240` 的联合 gate。按 smoke 吞吐率，预计 `01:40–02:10 NZST` 完成训练，在 `02:20 NZST` 前得到门控结论。

### 08-25 01:34 更新：共同下降训练未解决跨构型冲突

两项 240 epoch 正式训练和固定 checkpoint gate 均已完成。最后一块模型从 `00:37:14` 运行到 `01:10:44 NZST`，训练墙钟 `2007 s`，gate 于 `01:12:06` 完成；最后两块模型从 `00:37:19` 运行到 `01:32:29 NZST`，训练墙钟 `3307 s`，gate 于 `01:34:14` 完成。两项正式训练合计约 `1.48 GPU-h`，没有新增 DFT，也没有修改 q6/full-EPC。

| 参数范围 | checkpoint | 9 点 force RMSE | 最大力误差 | A' 投影力 RMS | centered energy RMSE | harmonic RMSE | T300/T450/T600 force RMSE | 最差归一化 gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 最后一块 | epoch 5，最优 | 47.006 | 856.725 | 35.788 | 18.286 | 12.088 | 15.099/15.499/16.905 | 4.284 |
| 最后两块 | epoch 5，最优 | 47.058 | 857.439 | 35.577 | 17.985 | 12.038 | 15.158/15.552/16.923 | 4.287 |
| 最后一块 | epoch 240/final | 55.476 | 1027.479 | 34.099 | 13.359 | 10.066 | 25.140/21.847/21.302 | 5.137 |
| 最后两块 | epoch 240/final | 56.975 | 1004.186 | 33.450 | 13.335 | 9.474 | 27.973/24.163/21.590 | 5.021 |
| 固定门槛 | — | ≤30 | ≤200 | ≤15 | ≤19.4 | ≤9.045 | 各自 ≤30 | ≤1 |

力的单位为 meV/Å，能量为 meV/config。两个分支的最优 checkpoint 都是 epoch 5，数值只比未调整的 depth-3 基线略好；继续训练后，harmonic 逐步下降，但 support RMSE 和最大误差持续上升。最后两块 final 的 harmonic 已降至 `9.474 meV/Å`，只比门槛高 `4.74%`，同时 support RMSE 和最大误差却恶化到 `56.975` 和 `1004.186 meV/Å`。因此不能为了接近 harmonic 门槛而选择晚期 checkpoint。

所有候选的 centered energy、ESS、A' 恢复力斜率和 thermal RMSE 基本保持在门槛内；A' 斜率相对误差仍低于 `0.4%`。失败集中在 support 总力、最大分量、A' 投影以及 harmonic 的联合约束，当前结果仍不支持调整长程项。

共同下降构造只保证每一步对当前 8 个 harmonic 构型和 1 个 support 构型是一阶下降方向，不能保证九个 support 构型之间互不抵消。此前的逐构型 A' 梯度审计已经显示，九个 support 构型两两梯度余弦的平均值为 `-0.106`，最小值为 `-0.990`。正式训练中，后续 support 步骤会撤销前面构型的改善；开放第二个块增加了 harmonic 调整能力，但没有形成能区分不同局域环境的独立参数方向。

本轮据此作出以下决定：

1. 两个 R2J 候选均不进入 LOCO、新 SSCHA、matched DFT holdout 或正式声子谱；
2. 不继续扫描 epoch、学习率、普通 loss 权重或相同末块范围；
3. 下一步先做高误差原子局域描述符的可分性检验，判断 support 与 harmonic 环境能否由现有局域表示区分；
4. 若可分，冻结 depth-3 和长程项，训练独立的能量守恒 residual 分支，并用平滑局域路由使其在 harmonic 环境趋近于零；若不可分，则先增加局域表示的感受野或结构信息，不追加同类 DFT 标签。

机器可读记录保存在：

- [最后一块训练与 gate](../results/graphene_physics_temperature/post_p4_feasibility/R2J_common_descent/balanced_last_block_seed83/checkpoint_development_gate.json)
- [最后两块训练与 gate](../results/graphene_physics_temperature/post_p4_feasibility/R2J_common_descent/balanced_last_two_blocks_seed83/checkpoint_development_gate.json)
- [梯度冲突审计](../results/graphene_physics_temperature/post_p4_feasibility/R2I_gradient_conflict/audit.json)

### 08-25 03:58 更新：R2K 六组局域描述符 screen 均未放行保守路由

R2K 完成了六条零优化步、零新增 DFT 的描述符可分性 lane。共同设置为：冻结 depth-3 residual expert 和 q6/full-EPC；用 E50 fixed-smearing 的 seed 0+1（40 个构型）拟合非因果 utility probe，用 seed 2（20 个构型）做 descriptor screen；9 个 support 构型按完整构型做 leave-one-configuration-out（LOCO）；25 个 harmonic validation 构型按奇偶索引拆成阈值校准和外部检查。这里的路由输入只取冻结 MACE `node_feats` 中带符号的 `l=0` 标量通道；lane 名称中的 `l≤1/l≤2` 表示底层 descriptor 模型包含的等变通道，不表示路由已经显式使用了 `l>0` 信息。

固定门槛为 support LOCO target/repair coverage 和正 utility retention 均不低于 `80%`，tension weighted R² 不低于 `0.5`、符号准确率不低于 `80%`；seed2 的 C² 正 utility retention 和 remaining-repair coverage 均不低于 `80%`、harmful exposure 不高于 `20%`，force RMSE/max 不高于 `30/200 meV/Å`，A' 投影 RMS 不高于 `15 meV/Å`。六条 lane 的关键数值如下；百分数列均为平方误差或平方张力加权比例。

| 路由描述符 | `2 × r_max × interactions` / 6×6 最短周期长度 | support LOCO：target / 正 utility / C² remaining | tension R² / 符号准确率 | seed2 C²：正 utility / remaining / harmful | seed2 C² force RMSE / max | seed2 C² A' RMS | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| depth-3，96 个 `l=0` 标量 | `30.00 / 14.76 Å` | `100.0% / 96.5% / 98.9%` | `0.118 / 53.5%` | `100.0% / 100.0% / 100.0%` | `15.40 / 101.12` | `16.35` | 感受野越过周期边界；tension、harmful 和 A' 失败 |
| `r_max=2 Å, l≤1`，32 维 | `8.00 / 14.76 Å` | `8.3% / 7.4% / 2.5%` | `-0.862 / 51.9%` | `10.9% / 5.3% / 34.0%` | `168.80 / 1290.08` | `158.26` | 失败 |
| `r_max=2 Å, l≤2`，32 维 | `8.00 / 14.76 Å` | `15.7% / 12.7% / 3.7%` | `-1.473 / 45.8%` | `21.1% / 10.3% / 7.0%` | `158.43 / 1290.08` | `145.71` | 失败 |
| `r_max=3 Å, h16, l≤1`，32 维 | `12.00 / 14.76 Å` | `18.1% / 18.1% / 4.1%` | `-0.027 / 55.4%` | `8.1% / 2.2% / 4.8%` | `171.58 / 1290.02` | `155.49` | 失败 |
| `r_max=3 Å, h16, l≤2`，32 维 | `12.00 / 14.76 Å` | `11.0% / 12.2% / 2.6%` | `-0.379 / 61.5%` | `3.5% / 1.1% / 0.0%` | `176.51 / 1290.08` | `161.35` | 失败 |
| `r_max=3 Å, h32, l≤1`，64 维 | `12.00 / 14.76 Å` | `20.0% / 23.6% / 5.8%` | `-0.304 / 53.5%` | `14.2% / 5.2% / 0.0%` | `165.21 / 1290.08` | `158.78` | 五个安全表示中相对最好，仍远低于门槛 |

力的单位为 `meV/Å`。五个 `r_max=2/3 Å` 短感受野 lane 的 interaction-diameter 上界为 `8/12 Å`，低于 `14.76 Å`，因而通过保守的无周期绕回检查；depth-3 的上界为 `30 Å`，不能作为 6×6 超胞上的严格局域路由。对同一中心原子和三近邻扰动分别构造 6×6 与 8×8 超胞时，六条 lane 的 score 差为 `1.99×10^-6–1.63×10^-5`，最大 C² gate 差为 `7.08×10^-5`。这个显式例子说明测试扰动本身稳定，但不能替代 depth-3 已失败的全局感受野边界。

六条 lane 的 pooled harmonic-external C² proxy RMSE 为 `7.48–8.10 meV/Å`，低于 `9.045 meV/Å` 上限；但每条 lane 都至少有单个 harmonic 构型的激活率或 route-proxy RMSE 超限。更关键的是，所有安全表示在 support LOCO 和 seed2 上都只能捕获很小一部分需要修复的环境；增加 `r_max`、通道数或底层 `l=2` 后没有形成接近 `80%` 门槛的趋势。因此当前瓶颈不是继续增加相同 DFT 构型，也不能由 l=0 标量路由的阈值或宽度微调解决。

本轮还确认了数据边界：E50 fixed-smearing 的 `60/60` 个几何与历史 R2C 几何精确重合，seed 0、1、2 分别对应历史 train、validation、test 几何。seed2 没有参与单条 probe 的系数拟合，但已经用于比较六个描述符家族，因此现在只是已打开的 development screen，不能再称为 external 或 blind holdout。冻结 expert 和各 descriptor 也在此前训练中看过全部 9 个 support 构型；support LOCO 只移除线性 probe 对被留出构型的拟合，不能消除底层表示的训练历史。六条 lane 均未新增 DFT 标签。

R2K 的 utility 标签为 `u_i = ||y_i||² - ||y_i-p_i||²`，直接使用参考 residual `y`，本身不是部署时可获得的输入。表中的 hard/C² force-space 结果也没有包含节点能量路由在真实保守力中产生的 `-ε∇g` 项和跨原子贡献。因此 R2K 只回答“当前冻结表示是否含有足够的局部分离信息”，不构成保守模型的力学验收。六条 lane 的 `router_screening_checks_satisfied`、repair predictability 和 deployment authorization 均为 false；R2L conservative-autograd 训练与部署测试没有放行，有限温验证仍未关闭。

下一步继续使用现有标签，把路由表示改为显式 `l>0` power invariants。对每个 interaction 的 `l=1/2` 等变块构造旋转不变量，例如 `P_ab^(l)=Σ_m h_am^(l) h_bm^(l)`，保留通道自功率并加入受控的通道/interaction 交叉 Gram 项，再与现有 `l=0` 标量、短键几何量联合做相同的 support-LOCO、harmonic-external 和 seed2 screen。该步不新增 DFT、不修改 depth-3、q6 或 full-EPC，也不训练可部署的保守路由模型。只有显式 power-invariant lane 通过全部固定 screen，才进入 R2L 的节点能量 C² gate、完整 autograd 力和有限差分一致性验收；若仍失败，则先扩大局域结构表示，而不是用 seed2 继续选择阈值或追加同分布标签。

机器可读结果位于 `results/graphene_physics_temperature/post_p4_feasibility/R2K_descriptor_separability/` 下六个 `fixed_e50_*_seed83/descriptor_separability.json`；每条记录同时保存 LOCO CSV、逐构型路由指标、图和不可部署的 router state。

### 08-25 04:44 更新：tensor power 和 diagonal-quadratic screen 仍未通过

上一节固定的显式张量信息 screen 已完成。每个 `l>0` 通道先构造同通道旋转不变量 `mean_m(h_m²)`，并与原有带符号 `l=0` 标量拼接；这一步只加入各通道的自功率，还没有加入不同通道之间的 Gram 交叉项。三条主 lane 仍使用固定 `alpha=10` 的线性 ridge probe，没有新增 DFT、优化 MLIP、修改 q6/full-EPC 或使用 seed2 拟合。其关键结果为：

| 主 power lane | 路由维数 | support LOCO：target / 正 utility / C² remaining | tension R² / 符号准确率 | seed2 C²：正 utility / remaining / harmful | seed2 C² force RMSE / max | seed2 C² A' RMS |
|---|---:|---:|---:|---:|---:|---:|
| `r3-h16-l≤1 power` | 48 | `26.6% / 30.6% / 8.3%` | `0.253 / 52.1%` | `14.3% / 4.3% / 0.0%` | `165.95 / 1282.03` | `162.37` |
| `r3-h16-l≤2 power` | 64 | `15.5% / 22.3% / 5.0%` | `-0.997 / 46.9%` | `6.8% / 2.4% / 19.6%` | `173.47 / 1290.08` | `160.70` |
| `r3-h32-l≤1 power` | 96 | `21.2% / 23.8% / 6.6%` | `-0.083 / 41.1%` | `16.3% / 5.8% / 0.2%` | `163.28 / 1290.08` | `155.70` |

力的单位为 `meV/Å`。与只用 `l=0` 的对应 lane 相比，power 特征在若干 utility 指标上有小幅收益，例如 h16-l1 的 support 正 utility 从 `18.1%` 增至 `30.6%`，h32-l1 的 seed2 C² 正 utility 从 `14.2%` 增至 `16.3%`。但三条 lane 的 support C² remaining coverage 仍只有 `5.0–8.3%`，seed2 只有 `2.4–5.8%`，远低于固定的 `80%` 门槛；force、A'、tension 以及逐构型 harmonic 条件也都未通过。结果说明自功率包含了一部分此前丢失的信息，但不足以分离需要打开 expert 的局域环境。

h16-l2 的第一次特征筛选曾把绝对标准差直接和 float32 epsilon 尺度比较，从而把小幅值的 `l=2` power 通道误判为数值常量。这个判定混淆了特征的绝对幅度和相对变化，不能作为删除 `l=2` 的物理依据。修复后只在 `harmonic_train + fixed-smearing seed0+1` 的 12,096 个原子记录上拟合 `population_std/RMS`，固定阈值为 `64×eps_float32 = 7.6294×10^-6`；support、harmonic validation、seed2、legacy mixed-smearing 和 target/utility 都不参与筛选。实际 `l=2` 通道的绝对标准差为 `3.61×10^-6–3.49×10^-4`，但相对标准差为 `0.258–0.852`，因此修复后的 64 个输入特征保留 `64/64`。表中的 h16-l2 数值来自这次 relative-variance 修复后的正式结果。

随后运行了三条事先固定的 diagonal-quadratic（Q）对照。Q 先用 training-only scaler 得到 `z`，再把 `[z, z²]` 交给第二个 scaler 和同样固定 `alpha=10` 的 ridge；48/64/96 维输入分别映射到 96/128/192 维。该映射只加入逐通道平方，没有通道交叉项，也没有扫描阶数、正则或阈值。三条 Q lane 的范围为：

- support target coverage `18.8–27.0%`、正 utility retention `16.9–23.0%`、C² remaining coverage `6.4–9.2%`；
- tension weighted R² 从 `-1.072` 到 `-0.152`，符号准确率 `38.0–46.8%`；
- seed2 C² 正 utility retention `16.2–23.1%`、remaining coverage `6.5–8.8%`，harmful exposure `28.7–34.0%`；
- seed2 C² force RMSE `156.16–163.73 meV/Å`、最大分量均约 `1290.08 meV/Å`，A' RMS `145.43–159.74 meV/Å`。

Q 的 seed2 正 utility 比相应线性 power lane 更高，但 harmful exposure 同时超过 `20%`，support/seed2 remaining coverage 仍比 `80%` 门槛低一个数量级，力和 A' 误差也没有接近验收线。三条 Q lane 的 pooled harmonic-external C² RMSE 为 `7.43–7.81 meV/Å`，但逐构型 harmonic gate 仍失败。三条 Q 的 router screen、repair predictability 和 deployment authorization 全部为 false，R2L conservative-autograd 继续不放行。

本轮同时收紧了机器可读记录。Q 结果使用严格 JSON：非有限诊断量写为 `null`，序列化禁止 `NaN/Infinity`；完整记录 feature schema、training-only pruning state、feature mask、Q map schema、probe state、各阶段维数及其 SHA256。三个输出均通过严格 JSON 解析和非有限 token 检查。每个输出目录还保存实际执行脚本的 `run_script_snapshot.py`，JSON 中记录的 snapshot SHA256 与文件实测一致，三条均为 `d494887f749c154ad632cb5b4120805f2c713b64948c1a62fd595e0202d7f88d`。这使后续 replay 可以区分特征提取、筛选、非线性 map 和 ridge 四个步骤，避免只凭当前工作树推断历史运行代码。

下一步只保留一条事先固定的 `r3-h16-l≤1 Gram-sketch16` lane。它对 interaction 0 的 `16x1o` 通道加入 16 个固定的纯 off-diagonal Gram cross sketches，同时保留 32 个 `l=0` 标量和 16 个 `l=1` self-power，因此未剪枝输入固定为 64 维；router map 固定为 linear，使这次相对主 power lane 的唯一变量就是跨通道 Gram 信息。该 lane 仍沿用同一 E50 seed0+1 fit、seed2 opened-development screen、support LOCO、harmonic split、门槛和严格快照，不新增 DFT，不修改 expert 或长程项，也不扫描 sketch 维数、投影、ridge、gate 阈值或 descriptor 家族。只有这唯一一条 lane 全部通过，才进入 R2L；若失败，本轮局域路由方向即停止，不再用 seed2 追加映射或参数选择。

### 08-25 05:22 更新：固定 Gram-sketch16 仅有小幅增益，冻结描述符路线停止

唯一的 `r3-h16-l≤1 Gram-sketch16 + linear` 正式筛选已经完成。64 维输入由 32 个带符号的 `l=0` 标量、16 个原有 `l=1` 自功率和 16 个新增的纯 off-diagonal 通道耦合量组成；新增量是 120 个 `a<b` Gram 元素的固定 16 维投影。投影不依赖标签或随机数，`R` 和 `Phi` 的 SHA256 分别为 `c2a4ac2867047609ca539a384556bd5ec1b204a9c65c736758b28b6deef7a044` 和 `929a48555356ff84cf5246adc1facd5f55e75f81e4c1f5bbefc0572ab210717b`，秩均为 16。执行脚本 SHA256 为 `32e99ee5e4fb8329b686d072d44eca50551524696eba0ee342d11fd507d1dd86`。

| 指标 | Gram-sketch16 结果 | 固定门槛 |
|---|---:|---:|
| support LOCO target tension coverage | `27.90%` | `≥80%` |
| support LOCO 正 utility retention | `30.88%` | `≥80%` |
| support LOCO remaining-repair coverage | `16.49%` | `≥80%` |
| support LOCO C² 正 utility / remaining | `25.55% / 8.27%` | 各 `≥50%`（逐构型也需通过） |
| tension weighted R² / 符号准确率 | `0.195 / 53.47%` | `≥0.5 / 80%` |
| harmonic activation / C² gate mean | `4.88% / 2.95%` | 各 `≤5%` |
| harmonic hard / C² force proxy RMSE | `8.17 / 7.75 meV/Å` | `≤9.045 meV/Å` |
| seed2 hard / C² 正 utility retention | `24.04% / 14.76%` | 各 `≥80%` |
| seed2 target / remaining tension coverage | `11.39% / 11.21%` | 各 `≥80%` |
| seed2 hard force RMSE / max | `157.63 / 1290.08 meV/Å` | `≤30 / 200 meV/Å` |
| seed2 C² force RMSE / max | `165.39 / 1290.08 meV/Å` | `≤30 / 200 meV/Å` |
| seed2 A' hard / C² RMS | `151.77 / 160.40 meV/Å` | `≤15 meV/Å` |

36 项固定筛选只通过 11 项，`screening_proxy_checks_satisfied=false`、`deployment_authorized=false`。与 48 维 self-power 基线相比，support target coverage 从 `26.6%` 增至 `27.9%`，正 utility retention 从 `30.6%` 增至 `30.9%`，没有形成接近门槛的变化；seed2 的正 utility 和力误差也没有改善。因此这条固定的 16/120 Gram sketch 不足以解决局部路由问题，不能放行 R2L conservative-autograd。该结果只排除了本次固定低维投影，不能外推为完整 off-diagonal Gram 空间没有信息。

同一模型和数据在 V100-A、V100-B、RTX 2060 上独立复算，support coverage 完全相同，seed2 hard force RMSE 的最大差小于 `2×10^-4 meV/Å`，A' hard RMS 的最大差小于 `3×10^-2 meV/Å`，三台机器的 36 项布尔判定一致。完整 JSON、CSV、PNG/PDF、router state 和实际执行脚本快照保存在 `results/graphene_physics_temperature/post_p4_feasibility/R2K_descriptor_separability/fixed_e50_r3_h16_l1_power_gramsketch16_linear_seed83_{v100a,v100b,rtx}/`。

下一阶段不再在这些冻结描述符上增加映射、阈值或 sketch 维数。先用现有 fixed-smearing 标签训练一个严格短程、无周期绕回的专门局域编码器；训练与验证按完整构型分组，底层表示也不能看到 outer-LOCO 留出的 support 构型。第一轮不新增 DFT，current S0 和 q6/full-EPC 继续冻结。只有新表示通过 support、harmonic 和 fixed-smearing 的分组筛选后，才实现并训练完整节点能量的 C² 保守路由。

### 08-25 06:02 更新：support-free replacement core 已在三台 GPU 启动

R2K 失败后，下一阶段先训练一条不含 support 的 replacement core。它不再叠加或初始化自 depth-3，而是直接替换现有 short-delta 分支；组合形式为 frozen v11 foundation + 新 core + frozen q6，full-EPC 的谱接口仍保持不变。第一轮只训练保守 MACE 的力，不使用能量标签，避免沿用 R2C 中受 support9 参与定零影响的能量规范。

训练集共有 164 个完整构型：20 个 E50 fixed-smearing seed0 精确标签、72 个 harmonic-train，以及去重后的 36 个 T300 和 36 个 T600 matched-operator auxiliary 构型。旧 T450 的 20 个标签与 seed0 几何完全重合，因此由 E50 精确标签替换，不作为第二份样本。E50 seed1 的 20 个构型和 25 个 harmonic-validation 构型只用于训练完成后的固定 checkpoint gate；seed2 和 support9 不参与梯度、尺度、学习率或 checkpoint 选择。92 个 thermal 训练几何与 seed1/seed2 的几何重合均为 0；旧 T450 与 E50 seed0 短程标签的差为 `0.599 meV/Å` RMSE、`1.643 meV/Å` 最大分量。

三组训练目标的 RMS 分别为 `204.231`、`7.369` 和 `197.126 meV/Å`。为避免 thermal 大目标淹没 harmonic，按训练集自身尺度构造 group-normalized MSE，使 E50 exact、harmonic、auxiliary 的总初始损失质量固定为 `0.4/0.4/0.2`；MACE batch=1 下对应的 config weights 为 `0.004687325/1.0/0.000698789`。seed1、seed2 和 support 不进入这些尺度或权重的计算。

模型是随机初始化的 MACE：`r_max=2 Å`、2 个 interactions、`16x0e+16x1o+16x2e`、`l_max=2`、24 个 radial basis、5 个 cutoff basis、correlation 3。新 core 的力依赖直径上界为 `8 Å`，小于 6×6 周期长度 `14.76 Å`；这一严格无绕回结论只适用于新 core，完整 `v11+core` 仍需单独做 6×6/8×8 数值一致性。训练固定为 240 epochs、batch 1、seed 83、学习率 `10^-3`、EMA 0.99，并显式使用 `ExponentialLR(gamma=1.0)`，使 validation 不改变优化轨迹，也不提前停止。

固定候选为 epoch `40/80/120/160/200/235` 和 final。进入 routed-tail 阶段必须同时满足：E50 seed1 force RMSE/max `≤30/200 meV/Å`、A' RMS `≤15 meV/Å`、恢复力斜率相对误差 `≤5%`；harmonic-validation RMSE/max `≤9.0447/200 meV/Å`。未通过时只保存 diagnostic model，不能读取 seed2 继续调参。即使 core 通过，也只放行 support outer-LOCO 的 C² 保守 tail 开发，不代表复合模型、SSCHA 或有限温声子谱已经完成。

V100-A 为事先指定的主运行，V100-B 和 RTX 2060 使用完全相同的 seed、数据和架构做稳定性复算，不用于挑选最优模型。三台任务于约 `06:02 NZST` 启动，冻结哈希为：数据 manifest `bb20a86a…`、prep `4172685e…`、launcher `8382faf1…`、evaluator `8d7d8438…`。三台均已进入实际训练，初始 validation force RMSE 为 `157.64 meV/Å`，epoch 0 降至 `101.77 meV/Å`。按既有同规模 MACE 吞吐，预计主训练约 `0.8–1.2 h`，随后固定 checkpoint gate 约 `0.1–0.2 h`。

### 08-25 06:42 更新：第一轮 core 退化到近零修正，已启动固定的 gate-normalized loss 修复

RTX 2060 上的第一条 R2M replica 已完成 240 epochs 和全部固定 checkpoint gate，进程 `EXIT_CODE=0`，但 `R2M_core_checkpoint_gate_failed`。epoch 40 到 final 的 E50 指标几乎不变；自动选择的 best diagnostic 是 final，它仍不能用于 routed-tail：

| 固定候选 | E50 force RMSE / max (meV/Å) | E50 A' RMS (meV/Å) | A' slope 相对误差 | harmonic RMSE / max (meV/Å) | 最坏门槛比 |
|---|---:|---:|---:|---:|---:|
| epoch 40 | `182.136 / 965.538` | `227.355` | `6.040%` | `7.235 / 56.901` | `15.157` |
| epoch 80 | `182.177 / 965.706` | `227.226` | `6.031%` | `7.223 / 57.213` | `15.148` |
| epoch 120 | `182.205 / 965.857` | `227.196` | `6.030%` | `7.222 / 57.227` | `15.146` |
| epoch 160 | `182.199 / 965.756` | `227.240` | `6.031%` | `7.219 / 57.260` | `15.149` |
| epoch 200 | `182.160 / 965.652` | `227.180` | `6.029%` | `7.223 / 57.269` | `15.145` |
| epoch 235 / final | `182.181 / 965.811` | `227.149` | `6.028%` | `7.220 / 57.253` | `15.143` |

零 core 在同一 validation 上的 E50 RMSE/max/A' RMS 为 `182.015/965.556/227.055 meV/Å`，harmonic 为 `7.236/57.082 meV/Å`，混合 RMSE 为 `101.576 meV/Å`。训练从 epoch 5 到 235 的混合 RMSE 一直在 `101.54–101.70 meV/Å`，因此失败不是 epoch 不够，而是模型被当前多目标损失稳定在近零修正。

对 epoch 40 做的分组梯度审计进一步定位了原因。未加权 E50 与 harmonic 梯度余弦为 `-0.7763`，E50 与 auxiliary 为 `0.9976`，harmonic 与 auxiliary 为 `-0.7727`。原来的 `1/RMS²` 权重使三组加权梯度范数约为 `0.2256/0.3275/0.0993`，方向相反的 E50 和 harmonic 在零修正附近明显抵消。这个结果排除了继续增加 epoch、改变 q6 或追加同类 DFT 标签；下一步只修改 loss 的固定相对权重。

R2N 使用同一训练/验证数据、同一随机初始化、架构、seed、240 epochs、constant scheduler 和固定 checkpoint gate，只把配置 MSE 改为按物理验收尺度归一化：

- E50 fixed-smearing：`0.3272244428`；
- harmonic：`1.0`；
- T300/T600 auxiliary：各 `0.04544783928`。

这些数值由 `lambda_g/(N_g L_g²)` 得到，其中 E50/aux 的 `L=30 meV/Å`，harmonic 的 `L=9.044673 meV/Å`；没有根据 seed1 扫描权重，也不使用动态 GradNorm。epoch-40 处按新权重估算的 E50/harmonic/aux 梯度范数为 `15.751/0.328/6.458`，合向量 cancellation ratio 为 `0.974`，能够离开本轮的零固定点。

R2N launcher SHA256 为 `978c013bd8938b7d246bb2135add1273c3f582c9b14c3d61b33d9230619cec2c`，继续固定 MACE `0.3.16` 的 config-weight、force MSE 和 reduce 语义，并拒绝读取 seed2/support 文件。它只在前序目录同时具备 `DONE`、`CORE_GATE_FAILED`、`EXIT_CODE=0`、失败 gate JSON 和对应 diagnostic model hash 时启动，也不使用 diagnostic model 初始化。RTX 的前序条件于约 `06:39 NZST` 满足，R2N 于约 `06:41 NZST` 启动；V100-A/B 的 R2M replica 继续运行，用于核对相同失败结论。R2N 通过前，routed-tail、SSCHA 和有限温声子谱仍不放行。

### 08-25 07:20 更新：gate-normalized core 仍存在 Pareto 冲突，转向低阶/高阶物理分解

RTX 上的 R2N 已完成全部固定候选，进程和 gate 标记一致、`EXIT_CODE=0`，但仍为 `R2M_core_checkpoint_gate_failed`。这条实验只改变了配置损失权重，因而能够直接检验上一节定位的梯度抵消是否是唯一问题。结果为：

| 固定候选 | E50 force RMSE / max (meV/Å) | E50 A' RMS (meV/Å) | A' slope 相对误差 | harmonic RMSE / max (meV/Å) | 最坏门槛比 |
|---|---:|---:|---:|---:|---:|
| epoch 40 | `145.885 / 796.542` | `193.459` | `4.856%` | `24.051 / 188.917` | `12.897` |
| epoch 80 | `142.723 / 791.406` | `187.977` | `4.793%` | `24.460 / 198.443` | `12.532` |
| epoch 120 | `140.239 / 777.798` | `185.141` | `4.789%` | `25.320 / 211.332` | `12.343` |
| epoch 160 | `138.352 / 779.623` | `182.284` | `4.681%` | `25.860 / 231.065` | `12.152` |
| epoch 200 | `132.348 / 791.705` | `179.009` | `4.198%` | `28.098 / 228.813` | `11.934` |
| epoch 235 / final | `125.631 / 754.812` | `166.100` | `3.421%` | `27.521 / 211.578` | `11.073` |

新 loss 明确打破了零修正：E50 RMSE 从约 `182` 降至 `125.6 meV/Å`，有限振幅 A' slope gate 也从失败变为通过。但每个固定 checkpoint 都在改善 E50 的同时破坏 harmonic，且 E50 force、A' 投影和 harmonic 三项仍远离联合门槛。因此不能再通过继续 epoch、在 seed1 上扫描权重或增加同类 DFT 标签解决；R2N 的 best diagnostic 也不授权 routed-tail、SSCHA 或部署。机器可读 gate SHA256 为 `b2e00ba1efff0dee62167a76d98dac87798580c7e15c49ec47bf12ffab23487e`，best diagnostic model SHA256 为 `d20b8281e18db3a196a4dcf2f5f2c185f02224d4f055d17d58655da63c1685e5`。

为判断普通局部 router 是否能消除冲突，随后只用 train 和 seed1/harmonic validation 做了不读取 seed2/support 的局部可分性审计。特征只含 O(3)/置换不变的径向、Gram、键伸缩、键角和非共面局部几何，不含原子数、cell、温度、degauss 或数据来源。`r<2 Å` 时，在 harmonic 假阳性约 `3.8%` 下，validation thermal 原子真阳性只有 `36.3%`；`r<3 Å` 时改善到 `61.25%`，harmonic 假阳性 `4.13%`、高振幅 harmonic 假阳性 `7.93%`，仍低于事先固定的 `80%` thermal coverage。按构型聚合可以完全区分，但这种全局统计会学习超胞/系综分布，不能用于可迁移的局部势。相同局部扰动放入 6×6/8×8 的特征差在 `10^-14` 量级，说明上述失败不是周期尺寸数值误差。

因此下一条不运行两个普通专家、不把 temperature/source 当输入，也不只把 `r_max` 从 2 改到 3。固定思路改为物理低阶/高阶分解：保留已经通过 harmonic force gate 的 frozen foundation+q6 低阶基座，新增一个保守的 anharmonic residual，并在结构上令其在平衡参考处的能量、力和 Hessian 为零。这样 harmonic Hessian 不再与有限振幅 thermal residual 争夺同一组参数；E50 seed0 和 T300/T600 auxiliary 只训练高阶残差，seed1/harmonic 仍只用于固定终点评价。具体 Taylor-null 参数化、reference/order mapping、6×6/8×8 一致性和 autograd 成本必须先通过实现审计，之后才允许占用 RTX/B 启动；当前 support9 outer-LOCO 继续保持 blocked。

### 08-25 08:59 更新：R2N 双节点正式失败，R2O Taylor-null 双节点 smoke 通过

V100-A 已完成 R2N 的独立正式复算。RTX 与 V100-A 在相同 seed、数据、架构、240 epochs 和固定候选下得到一致的失败结论；硬件浮点舍入使最终文件哈希不同，但门控数值在报告精度内相同：

| R2N 节点 / 固定候选 | E50 force RMSE / max (meV/Å) | E50 A' RMS (meV/Å) | A' slope 相对误差 | harmonic RMSE / max (meV/Å) | 最坏门槛比 |
|---|---:|---:|---:|---:|---:|
| RTX，epoch 235 | `125.6313 / 754.8124` | `166.1002` | `3.4214%` | `27.5210 / 211.5783` | `11.07335` |
| V100-A，final | `125.6313 / 754.8124` | `166.1002` | `3.4214%` | `27.5209 / 211.5783` | `11.07335` |
| 固定门槛 | `30 / 200` | `15` | `5%` | `9.0447 / 200` | `1` |

两条正式任务均为 `EXIT_CODE=0` 加失败 marker；RTX/V100-A gate JSON 的 SHA256 分别为 `b2e00ba1…` 和 `404f018b…`。R2N 只通过了 A' restoring slope，E50 总力、A' 投影和 harmonic preservation 没有联合通过。seed2 和 support9 未用于训练、尺度或 checkpoint 选择；两个 diagnostic model 均不授权 routed-tail、SSCHA 或部署。这把结论限定为：在当前 support-free replacement core 与固定 loss 下，单一参数化仍存在明显的低阶/有限振幅 Pareto 冲突；它不等价于完整方法已经失败。

R2O 随后改用 whole-energy live Cartesian Taylor-2 余项：

\[
R_\theta(x;x_0)=E_\theta(x)-E_\theta(x_0)-g_0\!\cdot u-
\frac12u^\mathsf{T}H_0u.
\]

其中参考构型 `x0` 独立构图，周期最小像整数映射在 autograd 外固定，位移 `u` 和 HVP 的 `grad_outputs=u` 保持可微。Stage 2 训练完整 encoder 与 readout，不是冻结 feature jet，也不是节点路由器。该结构按构造令参考点的能量、力和 Hessian 为零，同时保留同一局域 MACE 全能量的 Cartesian 三阶及更高阶响应。

冻结模型为 FP64 `r_max=3.2 Å / 2 interactions / 16x0e+16x1o+16x2e / lmax=2 / 24 radial basis / p=5 C² cutoff / correlation=3`，共 `42,096` 个参数。Stage 1 使用 E50 seed0 20、T300 36、T600 36，共 92 个 thermal 构型；Stage 2 另加按 `bond-length RMS≤0.003 Å` 唯一规则选出的 32 个 small-harmonic zero-tail 构型。E50 seed1 20 和 small-harmonic 12 只用于固定门控；seed2、support9 和全 25 个大振幅 harmonic 均不参与 checkpoint 选择。T600 auxiliary 对应 `degauss=0.003800173876 Ry`，是 matched-smearing 的 local-Mermin transferable delta，不能表述为与 E50 相同 smearing 的 DFT 标签。

RTX 2060 与 V100-A 的独立 `2+2 epoch` FP64 smoke 均得到 `DONE + SMOKE_PASS + EXIT_CODE=0`。两个节点的 Stage 2 四组归一化训练 MSE 逐位接近：epoch 1 为 `16.5897/9.37687/24.5651/20.4613`，epoch 2 为 `13.8981/8.24480/19.7911/21.8922`，顺序为 E50/T300/T600/small-harmonic。实测性能为：

| 节点 | Stage 1：184 updates | Stage 2：2 epochs | Stage 2 单 epoch | 峰值显存 | 端到端 smoke |
|---|---:|---:|---:|---:|---:|
| RTX 2060 | `35 s` | `67.29 s` | `35.97 / 31.07 s` | `657 MiB`（Stage 1），`485.92 MiB`（Stage 2） | `172 s` |
| V100-A | `75 s` | `112.17 s` | `61.97 / 49.57 s` | `474 MiB`（Stage 1），`485.92 MiB`（Stage 2） | `324 s` |

两个节点都通过了 ScaleShiftMACE 原始能量/位置梯度语义、非平凡 scale/shift、O(3)、平移、置换与跨周期像、能量—力和力—Hessian 有限差分、完整 `216×216` Hessian 对称性/ASR、6×6/8×8 局域一致性及 C² cutoff 检查。较大的实现残差仍低于固定阈值：O(3) 最大力差 `7.32×10^-7 eV/Å`，置换/像最大力差 `3.07×10^-7 eV/Å`，力—Hessian有限差分误差 `2.09×10^-6 eV/Å²`，6×6/8×8 中心力差 `5.58×10^-8 eV/Å`。参考点的最大力为 `3.44×10^-16 eV/Å`，Hessian 最大元 `1.07×10^-14 eV/Å²`，ASR row-sum `4.96×10^-15 eV/Å²`，Γ/K 频率漂移上界约 `1.82×10^-5 cm^-1`。

RTX 和 V100-A 的 smoke gate SHA256 分别为 `87b8d2ae…` 和 `411d6e00…`。两台机器的训练轨迹和门控数值一致，但模型 state hash 因硬件舍入分别为 `2dcb3867…` 与 `eac7ee8…`；本实验不把跨硬件逐字节相同作为要求。smoke diagnostic bundle 明确标记为不可部署。

smoke 只验证实现、数据隔离和实际吞吐，不是物理 gate。2 epoch checkpoint 的 E50 RMSE/max、A' RMS、slope error 和 small-harmonic RMSE/max 分别为 `114.071/525.240 meV/Å`、`207.705 meV/Å`、`5.171%` 和 `2.768/78.815 meV/Å`，没有通过正式门槛；这组数值不用于提前选择或否定 240-epoch 结果。

独立 smoke 复核已解除正式启动阻断，`80+240 epoch` 双节点正式训练已经启动，两边 preflight 均为 PASS：

| 节点 | 开始时间 | PID | OUT | 冻结 smoke gate |
|---|---|---:|---|---|
| RTX 2060 | `08:58:31 NZST` | `862210` | `/home/howardwang/phonon/results/graphene_physics_temperature/post_p4_feasibility/R2O_taylor_null_core/formal_r3p2_h16_l2_n24_seed83_rtx` | `87b8d2ae…` |
| V100-A | `09:00:17 NZST` | `51995` | `/data/graphene_r2o_core/formal_r3p2_h16_l2_n24_seed83_v100a` | `411d6e00…` |

V100-A 的 formal 前序来自 RTX 已完成的 R2N 失败目录，经 Tailscale 复制，共 72 个文件、约 71 MB，目录摘要为 `64321d8a…`；冻结 gate/model SHA256 为 `b2e00ba1…/d20b8281…`。R2N diagnostic model 只作为失败前序 provenance，不用于 R2O 初始化。RTX Stage 1 当前实测约 `7.3 s/epoch`；在得到 Stage 2 首个正式 epoch 前，端到端墙钟预算保守记为 RTX `2.5–3.5 h`、V100-A `3.5–5 h`，预计分别在约 `11:30–12:30` 和 `12:30–14:00 NZST` 得到完整 gate，实际时间以 Stage 2 首 epoch 更新。

正式 R2O 只有同时通过 E50 seed1 `30/200/A'15/slope5%`、small-harmonic `0.5/10 meV/Å`、reference-null 与 mechanics gate，才允许读取 support9 开始 outer-LOCO。若未通过，按固定停止规则分析失败项，不增加同类 DFT 或扫描 seed1 权重。有限温 SSCHA、matched DFT holdout 和 MLIP+full-EPC 正式声子谱仍未放行。

### 08-25 11:50 更新：R2O 实现门通过，但固定物理门未通过

RTX 的 R2O 正式任务已正常完成，终态为 `DONE + PRETRAIN_DONE + TRAINING_DONE + CORE_GATE_FAILED + EXIT_CODE=0`。没有运行错误；失败来自事先确定的物理精度门槛。gate JSON 的 SHA256 为 `924b13de54d2a918c16845524289036e117006e48ee454f364f442fd88ee8898`，epoch 240 diagnostic checkpoint 的 SHA256 为 `5cdc6b8e2c9719a715a66b8ab8f82c1bdede1a02982e24f3e629d175c48652b6`。该模型仍标记为不可部署。

7 个固定 EMA checkpoint 均未联合通过。最坏门槛比从 epoch 40 的 `4.499` 下降到 epoch 240 的 `1.674`，但最后 5 个 epoch 已接近平台。最优 epoch 240 的结果为：

| 固定检查项 | 结果 | 门槛 | 判断 |
|---|---:|---:|---|
| E50 seed1 force RMSE | `17.855 meV/Å` | `30` | 通过 |
| E50 seed1 force max | `92.465 meV/Å` | `200` | 通过 |
| A' restoring-slope 相对误差 | `0.670%` | `5%` | 通过 |
| A' projected RMS | `25.106 meV/Å` | `15` | 未通过 |
| small-harmonic RMSE | `0.4931 meV/Å` | `0.5` | 通过 |
| small-harmonic max | `16.647 meV/Å` | `10` | 未通过 |

reference-null、cutoff C²、原始能量/梯度一致性、O(3)、置换、有限差分、一般构型 Hessian/ASR 和 6×6/8×8 局域一致性全部通过。参考点力和 Hessian 残差分别在 `10^-16 eV/Å` 和 `10^-14 eV/Å²` 量级。因此本轮排除了 Taylor-null 实现、原子映射和构图量化是主要误差来源；不能据此进入 routed-tail、support9 outer-LOCO、SSCHA 或正式声子谱。

误差归因表明训练目标与验收量的几何不一致。epoch 240 EMA 在 E50 seed0 上的 A' RMS 已有 `20.140 meV/Å`，说明问题在训练域内已经存在；A' coherent direction 只占 E50 总力 SSE 的约 `0.915%`，普通逐分量 MSE 对它的权重很小。small-harmonic 最大误差来自 harmonic index 11 的 atom 86-y：tail 预测 `-10.609`、参考 residual `+6.038 meV/Å`，合成误差为 `-16.647 meV/Å`。相同的 `-10.609` 模式已出现在 small-zero 训练帧中，说明它同样是 max 指标被均方损失稀释，不是新构型外推失败。

纯 A' collective-coordinate 补丁不足以解释误差。固定的 `B2=[conj(Q)^2, |Q|²Q]` 线性投影只把 seed0 A' RMS 从 `20.140` 降到 `18.308 meV/Å`，应用到 seed1 时只从 `25.106` 降到 `24.766 meV/Å`。因此下一步不延长 R2O、不扫描权重、不增加同类 DFT，而是先做 R2P 零优化步梯度可行性审计。

R2P 的 primary hard set 固定为 176 个逐构型目标：20 个 E50 A' complex projector、92 个 thermal total-force、32 个 small-zero RMS 和 32 个 exact top-component correction。top-component 的保守余量为 `10-6.41582=3.58418 meV/Å`；shifted smooth-L∞ 会低估 128 原子构型的真实最大值，因而只作报告。R2P 在 EMA240 上求单位梯度的 max-min common-descent cone，要求 `t_primal>10^-3`，并同时通过 dual gap、KKT、simplex、非零梯度、top 唯一性和参数 state 前后哈希不变检查。它不执行 optimizer step，也不读取 seed2/support。只有逐构型 cone 通过后，才另行冻结一次短程 gate-aligned 训练合同；否则停止当前局域 Taylor-null 表示，转向显式 thermal-background-conditioned 表示。

### 08-25 13:31 更新：R2P 逐构型共同下降证书通过，冻结 R2Q 四步有限修复

R2P 已在 RTX 2060 上完成，终态为 `DONE + EXIT_CODE=0`。primary certificate SHA256 为 `54bf9056d8ac90641875f2dbe997336ffe909c3babf9a17440d6c2237791842c`。176 个 FP64 逐构型梯度组成 `176×42096` 矩阵；单位梯度凸包的最小范数方向给出：

- `t_primal=0.00991895648`；
- `t_dual=0.00992077217`；
- duality gap `1.816e-6`；
- active KKT slack `2.086e-8`。

四个 objective family 的最小方向余弦都约为 `0.009919`，高于固定 `10^-3` 门槛。参数 state 在计算前后保持 `09c29479…d236`，57 项 formal prediction parity 全部通过，最大差为 `4.73e-13 meV/Å`；seed2/support 未打开，也没有 optimizer 或参数更新。V100-A/B 随后对同一证书作硬件复算，`t_primal` 与 RTX 的差不超过 `1.3e-14`，gradient matrix、Gram 和 direction 的相对差为 `10^-13–10^-14`，32 个 exact-top 的 top/second index 完全相同。

这个结果只说明 EMA240 附近存在同时降低 176 个训练目标的一阶方向，不表示 held A' 已改善，也不授权部署。独立复算脚本 `audit_graphene_r2p_artifacts.py` 直接从落盘矩阵、方向和 receipt 重建了 primal/dual/KKT 结果，状态为 `PASS`。

下一步合同已经冻结为 R2Q 四步 trust-region 修复，合同 SHA256 为 `8b287433b1046d4c229d1d3bb12aeba6bb9d5b0180af2ebf7605802935f614f0`。选择四步而不是单步的定量依据是：seed0 A' objective 需要下降约 `44.5%` 才到 `15 meV/Å` 门槛，而单个最大步的一阶预期约为 `22%`。R2Q 事先固定以下规则：

1. 恰好完成 4 个 train-only accepted steps；每步在当前 state 重算同一 176-block cone；
2. 每步初始 global relative L2 update 为 `10^-3`，per-tensor guard 为 `2%`，只允许 `k=0..5` 的 6 个二分步长；
3. 选择最大的、同时通过全部 176 个 exact-objective Armijo 条件的步长；每个 accepted state 继续通过 reference E/F/full-H/Hessian symmetry/ASR/Γ–K null；
4. 第 4 步完整 checkpoint、state hash 和四级 receipt 原子冻结前，不打开 seed1 或 actual small-H；
5. 冻结后只对唯一 endpoint 做一次原 R2O held 与 mechanics gate。held 不能回调方向、步长、步数或 checkpoint；任何 gate 失败都停止当前表示，不追加第 5 步或扫描权重。

R2Q 预计消耗 RTX `0.25–0.50 GPU·h`。当前正在实现和复核 evaluator，尚未启动正式四步计算。R2P 通过没有解除 routed-tail、support9 outer-LOCO、SSCHA、matched DFT 或正式声子谱的阻断；只有 R2Q 唯一 endpoint 重新通过原 R2O 全部固定门槛后，才能进入下一阶段。

### 08-25 14:48 更新：R2Q 四步全部接受，但 held 改善不足，loss-only 路线停止

R2Q 在 RTX 上按冻结合同正常完成，墙钟 `1002.69 s`，峰值显存 `663.32 MiB`。四步共同下降锥和有限步均通过，选择的 `(k,eta)` 依次为 `(4,0.0135343)`、`(3,0.0270687)`、`(2,0.0541375)`、`(1,0.1082753)`；`t_primal` 从 `0.009919` 降到 `0.005922`，始终高于 `10^-3`。全部 176-block Armijo、trust replay、reference E/F/H、Hessian symmetry/ASR、Γ/K null 和 seed0-only mechanics 通过。endpoint 在读取 held 前冻结，checkpoint/receipt SHA256 分别为 `31dd053a…3eb5` 和 `810a45ab…43a1`。

唯一 held 结果为 `R2Q_FOUR_STEP_HELD_GATE_FAILED`，不是数值故障：

| 指标 | R2O 起点 | R2Q endpoint | 门槛 | 结果 |
|---|---:|---:|---:|---|
| seed1 total-force RMSE | `17.855` | `17.718 meV/Å` | `30` | 通过 |
| seed1 total-force max | `92.465` | `92.985 meV/Å` | `200` | 通过 |
| seed1 A' projected RMS | `25.106` | `23.977 meV/Å` | `15` | 未通过 |
| A' slope 相对误差 | `0.670%` | `0.640%` | `5%` | 通过 |
| small-H RMSE | `0.4931` | `0.4810 meV/Å` | `0.5` | 通过 |
| small-H max | `16.647` | `15.816 meV/Å` | `10` | 未通过 |

endpoint 净参数位移为 `9.09e-4`。训练侧 seed0 A' RMS 只从 `20.140` 降到 `19.119 meV/Å`，small-zero tail max 从 `10.609` 降到 `9.778 meV/Å`；held 的改善比例与训练侧相近。因此结论不是 seed1 过拟合，而是当前普通局域表示中 small-zero 约束限制了可用方向，四步共同下降只完成 A' 和 small-max 所需改善的约 `11%` 和 `12.5%`。按事先确定的停止规则，不增加 step 5、不扩大步长网格、不重读 held，也不在 A/B 重复这条确定性失败路径。

下一条固定为 R2R 多极化热背景条件化 Taylor-null 表示。它以参考邻域内位移协方差的第二不变量区分多模 thermal background 和 rank-1 harmonic 位移，不使用 `config_type`、晶格温度或 `degauss` 元数据。只读初查中，92 个 thermal train 的逐原子背景量最小为 `1.8339e-6 Å^4`，而 32 个 harmonic train 和 12 个已开放 small-H 为 0。R2R 先完成零参数的 O(3)/order-MIC/locality/rank/null 和设计矩阵秩审计；通过后才拟合冻结 encoder 上的 65 参数线性 readout。seed2/support 继续保持未打开，新 DFT 为 0。

以下各节保留 08-22 启动时的阶段规划和验收定义；当前执行结论以文首的逐时更新为准。

## 1. R0/R1 启动时的决定（已执行）

现在不应根据 E51 中约 `40 cm^-1` 的 K 点差异直接重调长程项，也不应立即重新训练一个更大的短程模型。第一步应是当前 S0 自洽分布上的小批量 DFT overlap 检验。

原因是 E51 的两条 450 K 曲线来自不同系综：

- MLIP 曲线来自当前 S0 在 `T_lat=450 K`、固定 `degauss=0.0019000869 Ry` 下的自洽 SSCHA；
- DFT-TDEP 曲线使用旧 P4 失败轨迹上的 60 个构型；
- 旧 P4 三个 seed 的 DFT-TDEP K 点频率跨度为 `23.465 cm^-1`，leave-one-seed-out 跨度仍为 `10.555 cm^-1`；
- 因而 `40 cm^-1` 主要说明两个系综不能直接作为热力学对照，不能据此判定长程项应加深或抬高。

当前最合理的顺序是：

1. 冻结现有 full-EPC 长程响应和当前 S0，完成同口径力与能量审计；
2. 从已经存在的 450 K 固定-smearing S0-SSCHA 系综中选择 12 个构型；
3. 用两台 V100 计算这 12 个构型的 DFT 力和能量；
4. 根据 overlap gate 决定“不重训”“只修训练目标”或“修改短程模型结构”；
5. 只有 matched-ensemble 的 K-referenced cusp 形状仍失败时，才重新检查长程响应的形状或矩阵接口。

## 2. 现有证据

### 2.1 旧 P4 的误差不能代表当前 S0

旧 P4 frozen predictor 在 60 个构型上的总力 RMSE 约为 `53.1 meV/Å`，最大分量约为 `744 meV/Å`。当前 S0 是随后重新训练的统一短程模型，不能把旧数值沿用为当前模型结论。

当前 S0 的已有验收结果为：

| 数据 | force RMSE (meV/Å) | 最大分量 (meV/Å) |
|---|---:|---:|
| 300 K development test | 26.295 | 203.582 |
| 450 K development test | 22.566 | 174.299 |
| 600 K development test | 27.343 | 124.152 |
| harmonic replay | 13.932 | 140.014 |

2026-08-22 的补充只读核对在旧 P4 的同一批 60 个几何上使用了当前 S0，并把电子算符固定为 `degauss=0.0019000869 Ry` 对应的 T300 q6 算符。相对于旧 smearing 的 DFT target，结果为：

| 范围 | force RMSE (meV/Å) | MAE (meV/Å) | 最大分量 (meV/Å) |
|---|---:|---:|---:|
| all 60 | 23.331 | 17.021 | 245.502 |
| seed 0 | 25.124 | 17.850 | 245.502 |
| seed 1 | 22.215 | 16.730 | 160.254 |
| seed 2 | 22.545 | 16.484 | 173.590 |

同一几何上，新旧 smearing 的 DFT 力差只有：

- RMSE `0.834 meV/Å`；
- MAE `0.596 meV/Å`；
- 最大分量 `2.293 meV/Å`。

因此当前 S0 对新固定-smearing 标签的精确 all-60 RMSE 必然落在约 `22.50–24.17 meV/Å` 之间。R0 仍要生成逐分量精确结果，但这个范围已经说明：现在没有依据把主要问题归因为长程项幅度，也没有依据先推倒当前 S0。

### 2.2 长程项当前应冻结

当前 full-EPC 长程响应已经满足以下数值检查：

- E48 中 q6 算符移除再装配的最大误差为 `5.6×10^-17 eV/Å²` 量级；
- full-EPC 响应相对更密 `nk=720` 的频率最大差在两个晶格温度上不超过 `0.55 cm^-1`；
- time-reversal、K-star、Hermiticity 和模式跟踪检查通过；
- 静态同晶格背景下，增大 smearing 时 K 点频率单调升高、cusp depth 单调减小，趋势符合当前电子响应模型；
- 把旧 P4 的长程力全局重缩放到最优值，只能消除约 `3.46%` 的平方力误差。

本轮冻结以下内容：

- 241 点 full-EPC 响应的 q 依赖和 smearing 依赖；
- K-star/mode projector 和矩阵装配方式；
- 采样时使用的保守 q6 力算符；
- 谱计算时“减去 q6、加回 dense full-EPC”的接口。

允许并行做能量—力一致性和 double-counting 审计，但在 matched-ensemble 数据出来前不重新拟合长程幅度、cusp 深度或温度锚点。

## 3. 模型与温度的定义

采样使用的势能写成

\[
E(\mathbf R;s)=E_{\mathrm{base}}(\mathbf R)
+\Delta E_{\mathrm{SR}}(\mathbf R)
+E_{\mathrm{LR}}^{q6}(\mathbf R;s),
\]

其中 `s` 是 `smearing/degauss (Ry)`。短程模型不输入 lattice temperature；`T_lat` 只通过 MD/SSCHA 的统计分布进入。

谱计算使用

\[
D_{\mathrm{eff}}(\mathbf q;T_{\mathrm{lat}},s)
=D_{\mathrm{SSCHA}}^{q6}(\mathbf q;T_{\mathrm{lat}},s)
-D_{\mathrm{LR}}^{q6}(\mathbf q;s)
+D_{\mathrm{LR}}^{\mathrm{full\ EPC}}(\mathbf q;s).
\]

短程微调的目标保持为可积分的能量模型：

\[
\mathbf F_{\Delta\mathrm{SR}}=
\mathbf F_{\mathrm{DFT}}(s)
-\mathbf F_{\mathrm{base}}
-\mathbf F_{\mathrm{LR}}^{q6}(s).
\]

不为 300、450 和 600 K 分别训练三个互不相关的 MLIP，也不把 `T_lat` 当作模型输入。

## 4. 数据清单和边界

现有可用数据：

| 数据 | 数量 | 本轮用途 |
|---|---:|---|
| E50 固定-smearing 450 K DFT 标签 | 60 个，3 seed×20 | 已开放 development；诊断和覆盖回放，不能再称为 blind holdout |
| S0 300 K thermal 数据 | 45 个唯一构型 | 短程与 harmonic 稳定性回放 |
| S0 600 K thermal 数据 | 45 个唯一构型 | 短程与高位移覆盖回放 |
| harmonic replay | 72 train + 25 validation | 防止 K/Γ 谐波曲率漂移 |
| X0 450 K 固定-smearing SSCHA | 300 个构型 | 当前 S0 自洽 overlap screen 的候选池 |
| full-EPC/DFPT K 邻域数据 | 241 点模型曲线及已有 direct DFPT | 长程响应冻结与最终谱线验证 |

新增 DFT 数据采用批次上限：

- 第一个 overlap screen：12 个构型；
- 每轮短程校正：最多 12 个新构型；
- 最终未用于训练的 matched holdout：12 个构型；
- 快速闭环共 36 个新标签；保守闭环最多 48 个；
- 达到 48 个仍不能满足门槛时停止加标签，先审查模型结构、能量目标和采样方法。

不再从旧 P4 分布增加标签。现有 60 个标签可以作为已付费的 development 数据，但不能替代当前 S0 自洽分布的测试。

## 5. 分阶段执行

### R0：冻结、精确审计和测试集选择

**墙钟：**1.5–2.5 h。  
**算力：**本地 CPU + RTX 2060 推理，小于 `0.5 GPU-h`。  
**新 DFT：**0。

工作内容：

1. 固定 base model、S0 checkpoint、T300 q6 operator、full-EPC response、结构映射和代码 SHA-256；
2. 对 E50 新标签生成当前 S0 的精确 total/short/LR force JSON 和逐构型 CSV；
3. 报告按 seed、位移、力幅度、原子和 Cartesian 分量分组的误差；
4. 对 6×6 K 点 A' 模式投影力误差，并回归 DFT/MLIP restoring-force slope；
5. 从现有 300 个 X0 SSCHA 构型中固定 12 个 overlap 点；
6. 在 DFT 返回前写入选择索引、结构哈希、DFT 参数和验收门槛。

12 个构型的选择规则：

- 4 个来自主概率质量区，检查常规热构型；
- 4 个来自 checkpoint committee 分歧较高但仍处于物理位移范围的区域；
- 4 个覆盖 A' 模式投影振幅的两端；
- 在 MACE descriptor 空间做 D-optimal/farthest-point 去冗余；
- 不读取 DFT 力后重新选择索引。

建议输出目录：

```text
results/graphene_physics_temperature/post_p4_feasibility/
R0_current_s0_fixed_smearing_audit/
R1_current_s0_overlap_screen/
```

R0 的结论是：当前 S0 在已开放数据上是否存在实现错误、明显的 A' 曲率偏差或长程重复加减。R0 不作最终热力学验收。

### R1：12 点当前分布 overlap screen

**墙钟：**DFT 1.5–2 h，合并与分析 0.5 h。  
**算力：**V100-A/B 各 6 个标签，总计约 `3.0–3.5 V100 GPU-h`。  
**DFT 条件：**72 atoms，`8×8×1` k 网格，`60/240 Ry`，Fermi–Dirac `degauss=0.0019000869 Ry`，固定 cell，`disk_io='none'`。

历史实测为每个构型约 13.3–16.4 min。两台 V100 并行时，6 个/机器的中心墙钟约 1.5 h，并预留到 2 h。

#### 力 overlap gate

- total-force component RMSE `≤30 meV/Å`；
- 最大分量 `≤200 meV/Å`；
- A' 投影力 RMSE `≤15 meV/Å`；
- A' restoring-force slope 相对 DFT 的偏差 `≤5%`；
- 误差不随总位移或 A' 振幅出现单调增大的系统趋势。

#### 热力学 overlap gate

S0 目前是 forces-only 训练，因此还要单独检查相对能量。移除全局常数后：

- 每个 72 原子构型的 centered energy-error RMSE 目标为 `≤0.5 k_B T_lat`，在 450 K 时约 `19.4 meV/config`；
- 用 DFT–MLIP 能量差估计的 importance-weight ESS/N `≥0.3`；
- 若力 gate 通过但能量 gate 未通过，归因为短程训练目标不足，不归因为长程响应失败。

R1 完成后得到第一个明确结论：

- 两个 gate 都通过：不重训当前 S0，进入 matched finite-T 参考；
- 力通过、能量失败：只做 energy+force 的保守短程 residual 校正；
- 力失败：进入短程模型结构/局域性消融；
- 力与能量都通过但 K-referenced cusp 仍失败：再检查长程形状或矩阵接口。

### R2A：R1 通过后的快速闭环

**适用条件：**力和热力学 overlap gate 都通过。

1. 复用 R1 的 12 个 DFT 点作为第一批 matched 数据；
2. 再选 12 个构型，检查 DFT-constrained TDEP/SSCHA correction 的收敛；
3. 用 24 点做受对称性和 ASR 约束的 DFT force-constant 回归，同时拟合 MLIP 在完全相同几何上的 force constants；
4. 两边都减去同一个 q6 响应，再加回同一个 full-EPC 响应；
5. 若 K 点和 near-K bootstrap 已收敛，冻结模型；否则只增加一批 12 点；
6. 最后计算 12 个未用于模型或门槛选择的 matched holdout。

快速路径使用 36 个新 DFT 标签；需要第二个 correction batch 时总数为 48。相对直接再做 60 个标签，标签数减少 40% 或 20%，并且每一个标签都来自当前模型的自洽分布。

**预计墙钟：**从 R1 开始约 12–24 h。  
**MLIP/SSCHA 算力：**约 `1–3 GPU-h`。  
**DFT 算力：**36 标签约 `9 V100 GPU-h`；48 标签约 `12 V100 GPU-h`。

### R2B：力通过、能量失败时的目标函数修复

不改变 full-EPC 长程项，也不先增大模型。对当前 S0 增加一个小的保守 residual：

\[
E_{\mathrm{SR,new}}=E_{\mathrm{SR,S0}}+\alpha\,\Delta E_{\theta}.
\]

训练和选择使用：

- DFT 相对能量和力联合损失；
- 现有 harmonic replay；
- E50 已开放数据作为低权重覆盖回放；
- A' Hessian projection、Γ/K 静态频率和最大力作为 checkpoint selection gate；
- lattice temperature 和 `degauss` 仍不作为局域模型输入。

只运行两个随机 seed。RTX 2060 上现有 240 epoch S0 实测约 1.06 h/次；本分支预算 `3–6 GPU-h`、墙钟 3–5 h。修复后重新生成 450 K SSCHA 系综，并用新的 12 点 matched holdout 验收。

### R2C：力 overlap 失败时的短程模型修复

按控制变量顺序运行，不做大规模网格搜索：

1. `loss-only`：保持 `2 interactions / 32x0e+32x1o / r_max=5 Å`，加入相对能量、力和 harmonic replay；
2. `depth`：只把 message-passing interactions 从 2 增到 3；
3. `capacity`：只把 hidden irreps 从 `32x0e+32x1o` 增到 `64x0e+64x1o`；
4. 每个候选最多两个 seed，先通过 development gate 再读取新的 matched holdout。

训练数据由以下部分组成：

- 现有 300/600 K thermal 数据；
- E50 的 60 个已开放固定-smearing 标签，按覆盖度加权；
- R1 失败后的 12 个当前分布标签；
- harmonic replay；
- 不新增旧 P4 分布标签。

选择门槛比旧 S0 的 `50/250 meV/Å` 更严格，采用 R1 的 `30/200 meV/Å` 和 A' slope gate。harmonic replay RMSE 不得超过 frozen base 的 `1.25×`，Γ/K 静态频率漂移不超过 `2 cm^-1`。

**训练预算：**`8–16 GPU-h`；V100-B 与 RTX 2060 可并行，预计 4–8 h 墙钟。  
**重新采样：**0.5–2 h。  
**新 matched holdout：**12 标签，约 1.5–2 h DFT 墙钟。

若三种受控消融和一轮新 holdout 都失败，不再增加几十个相同标签。下一步改为检查局域模型表达范围、显式保守 reciprocal-space coupling 或 SSCHA reference，而不是继续延长 epochs。

### R3：长程接口审计

该阶段与 R1/R2 并行，不修改已冻结的 full-EPC 参数。

工作内容：

- 对 q6 operator 做有限差分 energy–force 一致性；
- 检查 atom mapping、ASR、Hermiticity、K-star 和单位；
- 检查 SSCHA Hessian 中 q6 响应是否只减一次、dense full-EPC 是否只加一次；
- 对同一个 Hessian 分别输出 short-only、q6 和 full-EPC 三条曲线；
- 保持 dense full-EPC 为谱后处理模块，不把目前只在 q-space 定义的响应直接用于任意构型 MD 力。

**耗时：**2–4 h，本地 CPU/RTX；新 DFT 为 0。

只有在以下条件同时出现时才打开长程模型重拟合：

1. 当前分布的力与能量 overlap 已通过；
2. matched-ensemble DFT 与 MLIP 的 K 点共同绝对背景一致；
3. K-referenced cusp depth、斜率或 rounding 仍超过门槛；
4. 误差可以在 independent q points 上复现，而不是来自 TDEP seed spread。

### R4：受控谱线、DFT 对照和最终图

第一张正式图固定：

- `T_lat = 300/450 K`；
- 两者都使用 Fermi–Dirac `degauss=0.0019000869 Ry`；
- 相同 cell、相同 full-EPC response、相同 q path；
- MLIP+full-EPC 与 matched-ensemble DFT+full-EPC 以同一 absolute A' 形式叠加；
- DFT 离散点同时给出受物理约束的平滑拟合曲线；
- legend 放在坐标轴外，不遮挡 K 邻域。

固定-smearing finite-T 验收：

- K 点绝对误差 `≤5 cm^-1`；
- `|q-K|≤0.025 (2π/a)` 的 K-referenced shape RMSE `≤1 cm^-1`，最大误差 `≤2 cm^-1`；
- `d=0.003` 和 `d=0.025` 的 cusp depth 相对误差 `≤15%`；
- 300→450 K K 点频移误差 `≤5 cm^-1`；
- block/bootstrap 的 K 点 95% 区间宽度和 seed spread 均 `≤5 cm^-1`。

固定-smearing 闭环后，再保持每个 `T_lat` 的短程热背景不变，只替换电子长程响应，生成：

- `smearing=0` 的 full-EPC cusp；
- `0.0019000869/0.00285013035/0.0038001738 Ry`；
- 必要时增加 `0.005/0.010/0.020 Ry` 的模型预测。

这组图只比较 electronic smearing，不混入不同 lattice temperature。零-smearing 的长程部分用已有 direct DFPT/k-grid extrapolation 验证；在没有零-smearing DFT 热系综前，不把它表述为完整的零-smearing DFT-MD 验证。

## 6. 如何根据结果选择调整对象

| 观测结果 | 主要归因 | 下一步 |
|---|---|---|
| 当前分布 force RMSE/max 失败 | 短程势或局域表达范围 | R2C，先 loss，再 depth/capacity |
| 力通过，centered energy/ESS 失败 | forces-only 训练目标不足 | R2B，加入相对能量的保守 residual |
| 力、能量通过，绝对频率整体平移，K-referenced shape 通过 | 有限温短程自由能 Hessian/统计误差 | 增加 matched sampling 或改 SSCHA correction，不改 cusp 核 |
| 力、能量和绝对背景通过，但 cusp depth/slope 失败 | 长程 q 依赖或 projector/interface | 才进入长程模型修复 |
| DFT-TDEP seed/block spread 大于模型差异 | DFT 参考未收敛 | 增加当前分布的独立块，不调模型参数 |
| 仅旧 P4 DFT-TDEP 与当前 SSCHA 相差很大 | 系综不匹配 | 不作为任何参数的拟合目标 |

## 7. 算力、存储和时间预算

2026-08-22 16:30 NZST 的实时状态：

| 机器 | GPU 状态 | 可用空间 | 本轮职责 |
|---|---|---:|---|
| V100-A | 0%，5 MiB/32 GiB | `/data` 51 GiB | 每批 6 个 DFT labels |
| V100-B | 0%，5 MiB/32 GiB | `/data` 72 GiB | 每批 6 个 DFT labels；失败分支可训练一个消融 |
| RTX 2060 | 0%，415 MiB/8 GiB | 根分区 59 GiB | 推理、选择、短程小模型、SSCHA 和后处理 |
| 本地工作站 | CPU | 仓库工作区 | manifest、哈希、合并、TDEP/图表审计 |

存储规则：

- V100 保持至少 35 GiB 可用，不保留可重建的 QE response scratch；
- DFT single-point 使用 `disk_io='none'`，每批标签只回传输入、输出、结构、力、能量和 manifest；
- RTX 当前只有 59 GiB 可用，本轮新增数据控制在 5 GiB 内；
- 不复制旧 QE `tmp`、EPW restart 或无关 checkpoint；
- 新一轮 3000-frame MD 不是 R1 前置条件，因为已有 300 个当前 S0-SSCHA 构型可用。

从实际启动时刻计时：

| 时间点 | 应得到的结论 |
|---|---|
| T+2 h | 当前 S0 对 E50 新标签的精确误差、A' 投影误差和冻结的 12 点列表 |
| T+4–5 h | 当前 S0 在自洽 450 K 分布上是否通过 force/energy overlap；明确进入 R2A、R2B 或 R2C |
| T+12–24 h | 快速路径的 24–36 点 matched DFT/MLIP 曲线和第一版 K 点定量结论 |
| T+24–48 h | 修复分支的新模型、新自洽系综和未用于训练的 matched holdout 结论 |
| T+48–72 h | 300/450 K 固定-smearing 正式对比图，以及同一晶格温度背景下的零/有限 smearing 图组 |

这些时间假设两台 V100 持续在线且单点维持 13–17 min。任何阶段失败时，对应时间仍应给出明确的失败归因和停止点，不以继续增加标签代替结论。

## 8. 数据效率与可迁移方法的产物

本轮不仅要得到 graphene 图，还要留下一个可复用流程：

1. 用少量 harmonic/static 数据建立体系短程基座；
2. 用 q6 conservative operator 从 DFT 力中扣除电子长程响应；
3. 每个新体系先做 12 点自洽 overlap screen，而不是直接生成 60 点轨迹；
4. smearing 主要通过共享 full-EPC/低秩长程 adapter 迁移，不为每个 smearing 重训 MLIP；
5. 只有局域交叉-smearing residual 超过门槛时才增加局域 Mermin residual；
6. 用 12 点批次逐轮判断，单体系在 36–48 个 thermal labels 内关闭或停止；
7. 最终用未见 lattice temperature、未见 smearing 和未见 q points 分别验收短程、电子响应和组合模型。

建议对新体系记录三条独立学习曲线：

- thermal force/energy labels：`0/12/24/36/48`；
- full dynamical-matrix q points：`0/4/8/12`；
- 未见 smearing 的 K/cusp error。

这样才能回答“相对直接 DFPT/DFT-MD 节省多少数据和算力”，而不只报告 graphene 上的一次拟合。

## 9. 启动与停止规则

- R0 的模型、算符、12 个构型索引和门槛冻结前，不启动 R1 DFT；
- 不用旧 P4 的 `40 cm^-1` 频差拟合任何参数；
- 不再把 E50 的 60 个已开放构型称为 blind holdout；
- R1 通过前不启动完整 DFT-MD；
- 力失败时先修短程，力与能量通过但 cusp shape 失败时才修长程；
- 每轮最多 12 个新标签，累计 48 个仍失败则停止加标签；
- final matched holdout 在模型和阈值冻结后生成，不参与 checkpoint 选择；
- `smearing/degauss (Ry)` 和 lattice temperature 始终分别记录；
- 只有 fixed-smearing finite-T gate 通过后，才把零和更高 smearing 加入正式有限温图。

## 10. 第一批执行清单

1. 新建 R0 audit 脚本，精确评估当前 S0 + T300 q6 operator 对 E50 新标签；
2. 从 RTX 2060 现有 X0 SSCHA `xats_pop1.npy` 中导出 300 个构型；
3. 用 current/nearby checkpoints 的分歧、A' 模式振幅和 descriptor 覆盖选择 12 个点；
4. 写入 freeze manifest 和两个 6-label shard；
5. 两台 V100 并行运行 R1；
6. 合并后先输出 force/energy overlap 报告，再决定是否训练；
7. 不提前启动短程消融或新增 full-EPC 拟合。

本计划的近期关闭点是 R1：在约 4–5 小时的有效墙钟内，用 12 个新 DFT 点决定当前模型究竟需不需要改，以及应该改训练目标、短程结构还是长程接口。定量有限温谱的关闭点是 R4，目标墙钟为 1–3 天。
