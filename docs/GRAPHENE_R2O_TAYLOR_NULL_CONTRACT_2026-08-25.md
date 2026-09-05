# Graphene R2O 全能量 Taylor-null 实验合同与算力计划

## 当前状态

R2O 的数据、模型包装、训练器、固定检查点评估器和启动器已经实现并通过本地静态检查与 9 项单元测试。RTX 2060、V100-A 和 V100-B 的独立 `2+2 epoch` FP64 smoke 均得到 `DONE + SMOKE_PASS + EXIT_CODE=0`。RTX 主任务随后完成 `80+240 epoch` 正式训练；流程和全部实现门均通过，但固定物理 gate 未通过。终态为 `DONE + PRETRAIN_DONE + TRAINING_DONE + CORE_GATE_FAILED + EXIT_CODE=0`，gate SHA256 为 `924b13de54d2a918c16845524289036e117006e48ee454f364f442fd88ee8898`。V100-A/B 冗余复算已用 SIGTERM 在完整 epoch 边界后停止，checkpoint 和恢复记录保留。

smoke 的作用是验证 Taylor-null 高阶反传、数据隔离、力学不变性、恢复路径和实际吞吐。它不替代 E50 seed1 与 small-harmonic 的物理 checkpoint gate。正式任务的 diagnostic bundle 同样不授权 routed-tail、SSCHA 或声子谱部署。

本轮不再采用冻结 encoder 后拟合 feature jet 的方案。部署能量固定为同一个局域 MACE 全能量的 Cartesian 二阶 Taylor 余项：

\[
R_\theta(x)=E_\theta(x)-E_\theta(x_0)-D E_\theta(x_0)[u]
-\frac12D^2E_\theta(x_0)[u,u].
\]

这里的 `u` 是按 phonopy 原子顺序、经过周期最小像对齐后的实时 Cartesian 位移。原子置换和整数像在 autograd 外确定，图内保持 `du/dx=I`；HVP 的 `grad_outputs=u` 不得 detach。能量、力和 Hessian 都从完整标量余项求导，因此参考结构的能量、力和二阶力常数按构造为零，同时保留全部 Cartesian 三阶及更高阶响应。

## 数据冻结

Stage 1 使用 92 个热构型：E50 seed0 20 个、T300 36 个、T600 36 个。Stage 2 在相同 92 个热构型之外，加入由 `bond-length RMS <= 0.003 Å` 唯一规则选出的 32 个 harmonic-train 小振幅构型，目标 tail force 为零。E50 seed1 20 个只用于固定 gate；seed2 和任意 support 文件不在程序输入中。

目标恒等式已经逐构型复算：

- E50：`REF = DFT_TOTAL - FOUNDATION_BASE - FROZEN_Q6`，最大 extxyz 数值误差 `1.00000008e-8 eV/Å`。
- T300/T600 auxiliary：`REF = TOTAL - BASE - LONG_RANGE`，最大误差 `1.33e-15 eV/Å`。
- T300 的 `degauss=0.001900086938 Ry`，T600 的 `degauss=0.003800173876 Ry`。后者是 matched-smearing 的 local-Mermin transferable delta，不能称为与 E50 相同 smearing 的 DFT 标签。

小振幅 validation 共 12 个构型。其最大 MIC 原子位移为 `0.0300000023 Å`；`0.03 Å` 的物理界限使用 `1e-8 Å` 的坐标写出容差。零 tail baseline 为 `0.2517008 meV/Å` RMSE、`6.41582 meV/Å` max，固定 preservation gate 为 `0.5/10 meV/Å`。全 25 个大振幅 harmonic 构型只报告，不参与检查点选择。

## 模型与训练

模型固定为 `r_max=3.2 Å`、2 个 interaction、`16x0e+16x1o+16x2e`、`lmax=2`、24 个 radial basis、p=5 cutoff、correlation=3、FP64、无 pair repulsion。所有允许的训练和验证构型在 3.2 Å 下保持同一 edge set；观测到的第三壳最大距离为 `3.0669737 Å`，第四壳最小距离为 `3.5428117 Å`。两层 interaction 的力作用直径上界 `12.8 Å` 小于 6×6 面内晶胞长度 `14.76 Å`。

Stage 1 是 force-only 普通 MACE，从 seed83 随机初始化。92 个构型全部进入梯度；同一 thermal 文件仅作为 validation 别名，seed1 不参与 validation 或学习率控制。group mass 为 `0.50/0.25/0.25`，Adam、batch1、lr `1e-3`、gradient clip `10`、ExponentialLR gamma1、无 EMA、无 SWA。80 个 epoch 对应 zero-based epoch `0..79` 和准确的 `92×80=7360` 次 optimizer update。MACE CLI 的断点恢复会重复最后一个 zero-based epoch，因此 partial stage1 明确 fail closed，必须用新的 OUT 从 seed83 重跑；已通过 exact stage1 runtime/hash 的任务可以直接进入可恢复的 Stage 2。

Stage 2 从 exact Stage 1 权重继续训练同一个 MACE 的全部参数。group mass 为 E50/T300/T600/small-H `0.50/0.20/0.20/0.10`，对应 force scale `30/30/30/0.5 meV/Å`；AdamW、lr `1e-3`、weight decay `1e-6`、clip `20`、EMA `0.99`、gamma1。正式固定候选为 epoch `40/80/120/160/200/235/240`。`latest.pt` 保存 model、EMA、optimizer、scheduler、RNG、contract 和本 epoch metric；恢复时会核对并修复 metric/fixed-checkpoint 的原子写入窗口。

## P0 与物理 gate

smoke 必须同时通过以下实现检查，才允许安排正式训练：

- reference 的 `|E|<=1e-10 eV`、`max|F|<=1e-9 eV/Å`、`max|H|<=1e-7 eV/Å²`、Hessian translational ASR row-sum `<=1e-7 eV/Å²`，以及 Γ/K 频率漂移上界 `<=2 cm⁻¹`；
- 一般 E50 构型完整 `216×216` Hessian 的 antisymmetry 和 translational ASR row-sum均 `<=1e-7 eV/Å²`；
- proper rotation、improper rotation、全局平移、原子置换、跨周期像、6×6/8×8 局域一致性，以及能量→力、力→Hessian 两级有限差分；
- custom interaction energy 与官方 ScaleShiftMACE `interaction_energy` 的能量和 position-gradient 一致。除真实 checkpoint 外，还在不保存、不用于预测的临时 clone 中注入 `scale=2.25, shift=-0.375 eV/atom`，专门检查非平凡 scale/shift 语义；
- p=5 cutoff 的内侧值、一阶和二阶导数为零，三阶导数非零。

正式候选另需通过 E50 seed1 的 `30/200/A′15/slope5%` gate和 small-12 的 `0.5/10 meV/Å` preservation gate。全 25 harmonic 结果始终只报告。

## 算力与时间

Stage 2 的主要成本来自实时参考梯度、HVP 和 force-loss 高阶反传。先前 synthetic probe 给出的 9–11 h（RTX）和 18–20 h（V100-A）预算明显偏保守；正式排期改用实际 `2+2 epoch` smoke：

| 机器 | Stage 1 smoke | Stage 2 smoke | Stage 2 逐 epoch | 240 epoch 外推 | 正式端到端墙钟预算 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 2060 SUPER | 35 s / 184 updates | 67.29 s | 35.97 / 31.07 s | 约 2.25 h | 2.5–3.5 h |
| V100-A | 75 s / 184 updates | 112.17 s | 61.97 / 49.57 s | 约 3.72 h | 3.5–5 h |

RTX 的 Stage 1/2 峰值显存分别为 `657/485.92 MiB`，V100-A 为 `474/485.92 MiB`。两台机器 Stage 2 的归一化训练 MSE 在报告精度内相同；epoch 1 的 E50/T300/T600/small-H 为 `16.5897/9.37687/24.5651/20.4613`，epoch 2 为 `13.8981/8.24480/19.7911/21.8922`。正式预算包含 Stage 1、固定 checkpoint 物化、完整 Hessian、gate 和 I/O 余量，不把 synthetic probe 的单步下限继续当作实际吞吐。

## 双节点 smoke 结果

RTX 的 smoke 从 `08:44:32` 运行到 `08:47:24 NZST`，V100-A 从 `08:44:51` 运行到 `08:50:15 NZST`。gate JSON SHA256 分别为 `87b8d2ae7452d89f630c2968824b5343edb660858980b5df4b79143b0efbbcb1` 和 `411d6e00f071bd4695bf20166e9e9d850716e1884ee7e60534a4a4b036f8491f`。

两个节点的实现检查均通过。关键观测上界为：

- 官方 ScaleShiftMACE 与包装器的位置梯度差不超过 `5.33×10^-15 eV/Å`；
- proper/improper O(3) 最大力差分别为 `7.32×10^-7` 和 `5.44×10^-7 eV/Å`；
- 平移与置换/跨周期像最大力差分别不超过 `2.28×10^-14` 和 `3.07×10^-7 eV/Å`；
- 能量—力有限差分误差 `1.48×10^-7 eV/Å`，力—Hessian 有限差分误差 `2.09×10^-6 eV/Å²`；
- 一般 E50 构型 Hessian antisymmetry 与 ASR row-sum 不超过 `8.94×10^-15 eV/Å²`；
- 6×6/8×8 局域一致性的能量/中心力差为 `9.03×10^-12 eV` 和 `5.58×10^-8 eV/Å`；
- 参考点 `|E|≤8.88×10^-16 eV`、`max|F|≤3.44×10^-16 eV/Å`、`max|H|≤1.07×10^-14 eV/Å²`、ASR row-sum `≤4.96×10^-15 eV/Å²`，Γ/K 频率漂移上界 `≤1.82×10^-5 cm^-1`。

两个节点的数值轨迹一致，但 EMA model state SHA256 分别为 `2dcb3867…` 和 `eac7ee8…`。这是硬件舍入差异，本合同要求相同门控结论和数值容差，不要求不同 GPU 上逐字节相同。smoke bundle 的 `kind` 明确为 `R2O_diagnostic_bundle_not_authorized_for_deployment`。

2 epoch checkpoint 没有通过正式物理门槛：E50 RMSE/max 为 `114.071/525.240 meV/Å`，A' RMS 为 `207.705 meV/Å`，slope error 为 `5.171%`，small-harmonic RMSE/max 为 `2.768/78.815 meV/Å`。这是预期的短训练诊断；不能据此选择正式 checkpoint，也不能据此宣称 R2O 已通过或失败。

## 启动与恢复

启动器为：

```bash
bash scripts/v100/run_graphene_r2o_taylor_null.sh smoke
bash scripts/v100/run_graphene_r2o_taylor_null.sh formal
```

smoke 需要固定 feasibility JSON 和 `GO_FOR_TWO_EPOCH_SMOKE` marker。formal 还要求：R2N `DONE + TRAINING_DONE + R2N_CORE_GATE_FAILED + EXIT_CODE=0` 及固定 gate/model 哈希；完成且通过的 smoke `DONE + SMOKE_PASS + EXIT_CODE=0`；以及由人工复核后提供的 `R2O_EXPECTED_SMOKE_GATE_SHA256`。RTX 与 V100-A 已分别冻结上述 smoke gate 哈希，因此 formal 的最后一道方法学阻断已经解除；实际运行仍须在启动时把对应节点的 predecessor 和 smoke 哈希写入 `training_freeze.json`。

启动时会把 wrapper、trainer、evaluator 原子复制到输出目录，最后发布 runner snapshot。Stage 2 可从 exact `latest.pt` 恢复；`TRAINING_DONE` 后只重跑 evaluator。`DONE+EXIT_CODE=0` 只有在完整复核 freeze、模型、固定 checkpoint、gate、bundle 和互斥 marker 后才 no-op。成功结束时先移除 `RUNNING`、原子写 `EXIT_CODE=0`，最后发布 `DONE`。

本轮 formal 的实际运行记录为：

| 节点 | 开始时间 | PID | OUT | smoke gate SHA256 |
|---|---|---:|---|---|
| RTX 2060 | `08:58:31 NZST` | `862210` | `/home/howardwang/phonon/results/graphene_physics_temperature/post_p4_feasibility/R2O_taylor_null_core/formal_r3p2_h16_l2_n24_seed83_rtx` | `87b8d2ae7452d89f630c2968824b5343edb660858980b5df4b79143b0efbbcb1` |
| V100-A | `09:00:17 NZST` | `51995` | `/data/graphene_r2o_core/formal_r3p2_h16_l2_n24_seed83_v100a` | `411d6e00f071bd4695bf20166e9e9d850716e1884ee7e60534a4a4b036f8491f` |

V100-A 使用经 Tailscale 复制的 RTX R2N 已完成失败目录作为前序，共 72 个文件、约 71 MB，目录摘要 `64321d8a…`；固定 gate/model SHA256 为 `b2e00ba1…/d20b8281…`。该 diagnostic model 只提供 fail-closed provenance，不初始化 R2O。RTX Stage 1 启动后的当前实测约为 `7.3 s/epoch`；在 Stage 2 首个正式 epoch 返回前，完整结论窗口保守记为 RTX `11:30–12:30 NZST`、V100-A `12:30–14:00 NZST`。

## 当前结论和下一步

现在能得出的结论是：whole-energy live Taylor-2 的 FP64 高阶反传、数据隔离、reference/order/MIC、ScaleShift 语义、严格 harmonic null、恢复逻辑和固定力学检查均已通过。它成功把 E50 seed1 force RMSE/max 降到 `17.855/92.465 meV/Å`，A' restoring-slope 相对误差降到 `0.670%`，但 A' projected RMS 仍为 `25.106 meV/Å`，small-harmonic max 仍为 `16.647 meV/Å`，分别超过 `15` 和 `10 meV/Å` 的门槛。因此 R2O formal 明确未通过，不能声称已经准确复现有限晶格温度声子谱。

下一步严格按以下顺序进行：

1. 保留 RTX 的全部 formal checkpoint、gate 和 diagnostic bundle；V100-A/B 的冗余复算在主任务失败后停止，不继续消耗确定性重复算力。
2. R2P 的 176-block、零优化步 common-descent cone 审计已经完成。RTX primary certificate SHA256 为 `54bf9056d8ac90641875f2dbe997336ffe909c3babf9a17440d6c2237791842c`；`t_primal/t_dual=0.00991895648/0.00992077217`，duality gap 为 `1.816e-6`，KKT violation 为 `2.086e-8`。V100-A/B 的独立复算与 RTX 在 `10^-13–10^-14` 相对精度内一致。计算前后 model state 不变，seed2/support 未打开。
3. R2Q 已按合同完成 4 个 train-only trust-region steps，四步 cone、Armijo、trust、reference 和 state chain 全部通过。唯一 endpoint checkpoint SHA256 为 `31dd053a…3eb5`，但 held gate 未通过：seed1 A' RMS 为 `23.9769>15 meV/Å`，small-H max 为 `15.8157>10 meV/Å`；其余科学项和全部 mechanics 通过。该结果可靠地停止了当前表示上的 loss-only/继续加步修复。
4. 下一步改为 R2R 多极化热背景条件化表示。局域连续背景量只从参考映射后的位移协方差构造，在多模 thermal 构型上开启，在 rank-1 harmonic 位移上解析为零；不把来源、晶格温度或 electronic smearing 作为模型输入。先做零参数的几何分离、O(3)/order-MIC/locality/null 和线性设计矩阵审计，通过后才允许拟合冻结 encoder 上的 65 参数 readout。
5. seed1/small-H 已用于 R2Q endpoint，只能作为后续 development confirmation；seed2/support 继续保持未打开。在 R2R 新候选重新通过同一 E50/small-harmonic/reference/mechanics gate 并另经未打开轨迹验收前，routed-tail、support9 outer-LOCO、SSCHA、matched DFT holdout 和 full-EPC 正式声子谱保持 blocked。
