# Graphene R2O 通过后的 routed-tail 准备度审计

日期：2026-08-25

状态：只读审计完成。未启动训练，未打开 E50 seed2，也未读取 support9 的 extxyz 内容。下面关于 support9 标签的定义来自已经冻结的 R2M/R2O 合同和现有准备代码。

## 1. 审计结论

现有 `graphene_r2m_routed_tail` outer-LOCO 不能通过替换 core 路径直接绑定 R2O。结论为直接复用 NO-GO，需要新建 R2O 专用的 Taylor-null routed-tail 合同和代码命名空间。

这不是文件名或状态字段不同造成的表面问题。旧框架把 core 当作可以直接交给 `MACECalculator` 的 FP32、`r_max=2 Å` 普通 MACE；R2O 发布的是 FP64、`r_max=3.2 Å` 的 deployment bundle，只有经过固定 reference、原子顺序、MIC image 和 live-HVP Taylor wrapper 后的三阶及以上余项才是物理 core。R2O bundle 还明确禁止直接部署内部 raw MACE。

旧 routed tail 自身只对节点能量做 pristine carbon 定零，没有把 tail 的完整标量能量做 Taylor 二阶消去。它一般会在参考结构重新引入非零 Hessian。即使 support force 下降，也会破坏 R2O 用来解除 harmonic/thermal 冲突的结构约束。因此 R2O 后的 tail 也必须按构造满足参考点 `E=F=H=0`。

## 2. 旧框架不能直接使用的具体原因

现有代码有以下硬绑定：

- prepare 只接受 `R2M_core_checkpoint_gate_passed` 和 `selected_core_for_routed_tail_force_gate`，不能识别 R2O 的 gate 与 bundle。
- descriptor schema 固定 `r_max=2 Å`、raw width 160、64 个 invariant，并以 R2M 版本号发布。R2O 的 irreps 布局虽可能同宽，但 cutoff、FP64、固定参考图和 provenance 都不同，不能沿用同一个 schema hash。
- core prediction 使用 `MACECalculator(... default_dtype="float32")`。对 R2O 这样做得到的是 raw MACE，不是 Taylor remainder。
- trainer 从 current-geometry neighbour graph 取得 `node_feats`。R2O 必须使用固定 reference graph、species-aware assignment、MIC integer 和 source/reference order 映射。
- residual energy 目前减去普通 core energy；R2O 必须减去 Taylor remainder energy。
- replay 固定为旧 R2M 的 164 个构型，harmonic preservation 门槛为 `9.044673/200 meV/Å`。R2O 的对应训练 replay 是 thermal 92 加 small-harmonic 32，正式 small-harmonic 门槛为 `0.5/10 meV/Å`。
- 旧 tail 的能量、力虽由同一标量求导，但 tail 没有 Taylor-null，不能保持 R2O 的 exact reference null。
- 旧 evaluator/aggregate 没有按 R2O 合同重算 reference `E/F/H/ASR`、force-to-Hessian finite difference 和 Gamma/K drift。
- prepared manifest 和 fold manifest 记录并强制比较绝对路径。RTX 与两台 V100 的根目录不同，不能把一次准备出的同一份 folds 安全迁移到三台机器。若分别准备，又会引入不同路径、浮点预测和 fold hash。R2O 新合同必须改为相对路径加内容 hash。
- 现有 outer-LOCO 只在通过后授权 all9 final，但仓库中没有与 R2O 绑定的 all9 final trainer、完整复合模型 evaluator 或 deployment bundle。

因此旧代码只可作为策略和测试模式的参考，不能直接执行，也不能对它做少量参数替换后称为 R2O tail。

## 3. 正确的模型分解和 support residual

R2O 通过后，复合模型应固定为

\[
E_{\mathrm{comp}} = E_{\mathrm{v11}}^{\mathrm{frozen}}
+ R_{\mathrm{R2O}}^{(\ge 3)}
+ R_{\mathrm{tail}}^{(\ge 3)}
+ E_{q6}^{\mathrm{frozen}}.
\]

这里 `R_R2O` 是已通过 gate 的 whole-energy Taylor-2 remainder。新 tail 先定义一个局域、保守的 raw 标量能量

\[
U_\phi(R)=\sum_i g_i(R)\,[\epsilon_i(R)-c_C],
\]

再使用与 R2O 相同的 order/MIC-safe live-HVP 算法构造

\[
R_{\mathrm{tail}}^{(\ge3)}(R)=T_{\ge3}[U_\phi](R;R_0).
\]

Taylor 消去对能量是线性的，因此部署时也可以对 `U_R2O + U_tail` 一次做二阶消去；结果应与两个 remainder 相加在 FP64 容差内一致。router、query invariant、score、gate、node energy、`u` 和 HVP 的 `grad_outputs` 都不能 detach。

对 support 构型 `i`，先验证冻结标签分解

\[
F_i^{\mathrm{short}}
=F_i^{\mathrm{TOTAL}}-F_i^{\mathrm{BASE}}-F_i^{\mathrm{LONG\ RANGE}}.
\]

R2O-tail 的力目标只能定义为

\[
F_{i,\mathrm{tail}}^*
=F_i^{\mathrm{short}}-F_i^{\mathrm{R2O,Taylor}}.
\]

不能再次减 foundation 或 q6，也不能减内部 raw MACE force。完整预测用于 A' 和 force gate：

\[
F_i^{\mathrm{pred}}=F_i^{\mathrm{BASE}}+F_i^{\mathrm{LONG\ RANGE}}
+F_i^{\mathrm{R2O,Taylor}}+F_i^{\mathrm{tail,Taylor}}.
\]

能量先定义

\[
d_i=E_i^{\mathrm{short,parent}}-E_i^{\mathrm{R2O,Taylor}}.
\]

prepare 仍需复核 `E_short,parent` 与 `TOTAL-BASE-LONG_RANGE` 只差一个对 9 个构型相同的常数。该常数不能进入学习。对 fold `f`，只用 train8 定义

\[
\bar d_f=\frac1{8}\sum_{j\in\mathrm{train8}}d_j.
\]

能量损失必须同时中心化预测和目标，或使用完全等价的 train8 成对能量差：

\[
\left[R_\phi(R_i)-\overline{R_\phi}_{\mathrm{train8}}\right]
-\left[d_i-\bar d_f\right].
\]

held 能量误差在 epoch-240 EMA 和模型 hash 冻结后定义为

\[
e_h=\left[R_\phi(R_h)-\overline{R_\phi}_{\mathrm{train8}}\right]
-\left[d_h-\bar d_f\right].
\]

旧实现只中心化 target、没有同时中心化预测，不能直接复用。上面的定义消除了 DFT、foundation、force-only core 和不同 fold 的任意能量常数；450 K importance-weight ESS 使用九个 `e_h` 计算。outer-LOCO ESS 仍是九个不同 fold 模型的迁移诊断，all9 final 的 SSCHA ESS 才是部署阶段指标。

## 4. 可以复用与必须新建的部分

可以复用的内容限于：

- support9 原始文件及其冻结 hash；
- whole-configuration 9-fold outer-LOCO、held 不进入 scaler/gradient/gauge/selection、epoch 240 EMA 的方法学策略；
- `graphene_r2m_aprime_eval.py` 的物理定义和冻结 hash，前提是 operator/background/thermal-result hash 与 `lattice_temperature=450 K`、`degauss=0.001900086938 Ry` 合同一致；
- balanced replay、原子写 marker、completion recovery、NPZ 重新聚合等实现模式；
- 已经固定的 force、A'、relative-energy 和 ESS 门槛。

必须新建并使用新的 format/status/hash namespace：

1. R2O deployment bundle loader：核对 gate、bundle、model-state、wrapper、6x6/8x8 reference 和数据 manifest hash，禁止 raw MACE prediction。
2. R2O raw-feature helper：在固定 reference graph 上返回 query-live node features，发布 `graphene_r2o_raw_mace_invariants_v1` schema；cutoff、dtype、layout、reference/order policy 都进入 hash。
3. Taylor-null routed-tail module：`T_{>=3}[sum g(epsilon-c_C)]`，并验证与 combined raw scalar 一次 Taylor 消去的线性一致性。
4. R2O outer-LOCO prepare：只在 R2O formal gate 通过后打开 support9；一次准备、冻结相对路径和内容 hash，再经 Tailscale 原样同步到三机。
5. R2O fold trainer、evaluator、aggregator、provenance helper 和 launcher。旧 R2M status、artifact 和 prepared folds 一律拒绝。
6. all9 final trainer、完整 composite evaluator 和 combined deployment bundle。
7. 独立测试：bundle fail-closed、assignment/MIC/permutation、raw feature parity、reference `E/F/H/ASR`、O(3)、两级 finite difference、6x6/8x8、断点恢复、held 文件延迟读取和跨节点相对路径迁移。

R2O 内部 raw MACE 的 frozen node features可以作为 router/tail descriptor，但只承担描述符作用；它的 raw energy/force不得进入复合预测。feature scaler 每 fold 只可使用 train8 加 R2O replay-train，不能使用 held、seed1、full25 或 seed2。

## 5. 进入 9-fold 的 GO 前置

以下条件必须全部满足：

1. 选定一个唯一 R2O formal artifact。目录同时具备 `DONE`、`TRAINING_DONE`、`CORE_GATE_PASSED`、`EXIT_CODE=0`，且不存在 `RUNNING/FAILED/CORE_GATE_FAILED`。
2. gate 状态为 `R2O_core_checkpoint_gate_passed`，`postcore_or_deployment_authorized=true`；bundle kind 为 `selected_R2O_Taylor_bundle_for_postcore_validation`，所有文件和 semantic-state hash 一致。
3. 若 RTX 与 V100-A 都通过，按 support 打开前固定的主节点规则选择 artifact；不能看 support/seed2 后择优。两节点物理 gate 若结论不一致，应先解释数值差异，不做 cherry-pick。
4. 新 R2O-tail 代码、合同和测试完成静态复核；用不含 support 的 replay/reference 做双节点 FP64 smoke。smoke 必须证明 raw core 没有被直接部署、tail reference `E/F/H=0`、order/MIC 和恢复逻辑正确。
5. smoke 的代码 hash、wrapper hash和通过记录经独立复核后冻结。此后才允许 prepare 打开 support9。
6. prepare 只运行一次。三机收到相同 prepared-root tree hash、同一个 core bundle hash、同一 reference/operator/background/thermal-result hash和相同代码 snapshots。
7. 三机的 Python、NumPy、PyTorch、CUDA、cuDNN、`mace-torch==0.3.16` 语义版本满足冻结合同；GPU 名称可以不同，但必须记录，aggregator 不要求不同 GPU 的模型逐字节相同。

任一项不满足时不启动 fold。

## 6. 固定验收门槛

9-fold aggregate 必须同时满足：

- pooled support force RMSE/max `<=30/200 meV/Å`；
- 每个 held 构型分别满足 force RMSE/max `<=30/200 meV/Å`；
- support A' 投影 RMS `<=15 meV/Å`；
- support A' restoring slope 相对误差绝对值 `<=5%`；
- gauge-invariant relative-energy LOCO RMSE `<=19.4 meV/config`；
- 450 K importance-weight ESS fraction `>=0.30`。

每个 fold 冻结后还必须保持：

- E50 seed1 的 core+tail force RMSE/max `<=30/200 meV/Å`，A' RMS `<=15 meV/Å`，slope error `<=5%`；
- R2O small-harmonic validation 的 core+tail force RMSE/max `<=0.5/10 meV/Å`；full25 只报告；
- combined R2O+tail reference `|E|<=1e-10 eV`、`max|F|<=1e-9 eV/Å`、`max|H|<=1e-7 eV/Å^2`、Hessian ASR row sum `<=1e-7 eV/Å^2`、Gamma/K drift `<=2 cm^-1`；
- energy-force finite difference和force-Hessian finite difference均 `<=1e-5`（对应单位分别为 `eV/Å`、`eV/Å^2`）；
- Hessian antisymmetry 与 translational ASR 均 `<=1e-7 eV/Å^2`；
- rotation/reflection energy error `<=1e-6 eV`、force equivariance error `<=1e-5 eV/Å`，并通过 permutation/image/translation 检查；
- 6x6/8x8 tail energy差 `<=1e-7 eV`、tail 中心力差 `<=1e-5 eV/Å`、R2O+tail 中心力差 `<=1e-5 eV/Å`。

严格局域 size gate 只覆盖 R2O+tail。foundation 不声称理论无绕回；q6 继续使用冻结 Fourier 审计。

## 7. 三机 9-fold 调度与预算

prepared folds 只生成一次并原样同步。三台机器各自串行跑三个 fold，三批同时进行：

| 批次 | RTX 2060 | V100-A | V100-B |
|---|---:|---:|---:|
| 1 | fold 0 | fold 1 | fold 2 |
| 2 | fold 3 | fold 4 | fold 5 |
| 3 | fold 6 | fold 7 | fold 8 |

每台 GPU 同时只运行一个 fold。第一批到 epoch 10 时只更新吞吐、显存和预计结束时间，不调整 loss、epoch、架构、fold assignment 或 checkpoint。后续批次可以只根据已经记录的吞吐在未开始的机器间重新排队，不能根据任何 held 指标调度。

在实际 R2O smoke 吞吐和旧 tail 预算基础上，先冻结以下容量：

- 无 support 的双节点 adapter smoke：`0.25-0.5 h` 墙钟；
- 一次性 support residual/9-fold prepare：`0.15-0.35 GPU h`；
- 每 fold 训练加完整 evaluation：`1-2.5 GPU h`；
- 9-fold 总量：`9-22.5 GPU h`，三机墙钟 `3.5-8 h`；
- aggregate：`10-20 min` CPU；
- 通过后 all9 final 加完整 composite gate：`1.5-3 GPU h`，墙钟 `2-4 h`；
- 9 folds、日志、NPZ、Hessian、模型和 snapshots 共预留 `8 GB`。

上述是启动前容量规划。真正 ETA 只由第一批 epoch-10 和 evaluator 实测修正。三机若环境合同不同，先统一环境，不用分别准备数据来规避 hash 检查。

## 8. 禁止项和最短后续链

禁止：

- 把 bundle 内 raw MACE 交给 `MACECalculator` 后称为 R2O；
- 复用 R2M prepared folds、status、core prediction、absolute-path manifest 或训练 artifact；
- 使用不做 Taylor-null 的 routed tail，或用 `g*F` shortcut；
- detach query feature/router/gate/node energy、MIC 之外的连续变量，或 HVP 的 `grad_outputs`；
- held 进入 feature scale、gradient、energy gauge、epoch/checkpoint/replica 选择；
- 用 old harmonic `9.044673/200` 代替 R2O 的 `0.5/10`；
- 在模型选择完成前读取 seed2；
- outer-LOCO 失败后训练 all9 final、增加同类 DFT、扫描 seed1/support 权重或改 q6；
- 在 all9 final composite gate 前启动 SSCHA、matched DFT holdout 或声子谱；
- 把 support9 outer-LOCO 表述为 blind/external 测试。

最短安全链为：

```text
R2O formal PASS并冻结唯一 bundle
-> 新 R2O Taylor-null routed-tail 实现与双节点无-support smoke
-> 一次性准备并同步 support9 outer folds
-> 三机并行 9-fold + 独立 aggregate
-> 若且仅若 PASS，训练 all9 final tail
-> 完整 composite reference/mechanics/energy/ESS/size gate
-> order-safe 有限晶格温度 SSCHA
-> 新 matched DFT holdout
-> frozen v11 + R2O + tail + full-EPC/q6 正式声子谱。
```

outer-LOCO 若失败，结论应限定为当前 R2O descriptor、Taylor-null routed-tail 和固定 loss 对 support 构型的迁移未达到门槛；按当前停止规则分析模型/长程分解，不打开 seed2 或用更多同类 DFT 覆盖问题。
