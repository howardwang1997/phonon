# Graphene R2O Taylor-null routed-tail 本地 skeleton 合同

日期：2026-08-25

状态：本地核心模块和合成测试已实现。当前代码只允许读取经过内容 hash 固定的 formal passing R2O portable bundle；尚未读取 support9 或 E50 seed2，也没有生成 folds、训练 tail 或启动远端任务。

## 1. 当前实现范围

新增代码使用独立的 R2O 命名空间，不修改旧 R2M routed-tail：

- `scripts/smearing_kink/graphene_r2o_routed_tail.py`
- `tests/test_graphene_r2o_routed_tail.py`

核心合同包括：

1. loader 只接受 `R2O_core_checkpoint_gate_passed`、`postcore_or_deployment_authorized=true` 和 `selected_R2O_Taylor_bundle_for_postcore_validation`。它先核对 receipt、gate、bundle、wrapper、data manifest、6×6/8×8 reference、完成 marker 的相对路径与 SHA-256，再执行 `torch.load`。gate、bundle 和完成 marker 必须位于同一个 formal run root。smoke/diagnostic bundle、非零退出码、残留 `RUNNING/FAILED/CORE_GATE_FAILED`、绝对路径、路径穿越和 symlink 都会被拒绝。
2. bundle 内 raw MACE 只作为固定 reference graph 上的描述符和 Taylor raw scalar 来源。loader 会把参数冻结；代码没有把它包装成可直接部署的 calculator。
3. 新 descriptor schema 为 `graphene_r2o_raw_mace_invariants_v1`，固定 `r_max=3.2 Å`、raw width 160、invariant width 64、FP64、formal model-state/reference/wrapper hash、assignment/MIC 和 autograd policy。schema 和 query 只能从 `LoadedFormalR2O` 构造；query 根据原子数自动选择 formal 72/128-atom reference，并记录 atom count、reference artifact SHA-256、model-state SHA-256、receipt SHA-256、两向 assignment array SHA-256、MIC integer SHA-256 以及 frozen/native 两套 image-shift SHA-256。每次公开 raw/tail/source-order force API 调用都会复核 model、schema、query、完整 receipt chain，并由 MIC integer 和各自 graph cell 精确重算 image shift；任何 assignment 或 shift 的原位修改都会 fail closed。
4. tail 的 raw 标量能量为 `sum_i g_i (epsilon_i-c_C)`。正式 tail 输出只能由这个完整标量做 whole-energy Taylor-2 subtraction 得到。router、gate、node energy、query positions、位移和 HVP direction 保持 live；只有 assignment 和 MIC integer 离散量 detached。
5. outer-LOCO 固定为 9 folds、train8/held1、epoch-240 EMA 单一端点。能量损失分别减去 prediction 的 train8 均值和 target 的 train8 均值；held 能量也只使用同一 fold 的 train8 均值。held 只能在 epoch-240 EMA 和模型 hash 固定后读取。

模型 state hash 不使用缓存。即使参数已经 `requires_grad=False`，每个 provenance 边界仍重新计算 semantic state SHA-256，因此 `copy_` 和 `load_state_dict` 造成的原位修改会被检测。outer contract 的 source artifact 名称使用固定白名单，路径或名称只要含 `support`、`seed2` 或 `reserved` 就会直接拒绝；`leakage=false` 不能替代这个检查。

## 2. reference graph 的默认 dtype 语义

冻结的 R2O wrapper 在 `AtomicData` 构图时没有在函数内部切换 PyTorch default dtype。独立 evaluator 进程的实际 source default 是 `torch.float32`；随后连续张量再 cast 到 `torch.float64`。因此模型参数、autograd 和输出张量是 FP64，但 cell/shifts 的源值保留了一次固定的 float32 量化。6×6 和 8×8 reference 上观测到的 cell/shifts 最大差分别为 `3.61e-7 Å` 和 `7.91e-7 Å`。在不改变 frozen wrapper 的前提下，统一 geometry bound 事先固定为 `1e-6 Å`；6×6/8×8 都必须逐 query 报告并通过，不能局部绕过。该数值低于现有 permutation/image 力学门槛，但不能把它描述为从构图源头开始的 native FP64。

为了保持 formal gate 所验证的 core 语义，新 tail 不改 frozen core wrapper。代码明确固定以下两条分支：

- deployment 分支：`AtomicData source default=torch.float32 -> stored continuous dtype=torch.float64`；
- sensitivity comparator：`AtomicData source default=torch.float64 -> stored continuous dtype=torch.float64`，只用于配对敏感性检查，不部署。

构图 builder、wrapper SHA-256、两种 source dtype、`1e-6 Å` geometry bound/metric 和部署选择共同形成 `graphene_r2o_fixed_graph_semantics_v2` 的 canonical JSON hash。descriptor schema 和 outer-LOCO 合同都必须记录并核对该 hash。三台机器若得到不同 hash，不允许通过分别准备 folds 来绕过。

当前 portable receipt 固定的是路径和内容语义，尚未把 `torch`、`MACE`、`e3nn` 及底层运行时版本纳入 receipt。因此它还不是跨软件环境的行为等价证明。实际 formal tail smoke 前需要另行冻结并核对 runtime environment fingerprint；版本不同必须重新执行配对 native-FP64、E/F/H/ASR 和 mechanics gate，不能只凭相同 model-state hash 继续。

## 3. paired native-FP64 sensitivity gate

每个无-support smoke query 必须同时构造 frozen-semantics 和 native-FP64 graph，并在相同原子 assignment、相同物理 query、各自一致的 MIC cell 上比较。当前固定门槛为：

| 指标 | 门槛 |
|---|---:|
| raw node feature max abs difference | `1e-5` |
| raw scalar energy max abs difference per atom | `1e-6/72 = 1.388888888888889e-8 eV/atom` |
| raw scalar force max abs difference | `1e-5 eV/Å` |

energy sensitivity 使用逐原子量是因为 raw graph scalar 是 extensive sum。该门槛严格由原 6×6 total cap `1e-6 eV` 除以 72 得到，不按 8×8 观测值调参；因此 72/128 原子的 derived total caps 分别为 `1e-6 eV` 和 `1.777777777777778e-6 eV`。feature 和最大力误差不做 size normalization，localized 6×6/8×8 Taylor-remainder size energy gate 也继续使用 `1e-7 eV` total cap。

固定 seed-83 合成 R2O-shape MACE 的 6×6 image query 得到 feature/total energy/force difference `3.79e-7 / 1.49e-7 eV / 8.01e-7 eV/Å`；8×8 pristine query 得到 `7.75e-7 / 1.22e-6 eV / 1.55e-6 eV/Å`，其中 8×8 energy 为 `9.54e-9 eV/atom`。两种尺寸均通过统一门槛。这个数值只验证实现和门槛量级；必须用实际 formal passing model 和 actual tail smoke 重算，不能代替正式结果。

tail 尚未训练时比较 raw core；tail smoke 和每个 frozen fold 必须再比较同一完整 raw `core+tail` 标量。这个检查只回答冻结构图语义对 native FP64 的敏感性，不代替 combined Taylor-null 的 reference `E/F/H/ASR`、两级 finite difference、O(3)、permutation/image/translation 和 6×6/8×8 gate。

## 4. outer-LOCO 训练配方冻结

canonical outer contract 还固定以下内容；任何字段变化都会改变 contract hash：

- 每 fold 的 64 列 scaler 只使用 support train8 和 R2O replay-train 的原子 invariant。逐列计算 population mean、population std 和 RMS，scale 为 `max(std, 1e-6*RMS, 1e-8)`，不裁列。outer held、E50 seed1、harmonic full25、E50 seed2 和 reserved data 均不得进入 scaler。
- pristine gauge 使用 formal receipt 中 hash-bound 6×6、72-atom reference；`c_C` 是其 raw tail node epsilon 的平均值，symmetry-equivalent invariant relative spread 必须不超过 `5e-5`。
- 优化器固定 `AdamW(amsgrad=true)`，learning rate `1e-3`，weight decay `1e-6`，global gradient clip `20`，seed `83`，EMA decay `0.99`，无学习率调度。
- 每 epoch 8 个 optimizer steps：support train8 各一次；small-harmonic32 和 E50-seed0-20 各做 8 次平衡循环抽取；T300-36/T600-36 各做 4 次独立平衡抽取并交替进入 8 个 step。每个 step 用当前参数重新计算完整 train8 centered relative-energy MSE。
- force group mass 固定为 support/small-harmonic/E50/auxiliary `0.50/0.25/0.15/0.10`；force scale 为 `0.030/0.0005/0.030/0.030 eV/Å`。relative-energy weight/scale 为 `0.25/0.0194 eV`；harmonic/E50 gate-off weight 为 `0.05/0.02`，gate-off scale 为 `0.05`。
- replay 使用 seed-83 固定 permutation 和跨 epoch cyclic exposure，salt 固定为 support `11`、small harmonic `23`、E50 `37`、T300 `53`、T600 `71`；不能根据 held 指标改变 schedule。

## 5. 打开 support9 前仍需满足的条件

当前 skeleton 通过不授权 prepare。后续仍须依次完成：

1. formal R2O PASS，并按事先确定的主节点规则冻结唯一 bundle；
2. 用实际 passing bundle 生成 portable receipt，固定相对路径和内容 hash；
3. 使用无 support 的 replay/reference，在两台机器运行 FP64 tail smoke；
4. smoke 同时通过 raw feature parity、graph-semantics hash、paired native-FP64 sensitivity、tail reference `E/F/H=0`、combined/separate Taylor 线性、order/MIC、恢复和完整 mechanics gate；
5. 独立复核并冻结代码、receipt、schema 和 smoke hash。

只有以上步骤全部通过，prepare 才可以一次性读取 support9 并生成 9 个 outer folds。任何 skeleton 或 smoke 结果都不能作为已经通过 support outer-LOCO、all9 final 或声子谱验收的依据。
