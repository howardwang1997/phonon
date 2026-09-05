# Graphene R2P gate-aligned 梯度可行性审计合同（2026-08-25）

## 1. 目的与适用范围

R2O 的固定 epoch-240 EMA checkpoint 没有通过正式 core gate：E50 seed1 的 A′ 投影 RMS 为 `25.106 meV/Å`，高于 `15 meV/Å`；small-H 最大力误差为 `16.647 meV/Å`，高于 `10 meV/Å`。与此同时 E50 总力和斜率已通过。这说明继续按原损失延长训练不能直接回答模型是否存在同时改善 A′、热力总力和小振幅约束的一阶方向。

R2P 只做一次零优化步的梯度可行性审计。它在冻结的 R2O epoch-240 EMA 状态上计算逐构型、与 gate 对齐的参数梯度，并求这些单位梯度的 max-min common-descent cone。R2P 不训练、不更新参数、不扫描权重、不读取 seed2/support，也不授权 postcore、tail 或下一轮训练。

## 2. 固定输入与数据隔离

输入必须来自已经失败且明确不授权部署的同一个 R2O formal run：

- `core_checkpoint_gate.json` 状态必须为 `R2O_core_checkpoint_gate_failed`；
- 该 gate file 必须逐字节匹配冻结 SHA-256 `924b13de54d2a918c16845524289036e117006e48ee454f364f442fd88ee8898`，不能只相信 gate 内部自报的 input/metric 字段；
- 在任何 train/reference/bundle 打开或哈希前，formal root 必须同时满足：`EXIT_CODE` 精确为 `0\n`，`DONE` 精确为 `R2O_core_checkpoint_gate_failed\n`，空 marker `PRETRAIN_DONE/TRAINING_DONE/CORE_GATE_FAILED` 存在，`RUNNING/FAILED/PASSED/CORE_GATE_PASSED/SMOKE_PASS/SMOKE_FAILED` 全部不存在；failed/passed marker 必须落在 failed XOR arm；
- diagnostic bundle、training freeze、stage2 contract/runtime、data manifest、live/frozen wrapper 必须逐项 SHA-256 一致；
- checkpoint 固定为 epoch-240 EMA，semantic state SHA-256 为 `09c29479be6c8f207211614c2b3da20f09aba0702092a2b1bf8edab5a5e6d236`；
- primary 阶段只打开 92 个 thermal train、32 个 small-zero train 和 6×6/8×8 pristine reference；
- 任意输入或输出路径在存在性检查、打开或哈希前，逐路径分量拒绝 symlink、`..` 以及 `seed2|support|reserved|outer_fold` token；
- E50 seed1 和 12 个 actual small-H 只能在 primary certificate 已原子落盘并重新核对 SHA-256 后打开；它们不能参与 primary 梯度、方向或状态选择。

### 2.1 冻结构图语义

R2O trainer/evaluator 进程没有在 wrapper 内改变 PyTorch 默认 dtype。`AtomicData` 在 process default `torch.float32` 下构图，然后所有连续 graph tensor 转为模型的 `torch.float64`。R2P 必须复现这一语义：入口和每个实际 graph evaluate 前均要求 `torch.get_default_dtype()==torch.float32`，不允许调用 `torch.set_default_dtype`。模型参数、target、loss 和 autograd 仍为显式 FP64。

这与 native-FP64 graph construction 不同。R2P protocol 和 formal contract 同时保存该 graph semantic 的 canonical SHA-256；不能用 native-FP64 screening 数值替代正式 R2P 结果。

## 3. Primary hard set：176 个逐构型目标

所有目标梯度均在同一 42,096 参数的 EMA240 模型状态上计算。正梯度 `g_i=∇L_i` 经过逐块 L2 归一化；common direction `d` 表示更新时使用 `θ←θ−ηd` 的下降向量。每块的正比例系数不会改变归一化方向，但仍固定并记录，以便复算原始 aggregate loss。

### 3.1 20 个 E50 seed0 A′ complex projector

每个 E50 seed0 构型单独形成一块：

`L_A,i = |<mode_i, F_pred−F_REF>|² / (20 × (0.015 eV/Å)²)`。

实部、虚部、模长、mode norm、构型索引和 gradient norm 均逐项报告。20 个构型同时也各自进入下面的 total-force hard set。

### 3.2 92 个 thermal total-force

20 个 E50、36 个 T300 和 36 个 T600 各自形成一块：

`L_T,i = c_role × mean_j(F_pred,j−F_REF,j)² / (0.030 eV/Å)²`，

其中 `c_E50=0.50/20`，`c_T300=0.20/36`，`c_T600=0.20/36`。primary certificate 保留全部 92 个约束，不能用一个 aggregate thermal gradient 替代。

### 3.3 32 个 small-zero RMS

每个 128 原子的 exact-zero anchor 单独形成一块：

`L_Z,i = mean_j F_tail,j² / (32 × (0.0005 eV/Å)²)`。

这里 `F_tail` 是 R2O whole-energy Taylor2 remainder 的 source-order force；target 严格为零。

### 3.4 32 个 small-zero exact top-component

small-H gate 中 REF 最大分量为 `6.41582 meV/Å`，因此为使 combined error 保持在 `10 meV/Å` 内，zero-correction 的保守余量固定为

`m = 10−6.41582 = 3.58418 meV/Å`。

每个 anchor 的 hard objective 为

`L_M,i = max_j |F_tail,j|² / (32 × m²)`。

从 detached FP64 force 中按 flat source-order index 稳定选取 top 分量，再从 live tensor 的固定 index 构造平方目标。artifact 必须记录 top/second signed 和 absolute value、flat index、atom、Cartesian component、absolute/relative gap。

- exact tie 只按 detached FP64 absolute value 的严格相等判定，数值容差为零；只要 exact-active count 不等于 1 就 fail closed。单个平均次梯度不能证明 nonsmooth max 的全部 active branches 都下降；
- 在唯一 top 情况下，top 与 second 的 gap 若不大于 `1e-8 eV/Å`，审计仍 fail closed，不能用数值不稳定的单分支近似给出 hard certificate；成功记录必须包含 top 和 second 各自的 flat index、atom、Cartesian component、signed/absolute value 与 sign。

### 3.5 smooth-L∞ 仅作报告

另算 `tau=1 meV/Å` 的 shifted log-mean-exp，便于评估训练友好的 surrogate，但它不是 exact max 的上界。每个 small 构型有 128×3=`384` 个分量，最大低估界为

`tau ln(384) = 5.95064255 meV/Å`。

因此 smooth-L∞ cone 只能写入 report-only artifact，不能替代上述 32 个 exact-top hard blocks，也不能授权方向。

## 4. Common-descent cone 与数值证书

令 `u_i=g_i/||g_i||₂`，R2P 求

`t* = max_{||d||₂≤1} min_i u_i·d`。

其 simplex dual 为

`t* = min_{λ≥0, 1ᵀλ=1} ||Σ_i λ_i u_i||₂`。

固定算法如下：

1. 先验证原始 unit Gram `K=UUᵀ` 的 diagonal、symmetry 和 PSD，容差分别为 `1e-12`、`1e-12` 和最小特征值 `≥−1e-10`。只有检查通过后才允许数值对称化；禁止用填充对角线掩盖异常。
2. 主求解器为 simplex 上的 deterministic Frank–Wolfe：初始 vertex 固定为 index 0，linear-minimization tie 取最低 index，使用精确二次 line search；最多 200,000 次，内部 stationarity gap 容差 `1e-12`。
3. 以 uniform simplex 为初值运行 analytic-gradient SLSQP，作独立 cross-check。两个可行解中选择二次目标更低者；artifact 分别记录两者 objective、gap、success、iterations，并明确 selected solver。selected `iterations/converged` 必须对应被选中的 solver。
4. 对所选 `λ` 报告 simplex sum residual、minimum weight，以及 KKT slack `Kλ−||Uᵀλ||²1`。active slack 应接近零，inactive slack 应非负。
5. 构造 `d=(Uᵀλ)/||Uᵀλ||`，逐块重算 cosine；报告 primal minimum cosine、dual norm 和显式 cosine duality gap `dual_norm−primal_min`。

科学状态只有三种：

- `GO`：`primal_min>1e-3`，且 cosine duality gap 和 KKT 最大违反都不大于 `1e-3`；
- `NO_GO`：存在经过 simplex 可行性检查的 dual point 且 `dual_norm≤1e-3`，从而给出可靠的 `t*≤1e-3` 上界；
- `NUMERICAL_INCONCLUSIVE`：其余情况。求解器未充分收敛不能写成科学 NO-GO。

任一 gradient norm 不大于 `1e-14`、非有限值、Gram 审计失败或 exact-top exact/near tie 都直接 fail closed，不发布 primary 科学状态。

## 5. 落盘顺序与可复算链

formal output 必须是全新空目录。固定顺序为：

1. 写 `RUNNING`；
2. 复制当前 R2P script snapshot 并核对 byte SHA-256；
3. 在任何 parameter gradient 前写入 contract，包含全部 input hashes、protocol/hash、graph semantic/hash、parameter schema、runtime/software fingerprint；
4. 计算 176 个 gradients，复核 model state before/after 相同且所有 `.grad` 仍为 `None`；
5. 原子写入 gradient matrix、normalized Gram、common direction 和 `primary_certificate.json`；
6. 写 `PRIMARY_CERTIFICATE_FROZEN`，内容为 primary file SHA-256；
7. 重新核对 primary hash 后才打开 E50 seed1 与 actual small-H；
8. seed1 和 actual small-H 只做固定指标，并以 `1e-7 meV/Å` 绝对容差逐 aggregate/per-config 核对 frozen formal gate；这一步同时检查 float32-origin graph 语义，没有通过则整体 fail closed；
9. 另写 smooth-L∞ 和 actual-combined 0.5/10 report-only cones；不得重写 primary；
10. 最后写 `result.json`、`EXIT_CODE=0`、删除 `RUNNING` 并写 `DONE`。异常路径写 `EXIT_CODE=1` 和 `FAILED`。

矩阵 artifact 统一为 little-endian FP64 `.npy`，JSON 禁止 NaN/Infinity。result 明确保存 state hash before/after、optimizer 未实例化、optimizer step 未调用、seed2/support 未打开。

## 6. 报告项及其边界

Primary certificate 必须逐块保存 objective、gradient norm、common-direction cosine 和 dual weight。family aggregate 只能由逐构型梯度求和生成，用于与此前 aggregate screening 核对，不能替代 primary hard set。

Primary 落盘后的独立报告包含：

- E50 seed1 总力和 A′ RMS 的固定评价，不求梯度，不选择方向；
- 将 exact-top 替换为 shifted LME 的 report-only cone；
- 用 12 个 actual small-H combined error、`0.5/10 meV/Å` normalization 构造的 report-only cone。

这些 held/report-only 数值不能改变 primary certificate、primary direction 或三态结论。

## 7. 算力、预计时间与停止规则

正式审计只计划在 RTX 节点运行一次，预算小于 `0.1 GPU·h`。本地阶段只做脚本编译、合成 solver/autograd/path tests，以及 train/reference loader 合同检查；不在本地生成正式梯度结果。正式运行前需独立审查脚本、测试、合同和完整 SHA-256。

停止规则：

- 输入/hash/graph semantic/state/shape/role 任一不匹配：停止，不做梯度；
- exact-top exact/near tie、nonfinite/zero gradient、state 变化、`.grad` 被写入或 primary 原子落盘失败：停止，不打开 seed1/actual small；
- `NUMERICAL_INCONCLUSIVE`：只允许改进数值求解或复算同一冻结问题，不能训练；
- `NO_GO`：按 R2O 架构失败停止规则，停止当前局域 Taylor-null 表示，下一科学方向是显式 thermal-background-conditioned 表示；
- `GO`：只说明当前点附近存在覆盖 176 个 hard blocks 的一阶共同下降方向。下一步仍需另行冻结短程 gate-aligned constrained-loss 训练合同、步长和验收条件；R2P 本身不授权该训练。

## 8. 实现与当前本地验证

- 实现：`scripts/smearing_kink/graphene_r2p_gradient_feasibility.py`
- 合成测试：`tests/test_graphene_r2p_gradient_feasibility.py`
- 测试覆盖：176-block protocol、GO/NO_GO/hidden per-config conflict、primal-dual/KKT/simplex/Gram、SLSQP cross-check、exact top/tie/near-tie、smooth underestimation、formal-gate prediction parity、FP64 autograd 无 `.grad`/state mutation、严格路径和 report-only 打开顺序、原子 finite JSON、禁止 optimizer/backward/step/default-dtype mutation。

本地真实输入 preflight 只加载 train/reference 与 hash-bound diagnostic bundle，得到 92/32 个训练构型、72/128 原子 reference、42,096 个参数，state SHA-256 与预期一致。随后仅在 1 个 seed0 E50 和 1 个 small-zero anchor 上实际执行四条 autograd 路径，A′/total/RMS/exact-top gradient norm 分别为 `4.99496/2.31894/0.0388970/0.105032`；默认 dtype 保持 `torch.float32`，所有 parameter `.grad` 均为 `None`，state hash 前后均为 `09c29479...d236`。该 preflight 没有打开 seed1/seed2/support，也不是 176-block 正式科学结果。

当前状态仅为本地候选。独立 review 通过并由负责人明确授权前，不执行正式 RTX gradient audit。
