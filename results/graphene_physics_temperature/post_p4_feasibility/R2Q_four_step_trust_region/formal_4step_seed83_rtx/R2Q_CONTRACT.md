# Graphene R2Q finite trust-region 合同（2026-08-25）

## 1. 当前证据与唯一下一步

R2P 在冻结的 R2O epoch-240 EMA 状态上完成了 176-block、零优化步梯度审计。正式 primary certificate SHA-256 为 `54bf9056d8ac90641875f2dbe997336ffe909c3babf9a17440d6c2237791842c`，结论为 `GO`：

- 直接重算 `min_i U_i·d = 0.009918956481710891`；
- dual norm `0.009920772168250986`；
- cosine duality gap `1.81568654e-6`；
- active KKT slack 最大绝对值 `2.08571993e-8`；
- 176 个 cosine 全为正，最小值比 `1e-3` GO 阈值高 `0.00891895648`。

这证明 EMA240 附近存在一个同时降低 176 个 train-only hard objectives 的一阶方向，但没有证明有限步后仍保持共同下降，更没有证明 seed1 的 A′ 和 small-H gate 会通过。对正式 R2P direction 的有限步一阶估算还表明，单步预计只能把 A′ objective 降低约 `22%`，而当前 train-side 目标差距约为 `44.5%`；只做一步并立即打开 held，既不太可能达到目标，也会让 held 结果污染是否继续更新的决定。

因此唯一下一步事先固定为 **R2Q：连续完成恰好 4 个 train-only accepted trust-region steps**。每一步都在当前状态重新计算 176-block normalized common-descent cone，冻结该步 direction 后只沿这一方向做固定 Armijo/backtracking，并要求选中状态通过完整 reference gate；前 3 步不得打开 held，第 4 步还要通过只使用 reference 与固定 E50-seed0 train probe 的 pre-held implementation mechanics，之后才冻结唯一 endpoint。R2Q 不使用 optimizer，原始 EMA240 checkpoint 始终不可变，但会构造 4 个顺序依赖的显式 parameter states。它不是先做一步、看 held 后再决定是否继续的 A/B 流程。

每一步的方向、步长和接受判据都只由 train/reference 决定。第 4 步唯一 endpoint 落盘并 hash 固定后，seed1 和 actual small-H 各打开一次作最终报告。任一步不能产生可靠 cone 或没有有限 candidate 通过，或最终 endpoint 未通过正式 gate，都按 R2O 架构停止规则停止当前局域 Taylor-null 表示；不能追加第 5 步、更多 backtracking candidates、loss-weight 扫描或 held-driven 回退。

## 2. R2P 产物独立复算

独立复算脚本没有调用正式 R2P solver，而是直接读取落盘的 gradient matrix、direction、Gram、dual weights 和 JSON receipt：

- 脚本：`scripts/smearing_kink/audit_graphene_r2p_artifacts.py`；
- 结果：`results/graphene_physics_temperature/post_p4_feasibility/R2P_gradient_feasibility/epoch240_ema_exact_top_rtx_independent_recalculation/r2p_independent_recalculation.json`；
- 独立复算脚本 SHA-256：`8384cd02be0ab0b11e7514bc4aff3615c679dff4576cf4fc71664ed3c2377667`；
- 当前结果 SHA-256：`27f615fed81fd80d308b2f2e149652c9908b4ba10dcd158e2a976a17633361a0`；
- 复算状态：`PASS`。

独立检查得到：

- `||d||₂=0.9999999999999999`，由 dual combination 重建的 direction 与文件最大差 `4.42e-16`；
- 直接 `U Uᵀ` 与落盘 Gram 最大差 `1.83e-14`；
- 逐块 stored/direct cosine 最大差 `1.04e-16`，gradient norm 完全一致；
- simplex sum 精确为 1，86 个 dual weights 大于 `1e-10`；
- report-only JSON 中 primary hash 在前后均为 `54bf9056...1842c`，model state 仍为 `09c29479...d236`；两条 comparison cone 均保持 `can_authorize_primary_direction=false` 和 `can_authorize_parameter_update_or_training=false`；
- frozen formal gate prediction parity 最大差为 `4.73e-13 meV/Å`，远低于 `1e-7 meV/Å` 容差。

各 family 的直接下界和 dual mass 为：

| family | blocks | min cosine | active blocks | dual mass |
|---|---:|---:|---:|---:|
| A′ seed0 | 20 | 0.0099195980 | 11 | 0.0609307 |
| thermal total-force | 92 | 0.0099189565 | 47 | 0.5227857 |
| small-zero RMS | 32 | 0.0099199607 | 4 | 0.0390188 |
| small-zero exact-top | 32 | 0.0099189635 | 24 | 0.3772649 |

方向主要受 thermal total-force 和 exact-top 共同约束，不能把它简化成单独增加 A′ loss 权重。最大 dual weight 为 `0.0812641`，对应 `small_zero_exact_top:25`；其次为 `0.0664675` 的 `thermal_total_force:E50:12` 和 `0.0653322` 的 `small_zero_exact_top:20`。

## 3. R2Q 固定输入与隔离

R2Q 只接受以下 hash-bound 输入：

1. R2O failed formal receipt 和 EMA240 diagnostic bundle，model state 固定为 `09c29479be6c8f207211614c2b3da20f09aba0702092a2b1bf8edab5a5e6d236`；
2. R2P primary certificate `54bf9056...1842c`；
3. R2P gradient matrix file SHA-256 `566844a4dd4003b52560e1d2cb8ae1eb1d4b2e1011843b3e7aeefd9703885a38`，只作 step-1 parity anchor；
4. R2P common direction file SHA-256 `a6df2bc9395915d05739516e5bb7aba68bfdbcf6ff4673f3cdc837283e81d246`，只作 step-1 parity anchor；
5. 同一 92 thermal train、32 small-zero train 和 6×6/8×8 references；
6. frozen R2O wrapper 和 float32-origin graph、FP64 model/autograd 语义。

路径在任何 open/hash 前拒绝 `seed2|support|reserved|outer_fold`、symlink 和 traversal。4 个 train-only steps 与 endpoint 原子冻结全部完成前不打开 seed1 或 actual small-H；全程不打开 seed2/support。R2Q 不实例化 optimizer，也不调用 `optimizer.step()`；E/F/H、objective 和每步 cone 所需的 autograd 只用于求值。第 `j` 步的参数变化只允许为 `θ_j=θ_{j-1}−η_j d_j`，其中 `d_j` 必须由该步当前状态的 176-block cone 产生并在 candidate 评价前冻结。

## 4. 四步 schedule、步长与 tensor trust cap

参数和方向按 R2P parameter schema 展平。base state 实测

- `||θ₀||₂ = 216.54915896171457`；
- `||d_R2P||₂ = 1`。

每个 step `j=1,...,4` 都从该步当前状态 `θ_{j-1}` 和单位方向 `d_j` 独立定义初始 trial step：

`η_{j,0} = 10^-3 ||θ_{j-1}||₂/||d_j||₂`。

step 1 应复现 `η_{1,0}=0.21654915896171457`；后 3 步不能沿用这个数值，必须从各自实际 state norm 重算，但公式、global cap 和候选数量不变。

per-tensor relative L2 只作宽松安全 guard，不预先 clamp `η_{j,0}`：对每个 tensor 要求 `||ηd_t||₂/||θ_t||₂≤0.02`。任何 `||θ_t||=0` 且 `||d_t||>0` 的情况 fail closed，不临时引入 scale floor。step 1 的 `η_{1,0}` 最大 per-tensor relative update 约为 `1.55%`，低于 `2%` guard；后续 step 对每个 candidate 重新计算这一 guard。

每个 step 的 backtracking candidates 事先固定为

`η_{j,k} = η_{j,0} 2^-k,  k=0,...,5`。

step 1 即 `0.21654916, 0.10827458, 0.05413729, 0.02706864, 0.01353432, 0.00676716`。其最小候选的一阶最大 fractional objective change 仍约为 `2.17%`，避免用趋近于零的步长只验证连续性。每步都只能使用这 6 个相对候选；不能增加 k、改 global cap/per-tensor guard、提前少做一步，或按 seed1 表现重新定义 schedule。

## 5. 每步 cone 与 finite Armijo 验收

在每个 step `j` 开始时，从已 hash-bound 的 `θ_{j-1}` 重新计算与 R2P 定义完全相同的 176 个 train-only gradient blocks：20 A′、92 total-force、32 small-zero RMS、32 small-zero exact-top。normalized Gram、deterministic solver、cosine/KKT/duality-gap 检查和三态规则沿用冻结 R2P 合同；只有 `min_i U_i·d_j>1e-3` 且可靠 primal/dual gap、KKT、simplex/PSD 检查全部通过，才发布该步 direction。exact-top 在这里参与求梯度，因此 exact tie 或 `≤1e-8 eV/Å` near tie 仍按 R2P 规则 fail closed。

step 1 重算还必须满足以下 R2P parity：model state SHA 完全相同；176 个 block ID/顺序完全相同；每行 gradient norm 相对差 `≤1e-10`；每行 normalized-gradient cosine `≥1−1e-10`；新旧 common direction cosine `≥1−1e-10`；`min_i U_i·d` 绝对差 `≤1e-10`。dual weights 在非唯一 active face 上不要求逐元素相同，但新结果必须独立通过同一 duality-gap/KKT certificate。任一 parity 不满足即 `NUMERICAL_INCONCLUSIVE_AT_STEP_1`；R2P 文件不能直接替代本次重算。

该步每个 candidate 都从同一个 `θ_{j-1}` 重新实例化，设置

`θ_j(η_{j,k})=θ_{j-1}−η_{j,k}d_j`。

`θ_{j-1}` 和原始 EMA240 checkpoint 始终保持不变。对 candidate 只做 forward/force 评价，重新计算与该步 cone 完全相同的 176 个 exact objectives。令 `s_{j,i}=g_{j,i}·d_j`，固定 `c1=0.1`，每个 block 必须同时满足

`L_i(θ_j(η_{j,k})) ≤ L_i(θ_{j-1}) − c1 η_{j,k}s_{j,i} + ε_{j,i}`，

其中 `ε_{j,i}=10^-12 max(1,|L_i(θ_{j-1})|)` 只覆盖 FP64 数值舍入。不能用 family sum、平均值或 report-only smooth-L∞ 替代逐块判定。

candidate 还必须满足：

- 42,096 参数的 tensor 顺序、dtype 和 shape 不变；
- global relative L2 update 不超过 `1e-3`，每个 tensor 的 relative L2 update 不超过 `2e-2`；
- 32 个 exact-top objective 必须直接以 candidate 上的实际值 `max_j |F_tail,j|²/[32×(0.00358418 eV/Å)²]` 重算；`1/32` aggregate coefficient、correction margin 和 base/slope 定义必须与 R2P 完全相同，不能沿用 base argmax，也不能以 smooth-L∞ 代替；
- candidate 上只评价 exact-max 数值，不对 candidate exact-top 再求梯度，因此 exact tie 或 near tie 不构成拒绝条件；receipt 仍须记录 top/second 的 flat index、atom、Cartesian component、signed/absolute value、absolute/relative gap 和 exact-tie count；
- force/objective 全部有限；
- reference mapping、order/MIC 和 graph semantics 不变。

每个被实际评价的 candidate 还必须单独重算 6×6 pristine reference Taylor-remainder E/F，并保持 reference mapping、order/MIC 和 graph semantics。reference E/F 非有限或超过旧 null threshold 时必须立即给出 `NUMERICAL_INCONCLUSIVE_AT_STEP_j`，不能把它当作这个 candidate 的 Armijo/trust-cap 拒绝理由，也不能继续 backtracking 来寻找一个“能通过 reference”的步长。

在逐个 candidate 的 reference E/F invariants 均正常的前提下，每个 step 按 k 从小到大选择第一个、也就是最大的 176-block Armijo + trust-cap 通过步长，且一旦选中就不再检查更小步长。这个状态先标记为 provisional，不得因后续 reference 结果回退到更小步长。

每个 provisional selected state 都必须计算完整 reference gate：pristine-reference Taylor remainder E/F/full-H、H symmetry、translation ASR，以及 Γ/K frequency drift relative to frozen base；阈值见第 7 节。任一 reference 项非有限或超 threshold 都给出 `NUMERICAL_INCONCLUSIVE_AT_STEP_j`，不能归入 `TRAIN_NO_GO`，不能换用更小 candidate。reference 全部通过后，该 step 才成为 accepted。接受 step 1–3 后必须原子写入 chained recovery checkpoint、file/semantic state hashes、cone certificate、direction/hash、chosen `η_{j,k}`、176-block Armijo receipt 和完整 reference receipt；它们是恢复与审计产物，不是 held endpoint，也不得打开 held。下一步必须从上一 accepted checkpoint 精确加载并复核 hash。

只有 step 4 通过同样的完整 reference gate 并成为 accepted 后，才运行第 7 节列出的 pre-held implementation mechanics。这个 pass 的 nontrivial/O(3)/translation/permutation/FD/nonreference-full-H probe 必须固定为 hash-bound `E50-seed0 train[0]`；任何默认选择 `E50-seed1[0]` 的旧 helper 都必须显式覆盖或 fail closed，且 loader 要在打开路径前证明没有 seed1 token/source。任一 pre-held mechanics 检查非有限、超 threshold 或 probe/source 不符时，状态为 `NUMERICAL_INCONCLUSIVE_AT_STEP_4`，不能归入 `TRAIN_NO_GO`，也不能回退到更小步长或更早 step。

上述检查全部完成后，必须在打开 held 数据前原子冻结一套可重新加载的唯一 endpoint artifact，而不能只保存 semantic hash。artifact 至少包含：完整 step-4 model state/checkpoint 文件及其 file SHA-256、由实际参数和 buffers 重算的 semantic state SHA-256、base state/hash、四级 parent-state hash chain、4 份 cone/direction/dual certificate、parameter schema/hash、4 个精确 `η_{j,k}`、逐步 parameter update receipt、每步 176 个 before/after/Armijo margin、reference-null/pre-held-mechanics 结果、固定 seed0 probe identity/content hash 和输入/代码/环境 receipt。held evaluator 必须加载这一个实际 checkpoint 并复核 file/semantic hash；允许按四步 receipt 重放作 parity 检查，但不能以重算结果替代已冻结 artifact。没有通过的 candidate 不落盘。endpoint 必须保留 raw-MACE 直接部署禁令，且不能被称为正式通过模型。

## 6. R2Q four-step 三态与停止规则

- `FOUR_STEP_ENDPOINT_FROZEN`：4 个 steps 依次得到可靠 cone、Armijo/trust accepted candidate 和完整 reference PASS，step 4 还通过 seed0-only pre-held implementation mechanics，并已原子冻结唯一 endpoint；这是 train/reference 状态，不等于 held formal gate 通过；
- `TRAIN_NO_GO_AT_STEP_j`：step `j` 的 common-descent cone 得到可靠 `NO_GO`；或者 6 个 candidates 的 objective/reference 数值均有限、reference invariants 全部正常，但没有一个通过全部 176 个 Armijo constraints 和 tensor trust caps；
- `NUMERICAL_INCONCLUSIVE_AT_STEP_j`：输入/hash/state 不一致、cone 数值证书不充分、非有限值、candidate restore 失败，或任一 reference E/F/H、mapping/graph semantic、pre-held mechanics/probe 检查不通过。

任一步 `TRAIN_NO_GO` 都立即停止，且不打开 held；它表明当前 step 无法在事先固定的 common-descent/trust-region 合同内继续，不能扫描更多步长或减少总步数。`NUMERICAL_INCONCLUSIVE` 只允许修复同一计算，不能解释成物理 NO-GO。恰好完成 4 步后必须停止 train-side 更新；即使所有 train objectives 仍下降，也不能追加 step 5。

## 7. 唯一 endpoint 的一次 held 报告与旧 gate

只有 `FOUR_STEP_ENDPOINT_FROZEN` endpoint 的实际 checkpoint、file/semantic state hashes、完整四步 train-only chain 和 seed0-only pre-held mechanics receipt 全部原子落盘后，才能第一次打开 E50 seed1 与 12 个 actual small-H。两者各评价一次，不求参数梯度，不选择步长或 checkpoint，不再回写 four-step receipt 或 primary endpoint artifact。held report 必须独立落盘，并引用 endpoint artifact hash。

旧 formal mechanics 在 held 阶段必须再原样运行一次，其中 nontrivial probe 恢复为旧 gate 固定的 `E50-seed1[0]`；它不能复用 seed0-only pre-held probe 来声称旧 gate 通过。这个 seed1 probe 只能在 endpoint 已冻结后由 held evaluator 打开，其结果不能回写 endpoint、方向、步长或任何 train-side receipt。

最终沿用原 R2O formal gate 的全部固定科学阈值：

| gate | 固定阈值 |
|---|---:|
| E50 seed1 total-force RMSE | `≤30.0 meV/Å` |
| E50 seed1 total-force max | `≤200.0 meV/Å` |
| E50 seed1 A′ projected RMS | `≤15.0 meV/Å` |
| E50 seed1 A′ slope relative absolute error | `≤0.05` |
| 12 actual small-H force RMSE | `≤0.5 meV/Å` |
| 12 actual small-H force max | `≤10.0 meV/Å` |
| pristine-reference Taylor remainder energy | `≤1e-10 eV` |
| pristine-reference max force | `≤1e-9 eV/Å` |
| pristine-reference max Hessian | `≤1e-7 eV/Å²` |
| pristine-reference Hessian antisymmetry | `≤1e-7 eV/Å²` |
| pristine-reference Hessian translation ASR | `≤1e-7 eV/Å²` |
| Γ/K phonon drift relative to frozen base | `≤2 cm⁻¹` |

同一个 endpoint 在 pre-held 阶段用固定 `E50-seed0 train[0]` 完成下列 mechanics 检查，在 endpoint 冻结后的 held 阶段再用原 `E50-seed1[0]` probe 重跑旧 formal mechanics；两次门槛相同且都不因 R2Q 改写：

| mechanics gate | 固定阈值 |
|---|---:|
| reference raw-energy parity | `≤1e-10 eV` |
| reference raw position-gradient parity | `≤1e-9 eV/Å` |
| nontrivial-probe raw-energy parity | `≤1e-10 eV` |
| nontrivial-probe raw position-gradient parity | `≤1e-9 eV/Å` |
| O(3) proper/improper energy covariance error | `≤1e-6 eV` |
| O(3) proper/improper force covariance error | `≤1e-5 eV/Å` |
| translation energy error | `≤1e-9 eV` |
| translation force error | `≤1e-8 eV/Å` |
| permutation energy error | `≤1e-6 eV` |
| permutation force error | `≤1e-5 eV/Å` |
| force finite-difference error | `≤1e-5 eV/Å` |
| force–Hessian finite-difference error | `≤1e-5 eV/Å²` |
| nonreference full-H antisymmetry | `≤1e-7 eV/Å²` |
| nonreference full-H translation ASR | `≤1e-7 eV/Å²` |
| 6×6↔8×8 localized size energy error | `≤1e-7 eV` total |
| 6×6↔8×8 localized size max-force error | `≤1e-5 eV/Å` |

旧 full25 aggregate force RMS 参考值 `9.044673057081592 meV/Å` 和 max `200 meV/Å` 只作 report-only 对照，不参与 endpoint 通过/失败判定，也不能替代上表的 E50 seed1 和 actual small-H gate。

结果按以下边界解释：

- 全部 formal gates 通过：该唯一 endpoint 才可进入新的 postcore readiness 独立审查；R2Q 本身仍不授权 tail/support；
- 任一 E50 seed1 或 actual small-H scientific gate 未通过，而 reference/mechanics 全部正常：科学结论为当前 R2O four-step repair 未通过；停止，不增加 candidates 或 step 5，不回退到任一步已拒绝或更小的 `η_{j,k}`，不改变 endpoint，也不重复读取 seed1；
- 任一 held reference/mechanics/probe 检查非有限、超 threshold 或来源不符：状态为 `NUMERICAL_INCONCLUSIVE_HELD`，只允许修复并复跑同一个冻结 endpoint 的同一评价，不能解释成 `TRAIN_NO_GO` 或模型物理失败，也不能改方向、步长或 checkpoint。

held 结果在所有状态下都不得回调任一步 cone、direction 或步长选择；后续若考虑新表示，必须另立合同。

seed2/support 始终不打开。由于 held 数据只在唯一 endpoint 打开一次，它不能变成训练选择集。

## 8. 算力与时间

R2P 实测 primary gradient/certificate 为 `42.57 s`、峰值显存 `640.6 MiB`；完整 report-only 流程为 `374.47 s`。据此保守规划：

- R2Q 最多 4 次 176-block gradient cone、24 个 train-only forward candidates、4 次 selected-state full reference gate、1 次 step-4 seed0-only implementation mechanics 和唯一 endpoint held report：RTX 预计 `20–40 min`，`0.33–0.67 GPU·h`，显存 `<1 GiB`；每步 cone 和完整 reference 都重算，但不做 optimizer training；
- 双硬件 reproduction 只用于数值复现，不改变步长或状态选择。

当前只完成 R2Q 合同和 R2P artifact 独立复算。R2Q 尚未实现、prepare 或 launch。
