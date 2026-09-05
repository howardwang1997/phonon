# Graphene R2R 多极化背景条件化 Taylor-null 合同（R2R-0 review candidate）

## 1. 当前边界

R2Q 的四步 endpoint 已冻结，model-state SHA-256 为 `733e223805ed3cbc20b645615a8edbd8d60f3cd25898d7cf629848b0aee29d47`。R2R-0 只检查一个新的、零参数的条件化表示是否在几何、力学和线性可辨识性上值得进入后续 readout 审查。本阶段不拟合系数、不训练 encoder、不打开 E50 seed1、actual small-H、seed2 或 support，也不启动远端任务。

冻结 encoder 的梯度历史也写入 canonical provenance：R2O/R2Q encoder 更新只使用 thermal92 和 harmonic-zero32；E50 seed1 与 actual small-H 只在 R2Q endpoint 原子冻结后各评价一次，没有回写参数、方向或选择；seed2/support 从未打开。R2R-0 不更新 encoder。

允许输入只有 R2Q endpoint 的 checkpoint/receipt/marker、R2O 的 6×6/8×8 references、92 个 thermal train 结构和 32 个 harmonic-zero train 结构。七个文件的 SHA-256 已写入 canonical contract。public reference API 还绑定有序分数坐标、cell metric、元素和 PBC 的语义 fingerprint：

- 6×6：`2d9d97aa3a994f1bc4db0965a1837d46269b0f588fc19d8c96308848c58ef6ea`；
- 8×8：`c433781cb5b3270dcd6f78f1ef7af0a42f978a67219416dd15051795aede431e`。

这样可以允许整个 cell 和 positions 做同一个 proper/improper rotation，同时拒绝悄换展开、原子顺序或背景原点。

fingerprint 在分数坐标和 cell metric 各自舍入到 10 位小数后，把所有
IEEE 比较等于零的值显式写回 `+0.0`，再进入 JSON。这样 V100/BLAS 对
同一刚体变换产生的 `-0.0` 不会改变语义 SHA；任何非零坐标、cell、元素、
PBC 或原子顺序篡改仍然 fail-closed。该规范化只作用于 fingerprint
payload，不改变 production 几何、图或能量/力计算。
规范化严格发生在 `round(..., 10)` 之后，仅把 exact-zero 改成 `+0.0`；
任何非零值都不截断或钳制。

## 2. 固定 6 Å 背景

对 reference-order 节点 `i`，固定 reference 几何内 `r_ref<6 Å` 的 39 个邻居。背景权重是 degree-5 reverse smootherstep：

`w(x)=1-10x³+15x⁴-6x⁵,  x=r_ref/6<1`，cutoff 外为零。

这是 R2R 背景自己的 quintic smootherstep，不是 R2Q MACE `r_max=3.2 Å` 中 degree-7 的 `PolynomialCutoff(p=5)`。两者不能混用。

中心 `i` 以 weight 1 纳入局部样本。令 `W_i=1+Σ_j w_ij`，中心和邻居的归一化权重分别为 `1/W_i` 和 `w_ij/W_i`。对 live MIC-aligned displacement `u`，先计算中心在内的加权均值 `μ_i`，再计算

`C_i = [ (u_i-μ_i)(u_i-μ_i)^T + Σ_j w_ij (u_j-μ_i)(u_j-μ_i)^T ] / W_i`。

其中 `s_i=tr(C_i)`。多极化量直接按中心在内的所有 unordered sample pairs 求值：

`b_i = Σ_{j<k} p_ij p_ik ||(u_j-μ_i) × (u_k-μ_i)||² ≥ 0`。

这里没有额外 `1/2`。不使用 `0.5[(tr C)²-tr(C²)]`，避免 rank-1 附近的消减误差。两个平滑条件量冻结为

`a_i=-expm1[-b_i/(2.5e-8 Å⁴)]`，`c_i=s_i/(s_i+0.015 Å²)`。

只用允许的 train geometry 重算得到：thermal92 的 `min/median b=1.8338961456148413e-6 / 2.6026297280e-5 Å⁴`，nodewise median `s=0.014458039461185433 Å²`；harmonic-zero32 的 direct-pair `max b=2.2896187649234325e-34 Å⁴`。`s0` 是 thermal median 按 `0.005 Å²` 网格一次取整的 `0.015 Å²`；`beta=2.5e-8 Å⁴` 是 `{1,2.5,5}×10^n` 网格上满足 `min thermal b/beta≥50` 的最大值。geometry-only derivation SHA-256 为 `c97e5ea994ac0d034cb2c039daff6c01041b4efd8dd612d1b47dbcd6b8681211`，没有读取力标签。

背景自身 reference 半径为 6 Å。R2Q 第二 interaction 的 node scalar receptive radius 为 6.4 Å，因此组合 per-node scalar 的有效半径冻结为 6.4 Å，最大 interaction diameter 为 12.8 Å。`ReferenceNeighborhood.validate` 会逐 receiver 检查 sender 唯一，拒绝同一 source 的多个周期像；还会直接枚举 reference cell 的面内晶格平移。6×6/8×8 的最短面内平移分别为 14.76/19.68 Å，相对 12.8 Å 直径的余量分别为 1.96/6.88 Å。

## 3. 唯一 65 参数口径与部署组合

冻结 R2Q endpoint 对每个节点提供：

- ScaleShift per-node interaction energy `ε_Q,i`；
- 两个 interaction products 各 16 个 signed `0e` channels，合计 `φ_i∈R³²`。任何 `0o` channel 直接拒绝。

旧候选把 fixed carrier 写成 `T2null[Σ_i a_i ε_Q,i]`。这个公式已经判定为 NO-GO：reference 附近 `a_i=O(u⁴)`，因此外层 Taylor-2 counterterm 为零；thermal 上 `a≈1` 时得到的是 raw `Σ_i ε_Q,i(x)`，不能重放冻结的 R2Q whole-energy Taylor tail。只用 thermal92 geometry 的负回归诊断中，global index 0 的旧 carrier 与冻结 R2Q tail 的最大力差为 `0.5252 eV/Å`；全 92 构型 RMS 为 `0.11753 eV/Å`。这些量没有读取 force labels，只用于拒绝旧公式，旧接口不进入 production。

唯一 fixed carrier 改为逐节点 Taylor-null 后再乘 live weight：

`r_i(x)=ε_Q,i(x)-ε_Q,i(x0)-g_i(x0)·u-½uᵀH_i(x0)u`，

`G0(x)=Σ_i a_i(x) r_i(x)`。

实现不缓存逐节点 Hessian。令 `Eref_w=dot(a_live, ε_Q(x0))`，保持 `a_live` 与当前 `x` 的 autograd 连接，但把它视作 reference variable `x0` 的常数；随后计算 `g_w=∂Eref_w/∂x0` 和 `Hv_w=∂g_w/∂x0·u`。于是

`G0=dot(a,ε_Q(x))-dot(a,ε_Q(x0))-g_w·u-½u·Hv_w`，

严格等于 `Σ_i a_i T2null[ε_Q,i]`，并保留当前几何上 `da_i/dx` 的乘积法则。synthetic explicit-node 对照的 E/F/full-H 最大差小于 `1.2e-15`；真实 reference6 和 thermal92 index 0 上，corrected carrier 与冻结 R2Q whole Taylor tail 的 E/F/full-H 分别满足 `1e-10 eV / 1e-9 eV/Å / 1e-7 eV/Å²`。

完整条件化标量只有以下定义：

`G_R2R = G0 + Σ_i a_i [w0·φ_i + c_i (b1 + w1·φ_i)]`。

65 个后续可拟合参数和列顺序固定为：

1. columns `0:32`：`Σ_i a_i φ_i,k`，对应 `w0`；
2. column `32`：`Σ_i a_i c_i`，对应唯一 bias `b1`；
3. columns `33:65`：`Σ_i a_i c_i φ_i,k`，对应 `w1`。

没有 global intercept。零 readout 初始点是 `w0=b1=w1=0`，fixed carrier `G0` 保留。65 个参数项全部含 `a_i`；direct unordered-pair 定义给出 `b_i=O(u⁴)`、`a_i=O(u⁴)`，而 `c_i=O(u²)`，所以这 65 项各自的 reference value、Jacobian 和 Hessian解析为零，production design 可直接计算 raw 65-vector 及其当前位置 Jacobian。这个 zero-jet 优化严禁用于 fixed carrier。

R2R 是 R2Q Taylor-tail 的条件化 replacement。部署组合只能是 `frozen foundation + G_R2R + frozen q6`，不能再加一次旧 R2Q tail。thermal 上所有 `a_i=1` 时，线性求和使 `G0` 严格等于冻结 R2Q whole Taylor tail；exact rank-1 上 `a_i=0` 时回到 zero-tail baseline，不发生双计数。

zero-jet 数值门不使用少数 signed contractions 掩盖节点间抵消。6×6/8×8 都逐节点验证 global `a_i` value 和完整 Jacobian exact zero。随后每个节点从 production receiver 取 center+39 个 unique sender 和实际 normalized q，按 production sender order 建立独立 local direct-pair evaluator；在 fresh `z[40,3]=0` 图上计算 value、Jacobian 和完整 `120×120` Hessian，算完即释放。两种 reference 的 sender-order hashes 分别为 `4f0fcf5acf9669588bf7438fdb9eaf0b00dcff5c62c528b8d7637ffac3b744db` 和 `6e44497aa2b55fc433c8155ce08f9f9b2695d9adcf9b298046dfd291958ecc56`。q signature 直接绑定 center q、sorted neighbor-q multiset 的未舍入 little-endian FP64 bytes，以及 sender multiplicity/unique flag，不做容差聚类；6×6 有 71 个 exact signatures 覆盖 72 节点，hash 为 `19ca084d9f54c17864d059bffc2da71e290e87bd8bab9dbea4ac6395441d4440`；8×8 有 123 个覆盖 128 节点，hash 为 `b6cec9b3886a71c076c715daeaf2dbbc48a42986a7c0b76c0becc4a144a0fdb5`。两者所有 local value/J/full-H 最大值均为 0。固定 nonzero local base 按 direct b 缩放到 `b/beta=1`，再与 production selected `a_i` 和 mapped gradient 比较；value 差不超过 `1.11e-16`，gradient 差不超过 `3.56e-15 Å⁻¹`，local 外 global gradient 为 0。该审计 6×6/8×8 的观测墙时分别约 `0.98/2.00 s`，进程 `ru_maxrss` 原始值约 `592/584 MB`。

public node-observable API 每次调用都重新计算完整 `state_dict` semantic SHA，并要求等于 `733e2238...29d47`；不能只在 loader 或架构检查时验证一次。测试会原位修改一个参数并确认 API 立即拒绝。实现还逐值验证 `Σ_i ε_Q,i` 与正式 `mace_interaction_energy` 的 energy 和 position-gradient parity。

### 3.1 唯一 production 边界

正式 design 与 mechanics 共用同一个私有、hash-bound production context。`production_linear_design_query` 在内部完成 endpoint state、reference content、source↔reference assignment、周期像、current edge set 和 formal R2O graph 的复核；用同一个 live、reference-order 的 `x-image_shift-x0` 同时计算 R2Q observables 与多极化背景，并返回 source-order 的 corrected fixed-offset force column 和 65 个 parameter force columns。65 列只对 raw zero-jet parameter vector 求当前位置 Jacobian，不为每列重复 reference HVP。真实 thermal0 的 raw-65 forward/force-Jacobian 观测墙时为 `0.084/1.150 s`，shape 为 `216×65`，进程 RSS 约 `582 MB`。formal runner 不得退回手工拼 primitives。

`rigid_transform_probe` 的 public 原始输入协变采用 exact gate：调用者传入的 reference 和 live structure 的 positions/cell 必须逐数组等于 hash-bound baseline template 右乘 `Q.T` 的结果，因此 `1e-12` 的原始 positions 扰动会在 public 边界直接拒绝。ASE assignment/adapt/reorder 内部矩阵运算跨 BLAS 不保证 bitwise exact；这些派生 ordered live positions/cell 与 adapted reference positions 使用唯一命名的 machine-level 绝对门 `RIGID_INTERNAL_COVARIANCE_ATOL_A=1e-12 Å`（`rtol=0`），并在 receipt 记录三项 max 和通过布尔，不再使用旧 `2e-8` 容差。

formal graph 固定为 PyTorch default dtype 为 `float32` 时构造、随后按 endpoint dtype 使用的 R2O 语义；production 入口在 default dtype 改为 `float64` 时直接拒绝。未变换 6×6/8×8 baseline graph 的 semantic SHA-256 分别是：

- `3a0c26908a936042bda6f7dcaf7b2520e9a9f14373c8c61561bd6040c552256e`；
- `4728c7542ea0a89e3f0ca451f2bd5c87595207086e3a4f6e26c099571ff239ba`。

`baseline` mode 必须逐字节命中相应冻结 hash；builder 或 supplied graph 任一 tensor 被改动就拒绝。proper/improper O(3) 同步变换只能使用 v4 `rigid_transform_probe` mode。该模式先从 `baseline_reference_template` 重建并逐字节验证冻结 baseline graph，再把 baseline graph 的 `positions`、`shifts`、`cell` 以 endpoint dtype 右乘 `Q.T`；`edge_index`、`unit_shifts`、`node_attrs`、`batch`、`ptr`、`head`、`pbc` 保持逐字节不变。identity `Q` 必须精确重放 baseline graph hash。外部 supplied graph 只能逐 tensor 等于这份 covariant derived graph。

同一入口还要求显式传入 `baseline_structure_template`。live structure/reference 必须分别等于 baseline structure/reference 的同步 `Q` 变换，且 transformed 与 baseline 的 `reference_to_source`、`source_to_reference`、`image_integer_reference_order` 三组 assignment/MIC 数组逐字节相同。receipt 直接保存 baseline structure、baseline reference 的 full geometry semantic SHA、baseline assignment hash、Q hash，以及 baseline/derived graph hashes，不能只依靠 transformed geometry 反推 provenance。

背景不从 rotated native reference 重建 physics graph。rigid mode 直接复用 baseline `ReferenceNeighborhood` 的 receiver/sender/image topology、reference distances、quintic weights 和 normalization；live centered displacement q 向量随同步 structure/reference 旋转。native rotated background 只用 sorted physical edge identity-key set 核对拓扑等价性；只有该 exact key set 不等才拒绝。MACE native graph rebuild 也采用同一 exact physical edge identity-key set veto。两条 diagnostic 中的 vector/length、distance/weight/normalization 和 shortest-cell 差值只记录，不使用隐式数值阈值授权或拒绝。native rebuild 还保留为明确的 `diagnostic_only` receipt/API，用来记录 neighbor-list order变化、逐 key Cartesian 差、matched physical-edge multiset及 covariant-vs-native combined E/F；它不能成为 formal O3 gate 的输入。

rng=83 的真实 thermal92 index 0 / reference6 CPU regression 中，旧 native rebuild 的 proper/improper combined force covariance误差分别为 `2.67009836e-5/1.94847276e-5 eV/Å`，超过固定 `1e-5` 门槛。v4 covariant graph 下，proper 的 fixed/parameter/combined force误差为 `3.68e-14/5.82e-14/6.36e-14 eV/Å`，improper 为 `4.72e-14/6.50e-14/8.48e-14 eV/Å`；combined energy误差分别为 `5.33e-15/1.24e-14 eV`。该修复不改变 baseline 6×6/8×8 graph hash、thermal shard、系数或验收门槛。

所有 mechanics probe 固定使用 `p=linspace(-0.2,0.3,65,float64)`，其 JSON-list semantic SHA-256 为 `656ca438f398dc4373456439322d2c94883789e2f97ac1bbcad66b5bd5713b04`。组合必须调用 query 的 `canonical_combined_probe()`，评价 actual `fixed_offset + X@p`；完整 Hessian 只能调用 `production_combined_energy_force_hessian`。该入口先在同一个 context 内构造 corrected `G0` 与 raw 65-vector，再把 `[1,p]` 合成一个标量，只对这个标量计算一次 source-order F、full H、H antisymmetry 和 translation ASR。旧的 66 列 `create_graph=True` 后再求 full-H 路径已禁止；本地旧实现曾因内存放大以 exit 137 终止。reference 与 thermal92 E50 seed0 global index 0 是 node-energy parity probes；后者 absolute structure semantic SHA-256 为 `e64c2c5cc7681f710f01a22e9a374cf900c435387de16edfbb2037b025d941b9`。

wrap/order-MIC 分成两层。使用 frozen formal graph 的量化 cell 做 wrap 时，raw design 必须 exact replay；使用 archive native ASE cell 做 wrap 时，native cell 与 float32-origin graph cell 的末位差会传入 raw columns。因此 formal receipt 记录 raw-65 max-abs/relative 作为 diagnostic，但只允许组合 `fixed_offset+X@p` 的 `ΔE≤1e-6 eV`、`max|ΔF|≤1e-5 eV/Å` 决定通过。当前真实 endpoint/6×6 production smoke harness 得到 `ΔE=3.3490e-11 eV`、`max|ΔF|=4.6276e-9 eV/Å`；raw-65 force-design 的 max-abs/relative 为 `1.06381e-9 eV/Å / 4.77453e-9`。这组数只验证 production mapping/receipt 路径，正式 actual Taylor mechanics 仍须由后续 formal runner 按同一 gate 重算。

## 4. R2R-0 rank/condition 预检

R2R-0 的 force design/scaler/rank 只接收 92 个 thermal structures。harmonic-zero32 不进入 design scaler、SVD、OOF 或任何拟合；它单独记录 direct `b/a` 和 actual Taylor-remainder `E/F`。rank-1 receipt 的固定门槛是 `max b≤1e-20 Å⁴`、`max a≤1e-12`、`max |E_rem|≤1e-10 eV`、`max |F_rem|≤1e-9 eV/Å`。不能手工把 harmonic design 置零后声称解析通过。

65 个 thermal force-design columns 只除以各自 train RMS，不减均值。若任一 column RMS 不超过最大 column RMS 的 `1e-12`，即作为 zero/near-zero fail-closed 诊断；不删除该列后继续拟合。R2R-0 预检通过定义为：65 列全部非零、RMS-scaled rank 为 65、scaled condition number `≤1e8`。即使通过，也只说明表示的 train-side 线性可辨识性，不授权拟合或训练。

若后续另立 R2R-1 合同，ridge 只能作用于新增 65 个 residual coefficients，读取 thermal92 force gap。readout OOF 固定为四折连续切片，每折完整 holdout `5 E50-seed0 + 9 T300 + 9 T600`。harmonic-zero32 不进入这些 folds。由于冻结 R2Q encoder 的上游训练已见过相关语料，这只能解释为 conditional linear-readout OOF，不是 encoder-level 独立验证。

## 5. 不变性和适用范围

formal runner 不得自定或覆盖门槛。固定数值为：

| gate | threshold |
|---|---:|
| thermal92 `min b / min a` | `1e-6 Å⁴ / 0.99` |
| harmonic-zero32 `max b / max a` | `1e-20 Å⁴ / 1e-12` |
| harmonic-zero32 Taylor remainder `max E / max F` | `1e-10 eV / 1e-9 eV/Å` |
| R2Q node energy-sum / position-gradient parity | `1e-10 eV / 1e-9 eV/Å` |
| corrected carrier vs frozen R2Q `E / F / full-H` | `1e-10 eV / 1e-9 eV/Å / 1e-7 eV/Å²` |
| parameter zero-jet all-node `a / J / local full-H` | exact `0 / 0 / 0` |
| nonzero local↔production `a / mapped gradient` | `1e-14 / 1e-12 Å⁻¹`；`|b/beta-1|≤1e-12` |
| proper/improper O(3) energy / force covariance | `1e-6 eV / 1e-5 eV/Å` |
| translation energy / force | `1e-9 eV / 1e-8 eV/Å` |
| permutation energy / force | `1e-6 eV / 1e-5 eV/Å` |
| native-cell wrap/order-MIC combined probe energy / force | `1e-6 eV / 1e-5 eV/Å`；raw 65 列只作 diagnostic |
| force finite difference / force-Hessian finite difference | `1e-5 eV/Å / 1e-5 eV/Å²` |
| nonreference full-H antisymmetry / translation ASR | 各 `1e-7 eV/Å²` |
| localized 6×6↔8×8 full remainder total E / max F | `1e-7 eV / 1e-5 eV/Å` |
| reference remainder E / F | `1e-10 eV / 1e-9 eV/Å` |
| reference H max / antisymmetry / translation ASR | 各 `1e-7 eV/Å²` |
| reference Γ/K frequency drift | `2 cm⁻¹` |
| thermal design | 65 列全 active、rank 65、scaled condition `≤1e8` |
| quintic cutoff `r→6 Å⁻` value / first / second derivative | `1e-12 / 1e-11 Å⁻¹ / 1e-10 Å⁻²` |
| quintic cutoff `r→6 Å⁻` third derivative magnitude | `>1e-6 Å⁻³` |

R2R-0 tests 覆盖：

- degree-5 smootherstep 端点及中心在内的归一化；
- direct-pair `b≥0`、thermal/harmonic train geometry 统计和 rank-1 gate-off；
- proper/improper O(3) energy/force covariance、translation、source permutation、order/MIC；
- canonical `fixed_offset+X@p` mechanics、native-cell wrap sensitivity receipt、float32-origin exact replay；
- corrected live-weighted node carrier的 synthetic explicit-node E/F/full-H 等价、真实 R2Q E/F/full-H parity和旧 carrier负回归；
- 65 参数项的解析 `O(u⁴)` 证明、all-node global value/J exact zero、6×6/8×8 每节点 local direct full-H，以及非饱和 `b/beta=1` local↔production value/gradient parity；
- 6×6/8×8 localized background field parity和 6 Å locality；
- R2Q per-node ScaleShift energy/position-gradient parity；
- 65 列顺序、RMS-only/no-mean scaling、relative zero-column 和 rank/condition 逻辑；
- reference fingerprint、baseline/rigid-transform graph modes、default-float64 拒绝、graph/endpoint tamper fail-closed、held/support path fail-closed 和连续 thermal OOF folds。

归档 6×6/8×8 XYZ 的末位坐标有约 `1e-8 Å` 级打印差，field-level size parity 只作 implementation diagnostic。以后 deployment 的权威 size gate仍须评价相同 localized perturbation 的完整 Taylor-remainder total energy 和 max force，不能由 field parity 替代。

任何 exact rank-1 displacement field都解析满足 `b=a=0`。R2R-0 实际只用 harmonic-zero32 验证这条分支；held small set保持未打开。旧 full25 中只有 rank-1 subset 可以使用这条解析结论，其余 13 个大振幅 rank-3 structures 只作 report，允许 `a>0`，不能用作 null gate。

## 6. 当前实现状态

- 模块：`scripts/smearing_kink/graphene_r2r_multipolar_background.py`，SHA-256 `4db3665f945783086de5c7a2f6262843bbe7d03df2a32f9153e450200b25064c`；
- tests：`tests/test_graphene_r2r_multipolar_background.py`，SHA-256 `1255c5720cb0fe575fe7edd9d211035799feccebd922488bc33ebe20f7c36c08`；
- canonical contract SHA-256：`e589497c7b9f6a9d4ff0cdcc434cf07804fa0fd732d791dba811d4c7e1c5aecc`；
- 常规本地测试：`22 passed, 2 skipped`，包括 actual endpoint 的 identity/rng83 proper/improper v4 O3、fixed/parameter/combined 分解、native rebuild negative diagnostic 和 tamper fail-closed；同时通过 `py_compile` 和 `git diff --check`。

当前状态是 v5 public-input exact/internal-machine-tolerance hardening 后的 review candidate。formal runner/aggregate/launcher 已实现但没有 GO marker，也没有启动 attempt2；尚未生成正式 R2R-0 design matrix/rank receipt，不能称为 R2R 表示通过，更不能进入 tail、SSCHA 或正式声子谱。
