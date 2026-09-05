# Graphene R2O actual-tail 双节点 smoke 合同

日期：2026-08-25

状态：runner、双节点 evaluator 和 plan-only launcher 已完成本地实现。当前只用 synthetic portable formal receipt 和非零 mechanics tail 做了 CPU 集成测试；没有读取真实 support9 或 E50 seed2，没有生成 outer folds，也没有连接或启动远端节点。真实 formal bundle 是否可以进入双节点 smoke，需等本轮代码独立复核后另行决定。

## 1. 实现范围

新增文件为：

- `scripts/smearing_kink/graphene_r2o_tail_smoke.py`：输入/尾模型/环境合同 loader、完整 mechanics runner、节点结果验证、双节点聚合和 launch plan；
- `scripts/smearing_kink/run_graphene_r2o_tail_smoke.py`：单节点 CLI；
- `scripts/smearing_kink/evaluate_graphene_r2o_tail_smoke_dual.py`：双节点结果 CLI；
- `scripts/smearing_kink/launch_graphene_r2o_tail_smoke_dual.py`：只写 Tailscale argv 计划，不执行命令；
- `tests/test_graphene_r2o_tail_smoke.py`：synthetic portable receipt 的端到端测试。

smoke input 只接受正式 passing R2O receipt 所需的完整 gate/bundle/reference/hash 链，以及一个单独 hash 固定的非零 Taylor-null tail checkpoint。tail checkpoint 绑定 formal receipt、formal model state、descriptor schema、graph semantics 和自身 semantic state hash。checkpoint 明确为 `nonzero_pre_support_mechanics_smoke_only`，不可部署。

所有输入路径必须是 portable root 下无 symlink、无路径穿越的相对 POSIX 路径。路径中出现 `support`、`seed2`、`reserved` 或 `outer_fold` 会在模型加载前直接拒绝。运行中的本地 wrapper SHA 还必须等于 formal receipt 中的 wrapper SHA，避免只绑定模型权重却换了 Taylor 实现。

## 2. 运行环境指纹

runtime contract 固定以下语义字段并计算 canonical SHA-256：

- Python implementation/version；
- NumPy、PyTorch、mace-torch、e3nn、ASE、SciPy 和 torch-geometric 版本；
- PyTorch CUDA build、cuDNN version；
- default dtype、deterministic-algorithm 状态、float32 matmul precision、CUDA/cuDNN TF32 开关。

每个节点另行记录 platform、device type/index、GPU name、compute capability 和显存，并计算 execution-environment hash。双节点必须有完全相同的 semantic-environment hash；GPU 名称和 capability 可以不同。synthetic 模式允许 CPU 或 CUDA，真实 formal actual-tail 模式强制 CUDA。

## 3. 固定 mechanics protocol

smoke 只使用 formal receipt 内 hash 固定的 6×6 pristine reference，以及由代码按固定 seed 和固定数值生成的位移/旋转/原子置换，不读取外部 replay 或训练数据。复合模型为 formal R2O whole-energy Taylor remainder 与 tail whole-energy Taylor remainder 之和；内部 raw MACE 不作为直接预测器。

固定检查包括：

1. 6×6 reference 上复合模型的完整 `E/F/216×216 H`、Hessian antisymmetry、ASR 和 Gamma/K drift 上界；
2. 非零 tail 在普通位移 probe 上确实给出非零能量或力；
3. core/tail 分别 Taylor 消去与 combined raw scalar 一次消去的线性一致性；
4. energy-force 和 force-Hessian 两级中心有限差分；
5. proper rotation 和 improper reflection 的能量不变性与力协变性；
6. 全局平移，以及同时含 atom permutation 和跨晶胞 MIC image 的等价性；
7. 6×6 和 8×8 的 frozen graph geometry bound、actual `core+tail` paired native-FP64 feature/energy/force gate，以及 localized 6×6/8×8 Taylor-remainder size E/F gate；
8. 普通非参考 probe 的完整 216×216 Hessian antisymmetry 和 translation ASR；reference null Hessian 与单个 force-Hessian FD 不能替代这一项。

主要门槛沿用 R2O routed-tail readiness audit：reference `|E|<=1e-10 eV`、`max|F|<=1e-9 eV/Å`、`max|H|/antisymmetry/ASR<=1e-7 eV/Å²`、frequency drift bound `<=2 cm^-1`；普通 probe 的完整 Hessian antisymmetry/ASR `<=1e-7 eV/Å²`；两级有限差分 `<=1e-5`；O(3) energy/force `<=1e-6 eV / 1e-5 eV/Å`；order/MIC energy/force `<=1e-6 eV / 1e-5 eV/Å`。graph-source geometry bound 为 `1e-6 Å`。paired raw energy 使用严格由旧 6×6 cap 换算的 `|ΔE_raw|/N <= 1e-6/72 eV/atom`，feature/force 仍各用 `1e-5`；localized size gate 保持 total energy/central force `1e-7 eV / 1e-5 eV/Å`。

冻结 wrapper 保持不变。synthetic 6×6/8×8 graph-source geometry 观测为 `3.61e-7/7.91e-7 Å`，均低于事先固定的全局 `1e-6 Å` bound；bound、metric 和 source dtype 已进入 `graphene_r2o_fixed_graph_semantics_v2` hash。真实 formal smoke 必须重新报告两种尺寸，不能使用 synthetic 数值代替。

## 4. marker、恢复和双节点聚合

单节点开始后以 `O_EXCL` 创建 `RUNNING`。成功结果按顺序原子写入 metrics、runtime fingerprint、input snapshot、result receipt、`EXIT_CODE=0`，移除 `RUNNING`，最后发布 `DONE`。发生异常时写 `FAILED` 和 `EXIT_CODE=1`，不产生 passing receipt。

已完成目录只允许 validation-only recovery：重新核对 result receipt、全部 artifact hash、marker 内容、input snapshot、runtime/binding/source/protocol hash，不重新计算。任何未完成目录都拒绝原地续写，必须换新输出目录。

双节点 evaluator 要求两个不同 node label、不同 hostname 和不同 host-identity hash，且 input、formal core、tail、schema、graph semantics、runtime semantics、protocol 和 source hash 完全一致。同一台机器只改 label 不能通过。节点 GPU execution fingerprint 可以不同。聚合结果保存两个 node result 的逐字节 snapshot 和 hash，形成完整 receipt 链；双节点 probe energy/force 还需分别满足 `1e-6 eV / 1e-5 eV/Å` 的跨节点差异门槛。synthetic 测试可通过只在 Python API 暴露的 hostname override 构造两个假节点；formal mode 明确拒绝 override，CLI 也不提供该参数。

## 5. synthetic 集成结果

固定 seed-83、R2O-shape FP64 MACE 和非零 tail 的本地 CPU 测试得到：

| 检查 | 观测值 |
|---|---:|
| tail probe `|E|` / `max|F|` | `3.84e-8 eV / 2.90e-5 eV/Å` |
| reference `|E|` / `max|F|` | `1.41e-15 eV / 1.98e-15 eV/Å` |
| reference `max|H|` / antisymmetry / ASR | `1.36e-13 / 2.17e-15 / 1.23e-15 eV/Å²` |
| reference frequency drift upper bound | `8.12e-5 cm^-1` |
| nonreference full-H antisymmetry / ASR | `1.75e-15 / 1.14e-15 eV/Å²` |
| combined/separate energy / force difference | `3.62e-17 eV / 1.60e-18 eV/Å` |
| energy-force FD difference | `2.09e-10 eV/Å` |
| force-Hessian FD difference | `1.16e-8 eV/Å²` |
| proper O(3) energy / force difference | `1.17e-10 eV / 3.03e-8 eV/Å` |
| improper O(3) energy / force difference | `9.56e-11 eV / 2.14e-8 eV/Å` |
| order/MIC energy / force difference | `1.22e-16 eV / 6.33e-13 eV/Å` |
| 6×6 paired feature / total energy / energy-per-atom / force | `3.79e-7 / 1.46e-7 eV / 2.03e-9 eV/atom / 7.83e-7 eV/Å` |
| 8×8 paired feature / total energy / energy-per-atom / force | `7.75e-7 / 1.23e-6 eV / 9.60e-9 eV/atom / 1.54e-6 eV/Å` |
| 6×6/8×8 localized size energy / central-force difference | `2.19e-12 eV / 1.00e-8 eV/Å` |

这些数值只说明本地实现路径和合成合同通过，不能替代真实 formal model 在 V100-A/V100-B 上的结果。

## 6. launcher 边界与下一步

launcher 当前只接受两个位于 Tailscale `100.64.0.0/10` 的目标，生成 inert argv array 和 canonical plan hash。plan、每个 node record 都固定 `remote_execution_authorized=false`；代码中的 `remote_launch()` 无条件抛出 `PermissionError`。公网地址会被拒绝。

此外，formal-mode 单节点 CLI 即使被人工直接调用，也必须同时提供一个 hash 固定的独立 GO marker。marker 绑定本次 input manifest、完整 source/protocol hash 和两个授权 node labels，scope 只能是 real formal no-support smoke，并保持 `outer_prepare_authorized=false`。synthetic mode不需要也不接受该 marker；当前仓库没有生成或发布真实 GO marker。

因此本轮完成后仍不授权真实远端运行、support prepare 或 outer-LOCO。下一步依次为：独立复核本轮代码和 synthetic receipts；若通过，再把唯一 formal passing portable tree、runtime contract、tail checkpoint和 source snapshots 原样同步到两个节点；先验证两节点 hash 和环境语义，再单独授权真实 no-support smoke。真实双节点结果再次独立复核通过后，才讨论一次性 outer prepare。
