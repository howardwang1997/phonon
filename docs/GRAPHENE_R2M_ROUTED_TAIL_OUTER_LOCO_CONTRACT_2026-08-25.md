# Graphene R2M 保守路由尾项 outer-LOCO 合同

日期：2026-08-25

状态：代码框架已完成本地静态与合成测试，尚未启动。只有 support-free R2M core 通过既定 checkpoint gate，并给出唯一的模型路径和 SHA256 后，才能准备数据和训练。

## 1. 这一阶段回答什么问题

R2M core 不使用 support9 训练。如果它能通过 fixed-smearing seed1 与 harmonic gate，下一步需要判断：只用另外 8 个 support 构型训练的局域保守尾项，能否迁移到第 9 个完整构型，同时不破坏 core 已有的 fixed-smearing 和 harmonic 精度。

support9 按完整构型做 9 折 outer-LOCO。每折的 held 构型不进入特征尺度、梯度、能量定零或模型选择。训练终点固定为 epoch 240 的 EMA 状态，不根据 support 指标选择 checkpoint。若以后确实需要选择 epoch，必须另做严格 nested inner CV；不能把某个构型在其他 epoch 参加过训练的循环指标称为 inner holdout。support9 已在 R2C–R2K 开发中使用过，因此这里的 outer-LOCO 是本阶段的构型级迁移检验，不是新的 blind/external 测试。

本阶段不读取 E50 seed2。seed2 已在此前模型开发中打开，也不能继续用于 tail 的架构、阈值或 checkpoint 选择。

## 2. 模型组合

总模型保持以下分解：

```text
E_total = E_foundation(frozen)
        + E_core(frozen)
        + sum_i g_i(s_i(R)) [epsilon_i(R) - c_C]
        + E_q6(frozen)
```

foundation 和 q6 不进入 tail 的参数、尺度或损失。tail 只拟合 short-range target 与 frozen core 之间的差。

passing core 固定为 `r_max=2 Å`、2 interactions、`16x0e+16x1o+16x2e` 的 R2M MACE。它的拼接 `node_feats` 宽度为 160。路由输入只包含 O(3) 不变量：

- interaction 0 的 16 个 signed `0e`；
- interaction 0 的 16 个 `l=1` 通道自功率；
- interaction 0 的 16 个 `l=2` 通道自功率；
- interaction 1 的 16 个 signed `0e`。

因此输入固定为 64 维。schema 校验逐项固定每个 block 的 interaction、multiplicity、`l/p`、dimension、raw slice、operation、output slice 和 feature name；重新计算一个自洽 hash 不能掩盖错误切片。遇到 `0o`、不同的 irreps 顺序、不同 raw width、不同 core cutoff 或非有限 scaler/pristine 数值时直接停止。

tail 与 router 使用相互独立的 MLP：

```text
tail:   64 -> 64 -> 32 -> 1, SiLU
router: 64 -> 32 -> 16 -> 1, SiLU
```

gate 为固定区间 `[0,1]` 上的 quintic smootherstep：

```text
g(s) = smootherstep(clip(s, 0, 1))
```

它在两个边界的值、一阶导数和二阶导数连续。`c_C` 是同一 fold 的 tail 网络在 6×6 pristine graphene 上的平均节点能。它随 tail 参数更新，但对 query 坐标是常数。

力只能由完整标量能量求导：

```text
F_tail = -grad_R sum_i g_i [epsilon_i - c_C]
```

实现中没有 `g * F` 接口，也不 detach query 的 invariant、score、gate 或 node energy。因此 `-epsilon grad(g)` 和 MACE node feature 的跨原子项都会进入力。

## 3. 数据和能量定零

support 短程目标使用已有 support9 中的：

```text
PARENT_SHORT_RANGE_TARGET_forces
PARENT_SHORT_RANGE_TARGET_energy
```

已经核对，force 与 `TOTAL - BASE - LONG_RANGE` 的最大差不超过 `1.1e-8 eV/Å`。energy 与同一 raw 分解相差一个对全部 9 个构型完全相同的常数，因此构型间能量差可用。

对 fold `f`，先用 frozen core 计算 support 短程 residual：

```text
F_tail,target = F_short,target - F_core
E_tail,raw    = E_short,target - E_core
```

能量定零只使用该 fold 的 train8：

```text
b_f = mean_train8(E_tail,raw)
E_tail,target = E_tail,raw - b_f
```

held energy 只在 epoch 240 EMA 模型和文件哈希冻结后，用同一个 `b_f` 评价。

训练 replay 只使用 `data/graphene_r2m_support_free_core/train.xyz`：72 个 harmonic-train、20 个 E50 seed0 和 72 个 auxiliary 构型。现有数据根固定为 manifest/train/valid 的 SHA256 `bb20a86a…31e`、`789b65e1…519c`、`92ad1508…a6da`；先核字节 hash，再读取 extxyz，并再次检查 role 闭集和 seed。它们的 tail target 固定为零，用于保持 passing core。`valid.xyz` 必须恰好是 20 个 E50 seed1 和 25 个 harmonic-validation，只在 epoch 240 冻结后检查，不参与训练或 checkpoint 选择。

q6 operator、phonopy background 和 450 K corrected result 的路径与 SHA256 也在 fold 准备时冻结。A′ 定义已经移入无仓库相对导入的 `graphene_r2m_aprime_eval.py`：固定使用 K=`(1/3,1/3,0)` 的最高频 primitive mode、固定相位和归一化规则，并冻结 helper hash。它与旧 audit 实现在现有 72 原子资产上逐数组完全一致。evaluator 若收到不同文件或 helper 会在打开 held 前停止。

64 维 mean/scale 每 fold 单独由 train8 和 support-free replay-train 计算。scale 定义为 `max(std, 1e-6*RMS, 1e-8)`，不删列。held 与 seed2 不参与尺度。

## 4. 固定训练损失

每个 optimizer step 组成一个 4 构型 batch：1 个 support、1 个 harmonic、1 个 E50 seed0 和 1 个 auxiliary。四组 force loss 的质量固定为：

| 组 | loss 质量 | force scale |
|---|---:|---:|
| support | 0.50 | 30 meV/Å |
| harmonic | 0.25 | 9.044673 meV/Å |
| E50 seed0 | 0.15 | 30 meV/Å |
| auxiliary | 0.10 | 30 meV/Å |

另外加入：

- support train-gauge energy loss，权重 0.25、尺度 19.4 meV/config。19.4 meV 来自 R2C/R2G 已有 support gate，在所有 outer fold 之前固定，不由本轮 held 结果调节；
- harmonic gate-off penalty，权重 0.05、gate scale 0.05；
- E50 seed0 gate-off penalty，权重 0.02、gate scale 0.05。

optimizer 固定为 AdamW AMSGrad，`lr=1e-3`、`weight_decay=1e-6`、无 scheduler、gradient clip 20、EMA 0.99、seed 83。每个 epoch 让 train8 各出现一次，共 8 个 optimizer steps；三个 replay 组各自沿固定 permutation 循环，240 epochs 后同组任意两个构型的使用次数最多相差 1。总计 240 epochs。没有 checkpoint sweep、early stopping 或 validation-controlled learning rate。

## 5. 验收条件

9 折完成后，aggregate 逐 fold 按 prepared manifest 中的 fold path、manifest hash、held index 和 core hash 精确配对，不能只比较 fold/index 集合。它从 NPZ 重新计算 held、seed1、harmonic force/A′/gate/energy 指标，从 mechanics 原始字段重建其余 gate，并要求固定阈值和 JSON boolean 与重算结果一致。原始数组的 shape 和有限性也固定。九折的 optimizer/loss/model freeze、同名代码 snapshot hash 和软件环境必须一致，之后才聚合真正未见的 9 个 held 预测。必须同时满足：

- pooled support force RMSE/max `≤30/200 meV/Å`；
- 每一个 held 构型也分别满足 force RMSE/max `≤30/200 meV/Å`；
- support A′ 投影 RMS `≤15 meV/Å`；
- support A′ restoring slope 相对误差绝对值 `≤5%`；
- train-gauge energy RMSE `≤19.4 meV/config`；
- 450 K importance-weight ESS fraction `≥0.30`。

每一折冻结后还必须保持：

- E50 seed1 force RMSE/max `≤30/200 meV/Å`；
- E50 seed1 A′ RMS `≤15 meV/Å`，slope 相对误差绝对值 `≤5%`；
- harmonic-validation force RMSE/max `≤9.044673/200 meV/Å`；
- harmonic pooled 和逐构型 gate mean 都 `≤5%`。

力学检查为：

- energy-force finite difference 最大差 `≤1e-5 eV/Å`；
- full Hessian antisymmetry 最大值 `≤1e-7 eV/Å²`；
- rotation/reflection energy error `≤1e-6 eV`、force equivariance error `≤1e-5 eV/Å`；
- 同一中心扰动在 6×6/8×8 的 tail force 差 `≤1e-5 eV/Å`、tail node energy 差 `≤1e-7 eV`、core+tail force 差 `≤1e-5 eV/Å`。

这里的严格局域 size gate 只覆盖 core+tail。foundation 的理论感受野会超过 6×6 周期长度，因此完整 foundation+core+tail 只能另做数值 6×6/8×8 检查，不能声称理论无绕回。q6 继续沿用已经冻结的 Fourier 审计，不并入局域 gate。

outer-LOCO 全部通过只授权用完全相同的合同训练一个 all9 final tail，然后做完整复合模型 gate；它本身不授权 SSCHA、声子谱或部署。任何一项失败时，不训练 final tail，也不打开 seed2 调参。

## 6. 计算与存储预算

每个 outer fold 有 1920 个 optimizer steps，每步包含 4 个构型，并需要对 MACE node features 做二阶反传。第一个正式 fold 在 epoch 10 时只记录吞吐和显存，随后在同一进程继续到固定 epoch 240；不另跑 truncated pilot，也不根据这段数据改模型或选 epoch。

初始资源预算为：

- 三台 GPU 同时各跑一个 fold，分三批完成 9 折；
- 每 fold 预留 1–2 GPU 小时训练，以及 0.2–0.5 GPU 小时做 full Hessian 和其他 mechanics gates；
- 总预算约 11–23 GPU 小时，三机并行墙钟约 4–8 小时；
- 每 fold 模型、日志和 mechanics 数组预计小于 0.5 GB，总存储预留 6 GB。

以上时间是尚未用 passing core 实测前的容量规划。第一个正式 fold 到 epoch 10 时只修正时间估算；除非发现实现错误，不改变模型、loss 或 gate。

每折 launcher 首次运行时一次性冻结 shell、trainer、evaluator、模型模块、A′ helper 和 provenance helper；之后只执行 snapshot。运行环境要求 `mace-torch==0.3.16`、可用 CUDA 且只暴露 device 0，并冻结 Python/NumPy/PyTorch/CUDA/cuDNN 和 GPU 信息。`TRAINING_DONE`、epoch-240 EMA artifact、training freeze/metrics、evaluation NPZ/JSON、completion manifest、`DONE/FAILED/RUNNING/EXIT_CODE` 都按 hash 和状态联合校验。若实例恰好在写完 completion、写 DONE 前断连，重启只读验证既有 completion 后补 marker，不重跑 evaluator、不覆盖 completion。

## 7. 当前阻断条件

当前硬阻断是同一路 R2M support-free core 尚未通过并冻结。准备脚本要求同时提供：

- `selected_core.model`；
- 模型完整 SHA256；
- 状态为 `R2M_core_checkpoint_gate_passed` 的 gate JSON；
- gate JSON 中同一 artifact SHA256 和 `force_gate_allows_routed_tail_stage=true`。

任一项缺失或不一致时，数据准备直接失败，不会创建 fold，也不会启动训练。

这个 support9 outer-LOCO 只服务于通过上述 gate 的同一路 R2M core。R2N 或其他后续 core 即使结构相似，也不能复用这里的 prepared folds、launcher freeze 或结果；core path/hash 不同会 fail-closed。当前阶段保持 blocked，不运行 tail。
