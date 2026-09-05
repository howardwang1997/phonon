# Graphene 450 K 采样扩展方案

**日期：**2026-08-03  
**承接方案：**[`GRAPHENE_FD_CONDITIONAL_FALLBACK_PLAN.md`](GRAPHENE_FD_CONDITIONAL_FALLBACK_PLAN.md)

## 当前结果

三条轨迹各 120 帧时，point estimate 满足原门槛，但按冻结规则完成 1000 次 circular moving-block bootstrap 后，状态为 `insufficient_sampling`：

| 指标 | point estimate | 95% 上界 | 门槛 |
|---|---:|---:|---:|
| seed 0–1 全谱 MAE | 4.608 cm⁻¹ | 6.303 cm⁻¹ | <5 cm⁻¹ |
| seed 0–2 全谱 MAE | 3.555 cm⁻¹ | 5.305 cm⁻¹ | <5 cm⁻¹ |
| seed 1–2 全谱 MAE | 2.536 cm⁻¹ | 4.402 cm⁻¹ | <5 cm⁻¹ |
| seed 1 平均温度相对偏差 | 3.423% | 6.201% | ≤5% |

block 长度按冻结规则 `max(5, ceil(2*tau_int))` 得到 seed 0/1/2 分别为 8/14/8 个已保存帧。RTX 2060 上的快速充分统计量重放与原始三条 TDEP 谱的最大差为 `1.07e-5 cm⁻¹`。

## 一次性扩展设置

保持以下内容不变：

- 冻结的 v11、300 K delta、600 K delta 和 450 K 权重；
- provisional long-range operator 及其逐帧扣除方式；
- seeds 0/1/2、时间步、Langevin 设置和每帧间隔；
- 原先固定的 60 个 DFT 标注索引 `3,9,...,117`。扩展帧不进入 DFT force primary subset，也不用于模型或温度系数更新。

最差的 seed 0–1 指标按 `1/sqrt(N)` 缩放估算，达到门槛约需 2245 帧/seed。为留出数值余量，本轮把最终样本数一次性固定为 **3000 帧/seed**。不在 3000 帧之前反复检查并以“首次通过”作为停止条件。

三条轨迹从现有 checkpoint 继续，不重新平衡；完成后重算各 seed 和 pooled short-TDEP，并只进行一次相同的 1000 次 block-bootstrap。只有 point estimate 和 95% 上界同时通过，才重新创建 `READY_DFT_FORCE_LABELS`。若 3000 帧仍未通过，停止并重新评估采样模型，不自动继续追加。

预计 RTX 2060 增量运行约 30–32 小时。450 K 静态 DFPT 可在 V100-A 上继续并行；两台 V100 的 DFT force labels 在 bootstrap 通过前保持关闭。
