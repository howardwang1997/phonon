# Graphene 物理温度实验当前状态

**状态时间：**2026-08-06 03:09 NZST / 2026-08-05 23:09 HKT  
**研究目标：**用短程 MLIP 与物理长程修正定量复现有限晶格温度下的 graphene Kohn anomaly。`degauss (Ry)` 由电子响应公式引入，lattice temperature 由统计系综、TDEP 和 SSCHA/SCPH 引入。

本文是当前实验、算力和验收状态的快照。远端任务仍会继续推进，后续数值以新的状态记录和最终产物为准。

## 1. 当前结论

1. 450 K 冻结 holdout 的 `FD450_LINE` 已完成 19/19 个 q 点并传到 RTX 2060。
2. shard B 的 30 个 450 K DFT force labels 已完成并传到 RTX 2060。
3. shard A 正在 V100-A 上计算，当前完成 8/30 个标签。最近 8 个标签每个约 738–756 s，剩余计算约 4.5–5 h。
4. V100-B 上的 300/600 K、240 epochs 学习曲线已完成。240 epochs 在当前开发代理集上首次通过固定的 force gate。
5. 本次学习曲线没有读取 450 K P4 输入；汇总中的 `p4_450K_inputs_read` 为 `false`。
6. 当前结果支持“小模型在足够训练轮数下可以拟合现有 300/600 K residual 数据”。它还不能证明有限温 Kohn anomaly 已被定量复现，也不能代替 450 K C1、完整 S0 数据和声子谱验收。

## 2. 三台机器

| 机器 | 当前任务 | 实际计算状态 | GPU 状态 | 存储 |
|---|---|---|---|---:|
| V100-A | 450 K P4 shard A force labels | 8/30；正在运行 GPU `pw.x` | 98%，约 9.1 GiB | `/data` 可用 182 GiB |
| V100-B | 240 epochs MACE 学习曲线 | 已完成，服务正常退出 | 0%，空闲 | `/data` 可用 108 GiB |
| RTX 2060 | P4/C1 汇总等待器；旧 physical-FD postprocess 等待器 | 两个服务都在等待输入，没有计算进程 | 0%，空闲 | 根分区可用 72 GiB |

V100-A 上三个相关服务的状态为：

- `FD450_LINE`：已完成；
- 450 K DFPT follow-up：已完成并退出；
- shard A：active，完成后自动传到 RTX 2060。

RTX 2060 已收到 `FD450_LINE` 和 shard B，目前只缺 shard A。旧 postprocess 队列仍停在“wave3 failed，等待 long-range-subtracted residual model”，不能把这个 active 服务理解为 GPU 正在计算。

## 3. 240 epochs 学习曲线

### 3.1 固定设置

- 数据：300/600 K development-only joint data；训练集 216 个构型；
- 模型：两层小型 MACE，`32x0e+32x1o`，`r_max=5 Å`；
- 优化：`batch=1`，`lr=0.001`，seed 83；
- 训练：连续运行到 epoch 240，每 5 epochs 保存 checkpoint；
- 评估点：60、120、240 epochs；
- thermal gate：300 K 和 600 K test 分别满足 RMSE `≤50 meV/Å` 且 max `≤250 meV/Å`；
- harmonic replay gate：RMSE `≤38.51 meV/Å`。

训练从 2026-08-05 12:11 HKT 运行到 13:31 HKT，训练墙钟 4823 s，即 80.4 min。GPU 平均利用率 14.8%，最大 19%，显存峰值 554 MiB。服务已启用 checkpoint、失败重启和开机续跑，本次没有发生重启。

### 3.2 分组误差

| epochs | 300 K RMSE / max | 600 K RMSE / max | replay RMSE / max | proxy gate |
|---:|---:|---:|---:|:---:|
| 30 | 79.7 / 593.2 | 143.8 / 1036.9 | 36.5 / 221.2 | 未通过 |
| 60 | 57.7 / 556.7 | 95.1 / 784.1 | 31.4 / 291.1 | 未通过 |
| 120 | 32.4 / 336.8 | 41.7 / 232.1 | 17.9 / 242.9 | 未通过 |
| 240 | 24.2 / 244.2 | 25.7 / 115.9 | 11.0 / 105.3 | 通过 |

单位为 meV/Å。120 epochs 时两个 thermal RMSE 和 600 K max 已通过，但 300 K max 仍为 336.8 meV/Å；240 epochs 后全部代理门槛通过。因此当前开发训练的经验配置是 `batch=1, lr=0.001, 240 epochs`。

这个结论的适用范围有限：300 K 和 600 K test 各只有 3 个构型，数据使用的是现有开发代理 residual，而不是完成电子公式、交叉 `degauss` 判别和三温度合并后的正式 S0 数据。正式模型仍需在完整 physics-subtracted S0 development set 上重复验收。

## 4. batch 与吞吐实验

相同 30 epochs 下，batch 2/4/8/16/32 分别获得约 1.77/3.12/4.97/6.97/7.95 倍墙钟加速，但都没有达到相对 batch 1 固定的热扰动等效标准。随着 batch 增大，harmonic replay 误差下降，而 300/600 K thermal test 误差上升。

因此当前不再继续扩大 batch。正式 S0 从 batch 1 开始；如果以后需要重新优化吞吐，只能在完整 S0 development set 上重新固定候选和门槛。

## 5. 450 K P4/C1 链路

| 输入 | 状态 |
|---|---|
| 450 K frozen predictor | 已在读取任何 450 K target 前冻结 |
| `FD450_LINE` | 19/19 完成，RTX 侧 `REMOTE_DONE` 已存在 |
| shard B | 30 个标签完成，RTX 侧 `RAW_READY` 已存在 |
| shard A | 8/30，正在 V100-A 计算 |
| P4 merge/C1 | 服务 active，等待 shard A |

按最近标签的实测速度，shard A 剩余约 4.5–5 h。标签传输完成后，RTX 2060 会自动合并两片共 60 个构型，运行冻结模型预测、DFT-TDEP/bootstrap 和 C1 验收。若没有失败重试，预计约 5–7 h 后可以得到 C1 阶段性结论。

C1 关闭前继续执行以下边界：

- 不使用 shard A/B 或 450 K DFPT target 调参；
- 不改变冻结预测器、测试点和验收门槛；
- 300/600 K 开发实验的产物只用于训练配置和算力估计；
- C1 失败时先报告失败指标，不根据 P4 结果反复追加训练直到通过。

## 6. 物理温度方案所处阶段

目标方案已经在文档中固定，但完整实验尚未全部执行：

- 电子通道：以 Mermin 自由能和有限展宽电子极化公式引入 `smearing/degauss (Ry)`；
- 晶格通道：统一势能面不直接接收 lattice temperature，温度通过系综、TDEP 和 SSCHA/SCPH 引入；
- 交叉项：用固定的非对角 `(T_lattice, degauss)` 开发点判断是否需要局域 Mermin 项。

当前 240 epochs 模型只是 S0 训练可行性代理。它没有完成最终电子长程算子、450 K 开发数据和二维交叉项，因此不能作为正式物理温度模型发布。

## 7. 下一步队列

1. V100-A 完成 shard A 并自动传到 RTX 2060。
2. RTX 2060 一次性运行 P4 merge 和 C1 验收。
3. C1 关闭后再开放 450 K 结果用于下一阶段开发，执行电子响应 E0 和交叉 `degauss` force pilot E1。
4. 根据 E1 判断局域 Mermin 项为零或进入 E2 扩充。
5. 使用统一物理长程力扣除后的 300/450/600 K 数据训练正式 S0；从本次实测的 `batch=1, 240 epochs` 开始重新验收。
6. S0 通过后运行 L0 三温度 MD/TDEP 和 Q0 SSCHA/SCPH。

RTX 根分区目前只剩 72 GiB，低于后续 L0/Q0 预设的 100 GiB 启动门槛。P4 产物归档并核对后，需要先释放至少约 28 GiB，再启动大规模轨迹和 checkpoint 任务。

## 8. 主要产物

- [物理温度理论方案](GRAPHENE_PHYSICS_BASED_DEGAUSS_LATTICE_TEMPERATURE_PLAN.md)
- [实验与算力排期](GRAPHENE_PHYSICS_TEMPERATURE_EXPERIMENT_COMPUTE_PLAN.md)
- [30 epochs batch 收敛汇总](../results/graphene_physics_temperature/benchmarks/mace_convergence_30ep_v100b/convergence_summary.json)
- [batch 2/4 精化汇总](../results/graphene_physics_temperature/benchmarks/mace_convergence_refine_30ep_v100b/refinement_summary.json)
- [240 epochs 学习曲线汇总](../results/graphene_physics_temperature/benchmarks/mace_learning_curve_240ep_v100b/learning_curve_summary.json)

学习曲线的三个里程碑模型、分组评估、训练状态和 GPU 监控记录已经从 V100-B 同步到本地，并按远端摘要中的 SHA-256 完成核对。49 个可重建训练 checkpoint 仍保留在 V100-B，没有全部复制到本地。
