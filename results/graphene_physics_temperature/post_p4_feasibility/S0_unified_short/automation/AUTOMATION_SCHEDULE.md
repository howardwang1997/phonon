# S0 后续实验自动队列

更新时间：2026-08-09（新西兰时间）

## 当前执行

S0 replay weight 16 已完成。固定 checkpoint 扫描选择 epoch 215，三个开发温度和 harmonic replay gate 全部通过。RTX 2060 当前完成 450 K 的第三个正式 seed；450 K point/bootstrap gate 通过后，300 K 和 600 K 将在各自温度内部采用三机三 seed 并行。门槛顺序仍为 450→300→600 K。

## 自动分支

1. S0 全部 force gate 通过：自动冻结 base、delta、三个物理算符、代码哈希和 L0 轨迹设置，然后启动 L0。
2. S0 任一 gate 未通过：停止在 S0，不启动 TDEP，也不读取或生成 375/525 K 验证数据。
3. L0 先按 450→300→600 K 运行三个 seed、每 seed 120 帧的稳定性 pilot。三个温度的 pilot 都通过后从同一 checkpoint 扩展，不重新平衡。
4. 正式 L0 按 450→300→600 K 顺序运行，每个温度固定为 3000 帧/seed。每个温度只执行一次 point gate 和一次 1000-replicate block bootstrap。
5. 任一温度的 pilot、point gate 或 bootstrap 失败：停止队列，不自动放宽门槛，也不继续追加帧。
6. L0 三温度全部通过后自动启动 Q0。先运行 450 K、`200 configs × 最多 4 populations` 的 SSCHA benchmark；只有 SSCHA 主动返回收敛才运行 300/450/600 K、`300 configs × 最多 5 populations` 的正式量子计算。达到最大 population 但未收敛不算通过。
7. Q0 全部收敛后自动启动 X0。首个固定非对角条件为 `(T_lattice=450 K, operator temperature=300 K)`；实际 SSCHA Hessian 与公式重建 Hessian 比较，顶支最大差需 `<2 cm⁻¹`，K kink 相对变化需 `<20%`。
8. X0 通过时，用公式生成开发区间的 `3×3` 网格并写出 `READY_FOR_VALIDATION_FREEZE_REVIEW`。X0 超过门槛时只自动增加一个 seed 0、3000 帧的 classical-MD 诊断，随后停止，不自动扩展到三 seed。
9. 450 K 正式 gate 通过后，V100-A、V100-B、RTX 2060 分别运行 seed 0、1、2。每个 worker 从已经验收的 120 帧 checkpoint 继续，冻结模型、物理算符和脚本逐项校验 SHA-256。三个 3000 帧结果回收到 RTX 后，仍由原 L0 队列执行 pooled TDEP、point gate 和 1000 次 block bootstrap；通过后才按相同方式释放下一温度。

按当前实测速率，完整 L0 仍约需 4 天；动态完成时间以 `status.json` 为准。Q0 预留 12–24 GPU-h，X0 首条件预留 2–5 GPU-h；只有 X0 失败时才增加约一个 seed 的 MD 诊断。

## 三台机器安排

| 机器 | 当前任务 | 后续任务 | 启动条件 |
|---|---|---|---|
| RTX 2060 | 450 K seed 2；随后 300/600 K seed 2 | Q0 benchmark/正式三温度 → X0 首交叉点 | 每个前置 gate 通过、GPU 空闲、磁盘可用空间不少于 50 GiB |
| V100-A | 已完成隔离 smoke；等待 450 K gate | 300/600 K seed 0；以后保留 375 K V0 static DFPT | 同一温度的前置 gate 通过；验证点仍需 L0、Q0、X0 冻结 |
| V100-B | 已完成隔离 smoke；等待 450 K gate | 300/600 K seed 1；以后保留 525 K V0 static DFPT | 同一温度的前置 gate 通过；验证点仍需 L0、Q0、X0 冻结 |

V100 worker 位于隔离的 `/data/graphene_physical_s0_l0_dist/project`，不覆盖两台机器原有的 `/root/phonon`。正式 worker 只能在收到上一温度 gate 令牌后启动。三机协调由 Mac 上的用户级 launchd 任务执行，使用现有 Tailscale 地址和 SSH 密钥；Mac 睡眠时协调暂停，但远端 worker 和 checkpoint 不受影响，唤醒后继续收集。375/525 K 验证计算不会提前启动。

## 自动化边界

- 三个开发温度的 `smearing/degauss (Ry)` 均由 `k_B*T/Ry` 给出。
- 375/525 K 保持锁定；自动调度器无权提前释放。
- RTX 2060 上发现其他 GPU 任务或可用空间低于 50 GiB 时，L0/Q0/X0 自动等待。
- Q0 不把达到最大 population 当作收敛；X0 不在 gate 失败后自动消耗完整 30–120 GPU-h。
- RTX 2060 当前 `linger=no`。SSH 断开不影响本次服务；如果整机重启，需要重新启动一次 `graphene-physical-s0-autopilot.service`，已有训练和 MD 从 checkpoint 恢复。

动态状态文件：

- `RTX2060_status.json`
- `V100-A_status.json`
- `V100-B_status.json`
