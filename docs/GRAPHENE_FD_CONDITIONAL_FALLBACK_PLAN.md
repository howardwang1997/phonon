# Graphene physical-FD 温度条件化模型修订方案

**日期：**2026-08-02  
**承接方案：**[`GRAPHENE_FD_TRANSFERABILITY_PLAN.md`](GRAPHENE_FD_TRANSFERABILITY_PLAN.md)  
**当前状态：**P0、P1 已完成；P2 的 joint delta 未通过端点 development gate；尚未生成或读取任何 450 K DFT/DFPT 结果。

## 1. P2 结果与问题定位

P1 表明两个已有 delta 都不能直接跨温度使用：

| 模型 | 300 K force RMSE/max (meV/Å) | 600 K force RMSE/max (meV/Å) | 结果 |
|---|---:|---:|---|
| 300 K delta | `21.01/114.67` | `129.05/1135.93` | 600 K 未通过 |
| 600 K weighted delta | `44.05/359.98` | `43.31/210.74` | 300 K 最大分量未通过 |

P2 按事先固定的 `replay weight=12,16` 训练 joint delta。两个 sweep 均没有 checkpoint 同时通过 `50/250 meV/Å` 的热受力门槛和谐波保持门槛：

| replay weight | epoch | 300 K RMSE/max | 600 K RMSE/max | harmonic RMSE | 最差归一化分数 |
|---:|---:|---:|---:|---:|---:|
| 12 | 165 | `27.62/272.32` | `32.47/150.90` | `14.21` | `1.089` |
| 16 | 175 | `25.72/278.37` | `30.24/150.89` | `11.50` | `1.113` |

两条 joint 模型的 RMSE 和谐波保持均合格，失败项都是 300 K 的单分量最大误差。这里不把 `250 meV/Å` 放宽到 `280 meV/Å`，也不把失败 checkpoint 带入 450 K。P2 结果登记为 `no_joint_candidate_passed`。

## 2. 修订的短程预测模型

采用原方案第 8 节已经列出的温度条件化失败分支。冻结预测栈为

\[
E_{\rm SR}(T)=E_{v11}+[1-w(T)]E_{\Delta,300}+w(T)E_{\Delta,600},
\qquad
w(T)=\frac{T-300}{300},\quad 300\le T\le 600\ {\rm K}.
\]

这里两个 delta 都是已经完成端点验收的冻结模型。对固定的 lattice temperature，权重是常数，因此力严格来自同一个标量势能；在 300 K 和 600 K 分别精确退化为已有端点模型。该形式不是一个温度无显式输入的 joint MACE，但它是一个固定、守恒、带显式温度条件的预测栈。

450 K 使用 `w=0.5`。在任何 450 K target 生成前固定两个模型文件、权重公式、代码哈希和全部验收标准。450 K 失败后不得改权重并继续把同一批数据称为 holdout。

## 3. 长程项与采样算子

q-space correction 保持原先固定的 `rounded cusp + q²` 基底。六个系数只由 300/600 K 已有校准结果定义线性温度函数；450 K 使用端点系数的算术平均，不读取 450 K DFT-TDEP：

| 区域 | 450 K `c0` | 450 K `c1` | 450 K `c2` | 单位 |
|---|---:|---:|---:|---|
| Γ | `37373.8871` | `2549.9318` | `607123.3479` | squared-frequency expression |
| K | `-112737.8757` | `2867.0398` | `1408300.0115` | squared-frequency expression |

最终冻结文件以完整精度记录数值；表中只作便于核对的摘要。

用于 450 K MLIP-MD 采样的 provisional 实空间算子同样按端点 FC2 逐元素线性插值。该算子只改变采样势；保存的每帧总力随后减去完全相同的算子，再拟合 short-range TDEP。因此最终 q-space 预测不直接采用 provisional FC2。

## 4. 冻结前必须通过的检查

1. 条件化计算器在 `w=0/1` 时，与对应单端点模型的能量和力重放差小于 `1e-6 eV/Å`。
2. 300 K 端点继续使用已有 development force gate：`21.01/114.67 meV/Å`；600 K 端点继续使用冻结前定义的 15 构型 holdout：`42.81/215.09 meV/Å`。
3. 两个端点的 harmonic replay RMSE 分别为 `12.81` 和 `10.29 meV/Å`，均低于 `2×` frozen-v11。
4. 两个端点已有三 seed short-TDEP spread 和同温度 q-space 校准均通过；由于条件化模型在端点与原模型相同，不重复消耗轨迹来证明同一计算。
5. 冻结清单记录两个 delta、v11、两个采样算子、450 K 插值算子、端点 TDEP、端点 acceptance、温度函数、脚本、计划文档和 Git HEAD 的 SHA-256。
6. 冻结时再次确认本地、RTX 2060、V100-A、V100-B 均没有 450 K target 或标签产物。

任一项失败都停止，不启动 FD450 或 450 K force labels。

## 5. 执行顺序

### F1：冻结条件化预测栈

- 在 RTX 2060 上运行端点数值重放；
- 生成 450 K provisional operator、`temperature_law.json` 和 `freeze_manifest.json`；
- 将同一冻结清单同步到本地和两台 V100，核对哈希一致。

### F2：450 K on-policy MLIP 轨迹

- seeds `0,1,2`，每条保存 120 个 snapshots；
- seed 0/2：`dt=0.5 fs`、equilibration `3000`、stride `80`；
- seed 1：`dt=0.25 fs`、equilibration `6000`、stride `160`；
- mean temperature 相对 450 K 偏差 `≤5%`；
- 扣除 450 K provisional operator 后，三条 short-TDEP 的全谱两两 MAE `<5 cm⁻¹`。

轨迹或 seed gate 未通过时，不释放 DFT force labels。

### F3：450 K 静态 DFPT

冻结完成后即可与 F2 并行：

- `FD450_CONV`：Fermi–Dirac smearing/degauss `0.00285013035 Ry`，`k=120,144`，固定 8 个 Γ/K 点；
- 若 `k=120→144` 的 Γ/K 顶支最大变化 `<1 cm⁻¹`，运行 `FD450_LINE` 的 19 点 dense line，使用 `k=144`；
- DFPT 只检验中间 degauss 的静态电子响应，不参与有限温系数回归。

### F4：450 K on-policy DFT force/TDEP holdout

F2 通过后固定抽取每个 seed 的 `3,9,15,...,117` 共 20 个构型，总计 60 个。primary force subset 为每个 seed 的 `3,27,51,75,99`，共 15 个。

- V100-A：seed 0 的 20 个构型和 seed 2 的前 10 个；
- V100-B：seed 1 的 20 个构型和 seed 2 的后 10 个；
- 统一使用 `8×8×1` k 网格、`60/240 Ry`、Fermi–Dirac `0.00285013035 Ry`；
- primary subset force RMSE/max 必须 `≤50/250 meV/Å`；
- 60 个 DFT force 拟合 holdout DFT-TDEP，只用于评价冻结预测，不回写 450 K 系数；
- Γ/K line MAE 各 `<10 cm⁻¹`，Γ/K 顶支误差各 `<15 cm⁻¹`，K kink 相对误差 `<20%`；中心三点整块评价使用相同门槛。

通过 F4 只登记 C1，即 300–600 K physical-FD 路径上的未见温度插值通过。

### F5：独立 450 K physical-FD DFT-MD

仅在 F4 通过后启动。使用 `seed=45001`、`dt=0.5 fs`、至少 500 个 equilibration steps、60 个 snapshots、stride 20。先做前三帧 `k=8→12` force 检查，再按原方案完成 off-policy force、独立 DFT-MD TDEP 和 paired-force TDEP 验收。F5 通过后才登记 C2。

## 6. 失败后的处理

- 条件化端点重放失败：修复实现，不改变数学模型或门槛；
- 450 K MLIP 轨迹不稳定：保持模型与温度函数冻结，只按事先列出的时间步重试；
- 450 K force gate 失败：450 K 转为 development，下一次最终温度改为事先未计算的 375 K 或 525 K；
- force 通过而 q-space 失败：说明两端线性系数规律不足；450 K 可进入下一版温度函数训练，但必须另设未见温度；
- F4 通过而 F5 失败：结论停在 C1，不写成独立轨迹可迁移。

本轮先启动 F1、F2 和 F3。F4 由 F2 的固定 gate 自动释放，F5 继续保持关闭。
