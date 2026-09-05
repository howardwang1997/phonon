# Graphene K 点无人工展宽 cusp：仓库方法实验计划

**冻结日期：**2026-08-12  
**目标：**用本仓库的 MLIP 有限晶格温度背景与显式 q-space 长程电子修正，在不使用 `smearing/degauss` 的电子积分下，定量复现 graphene K 点 $A_1'$ 声子支的 cusp。

本实验的验收对象是 K 点附近的非解析形状，包括 K 点频率、两个入射方向的 cusp depth、单边斜率和 slope jump。结构稳定、没有虚频或全路径大体接近，均不能代替该验收。

## 1. 方法定义与结论范围

### 1.1 direct DFPT 参考

电子积分使用 Quantum ESPRESSO 的优化四面体方法：

```text
occupations = 'tetrahedra_opt'
```

输入中不出现 `smearing` 和 `degauss`。该结果记为 $D_{\mathrm{DFPT}}^{\mathrm{tetra}}(q)$。它表示有限 k 网格上的零人工展宽积分；只有完成 k 网格收敛后，才能作为本实验的无展宽数值参考。不得将其表述为已经证明严格的数学 $k\rightarrow\infty$ 极限。

### 1.2 仓库方法

现有 L0 classical TDEP 和 Q0 quantum SSCHA/SCPH 结果中已经含有有限 q6、有限实空间范围的电子算子。先将其移除：

\[
D_{\mathrm{short}}(q,T)=D_{L0/Q0}^{\mathrm{finite}}(q,T)
-D_{\mathrm{el}}^{q6\rightarrow R}(q,T).
\]

无展宽 direct DFPT 的最高光学支平方频率只在 development 点上拟合为 $\lambda_{0}^{\mathrm{DFPT}}(q)$。对每个温度对应的冻结静态短程 MLIP 基线，定义

\[
\Delta\lambda_0(q,T)=
\lambda_{0}^{\mathrm{DFPT}}(q)-
\lambda_{\mathrm{short,static}}(q,T).
\]

再沿连续追踪的 $A_1'$ 本征矢构造 Hermitian rank-one 修正：

\[
D_{\mathrm{repo}}^{0}(q,T)=D_{\mathrm{short}}(q,T)+
\frac{\Delta\lambda_0(q,T)}{s(q,T)}
|e_{A_1'}(q,T)\rangle\langle e_{A_1'}(q,T)|.
\]

这里 $s(q,T)$ 是动力学矩阵本征值到 cm$^{-2}$ 的数值换算因子。电子修正始终保留在 q 空间，不 Fourier 变换成有限范围力常数。

若 holdout 通过，可以得到两层结论：

1. 静态层面：仓库方法能预测未参与拟合的 K 邻域 direct-DFPT 无展宽点和 cusp 形状；
2. 有限晶格温度层面：同一个无展宽电子修正可以和 300/450/600 K 的 L0/Q0 背景组合，给出固定电子积分条件下的晶格温度预测。

第二层没有对应的有限温 direct DFPT/DFT-MD 无展宽参考，因此属于经过静态验证后的方法预测，不能写成有限温绝对精度已经由第一性原理验证。

## 2. q 点、development 和 holdout

采用原胞倒易分数坐标：

```text
Gamma = (0, 0, 0)
M     = (1/2, 0, 0)
K     = (1/3, 1/3, 0)
```

QE `ibrav=4` 的 Cartesian `2*pi/a` 坐标由

\[
(q_x,q_y)=(h,(h+2k)/\sqrt{3})
\]

得到。令 $d$ 等于 K→Gamma 路径的分数位移；K→M 使用两倍路径分数，使两个方向上的实际 $|q-K|$ 相同：

\[
q_{K\Gamma}(d)=((1-d)/3,(1-d)/3,0),
\]

\[
q_{KM}(d)=((1+d)/3,(1-2d)/3,0).
\]

固定划分见 `configs/graphene_k_cusp_nosmear/qpoints.tsv`：

| 用途 | 两个方向上的 $d$ | 唯一 q 点数 | 是否允许拟合使用 |
|---|---|---:|---|
| pilot | 0、0.007 | 3；K 在两台机器重复 | 是 |
| development 补点 | 0.015、0.023 | 4 | 是 |
| blind holdout | 0.003、0.011、0.019 | 6 | 否 |
| k 网格复核 | 0、0.007 | 3；K 在两台机器重复 | 只用于选网格 |

holdout 计算只在 development 模型和验收标准写入冻结 manifest 后释放。不能根据 holdout 改模型阶数、窗口或阈值。

### 2.1 事先确定的候选形状

拟合对象是 direct DFPT 的绝对平方频率，不是每个温度分别拟合的修正量。$r=|q-K|$，K 点截距在两个方向共享。候选模型为：

1. `linear_directional`：共享截距，K→Gamma 和 K→M 各有一个线性斜率；
2. `quadratic_shared`：在 1 的基础上增加一个两方向共享的二次项；
3. `quadratic_directional`：两个方向各有二次项，作为复杂度对照。

只用 development 点做 leave-one-distance-out 交叉验证，评分为 CV MAE 加每个额外参数 `0.10 cm^-1` 的固定复杂度惩罚。holdout 不参与模型选择。

## 3. 算力与存储安排

2026-08-12 盘点结果：

| 机器 | 当前资源 | 本实验任务 | 启动门槛 |
|---|---|---|---|
| V100-A `100.80.236.112` | 16 CPU、31 GiB RAM、`/data` 可用 117 GiB；无计算进程 | lane A direct DFPT；一半 k 网格点 | 可立即运行 |
| V100-B `100.123.220.57` | 16 CPU、31 GiB RAM、`/data` 可用 61 GiB；无计算进程 | lane B direct DFPT；构建增大 `npk` 的 QE | 可立即运行；全过程保持至少 35 GiB 可用 |
| RTX 2060 `100.105.21.7` | 16 CPU、46 GiB RAM、根分区可用 59 GiB；GPU 空闲 | 可选远程后处理和结果镜像 | 不复制 QE `tmp` 或大型 EPW restart |
| 本地工作站 | `phonon` Conda 环境 | 同步、哈希、矩阵组合、拟合、验收和绘图 | 所有 Python 使用 `conda run -n phonon` |

当前 conda QE 7.5 的 `npk=40000` 会使高密 k 网格的一般 q 点在 `set_kplusq` 失败。单独构建 `/data/qe-7.5-npk120k`，设 `npk=120000`；保留原二进制。新旧二进制要在 k192 的 K 和一个一般 q 点做交叉版本重放，最大频率差不得超过 `0.2 cm^-1`。

预算：

| 情形 | V100 box-hours | RTX GPU-hours | 远端新增数据 | 本地归档 |
|---|---:|---:|---:|---:|
| k192 通过，直接进入正式线 | 约 75–90 | <1，通常为 0 | 每台约 10–20 GiB scratch | <1 GiB |
| 需要 k240，并追加 k288 复核 | 约 120–150 | <1，通常为 0 | 每台约 20–30 GiB scratch | <2 GiB |

V100 的计费预算按整机墙钟计；当前 conda `pw.x/ph.x` 是 CPU 计算，GPU 空闲不表示任务没有运行。原始 SCF `tmp` 只留在远端，长期归档仅同步：

- `scf.in/out`、`ph.in/out`、`gr.dyn`；
- q 点 CSV、stage manifest、运行日志和开始/结束时间；
- QE 二进制 SHA-256、版本、`npk`、赝势 SHA-256；
- development 模型冻结 manifest、预测 CSV、指标 JSON 和最终图。

所有原始文件进入结果目录后不可覆盖；重试使用带时间戳的失败副本。

### 3.1 已冻结的本仓库输入数据

S5 不再临时挑选有限温数据。精确路径、NPZ 键、数组形状、字节数和 SHA-256 已写入 `configs/graphene_k_cusp_nosmear/data_inventory.tsv`，共 12 个文件、约 8.84 MiB：

- 300/450/600 K 的静态短程 MLIP 基线各一份，用于构造温度对应的 $\Delta\lambda_0(q,T)$；
- L0 classical TDEP 的 `pooled_fc2` 三份；
- 已收敛 Q0 quantum SSCHA/SCPH 的 `free_energy_fc2_eV_A2` 三份；
- 与 L0/Q0 内部哈希一致的 q6 电子算子 `delta_fc_full` 三份，只用于从有限晶格结果中移除旧的有限范围电子项。

组合前脚本必须重新计算 inventory 中每个 SHA-256。任何文件缺失、哈希变化、键名或 `72x72x3x3` 形状不符都停止 S5，不自动改用相似文件。

## 4. 阶段、时间与可获得的结论

时间从两台 V100 同时可运行且网络可用时计算。C0 实测 k192 一般 q 点为 4.9–5.3 h，精确 K 为 1.4 h；下表以此为基准并留出 SCF、同步和重试余量。

| 阶段 | 计算内容 | 墙钟时间 | 阶段结束后能回答的问题 |
|---|---|---:|---|
| S0 输入冻结与 QE 准备 | 冻结 q 点、脚本审计；构建 `npk=120000` QE | 1–3 h；可与早期分析并行 | 输入是否真正没有 `degauss`；高密一般 q 是否具备运行条件 |
| S1 k192 pilot | A/B 各自 SCF；K 在两机重复；分别计算 K→Gamma、K→M 的 `d=0.007` | 6–8 h | `tetrahedra_opt` 是否能稳定用于 graphene phonon；两个方向是否都出现 K 局部下凹；跨机器是否一致 |
| S2 k 网格门槛 | k240 重算 K 与两个 `d=0.007` 点；必要时再做 k288 | k240 9–12 h；若触发 k288 再加 14–18 h | k192 是否足够；否则 k240 是否可作为正式网格；排除 Dirac 点采样伪影 |
| S3 development | 在选定网格补 `d=0.015,0.023` 两方向四点；拟合并冻结模型 | k192：10–13 h；k240：15–19 h；拟合 <1 h | 不看 holdout 时，哪一个预声明的 cusp 形状受到 development 数据支持；是否值得释放 holdout |
| S4 blind holdout | 两方向 `d=0.003,0.011,0.019`，每台三个一般 q 点 | k192：15–18 h；k240：23–28 h | 仓库 q-space 方法能否在未见 q 点定量复现无展宽 K cusp |
| S5 L0/Q0 组合和全谱 | 300/450/600 K，L0/Q0；Gamma–M–K–Gamma 六支背景谱和 K 放大图 | 1–3 h，无新 DFT | 固定无展宽 K 区电子修正时，晶格温度背景如何改变绝对频率，同时是否保留 cusp |
| S6 可选方向/网格加密 | 只在 S4 的斜率或窗口稳定性未过门槛时触发 | 10–30 h | 区分 q 点不足与方法失效；不用于事后放宽门槛 |

最佳路径约 42–54 h 得到 blind-holdout 结论，约 45–57 h 得到最终图。若 k192 未收敛并触发 k288，完整结论约 75–100 h。

### 4.1 实际启动记录

- V100-B 的 `pilot_B_k192_qe75_conda` 于 `2026-08-12T16:31:24+08:00` 启动，SCF 于 `16:41:22` 完成，随后进入精确 K 点 `ph.x`；
- V100-A 的 `pilot_A_k192_qe75_conda` 于 `2026-08-12T16:31:49+08:00` 启动，SCF 于 `16:42:26` 完成，随后进入精确 K 点 `ph.x`；
- 跨机器 monitor 于 V100-A 的 `2026-08-12T16:47:29+08:00` 启动，每 300 s 检查两个 lane；两边完成后自动同步 bundle、跟踪 A1′ 模并生成 S1 图和验收报告；
- V100-B 于 `2026-08-12T16:51:28+08:00` 并行启动独立 QE 7.5 `npk=120000` 构建，于 `16:57:23` 完成；`pw.x/ph.x` 已通过匹配 NVHPC `mpirun -np 1` 的运行时启动门槛；
- 新构建的 `pw.x/ph.x` 已复制到 V100-A，A/B 哈希分别一致；A 上的 MPI 启动门槛也已通过；现有 conda QE 未修改；
- k192 新旧 QE 重放于 V100-A 的 `2026-08-12T17:08:51+08:00` 启动。该任务使用独立目录、`1 MPI x 4 OpenMP` 和原本空闲的 V100，与 CPU pilot 并行；它只读取事先固定的 development 点，不释放 holdout；
- B 的重放曾于 `17:09:41` 启动，实际 SCF 验证新二进制能读取 `npk=120000` 并进入 `tetrahedra_opt`，但原 pilot 的 `_ph0` 已占 24 GiB，使 `/data` 仅余 37 GiB。为保持 35 GiB 安全余量，该重放于进入声子阶段前停止，其 20 KiB 未完成 scratch 已删除；延迟门控于 `17:14:17` 启动，只在原 B lane 完整打包、移除可再生 `_ph0` 且可用空间至少 55 GiB 后自动重启；
- B 上旧 C1 k168/k216 的两份 `_ph0` 在核对 k168 `DONE`、manifest/CSV SHA-256、两个 `JOB DONE` 原始输出以及无活动进程后删除；只移除了 24.4 GiB 可重算响应波函数，输入、SCF、逐 q 输出、manifest 和 CSV 均保留。`/data` 可用空间由 37 GiB 恢复到 60 GiB；
- 两台机器的 `pw.x`、`ph.x` 和 C 赝势 SHA-256 分别完全相同。pilot 的跨机器 K 点差异因此可以直接解释为数值复现误差。
- 原 conda QE 的 k192 pilot 已于 `2026-08-13T01:17:01+08:00` 前完成。A/B 的 K 点 A1' 频率分别为 `1281.164584/1281.164583 cm^-1`；`d=0.007` 的 K→Gamma、K→M 频率分别为 `1289.247374/1289.306229 cm^-1`，对应 cusp depth 为 `8.082791/8.141646 cm^-1`。跨机器六支频率最大差为 `0.000006 cm^-1`，矩阵重放最大差为 `4.54e-6 cm^-1`，S1 全部门槛通过；图和报告保存在 `results/graphene_k_cusp_nosmear/S1_pilot/`。
- 独立构建的 QE 7.5 `npk=120000` k192 重放已于 `2026-08-13T07:47:28+08:00` 完成。它相对原 QE 的三个 A1' 频率最大差为 `0.000151 cm^-1`，低于 `0.2 cm^-1` 的二进制重放门槛；结果保存在 `results/graphene_k_cusp_nosmear/S0_qe_replay_k192/`。
- S1 与独立重放通过后，只清除了两台机器上已经完成任务的 `_ph0` 响应波函数 scratch；输入、SCF 输出、逐 q 输出、动力学矩阵、CSV 和 manifest 均保留。本操作释放约 111 GiB，scratch 若需要只能重算。清理后 V100-A/B 的 `/data` 可用空间分别为 111/128 GiB。
- S2 k240 于 `2026-08-13T08:49:32+08:00`（`12:49:32 NZST`）在 V100-A、`08:49:40+08:00`（`12:49:40 NZST`）在 V100-B 启动，使用独立 QE 7.5、`1 MPI x 4 OpenMP` 和 `tetrahedra_opt`。A 计算 K 与 K→Gamma `d=0.007`，B 计算 K 与 K→M `d=0.007`；两边 SCF 分别于 `13:02:02/13:02:23 NZST` 完成并进入 K 点 `ph.x`，development 和 holdout 仍保持锁定。`05:00–06:00 NZST` 完成的是先前的 k192 S1；S1 完成后经历了结果同步与双二进制审计、独立 QE 重放、scratch 安全清理和 S2 释放，因此 k240 没有在该时段之前启动。
- k240 A/B 分别于 `2026-08-13T23:21:03/23:56:44 NZST` 完成。k192→k240 的三点 MAE 为 `0.228786 cm^-1`，K 点差为 `0.588583 cm^-1`，K→Gamma/K→M cusp depth 相对变化为 `7.1587%/7.0959%`，最小模式重叠为 `0.999849`；全部通过事先确定的 S2 门槛，因此正式网格冻结为 k192，不触发 k288。
- S2 bundle 同步并通过哈希和矩阵审计后，两台机器各删除了 `29 GiB` 的 k240 `_ph0` 可重算响应 scratch；持久结果完整保留。S3 development 于 `2026-08-14T03:31 NZST` 启动：A 计算 K→Gamma `d=0.015` 与 K→M `d=0.023`，B 计算 K→M `d=0.015` 与 K→Gamma `d=0.023`。两边复用已审计的 k192 SCF，并已进入实际 `ph.x`；blind holdout 仍锁定。
- S3 development 的 A/B 通道分别于 `2026-08-14T13:43:46/13:59:15 NZST` 完成。K→Gamma 在 `d=0.015/0.023` 的 A1′ 频率为 `1293.790744/1298.931402 cm^-1`，K→M 为 `1294.043122/1299.445485 cm^-1`；全部模式重叠不低于 `0.999757`。
- 在 holdout 计算前，三个事先确定的候选模型只用 pilot 与 development 数据完成 leave-one-distance-out 选择。固定模型为 `linear_directional`，CV MAE 为 `1.896440 cm^-1`，通过 `3 cm^-1` 门槛；冻结 manifest 和模型 SHA-256 分别为 `ab1e2925e4672d7c31f44d03ce277630541dcb71257c0074c399c722c9636906` 与 `5fd2decc5d7b80706b37befec2b62bcf61220caf376159c83298eb84e34789f6`。development 对比图已写入 `results/graphene_k_cusp_nosmear/S3_development_freeze/`。
- 固定模型与验收阈值核对后，S4 blind holdout 于 `2026-08-14T14:52:47 NZST` 在 V100-A、`14:52:33 NZST` 在 V100-B 启动。A 计算 `KG_d003/KM_d011/KG_d019`，B 计算 `KM_d003/KG_d011/KM_d019`；两边均复用 k192 SCF，并已确认第一个一般 q 点的 `ph.x` 实际运行。现有 `/data` 可用空间为 `87/104 GiB`，高于 `75 GiB` 启动门槛，因此本阶段未删除 S3 的 `19 GiB` 响应 scratch。
- S4 的 A/B 通道分别于 `2026-08-15T06:26:00/06:45:53 NZST` 完成，自动监控于 `06:47:50 NZST` 完成同步、哈希审计、固定模型验收和最终绘图。六点频率 MAE 为 `1.815196 cm^-1`、最大绝对误差为 `3.766638 cm^-1`，两方向单边斜率相对误差为 `7.324%/6.876%`，slope-jump 相对误差为 `7.098%`，最小模式重叠为 `0.999736`，Hermiticity 与 rank-one replay 门槛均通过。
- 正式 S4 状态为 `failed_blind_holdout`。唯一失败项是逐点 cusp-depth 相对误差：`d=0.003` 的 K→Gamma/K→M direct depth 为 `6.224609/6.251738 cm^-1`，冻结模型只预测 `2.457971/2.518486 cm^-1`，相对误差为 `60.512%/59.715%`；`d=0.011` 的相对误差降至 `15.117%/14.176%`，`d=0.019` 为 `0.924%/0.779%`。结果表明误差集中在极近 K 区域，不能通过整体 MAE 或绘图平滑掩盖。
- S4 原始 bundle 和最终图归档后，两台机器各删除了约 `19 GiB` 的 k192 `_ph0` 可重算响应 scratch，所有持久输入和输出保留，可用空间恢复至 `101/121 GiB`。S4a 于 `2026-08-15T14:40:12/14:40:23 NZST` 启动，V100-A/B 分别计算 k240 `KG_d003/KM_d003`；两边复用 S2 的 k240 SCF，并已进入实际 `ph.x`。判定门槛固定为频率变化不超过 `1 cm^-1` 且 cusp-depth 相对变化不超过 `10%`；通过则归因为 v1 模型形状不足，否则先做 k288，不重拟合。
- 等待 S4a 时，用已经揭盲的 S4 数据做了 v2 开发性试验；这些点从此只属于 development，不能再作为 v2 独立验证。第一版无约束 two-scale exponential 虽得到 `0.353970 cm^-1` 的 leave-one-distance-out MAE，但尺度压到 `1e-5`，在 K 旁形成近似台阶，与文献中的连续有限斜率 K cusp 不符，因此该版本和原图均作废。修正版按 `omega(K+q')=omega_K+alpha_K|q'|+O(q'^2)` 限制 K 点频率连续、单边斜率有限，并规定 crossover 尺度不小于最小采样距离 `d=0.003`。选出的 finite-slope rational 模型 leave-one-distance-out MAE 为 `0.809765 cm^-1`，相对线性 v1 的 `1.720537 cm^-1` 降低约 `52.9%`；全数据拟合 MAE/最大误差为 `0.491142/1.167515 cm^-1`，K→Gamma/K→M 的 inner-to-outer slope ratio 相对误差为 `7.020%/5.554%`，K 点单边斜率均为有限值 `2891.95/2911.11 cm^-1 per d`。选定尺度仍位于 `0.003` 的分辨率下界，因此只作为物理约束后的开发模型，等待 k240 的 S4a 判定后再确定是否补更近 K 的点。

pilot 完成前不释放 development 或 holdout。S1 通过后只释放 k 网格门槛；S2 选定正式网格、S3 冻结模型后才释放 blind holdout。

## 5. 每阶段验收和停止条件

### S1 pilot

- 两台 SCF 和四个 `ph.x` 任务均正常结束；
- 输入包含 `tetrahedra_opt`，且不含 `smearing`/`degauss`；
- `gr.dyn` Hermiticity 打印精度残差不超过 `5e-7`；
- 动力学矩阵重建频率误差不超过 `0.1 cm^-1`；
- 两台独立 K 点的六个频率最大差不超过 `0.2 cm^-1`；
- 追踪的 $A_1'$ 模在两个 `d=0.007` 邻点均高于 K。

若四面体权重、SCF 或 `ph.x` 本身不稳定，停止扩大 q 点。先比较 `tetrahedra_opt` 的网格/平移设置，不以极小有限 `degauss` 冒充通过。

### S2 k 网格

- 最密两个候选网格的三个 A1′ 点 MAE不超过 `1 cm^-1`；
- K 点绝对差不超过 `1 cm^-1`；
- 两个方向在 `d=0.007` 的 cusp depth 变化均不超过 10%；
- `npk` 新编译版与原 QE 7.5 的 k192 重放差不超过 `0.2 cm^-1`。

k192 对 k240 通过时，正式 development/holdout 使用 k192；否则必须等待 k288 对 k240 的复核。

### S3 development

- 所选模型的 development leave-one-distance-out MAE不超过 `3 cm^-1`；
- 两个方向的 K 点单边线性系数均对应 K 局部极小；
- 所有相邻模式本征矢重叠不低于 0.95；
- 模型、参数、输入哈希和 S4 验收标准写入冻结 manifest 后，才创建 holdout release marker。

### S4 blind holdout

- holdout K-window MAE `<10 cm^-1`，最大绝对误差 `<15 cm^-1`；
- 两个方向各自的 cusp depth 相对误差 `<20%`；
- 两个方向的单边斜率相对误差 `<20%`；
- 合并后的 slope-jump 相对误差 `<20%`；
- Hermiticity `<=1e-10`、rank-one 标量重放误差 `<=1e-6 cm^-1`；
- time-reversal 频率差 `<=1e-6 cm^-1`，K-star 频率散布 `<=1 cm^-1`；
- 不允许通过样条平滑或改变绘图窗口制造 cusp。

K 点参与 development，是两方向共享的物理锚点；K 点重放属于构造检查，不计入独立 holdout 精度。独立结论由六个 holdout 点和方向斜率给出。

## 6. 数据目录与状态标记

远端工作目录：

```text
/data/graphene_k_cusp_nosmear/
  sets/k192_tetra/                 # SCF tmp 和逐 q 原始输出
  sets/k240_tetra/
  sets/k288_tetra/
  lanes/<stage>_<lane>_k<grid>/    # CSV、manifest、日志、DONE/FAILED
  qe_build/                         # 独立 QE 构建日志和审计
  RELEASE_CONVERGENCE
  SELECTED_KGRID
  RELEASE_DEVELOPMENT
  RELEASE_HOLDOUT
```

本地结果目录：

```text
results/graphene_k_cusp_nosmear/
  raw/<stage>/<lane>/
  S1_pilot/
  S2_kgrid/
  S3_development_freeze/
  S4_holdout/
  S5_finite_lattice/
```

`DONE` 只表示该 lane 的原始计算完整；`PASS` 表示本地审计通过。后续阶段只读取 `PASS` 和冻结 manifest，不读取日志中的人工判断。

## 7. 最终交付

1. direct `tetrahedra_opt` 的 K→Gamma 和 K→M 原始点与 k 网格收敛图；
2. 静态仓库方法 development/holdout 预测图，holdout 使用不同标记；
3. 300/450/600 K 的 L0/Q0 全 Γ–M–K–Γ 六支背景谱，并只在经过验证的 `d<=0.023` K 窗口叠加无 degauss q-space 结果；窗口外不标成无展宽电子响应；
4. 每个全谱面板附 K 区放大图，直接点可见，legend 位于数据区外；
5. cusp depth、两个单边斜率、slope jump、K 频率、MAE 和对称性审计表；
6. 原始输入、二进制/赝势哈希、模型冻结 manifest 和阶段报告。

图中电子条件统一写为 `electronic integration: tetrahedra_opt (no degauss)`；晶格温度单独写 `lattice temperature (K)`。不换算或标注虚构的电子温度。
