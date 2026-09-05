# Graphene 物理温度模型实验与算力排期

**对应理论方案：**[`GRAPHENE_PHYSICS_BASED_DEGAUSS_LATTICE_TEMPERATURE_PLAN.md`](GRAPHENE_PHYSICS_BASED_DEGAUSS_LATTICE_TEMPERATURE_PLAN.md)

**状态基线：**2026-08-03 11:48 HKT / 15:48 NZST

**适用机器：**V100-A、V100-B、RTX 2060 和本地工作站

## 1. 排期目标

本排期服务于两个独立物理通道及其组合验证：

- 电子通道：用 Mermin 自由能和有限展宽电子极化引入 `degauss (Ry)`；
- 晶格通道：用统一非谐势、统计系综和 TDEP/SSCHA/SCPH 引入 lattice temperature；
- 二维组合：在非对角 \((T_{\mathrm{lat}},s)\) 条件上检验两种效应能否分离，以及交叉项是否可忽略。

算力安排遵循以下原则：

1. 当前 450 K 冻结实验先按原方案完成，新模型不能改写它的输入、标签或验收标准。
2. 开发阶段先用已有结果和少量交叉标签判断模型结构，再决定是否扩充数据。
3. 验证标签只在模型、验证点、构型索引和门槛全部固定后生成。
4. V100 优先用于 DFT/DFPT；RTX 2060 优先用于 MLIP、轨迹、TDEP 和 SSCHA。
5. 不直接运行完整的 \(3\times3\) DFT-MD 网格。只有交叉项显著时才增加耦合轨迹。
6. 所有远程连接、提交和同步均通过 Tailscale `100.x` 地址。

## 2. 当前三台机器的实测状态

### 2.1 硬件与队列

| 机器 | 硬件 | 当前任务 | 当前可用空间 | 实测速度 |
|---|---|---|---:|---|
| V100-A | V100 32 GB；16 CPU；31 GiB RAM | `FD450_CONV`，随后自动 `FD450_LINE` 和 shard A | `/data: 171 GB`，清理后 64% | 低 `degauss`、\(k=144\) 的 K 邻域 `ph.x` 约 2.2–2.4 h/q point |
| V100-B | V100 32 GB；16 CPU；31 GiB RAM | 等待 450 K bootstrap，随后 shard B | `/data: 108 GB` | 72 原子、\(8\times8\times1\) 的 force label 约 14–16 min/构型 |
| RTX 2060 | RTX 2060 SUPER 8 GB；16 CPU；46 GiB RAM | 450 K 轨迹扩展到 3000 帧/seed | 根分区 73 GB | seed 0/2 约 350 saved frames/h；seed 1 约 190 saved frames/h |

RTX 2060 的 `phonon` Conda 环境已经安装 `sscha`、`cellconstructor`、`mace`、`torch` 和 `phonopy`，可直接承担后续 MLIP 与晶格温度任务。

### 2.2 当前冻结实验的剩余时间

以下估计基于上述时间点的实测进度，不包含网络中断和磁盘故障：

| 机器 | 剩余工作 | 预计剩余墙钟 |
|---|---|---:|
| RTX 2060 | seed 1 从 764 到 3000、seed 2 从 120 到 3000、TDEP 与 bootstrap | 约 21–23 h |
| V100-B | 等 bootstrap 后计算 30 个 force labels | 从当前起约 28–30 h，其中真正计算约 7–8 h |
| V100-A | `FD450_CONV` 剩余 K 点、约 40–45 h dense line、30 个 force labels | 约 55–65 h |

按当前速度，RTX 2060 预计在 2026-08-04 中午至下午早段 NZST 完成。V100-B 预计在 8 月 4 日晚间完成；V100-A 预计在 8 月 5 日晚间至 8 月 6 日凌晨完成。

V100-A 的磁盘门槛已于 2026-08-03 12:09 HKT 达到，当前时间估计不再受原先的零可用空间限制。

## 3. 启动前的存储门槛

### 3.1 当前风险

V100-A 的 `/data` 为 492 GB。状态检查时使用 468 GB，`df` 显示可用空间为 0。主要占用包括：

- `/data/graphene_fd_force_convergence`：173 GB；
- `/data/results`：103 GB；
- `/data/graphene_physical_fd_dfpt`：68 GB；
- `/data/phonon_offload`：34 GB。

核对旧 force-convergence 服务、`DONE`、进程占用和本地/远端 summary 哈希后，已清理三个 `pwscf.wfc1` 及 222 个分布式 `wfc*.dat`。旧任务目录由 173 GB 降至 102 MB，`/data` 现有 171 GB 可用；输入输出、XML、charge density、summary、日志和完成标记均保留。清理记录见 `results/graphene_fd_force_convergence/A/cleanup_2026-08-03.json`。

### 3.2 正式门槛

任何新任务启动前要求：

| 机器 | 最低可用空间 | 原因 |
|---|---:|---|
| V100-A | 100 GB | dense DFPT line、QE save 目录和失败重试 |
| V100-B | 100 GB | 交叉 force labels 与验证 DFPT |
| RTX 2060 | 100 GB | 多温度轨迹、checkpoint、SSCHA ensemble |

当前 V100-A 已满足门槛；RTX 2060 仍需要在当前轨迹同步后再释放约 30 GB。后续存储处理单独执行，不自动删除文件。

建议顺序：

1. 对完成任务的 JSON、CSV、输入、输出和最终矩阵做本地/远端 SHA-256 对照；
2. 确认对应 systemd service 已结束且没有 `pw.x/ph.x` 使用目录；
3. 只清理可重建的 `*.wfc*`、临时 charge-density 和失败 attempt 目录；
4. 保留输入、日志、最终 dynamical matrix、summary 和 manifest；
5. 清理后再次执行 `df`，达到门槛才释放下一阶段。

## 4. 实验依赖关系

```text
P0 当前 450 K 冻结实验
        |
        +--> R0 数据/存储审计
        |
        +--> E0 电子极化公式开发 ----> E1 交叉 degauss force pilot
                                            |
                                      局域项判别 gate
                                       /            \
                                  可忽略          需要局域项
                                    |                 |
                                    |          E2 扩充交叉标签
                                    +--------+--------+
                                             |
                                      S0 统一短程 MLIP
                                             |
                              L0 classical TDEP / Q0 SSCHA
                                             |
                               X0 二维开发网格与交叉项
                                             |
                                      冻结模型和门槛
                                             |
                        VP 验证轨迹 + V0 静态 DFPT
                                             |
                                  V1/V2 force/TDEP 验证
                                             |
                                  I0 独立 DFT-MD 验证
```

`E1` 和 `E2` 是开发数据，可以根据事先写明的 gate 决定是否扩充。`V0/V1` 是未见验证数据，不能根据中间结果反复增加标签直到通过。

## 5. 开发阶段的具体实验

### R0：数据与软件能力审计

内容：

- 封存当前 300/450/600 K 的模型、轨迹、DFPT 和 DFT-TDEP；
- 审计已有 `graphene_epw`、`graphene_epw3`、电子能带和电子—声子矩阵元能否复用；
- 确认 QE 输出中的 Mermin free energy、电子熵项和 force 定义；
- 固定参考展宽 \(s_0=0.00285013035\ \mathrm{Ry}\)；
- 建立 development、validation_locked 和 independent_dftmd 数据边界。

算力：

- 本地 CPU：4–8 h；
- RTX 2060：最多 1 GPU-h 用于矩阵重放；
- 不需要新 DFT。

产物：

- `results/graphene_physics_temperature/inventory/inventory.json`；
- `results/graphene_physics_temperature/freeze_manifest.json` 第一版；
- EPW/band-sum 的 go/no-go 记录。

### E0：有限展宽电子响应原型

先用已有 300/450/600 K 静态 DFPT line 开发两种模型：

1. 主模型：固定电子结构和电子—声子矩阵元后的 finite-smearing band sum；
2. 低成本备选：共享 \(v_F,\mu,g_\Gamma,g_K\) 的 Dirac 极化模型。

当前 `rounded cusp + q²` 线性系数模型只作消融对照。

算力：

- 本地或 RTX 2060：8–16 h；
- 若已有 EPW 数据不完整，先做一个 \(s_0\) 的 EPW feasibility pilot：12–24 V100 box-h；
- 只有 pilot 能在 dynamical-matrix 层面重放 \(s_0\)，才扩展到收敛 band sum；预计再需 12–36 V100 box-h。

停止条件：

- \(s_0\) dynamical matrix 无法在统一模规范下重放；
- k 网格或 Wannier 插值误差已经超过最终 10/15 cm⁻¹ 门槛；
- 需要为每个 `degauss` 独立拟合参数才能通过开发点。

上述情况发生时先修电子模型，不进入验证点。

### E1：交叉 `degauss` force pilot

从 300、450、600 K 各固定 5 个已有热构型。每个构型补齐三个开发 `degauss (Ry)`：

\[
\{0.0019000869,\ 0.00285013035,\ 0.0038001738\}\ \mathrm{Ry}.
\]

已有的对角标签直接复用。每个构型通常只需补两个条件，因此新增：

\[
3\ T_{\mathrm{lat}}\times5\ \text{构型}\times2\ s
=30\ \text{DFT single points}.
\]

算力：

- 总计约 8–10 V100 box-h；
- 两台 V100 各 15 个时，墙钟约 4–5 h；
- 如果 V100-A 仍执行 dense line，可由 V100-B 单独运行，墙钟约 8–9 h；
- RTX 2060 后处理约 1–2 h。

判别：

- 先扣除物理 Kohn-anomaly 谐波力；
- 若剩余 force RMSE 满足主计划第 8 节门槛，冻结局域项为零；
- 未满足时进入 E2。

### E2：局域 Mermin 自由能扩充，条件执行

每个晶格温度扩展到 20 个固定索引构型。相对 E1 需要新增 15 个构型/温度，每个补两个 `degauss`，即：

\[
3\times15\times2=90\ \text{additional DFT single points}.
\]

算力：

- 总计约 23–28 V100 box-h；
- 两台 V100 平分后约 12–14 h 墙钟；
- 局域熵/自由能模型训练与积分检查约 8–20 RTX-2060 GPU-h。

E2 完成后只允许冻结一个主局域模型和一个无局域项消融，不做大规模超参数搜索。

### S0：统一短程非谐 MLIP

训练数据包括三个开发晶格温度的构型，但模型不接收 \(T_{\mathrm{lat}}\)。所有标签使用同一个物理 Kohn 算子做扣除；需要时加入 E2 的可积分 Mermin 局域项。

先运行固定 10 epochs benchmark，再据实测速率一次性确定训练 epochs 和墙钟预算。

算力预留：

- RTX 2060 主模型与一个消融：30–70 GPU-h；
- 如果 10 epochs benchmark 外推单个模型超过 36 h，且没有 DFT 验证任务等待，可把冻结训练转到一台空闲 V100，预计 8–16 V100 GPU-h；
- 不允许同时占用两台 V100 做 MLIP sweep。

验收：

- 未见构型 force RMSE/max `≤50/250 meV/Å`；
- harmonic replay 不超过冻结门槛；
- energy–force 和 `degauss` 积分闭合检查通过。

### L0：晶格温度开发

第一轮只在参考 \(s_0\) 运行三个晶格温度，不跑完整 \(3\times3\) 组合：

\[
T_{\mathrm{lat}}\in\{300,450,600\}\ \mathrm K.
\]

每个温度使用三个固定 seed，按当前 3000 帧统计标准估算：

- seed 0/2：各约 8–9 h；
- seed 1：约 15–16 h；
- 单个温度总计约 30–32 RTX-2060 GPU-h；
- 三个温度约 90–96 RTX-2060 GPU-h。

三温度轨迹顺序运行。每个温度结束后可以做开发诊断，但最终 sampling gate 只在固定帧数上评价一次。

### Q0：quantum SSCHA/SCPH

先在 450 K 做一个固定 population 数和迭代数的 benchmark。当前环境已有 SSCHA 依赖，历史小体系任务约 15–30 min/(条件)，但本项目要求完整 Γ/K 定量谱，预算按更保守的多 population 收敛计算。

算力预留：

- 450 K benchmark：2–4 RTX-2060 GPU-h；
- 300/450/600 K 正式 quantum 计算：10–20 RTX-2060 GPU-h；
- 若单个温度在固定最大迭代数内不收敛，停止并分析，不无限增加 population。

classical TDEP 与 quantum SSCHA/SCPH 分别报告。两者共用势能面，但不共用物理结论。

### X0：电子—晶格交叉项

先用 E0/E2 的自由能模型在 MLIP 层面计算二维开发网格。若第一个非对角条件的交叉项对 Γ/K line 影响小于 2 cm⁻¹，则其余组合使用公式重建，不增加完整三 seed 轨迹。

如果交叉项超过 2 cm⁻¹：

- 对需要的非对角条件先运行一个固定 seed 的开发轨迹；
- 根据事先固定的条件列表补齐三 seed；
- 预留 30–120 RTX-2060 GPU-h，取决于需要补算的组合数。

这个分支只影响开发算力。验证阶段仍按冻结后的固定条件一次评价。

## 6. 未见验证阶段

验证点保持为：

\[
T_{\mathrm{lat}}\in\{375,525\}\ \mathrm K,
\qquad
s\in\{0.002375108625,\ 0.003325152075\}\ \mathrm{Ry}.
\]

在任何验证 DFT 生成前，保存模型、公式参数、代码哈希、构型索引和门槛。

### VP：冻结模型的验证轨迹

验证构型必须由冻结后的统一模型生成，不能直接复用开发温度的轨迹。VP 与 V0 static DFPT 可以并行运行。

最小 E1+L1 结论只生成两个 physical-FD 对角条件：

- \((375\ \mathrm K,0.002375108625\ \mathrm{Ry})\)；
- \((525\ \mathrm K,0.003325152075\ \mathrm{Ry})\)。

每个条件使用三个固定 seed 和 3000 帧/seed，预计约 30–32 RTX-2060 GPU-h；两个条件合计 60–64 GPU-h。

如果目标包括 X1，再生成两个交换后的非对角条件，额外需要 60–64 RTX-2060 GPU-h。VP 的轨迹设置、DFT 构型索引和 bootstrap 规则在运行前一次固定。

### V0：未见 `degauss` 静态 DFPT

两台 V100 分工：

- V100-A：`0.002375108625 Ry`；
- V100-B：`0.003325152075 Ry`。

每个条件先完成与当前一致的 \(k=120\rightarrow144\) 八点收敛 gate；通过后自动运行 \(k=144\) dense line。

按当前低 `degauss` 实测速率：

| 子任务 | 单条件 box-h | 两条件总 box-h | 两台并行墙钟 |
|---|---:|---:|---:|
| k 网格收敛 | 27–32 | 54–64 | 27–32 h |
| dense line | 40–45 | 80–90 | 40–45 h |
| 合计 | 67–77 | 134–154 | 67–77 h |

任一 k 网格 gate 失败时，对应 dense line 保持关闭，先增加 k 网格，不读取不收敛结果作为模型成败。

### V1：四个组合的 force screen

固定四个条件：

- \((375\ \mathrm K,0.002375108625\ \mathrm{Ry})\)；
- \((525\ \mathrm K,0.003325152075\ \mathrm{Ry})\)；
- \((375\ \mathrm K,0.003325152075\ \mathrm{Ry})\)；
- \((525\ \mathrm K,0.002375108625\ \mathrm{Ry})\)。

每个条件固定 15 个构型，共 60 个 DFT single points。

最小方案可以在同一晶格温度的对角 VP 构型上分别计算两个 \(s\)，因此 V1 不要求先生成非对角轨迹。只有 V3 的非对角 DFT-TDEP 需要独立的非对角 VP 轨迹。

算力：

- 总计 15–18 V100 box-h；
- 两台平分后约 8–9 h 墙钟；
- 只做一次 force gate，不用这些结果调参。

### V2：对角条件 DFT-TDEP

V1 通过后，把两个 physical-FD 对角条件各扩展到 60 个标签。已有 15 个/条件，因此新增 90 个标签：

- 总计 23–28 V100 box-h；
- V100-A/B 各负责一个温度，墙钟约 12–14 h；
- DFT-TDEP 拟合、矩阵比较和 bootstrap 在 RTX 2060 上约 2–4 h。

### V3：非对角条件 DFT-TDEP，X1 结论需要时执行

如果目标包括“电子与晶格温度已经在二维条件上解耦”的 X1 结论，则两个非对角条件也各扩展到 60 个标签：

- 新增 90 个标签；
- 23–28 V100 box-h；
- 两台并行约 12–14 h。

只要求 E1/L1 时可以不执行 V3，但结论必须限制在各自的一维通道。

### I0：独立 DFT-MD

所有静态、force 和 on-policy DFT-TDEP gate 通过后，运行至少一条独立 DFT-MD。首选非对角条件：

\[
(450\ \mathrm K,0.0019000869\ \mathrm{Ry}).
\]

沿用 72 原子、checkpointed 60 snapshots 的规格。根据现有 300/600 K DFT-MD 实测，单条轨迹预留：

- 240–336 V100 box-h；
- 10–14 天墙钟；
- 一个 V100 独占，另一台可做分析或第二条轨迹。

如需同时验证 physical-FD 对角点和非对角点，两台 V100 可各运行一条，墙钟仍约 10–14 天，总算力为 480–672 V100 box-h。

I0 是全计划最昂贵的部分，只在前序全部通过后启动。

## 7. 算力总预算

以下预算不含当前 P0 剩余任务，也不含失败后的方法重构。

### 7.1 最小定量方案：E1 + L1

包括电子公式、交叉 pilot、统一 MLIP、三个开发晶格温度、两个对角验证轨迹、两个未见 `degauss` 的 static DFPT、四条件 force screen 和两个对角 DFT-TDEP。

| 算力 | 预算 |
|---|---:|
| V100，已有 EPW 数据可复用 | 180–210 box-h |
| V100，需要重建 EPW/band sum | 205–270 box-h |
| RTX 2060 | 200–275 GPU-h |
| 本地 CPU | 20–40 CPU-h |
| 两台 V100 并行后的关键墙钟 | 约 4–7 天 |
| RTX 2060 顺序关键墙钟 | 约 9–12 天 |

两类任务可以重叠。当前实验完成、磁盘达标之后，最小定量结果预计需要约 12–16 天。

### 7.2 二维解耦方案：增加 X1

根据局域项和交叉项大小，额外预算：

| 算力 | 额外预算 |
|---|---:|
| V100 | 25–60 box-h |
| RTX 2060 | 90–185 GPU-h |
| 墙钟 | 约 3–8 天 |

其中 60–64 GPU-h 是两条非对角验证轨迹的固定成本。交叉项不显著时接近下界；开发阶段需要多个耦合轨迹时接近上界。

### 7.3 独立轨迹方案：增加 C2

每条独立 DFT-MD 额外需要 240–336 V100 box-h 和 10–14 天墙钟。完整 E1+L1+X1+C2 从当前冻结实验完成后算起，预计约 4–6 周。

## 8. 推荐机器队列

### 当前到 P0 完成

- V100-A：只运行 `FD450_CONV → FD450_LINE → shard A`；
- V100-B：等待 bootstrap 后运行 shard B；
- RTX 2060：只完成 3000 帧/seed、TDEP 和 bootstrap；
- 本地：只读监控、同步和文档，不生成新标签。

### P0 完成后的第 0–2 天

- 本地：R0 inventory、公式原型和 freeze manifest；
- V100-B：E1 的 30-label pilot；V100-A 完成存储审计；
- RTX 2060：E0 Dirac/band-sum 原型及矩阵重放；
- 两台 V100 都空闲且存储达标时，E1 改为 15+15 并行。

### 第 2–7 天

- RTX 2060：S0 统一模型，随后 L0 三温度轨迹和 Q0；
- V100-A/B：只在 E2 被触发时计算额外交叉标签；否则保持空闲或做开发 EPW；
- 本地：交叉项分析，准备 validation freeze。

### 第 7–16 天

- V100-A：375 对应 `degauss` 的 V0；
- V100-B：525 对应 `degauss` 的 V0；
- RTX 2060：先完成 VP 的两个对角验证轨迹；需要 X1 时再完成两个非对角验证轨迹；
- V0 完成后，两台 V100 并行 V1/V2。

### 第 16 天以后

- 达到 E1/L1 即先形成阶段性结果；
- 需要 X1 时追加 V3；
- 全部门槛通过后再把一台或两台 V100 分配给 I0 独立 DFT-MD。

## 9. 调度与失败恢复

- 每台 V100 同时只运行一个 QE 主任务，`pw.x` 与 `ph.x` 不并发。
- 每个 q point、force label 和 MD snapshot 完成后写原子级 checkpoint。
- systemd service 必须幂等：`DONE` 存在时不重算，partial 文件不能冒充完成结果。
- V100 节点可用空间低于 50 GB 时不启动新 QE point；低于 20 GB 时当前 supervisor 在安全 checkpoint 后停止。
- RTX 2060 可用空间低于 50 GB 时不开始新 seed 或 SSCHA population。
- 同一验证条件不因中间误差接近门槛而追加帧或标签。
- 失败重试只允许修复基础设施或数值收敛；修改物理公式后，原验证点转为开发数据。

## 10. 当前建议

1. 暂不启动新物理模型实验，先让 450 K 冻结实验完成。
2. V100-A 已有 171 GB 可用，满足 `FD450_LINE` 的存储门槛；运行期间继续监控。
3. 当前结果封存后，先做 30-label 交叉 `degauss` pilot，而不是直接跑九个条件的 DFT-MD。
4. 同步开发 E0 电子极化公式；只有电子公式和局域项判别都通过，才训练统一短程模型。
5. 最小目标先完成 E1+L1，预计当前实验结束后约 12–16 天。
6. 独立 DFT-MD 作为最后一关；完整 E1+L1+X1+C2 预计需要约 4–6 周。
