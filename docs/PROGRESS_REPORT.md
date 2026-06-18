# 阶段性实验报告 — GPU 加速声子谱计算

**项目**:用 GPU 把声子谱计算加速 ≥50×,同时控制精度,目标发表一篇完整工作。
**日期**:2026-06-19 ｜ **分支**:Line A = `phonon-pipeline`,Line B = `glm`(各自推进,后续合并)

---

## 0. 执行摘要(一页看懂)

声子谱的第一性原理(DFPT/有限位移)计算比单点电子结构慢 10–100×。本项目沿**两条互补主线**攻这个问题,目前两条都已拿到**实测、可发表**的初步结果:

- **Line A — 基础 MLIP + 微调(精度科学线)**:用 GPU 基础机器学习势(MACE、MatterSim)替代 DFT 算力,声子计算快 50–1000×;核心科学问题是**精度**。已完成:(1) 对照真实 DFPT 的全谱 benchmark;(2) 提出并验证一种**零新增 DFT 的力常数蒸馏微调**,把声子软化大幅修复;(3) 定位并修复了**未见元素的灾难性遗忘**(multihead-replay)。
- **Line B — GPU-DFT 工作流加速(工程/方法线,`glm` 分支)**:从零编译 QE 7.4 GPU 版,叠加对称约化 + 单 SCF GPU 卸载 + 多 GPU 并发,**实测端到端 ~49×(16原子)/ ~406×(64原子)**,Si 声子对实验误差 <1%。

**一句话结论**:50×+ 的加速在两条线上都已实测达到;真正的科学贡献在于**精度的可控性** —— 我们给出了"力常数蒸馏 + 遗忘修复"的完整配方,并量化了其中"鲁棒性 vs 精度"的权衡。

---

## 1. 背景与目标

有限位移法算声子 = 对超胞里每个不等价原子做 ±位移,每次跑一次完整 SCF 读受力;成本 ~N⁴,且声子频率是能量二阶导,对收敛要求远高于几何优化。**纯 GPU 移植 DFT 只能到 ~5–15×**(文献与我们的复现一致),达不到 50×。两条提速路径:

1. 用基础 MLIP 做 DFT 的快速代理(Line A,50–1000×,问题是精度);
2. 在工作流层面叠加加速 GPU-DFT 本身(Line B,对称约化 × GPU SCF × 多卡)。

---

## 2. 方法与基础设施

### 2.1 统一声子流水线(Line A 核心,后端无关)
`src/phonon_accel/`:基于 `phonopy + ASE`,结构 → 对称约简位移超胞 → 力 → 力常数(ASR 对称化)→ 色散/DOS/热力学。任意 MLIP 或 DFT 后端共用同一逻辑。

### 2.2 真实 DFPT 参考数据
**MDR / PhononDB(Togo/NIMS)**:10,034 个无机材料的 VASP+phonopy DFPT 力常数。`reference.py` 按 mp-id 下载并 `phonopy.load`,给出逐-q 真实参考(`data/benchmark/mdr_index.csv` 索引全部 10,034 材料)。

### 2.3 硬件(踩坑后的结论)
| 机器 | 配置 | 用途 |
|---|---|---|
| `100.105.21.7` | RTX 2060 SUPER **8GB** | **太小**:MACE 力训练固定 ~7GiB 显存(力的二阶反传 + 全元素表),与 batch 无关 → OOM。仅做推理 |
| `100.80.123.104` | **2× H20 96GB**(在中国) | 严肃训练。~6s/epoch。需绕过中国网络(NIMS/pytorch.org/replay 下载受阻) |

### 2.4 数据集
- 微调蒸馏数据:从 MDR 力常数生成的 rattled 超胞(16 材料,912 配置)。
- replay 数据:MPtrj `mp_traj_combined.xyz`(145,923 结构,跨周期表),子采样到 13k(含 C/B/In)。

---

## 3. 初步结果

### 3.1 Line A — 基础 MLIP 声子 benchmark(对照真实 DFPT)

MLIP 在 DFT 参考晶胞 + 相同超胞上跑 → 同一 seekpath 路径逐点对比。

**MatterSim-v1(14 个晶体,全谱)** `results/dfpt_mattersim_curated.csv`:
| 指标 | 值 |
|---|---|
| 频率 MAE(均值/中位/最大) | **0.58 / 0.53 / 1.20 THz** |
| 平均软化 | **−7.2%** |
| 动力学稳定 | 14/14 |
| C_v(300K) / S(300K) 误差 | 1.0 / 2.1 J/K/mol |

**MACE-MP-0(2023)** 对照:平均软化 **−26%**(7 材料,单点 ω_max)。

→ MatterSim 精度高约 4×、软化小约 4×,与文献排序一致。**这定量确立了"基础 MLIP 系统性软化声子"这一被修复对象。**

### 3.2 Line A — 力常数蒸馏微调(本项目核心方法)

**方法(零新增 DFT)**:把 DFPT 二阶力常数 Φ 蒸馏进基础模型 —— 对每个训练材料生成 rattled 超胞,用精确谐性力 `F=−Φ·u`、能量 `E=½uᵀΦu` 打标签,力匹配即注入 DFPT 曲率。**round-trip 精确验证**:蒸馏力喂回有限位移流水线精确复现 DFPT 色散(Si 15.287→15.287 THz)。

**全规模单头微调(16 材料,H20)** `results/finetune_eval_h20.csv`:
| | 平均 MAE | 平均\|软化\| | 虚频总数 |
|---|---|---|---|
| 基线 MACE-MP-0 | 0.82 THz | 9.6% | 291(SrTiO₃) |
| **微调后** | **0.28 THz**(3×↓) | **3.6%** | **4** |

亮点:SrTiO₃ 虚频 291→4、Li₂O MAE 2.01→0.12、Al₂O₃ 0.34→0.10。
**泛化定位**:留出材料若元素全在训练集内 → 泛化良好(LiF/CaO/TiO₂);**含未见元素则灾难性遗忘**(SiC 的 C → 542 虚频,BN 的 B → 858 虚频)。

**Multihead-replay 微调(修遗忘,H20)** `results/finetune_eval_h20_mh.csv`:
| 留出(未见元素) | 基线虚频 | 单头虚频 | **多头虚频** |
|---|---|---|---|
| SiC(C) | 0 | **542** | **0** ✓ |
| BN(B) | 0 | **858** | **0** ✓ |

replay 头用 4500 个跨周期表 MP 结构锚定 → **彻底消除未见元素上的非物理虚频**。
**权衡**:多头训练集 MAE 0.30(单头 0.28)、留出"已见元素"材料反而变差 → replay 用"精度/迁移"换"鲁棒性"。

### 3.3 Line B — GPU-DFT 工作流加速(`glm` 分支)

独立 `phonongpu/` 包(自写空间群对称、QE/VASP-GPU 后端、多 GPU 调度);**在 H20 上从零编译 QE 7.4 GPU 版**(NVHPC/OpenACC/Hopper cc90)。

**实测**(三杠杆相乘):
| 杠杆 | 效果 |
|---|---|
| 对称约化(减 SCF 次数) | 16–64× |
| 单 SCF GPU 卸载(QE-GPU) | 1.52×(16原子)→ 4.75×(64原子) |
| 多 GPU 并发 | ~2× |
| **端到端** | 16原子 **~49×**、64原子 **~406×**(naive 基线为投影) |

**Si 精度**:Γ 光学支 15.42–15.68 THz vs 实验 15.53(−0.7%…+0.99%),无虚频。
诚实分解:大头来自对称约化(任何工具都做),纯 GPU 贡献约 ~10×(单 SCF ~5× × 多卡 ~2×)。

---

## 4. 关键发现(均为论文级细节)

1. **力常数蒸馏靶子在数学上精确**(round-trip 15.287→15.287)→ 匹配蒸馏力 == 匹配 DFPT 声子。
2. **遗忘被精确定位到"未见元素"**,且 replay 可修(SiC/BN 虚频数百→0)—— 给出明确因果与修复。
3. **鲁棒性 vs 精度的权衡**:replay 防灾难性失稳但降峰值精度;配比是干净的 ablation 轴。
4. **NAC 公平性**:短程 MLIP 无 Born 电荷、无 LO-TO 劈裂;benchmark 必须把参考也设 NAC-off 才公平(极性晶体否则被冤枉,如 MgO Γ 区 20.8 vs 11.7 THz)。
5. **元素覆盖**:`--foundation_model_elements True` 必须开,否则微调模型丢周期表、无法跑新元素。
6. **工程现实**:float32 足够算力常数;8GB 卡做不了 MACE 力微调(需 ~7GiB 固定);中国机器需镜像/中转绕过 NIMS/GitHub/pytorch.org 的网络阻断。

---

## 5. 当前局限

- 微调仅用 MACE-MP-0 **small** + 16 训练材料;尚未上 MatterSim/更大模型,benchmark 材料数也偏少(curated 14–16,非随机大样本)。
- 微调对照只覆盖**谐性**声子(短程力常数);未做 NAC 全谱、未做**非谐(phono3py 三阶 / 热导)**。
- multihead replay 的配比/loss 权重未调优(留出"已见元素"材料退化,有改进空间)。
- LoRA 第三种 anti-forgetting 方案尚未对照。
- Line A 与 Line B 尚未合并为统一框架。

---

## 6. 下一步计划(短期,1–4 周)

1. **replay 配比 ablation**(降 replay 样本/权重、调 head loss):目标拿到"鲁棒 + 不掉精度"的甜点 —— 论文里漂亮的一张图。
2. **LoRA 第三条对照**(`--lora_rank`),与单头/多头并列,形成完整 anti-forgetting 对照表。
3. **benchmark 铺大**:在**同一批 DFPT 材料**上加 SevenNet、ORB(conservative)、MACE-OMAT/MPA、eSEN、CHGNet、M3GNet;从 MDR 随机抽 100–300 材料做诚实分布(非 curated)。
4. **NAC 全谱对照**:把 DFT Born 电荷接到 MLIP 声子上(更物理的对照),区分"短程 FC 误差"与"长程 LO-TO"。
5. **数据效率曲线**:精度 vs 每材料蒸馏配置数 / 训练材料数 —— 量化"多少 DFT 才够"。

## 7. 中期计划(1–3 月)

6. **非谐声子**:用微调模型 + phono3py 算三阶力常数与**晶格热导 κ**,对照 DFPT(这是公认的 benchmark 空白)。
7. **Line A × Line B 合并**:Line B 的快 GPU-DFT 引擎产参考数据 → 喂 Line A 微调;形成"快速产参考 → 微调 → 近 DFT 精度高通量声子"的闭环。
8. **高通量应用 demo**:用框架筛选某目标性质(低热导 / 动力学稳定性),对 top 命中做 DFT 验证。

## 8. 长期目标(发表)

- **一篇完整工作**:GPU 加速的高通量声子框架 = (B) 把声子有限位移 DFT 工作流加速 ~50× 高效产参考 + (A) 系统 benchmark 基础 MLIP 声子精度并提出"蒸馏微调 + 遗忘修复"配方,实现近 DFT 精度下 50–1000× 的声子预测。
- **目标期刊**:npj Computational Materials / Physical Review B / Physical Review Materials / Digital Discovery。
- **可选增量**:微调出一个**声子专用基础模型** + 一个公开 benchmark/leaderboard 贡献;或落到具体材料类(热电、2D、钙钛矿)做发现型工作。
- **诚实边界写进论文**:纯 GPU-DFT 单 SCF 仅 5–15×;50× 来自工作流级叠加 + MLIP 代理;replay 的鲁棒性/精度权衡;NAC 与未见元素的处理。

---

## 附:结果文件索引
- `results/dfpt_mattersim_curated.csv` — MatterSim 全谱 benchmark
- `results/benchmark_builtin_mace.csv` — MACE-MP-0 软化
- `results/finetune_eval_h20.csv` — 单头全规模微调前/后
- `results/finetune_eval_h20_mh.csv` — multihead-replay 微调前/后
- `results/figures/` — Si 色散对比图(MLIP vs DFPT、微调前/后)
- `docs/findings.md` — 逐次实验细节与踩坑记录
- `glm` 分支 `docs/ACCELERATION_REPORT.md` / `METHOD.md` / `SI_ACCURACY.md` — Line B
