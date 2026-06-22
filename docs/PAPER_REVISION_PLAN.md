# PAPER.md — 问题清单与修改计划

> 评审基于 `docs/PAPER.md`（commit b84f271 时的状态），并与 `results/figures/`、
> `RESULTS_SUMMARY.md`、`CAMPAIGN_FINDINGS.md`、`LINE_B_FINDINGS.md` 交叉核对。
> 标注：**P0** = 投稿前必须解决；**P1** = 强烈建议；**P2** = 锦上添花。

---

## 0. 总体判断

科学内容是扎实的（breadth law、coverage acquisition、κ 恢复、两个 honest negative），
故事主线（small-DFT + large-MLIP 闭环数据策略）也立得住。**当前稿子的问题不在结果本身，
而在"呈现"**：缺主图、图都是工作图（working plots）、引用过薄、Introduction 没把
"我们和已有 fine-tuning 工作的区别"讲清楚、若干段落是"声明"而非"证据"。下面按四大类
（用户点名的）+ 三类额外问题（一致性 / 科学严谨性 / 格式）逐条列出。

---

## 1. 缺少主图（主图 / 概念图）—— P0

**问题**
- 全文 9 张图全是数据图（bar / scatter / dispersion），**没有一张概念/流程/overview 主图**。
  npj Comput Mater / Nat Comput Sci 的 Figure 1 几乎都是"方法 + 故事"的示意图（graphical
  abstract 级别）。现在的 Figure 1 是 `mattersim_summary.png`，一张数据 bar 图。
- 现有图全部是 **working-quality**：
  - `mattersim_summary.png`（被当作 Figure 1）标题里直接印着 CSV 路径
    `results/dfpt_mattersim_curated.csv — MLIP vs DFPT (14 materials)`。
  - 多张图标题是句子（"Breadth fixes transfer where depth cannot"、"Phonon softening is
    universal…"）——这些应该是 caption，不是图标题。
  - 默认 matplotlib 字体/配色、无 panel 标号（a/b/c）、轴标签生硬
    （"# training materials / # configs per material (log2)"）。

**要做**
1. **Figure 1 概念主图（✅ 已完成）**：`scripts/fig1_overview.py` → `results/figures/fig1_overview.png`。
   左→右叙事 + 闭环：① 问题（PES 曲率欠预测，inset 草图）→ ② FC distillation（公开 DFPT Φ →
   谐波 E/F 标签，inset Φ 矩阵，zero new DFT）→ ④ 收益（inset 真实 Si 色散 + 真实 κ 散点）；
   底部 ③ 闭环数据引擎（breadth≫depth、coverage≫uncertainty）回馈 ②。payoff/κ 散点与 Si 色散
   inset 复用真实数据（`si_dispersion.npz` / `sc4_*.json`），所以主图也是 evidence-anchored。
2. **数据图合并为 3 张 story-driven 大图（✅ 已完成）**——见下方"图合并方案（已锁定）"。
3. 重排图序：Fig 1 = 概念图（✅）、Fig 2/3/4 = 三张合成图（✅）。

### 图合并方案（已锁定 + 已生成）
统一风格模块 `scripts/plot_style.py`（字体/配色/panel 标号）。三个合成脚本，全部跑通：

| 新图 | 脚本 | 面板 | 数据源 | 替换的旧图 |
|---|---|---|---|---|
| **Fig 2** 失效 & 谐波修复 | `scripts/fig2_failure_repair.py` | (a) MACE 逐材料软化 (b) 跨模型普适 MatterSim/SevenNet (c) Si 色散 before/after | `benchmark_builtin_mace.csv`, `xmodel_baseline.csv`, `figures/data/si_dispersion.npz` | `mattersim_summary` + `xmodel_softening` + `Si_before_after` |
| **Fig 3** 数据效率定律 | `scripts/fig3_data_laws.py` | (a) breadth vs depth (b) depth×breadth 热图 (c) coverage vs random vs uncertainty | `results/ablation/eval_*.csv` | `depth_vs_breadth` + `depth_breadth_surface` + `acquisition_comparison` |
| **Fig 4** 下游 κ | `scripts/fig4_kappa.py` | (a) κ 散点 base→FT→exp (b) Si κ supercell 收敛 | `results/kappa/sc4_*.json` + 收敛记录 | `kappa_benchmark` + `kappa_si_convergence` |

- 输出：`results/figures/fig2_failure_repair.png` / `fig3_data_laws.png` / `fig4_kappa.png`。
- Si 色散面板的能带数据由 `scripts/cache_si_dispersion.py`（phonon-mace 环境）一次性缓存到
  `results/figures/data/si_dispersion.npz`，合成图直接读取，避免每次重跑 MLIP。
- **副带修复**：Fig 2a 现在用的是**真正的 MACE** 数据（`benchmark_builtin_mace.csv`，
  Si −28% / Ge −41% / median −28%），顺手解决了 §5 里"正文讲 MACE、旧图却是 MatterSim"的图文不符。
- **下放 SI**：anti-forgetting（`antiforgetting_errorbars`）→ SI；DFT 引擎加速 → 保留为表格。
- **PAPER.md 接线（✅ 已完成）**：Fig 1(概念) 引用进 Introduction + Contributions 后；旧 Fig 1/1b/2
  → **Fig 2** (a/b/c, §2.1–2.2)；旧 Fig 3/4/5 → **Fig 3** (a/b/c, §2.3–2.4)；旧 Fig 7/8 → **Fig 4**
  (a/b, §2.7)；旧 Fig 6 anti-forgetting → 新增 **## Supplementary Information** 的 **Supplementary
  Fig. S1**。所有正文 callout（Fig. 2a/2b/2c/3a/3b/3c/4a/4b）已加 panel 字母，无悬空引用。

---

## 2. 内容空洞（声明 > 证据）—— P0/P1

逐处列出"只给了结论数字、缺少支撑图表"的地方：

- **§2.2 in-domain MAE ~0.10 THz（P0）**：这是核心精度声明，但全文只有一句话 + 一张 Si
  dispersion 图。缺一张"in-domain MAE 跨材料分布"图/表（哪怕 box plot 或 per-material 柱状）。
- **§2.2 "preserves the base model's universality"（P1）**：声称抗遗忘后保留了基模型通用性，
  但没有任何"通用性保留"的量化（例如在通用 energy/force test set 或 Matbench-Discovery 上
  base vs fine-tuned 的指标）。需要一个小表或一句带数字的证据，否则是空声明。
- **§2.6 DFT 引擎整节（P0）**：是"组件描述"而非"结果"。
  - 加速表把 **measured（reuse ~1.5×）**、**理论 scaling（non-diagonal 141–1152×，其实是
    N³→N 的理论比，不是实测墙钟）**、**literature（GPU-SCF 5–15×，本工作未实现）** 混在一张表里，
    量纲/性质不一致，读者会误读成"我们实测了 1152×"。
  - GPU 部分明确没跑出来（FP64 硬件不够），所以 Abstract 把"GPU/CPU DFT engine"列为
    contribution (3) 是**半兑现**。要么把引擎降格为 Methods/SI 的工具描述，要么补一个真正的
    结果（例如"用引擎生成 N 个新材料的 FC，喂给 distillation，held-out 改善 X"，把它和
    breadth law 闭环连起来——这才是它在故事里的价值）。
- **§2.7 NAC 极性材料（P1）**：只有 MgO 一个例子（51 vs 55–60），却写成"For polar materials,
  NAC restores the correct κ"——N=1 推广。至少再加 1–2 个极性材料，或把措辞限定为"示例性"。
- **§2.8 三阶（P0，见 §5 科学严谨性）**：现在的版本把它从"负结果"重诊断成"方法其实对，只是
  参考态没收敛、我们没算力验证"——本质是**deferred / 非结果**。作为"two instructive negative
  results"之一来撑，论证偏弱。

---

## 3. Introduction 不够清晰 —— P0

**问题**
- 信息密度高但主线不清：lattice dynamics 应用 → 成本 → foundation MLIP → 软化机制 →
  fine-tuning → Q1/Q2，一气呵成但没有清晰的 **gap → 我们的贡献 → roadmap** 三段式。
- **与已有工作的区别讲得太晚**：refs [8,9]（PFT、parameter-efficient FT）已经证明
  "fine-tuning 能修声子"。读者读 Intro 时会问"那这篇不就是别人做过的 fine-tuning 吗？"
  真正的差异化（① zero-DFT 的 FC distillation；② data-efficiency 两条定律；③ 一路打到
  converged κ）要到 Discussion 的 "Relation to prior work" 才说清。**必须提前到 Intro**。
- "curvature-supervision gap" 这个核心概念是**断言**的，没解释机制（为什么 energy/force
  训练会欠表达曲率）——哪怕一句直觉解释。
- 闭环"data engine"这个统一愿景在 Intro 里被埋没了，而它恰恰是把四个 contribution 串起来
  的灵魂。

**要做**
1. 重写为清晰三段式：**(1) 为什么声子重要 + 为什么贵**（应用 + cost，1 段）→ **(2) foundation
   MLIP 的希望与软化失效 + fine-tuning 已能修谐波**（现状，1 段）→ **(3) 但"该花多少/哪些
   DFT"和"能否传到 κ"没人答 → 我们答这两个，并给出闭环数据策略**（gap + 贡献，1 段）。
2. 在 Intro 内**显式一句**区分本工作 vs [8,9]：他们证明"能修"，我们解决"怎样 data-efficient
   地修 + 是否传到 anharmonic transport + 用 zero-DFT 力常数蒸馏"。
3. 给 FC distillation 一句**前置定义**（不必等到 Contributions）。
4. 用 1–2 句把"closed-loop data engine"作为统一框架点出来，让四个贡献有共同母题。

---

## 4. 引用 / 文献调研太少 —— P0

**问题**：参考文献只有 **14 条**，对 npj/Nat Comput Sci 量级（通常 35–60 条）明显偏薄；
而且**正文点名用到的东西没引**。

**正文提到但未引用（必须补）**
- **SevenNet / ORB / CHGNet / M3GNet**：Intro 里点名，且 **SevenNet 实际用在 §2.1 结果里**，
  却没有 reference。（SevenNet: Park et al.；ORB: Neumann et al.；CHGNet: Deng et al. 2023;
  M3GNet: Chen & Ong 2022）
- **LoRA**（§2.5 当方法用）：Hu et al. 2021。
- **灾难性遗忘 / replay / 持续学习**（§2.5）：EWC（Kirkpatrick 2017）或 experience replay。
- **四声子散射 / BAs 反常**（§2.7 解释 BAs 残差）：Lindsay et al.（BAs 预测）、四声子
  Feng & Ruan、BAs 实验确认（Kang/Li/Tian 2018 Science）。
- **symfc / 压缩感知力常数 与 D3Q/thermal2**（§2.8 提到三条路线）：Togo symfc、压缩感知
  晶格动力学（Zhou/Nelson/Ozolins）、Paulatto et al. D3Q。
- **DFPT 基础**：Baroni et al. RMP 2001。
- **κ 实验值**：§2.7 表格里每个实验 κ 都**没有出处**——reviewer 必然要求逐个引用。

**领域综述缺口（建议补，增强 literature review）**
- foundation/universal MLIP 全景：GNoME、eqV2/eSEN、MACE-MP 系列、Matbench-Discovery
  benchmark（Riebesell et al.）。
- 预训练数据：MPtrj / Materials Project（Jain 2013）、Alexandria。
- MLIP 主动学习：Podryabinkin & Shapeev（MTP-AL）、Vandermause FLARE、Zhang DP-GEN
  ——支撑"uncertainty sampling 是 textbook default"的论断。
- 直接声子/属性预测的 ML 工作（对比"我们走 force-constant 路线"）。
- 高通量热电 / 热管理筛选动机（Intro 的应用动机现在零引用）。

**要核实（credibility 风险）**
- [8] arXiv:2604.01017、[9] arXiv:2601.07742 —— 这两个 2026 的 arXiv 号要逐一确认能解析到
  真实文献；解析不了会被当成编造引用，杀伤力极大。底部"已核对"那句话要对得上。

**要做**：把 reference 扩到 ~35–50；先补"正文已用但没引"的（P0），再补领域综述（P1）；
为每个实验 κ 值加出处（P0）。

---

## 5. 科学严谨性 / 一致性问题 —— P0/P1

- **§2.8 三阶诊断与支撑文档矛盾（P0 记录用）**：`PAPER.md` 说"模型忠实复现欠收敛的 sc2
  fc3，NOT curvature-aware loss"；但 `CAMPAIGN_FINDINGS.md`/`RESULTS_SUMMARY.md` 仍写
  "degrades the Hessian, needs a curvature-aware loss"。paper 是最新的，但**支撑文档要同步
  更新**，否则共同作者/审稿补充材料会自相矛盾。
- **"113" 从哪来（P1）**：§2.8 写 "Si 113 → 42–47"，但 §2.7 的 fine-tuned Si κ 是 109.9
  (sc2)/114 (sc3)/143 (sc4)。这个 113 没在前文定义，读者对不上。需注明它是哪个 supercell /
  哪个设置下的二阶-only 值。
- **Figure 1 与正文不匹配（P0）**：§2.1 正文讲的是 MACE、全库 10,034 材料、Si 11 vs 15.5；
  但 Figure 1 (`mattersim_summary.png`) 实际是 **MatterSim**、**14 个材料的 curated 子集**、
  mean −7.2%。图文对象不一致。要么换成真正的 MACE/全库失效图，要么改 caption 与正文。
- **Figure 1 vs Figure 1b 冗余（P1）**：softening 这个点被讲了三遍（正文表 + Fig 1 + Fig 1b）。
  Fig 1 用的也是 MatterSim 数据，和 Fig 1b 重叠。建议合并成一张 Fig 2（multi-panel：
  MACE 全库分布 + 跨模型 per-material）。
- **breadth floor 被一个材料主导（P1）**：1.3 THz 的"floor"是被 BN（单材料 4.43 THz）抬高的
  mean；median ~0.7。headline 用的是被离群值放大的 mean。建议 headline 改用 median，并明确
  floor 由 BN 主导。
- **held-out 只有 6 个材料（P1）**：两条 data-efficiency 定律都建立在 6 个 held-out 上，其中
  BN 一个就主导 mean。统计稳健性弱，limitations 里要更显著地承认（现在只在内部 doc 提了）。
- **κ "toward experiment" 框定偏乐观（P1）**：headline 强调 FT/baseline 比值（1.2–3.2×），
  但除 Si/C 外多数材料离实验仍 2–3×（Sn 1.2 vs 11，BAs 130 vs 1300，Ge 20 vs 60，
  InP 28 vs 68）。"improves toward experiment"方向对，但绝对吻合差。main text 要更直白
  （现在只在 "honest residuals" 一句带过）。
- **Si 达到实验缺收敛 DFT 锚点（P1）**：FT 在 sc4=143 对上实验 140，但**没有收敛的 DFT-κ
  做交叉验证**（DFT 只在 sc2 跑出 48）。FT-RTA 在 sc4 对上实验可能含误差抵消（isotope /
  boundary / 4-phonon）。paper 已承认 anchor 用实验，但这是真实软肋，limitations 要点明
  "experiment 含 RTA 不含的物理，吻合可能部分来自抵消"。

---

## 6. Abstract —— P1

- 过长：一个 ~350 词的巨段。Nature 系一般 150–200 词。建议压缩、分层（问题→方法→两条定律→
  κ 收益→开源），把关键数字（0.10 THz、breadth knee、Si κ 143 vs 140、1.2–3.2×）做成可
  扫读的骨架。
- 把"两个负结果"压成半句，避免在 abstract 里和正面结果抢戏。

---

## 7. 结构 / 格式 —— P1/P2

- **图编号**：Figure 1 / 1b 不规范。统一为 Fig 1（overview）、Fig 2（softening，含 a/b
  panel）、Fig 3…，正文引用同步。
- **正文图过多 + 无 SI**：8 张数据图全塞正文。建议主文留 4–6 张（overview、softening、
  breadth/acquisition、κ），其余（depth×breadth surface、anti-forgetting、Si convergence）
  移 Extended Data / SI。需新建 SI 大纲。
- **表格偏多**：softening 表、breadth 表、acquisition 表、κ 表、convergence 表、speedup 表。
  可把部分并入图或移 SI。
- **占位符未填**：作者、单位、通讯作者全是 [TBD]——投稿前必填。
- **小标题层级**：§2.8 当节但内容是"两个负结果回顾"，和 §2.4/§2.8 有重复（uncertainty 在
  §2.4 已讲）。考虑把负结果整合，避免重复。

---

## 8. 修改计划（按优先级排期）

### 阶段 A — P0（投稿前必须）
1. **画 Figure 1 概念主图**（overview/schematic）。— §1 ✅
2. **修 Figure 1 图文不匹配**：§2.1 失效图换成真正 MACE 版本（Fig 2a 用 `benchmark_builtin_mace.csv`）。— §5 ✅
3. **重写 Introduction**：四段式、提前差异化 vs [8,9]（[9]=PFT 监督 Hessian；我们复用现成 Φ、zero new DFT）、
   点出闭环愿景、前置定义 FC distillation、解释 curvature-supervision gap 机制。— §3 ✅
4. **补引用（14→31 条）**：SevenNet[17]/ORB[18]/CHGNet[16]/M3GNet[15]/MP[20]/Matbench[19]/DFPT[21]/
   LoRA[22]/EWC[23]/AL[24,25]/BAs[26,27]/四声子[28]/压缩感知[29]/D3Q[30]/热电[31]；κ 实验值加出处；
   **[8][9] arXiv 号已 WebSearch 核实为真**（2604.01017 / 2601.07742）。引用一致性已校验（全引全定义，无悬空）。— §4 ✅
5. **§2.2 补 in-domain MAE 跨材料证据**：加 8 材料 baseline→FC-distilled 表（median 0.80→0.09 THz）。— §2 ✅
6. **§2.6 加速表拆分**：measured(~1.5×) / 理论 scaling(141–1152×) / literature(GPU 5–15×, 未实现) 三类分列 +
   "Realised here?" 列。— §2 ✅
7. **同步 §2.8 诊断到支撑文档**，并在正文交代 "113" 的来历。— §5 ✅
   - PAPER.md §2.8：定义 "113" = 谐波(二阶)FC-distilled 在 2×2×2 小 supercell 的 κ（cf. §2.7 sc2≈110）。
   - 已同步纠正诊断到 `LINE_B_FINDINGS.md`、`RESULTS_SUMMARY.md`、`REMAINING_EXPERIMENTS_PLAN.md`（旧的
     "curvature-aware loss" 说法 → "忠实复现欠收敛 sc2 fc₃，需 converged 参考而非新 loss"）。

### 阶段 B — P1（强烈建议）
8. **所有数据图重做为出版级**（统一 `plot_style.py`，去 CSV 标题/句子标题，加 panel 标号）。— §1 ✅
9. **补领域综述引用**（foundation MLIP 全景、AL、四声子、热电筛选动机），reference 14→31。— §4 ✅
10. **headline 用 median 而非被 BN 主导的 mean**：§2.3 表加 median 行（floor 0.77–0.85）、正文 lead median
    ~0.8、Fig 3a 同时画 median(实线)+mean(虚线, BN-inflated)、caption 同步。— §5 ✅
11. **κ 框定更诚实**：§2.7 改"moves κ toward experiment, 1.2–3.2× over baseline"，直说除 Si/C 外仍 ~1.5–3×
    偏低；limitations (i) 补"无收敛 DFT 锚点 + 实验含 RTA 不含物理→可能误差抵消"，新增 (vi) held-out 仅 6 材料。— §5 ✅
12. **NAC 段**：限定为"唯一测试的极性材料 MgO，~10% 内"，泛化留 future work。— §2 ✅
13. **压缩 Abstract**：340→**227 词**，分层（问题→Q→方法→两定律→κ 收益→负结果→开源），floor 改 median。— §6 ✅
14. **Figure 1/1b 合并**，消除 softening 三重表述。— §5 ✅

### 阶段 C — P2（锦上添花）
15. **SI 大纲 + 次要表下放**：新增 SI *Outline*；把 §2.1 跨模型 softening 表移为 **Supplementary
    Table S1**（与 Fig 2b 重复），§2.1 正文改为引用。主文图 = 4（Fig 1–4）。— §7 ✅
16. **统一表编号**：主文 6 张表加 **Table 1–6** 题注 + SI 加 **Supplementary Table S1**；图编号 Fig 1–4 + S1。
    — §7 ✅（**作者/单位占位符 [TBD] 未填**：无法编造作者，需你补真实作者/单位/通讯。）
17. **去重两个负结果**：§2.8 retitle "Instructive negative results"，uncertainty 收成一句指回 §2.4，
    三阶作为本节详述（去掉并列 bullet 的重复）。— §7 ✅

---

## 进度总览
- **阶段 A（P0）**：7/7 ✅
- **阶段 B（P1）**：7/7 ✅（含 8/9/14 由图合并+引用工作一并完成）
- **阶段 C（P2）**：3/3 ✅（唯一遗留：作者/单位占位符需你本人填写）
- 主文：Fig 1 概念图 + Fig 2/3/4 合成图 + Table 1–6；SI：Fig S1 + Table S1。
- 引用 31 条，全引全定义零悬空；Abstract 227 词；median/κ 框定一致诚实。

---

## 9. 备注：内容上真正的"硬缺口"（需要补实验，不只是写作）

以下不是写作能补的，需要评估是否补算/或在 limitations 里讲清：
- **converged DFT-κ 锚点**（Si sc≥4）—— 现在完全靠实验对标，是最大科学软肋（需 V100/A100）。
- **三阶 distillation 的 converged 验证** —— 现为 deferred；若不补，§2.8 作为"negative result"
  的论证力有限，建议在文中更明确地定位为"compute-bound open item"而非"failure"。
- **cross-model fine-tuning** —— 软化的普适性已证（MatterSim/SevenNet），但 distillation 只在
  MACE 上做。承认为 future work（已在 limitations，OK）。
- **更大的 held-out 测试集** —— 6 个材料太少，若能扩到 20–30 个会显著加固 data-efficiency 定律。
