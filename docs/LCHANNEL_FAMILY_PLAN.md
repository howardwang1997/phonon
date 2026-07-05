# (L)-通道家族筛查 + 收尾规划 · 2026-07-05

**分支** `smearing-kink-ml` · **定位** 这是支线(`smearing-kink-ml`="准确算两种声子谱")的 **(L) 含温度半边**,直接服务 Part II 发现((E)/(L) 起源判别)。

---

## 背景与目标

支线 = 准确算两种声子谱 + 用它们判 CDW/Kohn 起源:
- **(E) 含 smearing 谱**(电子温度 T_el):✅ 家族级完成(graphene kink + 6 CDW soft-mode,MLIP+长程项)。
- **(L) 含温度谱**(晶格温度 T_lat):⚠️ 本文档的主线 —— 家族级铺开。

**(L)-通道的发现目标**:对每个 CDW 材料,软模/kink 随 **T_lat** 怎么演化?和 (E) 通道对比 → 判起源是电子还是晶格非谐。**预期主结论**:2H CDW 的 (L) 通道弱(软模随 T_lat 基本不动)→ 温度依赖主走电子 = 发现头条。

---

## 四个层次(T0–T3)

| 层次 | 方法 | 状态 | 机器 |
|---|---|---|---|
| **T0 谐振 triage** | foundation MACE 有限位移 fc₂ → 软模 | ✅ 完成(8 材料) | 2060 |
| **T1 SSCHA-MLIP** | 自由能 Hessian vs T_lat → 软模(T_lat) | 🔵 跑中(6 CDW) | 2060 |
| **T2 TDEP-MD** | MLIP-MD → 有效 fc₂(T) + fc3 非谐诊断 | 🔵 排队(等 T1)+ 配 mace 后并行 | 2060/Box A/Box B |
| **T3 严格 (L)** | Path-P 非谐 DFT → 非谐微调 → SSCHA(DFT 验证) | ❌ 本文档规划 | Box A/B DFT + 2060 微调 |

### T0 结果(已完成)
foundation MACE-MP medium,8 材料:
- 2H CDW(NbSe₂/NbS₂/TaS₂/TaSe₂)+ TiSe₂ + gapped 对照:**全 marginal**(foundation 看不见软模)→ 与电子起源自洽。
- **1T-VSe₂:真软模 −0.96 THz @ M** → (L) 异常候选,T3 深挖。
- 诚实:foundation 对 2D 不准,T0 是暗示性筛查。

### T1 结果(进行中)
SSCHA(foundation MACE,5 档 T_lat)。NbSe₂ 测试:裸 fc₂ 软模 −56.5 cm⁻¹ 在 T≈100K 被 (L) 非谐 **heal 到 0** → 给出 (L)-crossover 温度。家族 6 CDW 跑完后给每个材料的软模(T_lat)。

### T2 结果(排队)
td_anharmonic.py:foundation MACE Langevin MD → 有效 fc₂(T) + **fc3 非谐分数**(anharmonicity 排名)。输出 `minfreq_thz, anharm_frac_cubic, anharm_frac_force, fc3_norm`。给家族"哪个材料最非谐"的定量排名(与 T1 互补)。

---

## T3 — 严格 (L)-通道(本文档主交付:规划)

**目的**:T0–T2 是 MLIP triage(foundation 对 2D 不准 → 定性);T3 是 **DFT 验证的定量 (L)-T-演化**,达到可发表门槛。

### 框架修正(2026-07-06):T3 不是"租算力",是"非谐微调家族铺开"
Path-P 非谐微调 = **手上算力**(V100 DFT + 2060 微调/SSCHA),不是 rental。这是 (E) 通道"FC-蒸馏 backbone + 长程项"的 (L) 类比。**全家族 6 个 CDW 材料铺开**(NbSe₂ 复用既有 Path-P 结果)。详见 `~/.claude/plans/...moonlit-marshmallow.md`。

### T3 执行进度表(每材料:fc₂ → Path-P → 微调 → SSCHA → 验证)

| 材料 | fc₂(dg0.015) | Path-P 数据 | 微调 MACE | SSCHA(T_lat) | 验证 |
|---|---|---|---|---|---|
| 1T-VSe₂ | 🔵 Box A(4×4) | ⏳ | ⏳ | ⏳ | ⏳ |
| NbS₂ | ✅(既有) | 🔵 Box A | ⏳ | ⏳ | ⏳ |
| 2H-TaS₂ | 🔵 Box B(3×3) | ⏳ | ⏳ | ⏳ | ⏳ |
| 2H-TaSe₂ | ✅(既有) | 🔵 Box B | ⏳ | ⏳ | ⏳ |
| 1T-TiSe₂ | ✅(既有) | 🔵 Box B | ⏳ | ⏳ | ⏳ |
| NbSe₂ | ✅(既有) | ✅(既有) | ✅(既有 Stage C) | 🔵 复用 | ⏳ |

**编排**:`scripts/smearing_kink/run_L_boxA.sh` + `run_L_boxB.sh`(V100:fc₂→Path-P,带 relay)+ `run_L_2060.sh`(2060:监听数据→微调→SSCHA)+ `relay_to_2060.sh`(box→2060 推 Path-P 数据)。

### 每材料流程(复用 NbSe₂ Path-P 那套)
| 步 | 内容 | 机器 | 估时 |
|---|---|---|---|
| a | Path-P 非谐 DFT:CDW 软模方向双井 + 热位移的力(~69 SCF) | Box A/B GPU-DFT | 2–3 h/材料 |
| b | 非谐微调 MACE(`finetune_pathp_nbse2.sh`,energy-aware) | 2060 | 0.5–1 h/材料 |
| c | SSCHA(T_lat) → 软模(T_lat) | 2060 | 15–30 min/材料 |
| d | foundation vs 微调 backbone 软模;(L)-crossover vs 实验 T_CDW | 本地 | — |

### 数据/算力需求
- **数据**:Path-P 非谐 DFT 力(V100 GPU-DFT,~69 SCF/材料)。瓶颈但手上可做。
- **算力**:Box A/B(V100 GPU-DFT)+ 2060(微调 + SSCHA)。**不租 H100**(只有 C2 cutoff-scaling、C3 Engine-1 才租)。
- **诚实边界**:foundation/distilled MLIP 的 (L)-TDEP/SSCHA 是 triage;Path-P 非谐微调模型给出的 (L)-T-演化才是定论级。

### 验收
VSe₂ 的 (L)-软模(T_lat) 定量曲线 + 判定 (L) 分量是否显著 → 坐实/否定"T0 的 VSe₂ 异常"= 一个具体的非平凡 origin 发现。

---

## 收尾(支线剩余,纯分析或小工程)

做完 T0–T3 后,支线还差这些(写进 Part II):

1. **家族 (E)/(L) origin-map 图**(发现头条图):每个材料 (E)-melting 斜率 vs (L)-crossover/斜率。等 T1–T3 数据。纯分析。
2. **Sharp-transition 窗口**:定位每个材料 dk/dT 最大的 T 窗口(execplan 的科学目标)。纯分析。
3. **统一真实温度** `kink(T_el=T, T_lat=T)`:把 (E)+(L) 合成物理温度曲线。至少 graphene/NbSe₂ 做一次(小收尾)。
4. **Engine 1 ML-EPW(E12)**:(E) 物理忠实仲裁器。大工程,可能需租算力。属支线后期。
5. **Baseline emulator(E11)**:kink(T_el,T_lat) 的 GP/Δ-ML。先于 Engine 1,近场。

---

## 当前机器分配(执行中)
- **2060**:T1 SSCHA(跑中)→ T2 TDEP(排队);mace stack 已就绪。
- **Box A/B**:正在装 mace stack(torch2.6.0+cu124 / mace0.3.16);装好后分担 T2 的 6 材料(每台 2 个)。
- **本地**:本文档 + 后续 origin-map 分析。
