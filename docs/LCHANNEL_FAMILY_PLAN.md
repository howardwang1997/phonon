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

### 选材(只做 1–2 个,~3 天/材料)
1. **1T-VSe₂**(必做):T0 露出的 (L) 异常(M 点软模)→ T3 验证它是不是真有 (L) 分量 = **非平凡发现**。
2. **一个 2H 对照**(NbSe₂ 已有 SSCHA,可选 TaS₂):证明 2H 的 (L) 弱。

### 每材料流程(复用 NbSe₂ Path-P 那套)
| 步 | 内容 | 机器 | 估时 |
|---|---|---|---|
| a | Path-P 非谐 DFT:热位移超胞(2–3 个 T_lat)的力/能 | Box A/B GPU-DFT | 1–2 天 |
| b | 非谐微调 MACE(Path-P 力 + 曲率监督) | 2060 | 0.5–1 天 |
| c | SSCHA(T_lat) 用非谐微调模型 → 软模(T_lat) vs DFT | 2060/Box | 几 h |
| d | 与 (E)-melting 合成 → 该材料 (E)/(L) origin 结论 | 本地 | — |

### 数据/算力需求
- **数据**:Path-P 非谐 DFT 力(贵,V100 GPU-DFT,~每材料 1–2 天)。这是 T3 瓶颈。
- **算力**:Box A/B(V100 GPU-DFT,产 Path-P)+ 2060(微调 + SSCHA)。
- **诚实边界**:foundation/distilled MLIP 的 (L)-TDEP/SSCHA 是 triage;只有 Path-P 非谐微调模型给出的 (L)-T-演化才是定论级。T3 只对 VSe₂(+1 对照)做。

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
