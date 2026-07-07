# 三机 Phase-2 规划:origin 收尾 + 统一 kink(T_el,T_lat) 方法 · 2026-07-07(7-08 更新)

**分支** `smearing-kink-ml` · **快照 2026-07-08 00:00** · 战略锚点 = `docs/UNIFIED_KINK_GOAL_2026-07-08.md`。
**MISSION(2026-07-08 重申)**:用 **一个统一 MLIP+长程微调模型** 复现科恩反常 kink 在 (E) smearing(T_el) 和 (L) 温度(T_lat) 两轴的变化。(E) kink(T_el) ✅ 已准;**(L) kink(T_lat) via MLIP+LR = 核心待交付**(本文 Track B)。origin 发现已坐实(2H 电子 / 1T-VSe₂ 晶格),前沿 = 把方法从 origin 级升到 kink(T_el,T_lat) 准确级。

**当前在跑**:B1 VSe₂ MD-TDEP(2060,GPU 86%)· A2 NbS₂ 4×4(Box A)· TiSe₂ Path-P(Box B 61/69)· NbSe₂ NQ=4 EPW(Box A,排队 A2 后)。

---

## 两条平行轨道

| 轨道 | 目标 | 服务 |
|---|---|---|
| **A. origin 收尾** | TiSe₂ 第二个 1T 点 + Family EPW 第一性原理背书 + 2H "~0" 收敛测试 | Part II 发现 |
| **B. 方法准确度** | MD-TDEP 过相变 overdamped 谱 + (E)+(L) 统一长程项 + DFT-direct 仲裁 | Part I 方法 |

---

## 实验清单(按优先 + 机器)

### 轨道 A(origin 收尾)

| # | 实验 | 机器 | 净算力 | 状态/触发 |
|---|---|---|---|---|
| A1 | **TiSe₂ (L)-验证**(Path-P→微调→SSCHA→fine-T)| Box B→2060 | ~1.5 GPU-hr | 🔵 Box B 61/69(~22:00)|
| A2 | **2H 4×4 收敛测试**(NbS₂/TaS₂,验 "~0" 非欠收敛)| Box A→2060 | ~10 GPU-hr | ✅ **Box A 现在空,可起** |
| A3 | **Family EPW stage 2**(NbS₂/TaS₂/TaSe₂/TiSe₂ λ(T_el))| Box A/B | ~14 GPU-hr | ⏳ 需写 per-material 驱动 + 先 vetting NbSe₂ λ |

### 轨道 B(方法准确度 — 本次讨论新增)

| # | 实验 | 机器 | 净算力 | 状态/触发 |
|---|---|---|---|---|
| B1 | **MD-TDEP VSe₂ 过相变谱**(MLIP-MD→fc₂(T)+fc₃ linewidth,抓 overdamped)| 2060 | ~4 GPU-hr | ✅ **2060 现在空,VSe₂ ft.model ready,可起** |
| B2 | **(E)+(L) 统一长程项**(Friedel 项 T_lat-条件化,fit 在 B1 的 fc₂(T))| 2060+本地 | ~3 GPU-hr + code | ⏳ 依赖 B1 出 fc₂(T) |
| B3 | **DFT-direct VSe₂ (L) 仲裁**(DFT-MD-TDEP,金标准,不经 MLIP)| Box A/B | ~15 GPU-hr | ⏸ gated(贵,只 VSe₂)|

### 已完成(本期)
NbSe₂ EPW λ(T_el)=17.2/116.6 @ dg0.03/0.04(Box A 18:19)、MoS₂ gapped 对照(FLAT)、origin-map 头图、VSe₂ (E)-deploy reframe、A1 origin-map、F4 VSe₂ reframe。

---

## 三机排程(分阶段)

### Phase 1 — 现在(18:30)→ 今晚
```
Box A  ── A2: NbS₂ 4×4 Path-P(~2.5h)→ TaS₂ 4×4(~2.5h)──┐  ← 决定性诚实测试
Box B  ── A1: TiSe₂ Path-P 61→69(~22:00 完)──────────────┤
2060   ── B1: MD-TDEP VSe₂(5 档 T,过 110K 相变)──────────┤  ← overdamped 谱,关键方法实验
                                                          │
~22:00 Box B 完 → relay TiSe₂ → 2060(在 B1 后)────────── ┘
```

### Phase 2 — 明天(7/8)
```
Box A  ── A3 stage 2: 写 per-material EPW 驱动(不同原子→pseudo+Wannier)→ NbS₂ EPW
Box B  ── A2 续 / A3 stage 2 并行(TaS₂/TaSe₂ EPW)
2060   ── A1: TiSe₂ 微调+SSCHA+fine-T(~1.5h)→ B2: (E)+(L) 统一 code+fit(用 B1 的 fc₂(T))
本地   ── NbSe₂ λ=17 vetting(q-grid / soft-mode divergence 检查)
```

### Phase 3 — 7/9–10
```
B2 (E)+(L) 统一:训 T_lat-条件化长程项 → 验证统一 MLIP+LR 算 (T_el,T_lat) kink
A3 Family EPW 收尾(TiSe₂ 最后一个)
B3 DFT-direct VSe₂ 仲裁(gated,确认 B2/B1 的 MLIP 结果可信)
```

---

## 算力账

| 项 | 净 GPU-hr | 机器 |
|---|---|---|
| A1 TiSe₂ 微调+SSCHA+fine-T | ~1.5 | 2060 |
| A2 2H 4×4 Path-P(NbS₂+TaS₂)+SSCHA | ~10 | Box A→2060 |
| A3 Family EPW stage 2(4 mat)| ~14 | Box A/B |
| B1 MD-TDEP VSe₂ | ~4 | 2060 |
| B2 (E)+(L) 统一 fit+train | ~3 | 2060 |
| B3 DFT-direct VSe₂(gated)| ~15 | Box A/B |
| **总计** | **~47 GPU-hr** | ~3–4 天 |

沉没:NbSe₂ EPW(已完)、TiSe₂ Path-P(在跑)。

---

## 关键路径 + 依赖

```
TiSe₂(Box B)─┐
             ├─→ TiSe₂ SSCHA(2060)──→ 第二个 1T origin 点(A1)
             │
B1 MD-TDEP VSe₂(2060)─→ fc₂(T)数据 ─→ B2 (E)+(L) 统一(T_lat-LR)
                                            │
                                            └─→ 统一 MLIP+LR 算 kink(T_el,T_lat)【Part I 核心】
A2 2H 4×4(Box A)─→ SSCHA ─→ 决定 2H "~0" 真实性【最大诚实边界】
A3 EPW stage 2 ─→ (E) 第一性原理家族背书(需 NbSe₂ λ vetting 先)
```

**最决定性、最便宜、可立即起的三件**:A2(2H 4×4)、B1(MD-TDEP VSe₂)、A1 收尾(TiSe₂)。

---

## 决策点 / gates

1. **NbSe₂ λ=17/116 vetting — DONE (2026-07-07)**:λ=17 是**软模 divergence artifact**,不是物理值。证实:divergent λ 来自 CDW q=(1/3,1/3)(**正在 NQ=3 网格上**)的软模(ω~2 meV → λ∝γ/ω² 发散,per-mode λ=150–1500)。VSe₂ λ=2.12 干净是因为其 CDW q=(1/4) 不在 NQ=3 网格 → 软模没被采样。**定性结论稳**:NbSe₂ 在 CDW q 强 EPC(软模增强),与电子起源一致。**A3 stage 2 NQ 策略**:每材料选与 CDW q incommensurate 的 NQ(NbSe₂/TaS₂/TaSe₂ 3×3 CDW → NQ=4/5;TiSe₂ M 点 2×2 → NQ=3/5;VSe₂ 4×4 → NQ=3 已完)。文章报"强耦合 + 软模增强",不报绝对 λ。
2. **A2 结果决定 2H 诚实边界**:4×4 若仍 ~0 → "2H 电子起源,(L) 真惰性"坐实;若出软模 → 3×3 是元凶,2H 需重判。
3. **B3 DFT-direct gated**:只在 B1/B2 出结果后,若 MLIP (L)-谱与 VSe₂ crossover 有张力才启动(贵)。
4. **B2 (E)+(L) 统一 = 升级旗舰 spine 的 gate**(对应 memory promotion gate):统一项算准 kink(T_el,T_lat) → 才把旗舰 reframe 到它;否则维持 sub-line。

---

## 一句话
两台机现在空(Box A + 2060),立即起三件最决定性的事:**A2 2H 4×4 收敛测试**(Box A,关诚实边界)、**B1 MD-TDEP VSe₂ 过相变谱**(2060,抓 overdamped)、**A1 TiSe₂ 收尾**(Box B→2060,第二个 1T 点)。之后 B2 (E)+(L) 统一长程项 = Part I 方法核心 + 升级 gate。~47 GPU-hr,3–4 天。
