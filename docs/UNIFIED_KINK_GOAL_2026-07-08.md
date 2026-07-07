# 支线核心目标(2026-07-08 重申):统一 MLIP+长程项 → kink(T_el, T_lat)

**分支** `smearing-kink-ml` · **一句话目标**
> 用 **MLIP + 长程微调(一个统一模型)**,准确复现 **科恩反常点的 Kink** 在 **(E) 含 smearing 谱(T_el)** 和 **(L) 含温度谱(T_lat)** 下的**变化**——即 `kink(T_el, T_lat)`。

本文档是支线的**战略锚点**(re-assert;近期执行漂进了 soft-mode/SSCHA,这里拉回 kink)。操作细节见 `docs/THREE_MACHINES_PHASE2_2026-07-07.md`;算力/数据见 `docs/COMPUTE_DATA_MASTERPLAN.md` §3/§4。

---

## 目标拆解:统一模型 × 两条 kink 轴

| | (E) kink(T_el) | (L) kink(T_lat) |
|---|---|---|
| **物理** | Fermi 面模糊 → Friedel 振荡幅度 B(T_el) 衰减 | 晶格随 T → 电子结构(k_F)漂移 + 非谐重整 → Friedel kink 漂移 |
| **长程项条件化** | ✅ 已做(T_el-conditioned Friedel,`friedel_calc.py`)| ❌ **未做**(LR 项没 T_lat 依赖——**核心缺口**)|
| **现状** | graphene MAE 0.31 + 5 CDW soft-mode 0.011–0.16,**已准** | 只有 SSCHA-on-短程-MACE(VSe₂ crossover=110K),**没走 MLIP+LR 路线** |
| **还差** | 收尾(family EPW 仲裁、family deploy polish)| **(L)-LR 项 + kink(T_lat) 读出 + 统一**(本文档主线)|

**关键**:目标是**一个**统一 MLIP+长程模型同时出 kink(T_el) 和 kink(T_lat),不是两套方法。当前 (E) 侧已用 MLIP+长程,(L) 侧还在用 SSCHA(短程)——必须把长程项扩到 T_lat,两边才能合一个模型。

---

## 当前 vs 目标(诚实)

✅ **(E) kink(T_el) via MLIP+长程**:graphene K-cusp(MAE 0.31)+ 5 CDW 软模(TaSe₂ 0.011 / TaS₂ 0.035 / TiSe₂ 0.034 / NbS₂ 0.071 / NbSe₂ 0.160)+ MoS₂ gapped 对照(FLAT)。**这条轴的方法已验证**。

⚠️ **(L) kink(T_lat) via MLIP+长程**:**没做**。现在 (L) 只有 SSCHA-on-短程-MACE(给软模 crossover,无长程项)。要达目标,必须:
1. 拿到 (L) 的 kink(T_lat) DFT-target 数据(MD-TDEP 的 fc₂(T_lat))。
2. 把 Friedel 长程项从 T_el-条件化扩到 **T_lat-条件化**(或 T_lat→等效电子展宽映射)。
3. 用 MLIP+长程(不是 SSCHA)读出 kink(T_lat),对 DFT 验证。
4. 合成一个 (T_el, T_lat) 双条件化模型,出 kink 2D 面。

⚠️ **过相变 overdamped**:kink 在 T_CDW 处深化到 0,T_CDW 以上可能 overdamped。MLIP+长程是谐振层面 → 给 kink **频率**(T),不给 linewidth/overdamping(那要 MD-TDEP,B1 做)。文章须框此边界。

---

## 实验规划(服务统一 kink 目标)

### Track 1 — (L) kink(T_lat) 数据(统一模型的燃料)
| # | 实验 | 机器 | 算力 | 状态 |
|---|---|---|---|---|
| **B1** | VSe₂ MD-TDEP(fc₂(T_lat)+fc₃,过 T_CDW=110K)| 2060 | ~4 GPU-hr | 🔵 **跑中(GPU 86%)** |
| B1b | TiSe₂ MD-TDEP(第二个 1T)| 2060 | ~4 GPU-hr | ⏳ TiSe₂ SSCHA 后 |
| S-a | graphene MD-TDEP kink(T_lat)(最干净的 kink 案例)| 2060 | ~3 GPU-hr | ⏳ |

### Track 2 — 统一模型(目标本身,核心)
| # | 实验 | 机器 | 算力 | 状态 |
|---|---|---|---|---|
| **B2a** | Friedel 长程项扩 T_lat-条件化(B(T_lat),κ(T_lat),or T_lat→T_el_eff)| 本地+2060 | code | ⏳ B1 出 fc₂(T_lat) 后起 |
| **B2b** | 在 B1 的 fc₂(T_lat) 上 fit/train T_lat-LR 项 | 2060 | ~2 GPU-hr | ⏳ |
| **B2c** | MLIP+长程读出 kink(T_lat)(VSe₂/TiSe₂/graphene),对 DFT 验证 | 2060+本地 | ~1 GPU-hr | ⏳ |
| **B2d** | **统一 (T_el,T_lat) 双条件化模型 → kink 2D 面**(graphene S-c)| 2060+本地 | ~2 GPU-hr | ⏳ **= 支线交付物** |

### Track 3 — 仲裁 / 背书
| # | 实验 | 机器 | 算力 | 状态 |
|---|---|---|---|---|
| NbSe₂ EPW NQ=4 | 干净 (E) λ(incommensurate,不发散)| Box A | ~7 GPU-hr | 🔵 **排队(A2 后)** |
| Family EPW stage 2 | NbS₂/TaS₂/TaSe₂/TiSe₂ λ(T_el)| Box A/B | ~14 GPU-hr | ⏳ per-material 驱动 |
| B3 | DFT-direct kink(T_lat) VSe₂(金标准,不经 MLIP)| Box A/B | ~15 GPU-hr | ⏸ gated |

### Track 4 — 支撑(在跑)
| # | 实验 | 机器 | 状态 |
|---|---|---|---|
| A2 | 2H NbS₂ 4×4 收敛测试(验 (L) "~0" 非欠收敛)| Box A | 🔵 跑中 |
| A1 | TiSe₂ Path-P → SSCHA(第二个 1T origin 点)| Box B→2060 | 🔵 61/69 |

---

## 三机分配 + 关键路径

```
2060  B1(VSe₂ TDEP)─→ B2a/b/c(T_lat-LR + kink(T_lat) 读出)─→ B2d(统一 2D 面)【支线主线】
Box A A2(NbS₂ 4×4)─→ epwNQ4(NbSe₂ 干净 λ)─→ B3/Family-EPW-stage2
Box B TiSe₂(Path-P)─→ relay 2060 SSCHA + B1b(TiSe₂ TDEP)
本地  B2a code(T_lat-LR);NbSe₂ λ vetting ✅;origin-map/MoS₂ 对照 ✅
```

**关键路径 = B1(fc₂(T_lat) 数据)→ B2(T_lat-LR + 统一)**。B1 在跑,~2–4h 出数据 → B2a 今天就能起。**统一 kink(T_el,T_lat) 模型是支线真正交付物**(= 升级旗舰 spine 的 gate)。

---

## 算力 / 时间

| 项 | GPU-hr | 机器 |
|---|---|---|
| Track1(B1+B1b+S-a MD-TDEP)| ~11 | 2060 |
| Track2(B2a-d 统一模型)| ~5 + code | 2060+本地 |
| Track3(NQ4 + Family EPW + B3)| ~36 | Box A/B |
| Track4(A2 + TiSe₂)| ~12(sunk/在跑)| Box A/B |
| **新增总计** | **~50–55 GPU-hr** | ~4–5 天 |

沉没/在跑:NbSe₂ EPW NQ=3(已完,λ 发散已 vet)、TiSe₂ Path-P(在跑)。

---

## 验收(怎么算"复现出 kink 变化")
1. **(E) kink(T_el)**:MLIP+长程 vs DFT,graphene K-cusp MAE 已 0.31;family soft-mode 已 0.011–0.16。✅ 基本到位。
2. **(L) kink(T_lat)**:MLIP+长程(T_lat-条件化)vs DFT-MD-TDEP 的 kink/crossover,VSe₂ 在 T_CDW=110K 处的演化。**= B2c 的交付**。
3. **统一 2D 面**:一个模型出 graphene `kink(T_el,T_lat)` 面,两条轴都对。**= B2d 的交付 = 支线 flag**。
4. **诚实边界**:过相变 overdamped 须框为方法上限(谐振层面),不掩盖。

## 一句话
(E) kink(T_el) 已用 MLIP+长程做准;**把同一个长程项扩到 T_lat(B2)+ 用 MD-TDEP(B1)喂 (L) 数据**,就达成"一个统一模型复现 kink(T_el, T_lat)"的支线目标。~50 GPU-hr,4–5 天,主线在 2060。
