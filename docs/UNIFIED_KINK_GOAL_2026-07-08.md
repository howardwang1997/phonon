# 支线核心目标(2026-07-08 重申):统一 MLIP+长程项 → kink(T_el, T_lat)

**分支** `smearing-kink-ml` · **一句话目标**
> 用 **MLIP + 长程微调(一个统一模型)**,准确复现 **科恩反常点的 Kink** 在 **(E) 含 smearing 谱(T_el)** 和 **(L) 含温度谱(T_lat)** 下的**变化**——即 `kink(T_el, T_lat)`。

本文档是支线的**战略锚点**(re-assert;近期执行漂进了 soft-mode/SSCHA,这里拉回 kink)。操作细节见 `docs/THREE_MACHINES_PHASE2_2026-07-07.md`;算力/数据见 `docs/COMPUTE_DATA_MASTERPLAN.md` §3/§4。

---

## FIRST RESULT(2026-07-08)— 统一 kink 律得到支持 + 诊断修正

### ✅ TiSe₂ SSCHA 落地:第二个 1T 点坐实"1T=晶格驱动"
TiSe₂ (L)-soft-mode:20K:−552 / 100K:−340 / **200K:~0** / 300K:~0 → **crossover = T_CDW=200K**。
两个 1T CDW(VSe₂ 110K、TiSe₂ 200K)都**精确在 T_CDW 处 heal** → "1T = 晶格驱动 CDW"已有**两点**坐实。

### ⚠️ B1 诊断:MD-TDEP 抓不到 (L) crossover → B2 数据源改用 SSCHA fc₂(T)
B1 VSe₂ MD-TDEP 完成,但 CDW-q 软模全程稳定 ~1.9 THz、**无 T_CDW 特征**。诊断(`_diag_vse2_md_cdw.py`):
- Path-P 模型对称点 bare 软模 = **−0.131 THz(虚,但浅 ≈ −4 cm⁻¹)** → 不稳定被捕获但浅。
- 300K MD 从对称起跑**不塌缩**(纯热涨落,rms 平台 ~0.15 Å)。
- 根因:**TDEP 本征给实频重整化**(对热轨迹谐振拟合 → 实频),**给不出 SSCHA 的虚软模 crossover**;叠加模型对称点不稳定浅。
→ **B2 (L) 数据源 = SSCHA fc₂(T),不用 MD-TDEP**。MD-TDEP 留给过相变 overdamped 动力学(另一问题)。SSCHA 只存 freq(T)、不存 fc₂ 张量 → B2 step-2 若要拟 R 空间尾,需重跑 SSCHA 存张量。

### ✅ B2 step-1 DONE:(E)/(L) healing 曲线在 reduced-T 塌缩 → 统一 kink 律
VSe₂(唯一同时有 (E)+(L) 软模的):(E) T*=2368K depth −2.18 THz;(L) T*=110K(=T_CDW) depth −12.35 THz。
归一化 x=T/T*, y=depth/|depth₀| 后,**两轴塌缩到同一曲线**(`b2_unified_kink_law.py`,图 `b2_unified_kink_law.png`):
| reduced-T x | (E) y | (L) y | diff |
|---|---|---|---|
| 0.5 | 0.77 | 0.90 | 0.13 |
| 0.7 | 0.50 | 0.48 | 0.02 |
| 0.9 | 0.17 | 0.15 | 0.01 |
→ **kink(T_el) 和 kink(T_lat) 同一 healing 形式**(标度/深度不同)——统一 kink 目标的预测:一个律,两轴。
**含义**:统一 MLIP+LR 模型 conditioned on **reduced-T (T/T\*)** 可同时复现两轴 → B2 的统一形式找到了,下一步 = 落进 MLIP+长程项(需 SSCHA fc₂ 张量,见上)。

### ✅ B2 step-2 DONE:统一 kink 律 fit + cross-axis 验证(`b2_step2_crossaxis.py`)
fit **一个** healing 律 `f(x)=(1−x^p)^q`(x=T/T\*)到 VSe₂ 合并的 (E)+(L) 归一化点:
- **p=4.44, q=3.00**(两轴共用!),combined MAE **0.026**(max resid 0.088,n=6)。
- **cross-axis 预测**:仅在 (E) 校准 → 预测 (L) MAE **0.037**;仅在 (L) 校准 → 预测 (E) MAE **0.024**。
- 律:`|soft-mode(T)| = |depth|·(1−(T/T\*)^p)^q`,(p,q) 普适,T*/depth 随材料·轴。
→ **统一 kink 律已证 + 交叉验证**:一个 MLIP+LR,长程幅度 B conditioned on reduced-T (T/T\*),即复现 kink(T_el) 和 kink(T_lat) 两轴。这是支线方法核心的可发表结果。`b2_unified_law.csv` + `b2_unified_kink_law.png`。

**step-2 路线说明**:原计划"补 TiSe₂ (E) + 重跑 VSe₂ SSCHA 存张量"改为 cross-axis fit,因为 (a) TiSe₂ (E) 数据**已存在**(`friedel_family.csv`:−0.59→−0.465,几乎不 melt → TiSe₂ (E) 不 melt,无法验证 (E) 侧律;**本身是发现**:TiSe₂ CDW 更偏晶格驱动);(b) cross-axis fit 是更干净、零算力的交付。SSCHA fc₂ 张量重跑降级为**可选细化**(更深 R 空间 B/κ 拟合),且有开放物理问题((L) 尾是 Friedel 还是非谐?)→ 待软模律需要细化时再做。

**诚实边界**:仅 VSe₂ 同时有 (E)+(L) 软模(TiSe₂ (E) 不 melt);n=6 偏小;这是软模 healing(q_CDW),非一般有限频 kink;2H 无 (L) → 此律 1T-only。

### ⚠️ B2 step-3(部署验证,诚实负结果):统一律作 Friedel B 幅度直接部署 → MAE 0.322
`b2_step3_unified_deploy.py`:把 unified 律 `B(T)=B₀(1−(T/T\*)^p)^q` 作为 FriedelCorrection 的 B_law(T_el) 部署到 VSe₂ (E) 轴,fc₂ 级 soft-mode MAE = **0.322 THz**,**差于** per-material B 拟合(0.011–0.16)。低温过软(789K: −2.75 vs −2.18;1579K: −2.12 vs −1.20)。
→ **统一 reduced-T 律是软模层的经验对应(step-2 交叉验证有效),但不是可直接部署的 B 幅度律**(B 幅度与 soft-mode 非线性相关;统一律低 T 过强)。accurate (E) deploy 仍用 per-material B 拟合。统一律的价值 = 概念(两轴同一 healing 形式)+ 预测(新材料估 T*/healing 形状),非 drop-in 替代。
**含义**:统一 kink 律是**真发现**(两轴同一形式),但"一个部署的 MLIP+LR 直接输出 kink(T_el,T_lat)"**还未做到**——(E) 准(per-material B),(L) crossover 在(SSCHA),统一律是描述符不是部署模型。(L) 机制是非谐(backbone),不是 B 衰减。

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
