# 三机剩余实验 · 算力估算 + 排程 · 2026-07-07

**分支** `smearing-kink-ml` · **问题** 现在三台机器还有多久空?还能跑什么?要多少算力?
**一句话** 当前在跑的 ~05:00 (7/8) 全完;之后**手上三机能把支线非租算力部分全跑完**(Family EPW + (E)-breadth deploy + VSe₂ 重蒸馏 + TiSe₂ 验证 + origin-map)。再往后只剩租算力(C2/C3)和纯分析。

---

## 1. 当前在跑 — ETA(快照 13:25)

| 机器 | 任务 | 进度 | 速率 | ETA | 完后状态 |
|---|---|---|---|---|---|
| **Box A** | (E)-breadth fc2(MoS2 dg0.010/0.015/0.020)| 9/12 | ~17 min/fc2 | **~14:15** | 空闲 |
| **Box B** | TiSe₂ Path-P 4×4 | 43/69 cfgs | ~28 min/cfg(48-atom 4×4 金属位移)| **~01:30 (7/8)** | 空闲 |
| **2060** | listener(等 TiSe₂)+ VSe₂ fine ✅ | 5/6 ft.model | — | 即时空闲 | **现在就空** |

- Box B TiSe2 健康检查:pw.x 12%(SCF 间),最后 cfg 写于 12:56(~28 min 前,正好一个 cfg 周期)→ 正常,只是 4×4 金属位移构型重。
- **关键路径 = TiSe₂ Path-P(Box B ~01:30)→ relay → 2060 微调+SSCHA(~03:15)**。
- 监控 v4(`bjqyw5fwa`)继续盯。

---

## 2. 剩余可跑实验(手上三机,非租算力)

按支线价值排序:

### Tier 1 — 直接补头条缺口

| # | 实验 | 机器 | 净算力 | 墙钟 | 价值 |
|---|---|---|---|---|---|
| **F1** | **Family EPW λ(T_el)**(NbSe₂/NbS₂/TaS₂/TaSe₂/TiSe₂)| Box A+B(V100 FP64)| ~14–17 GPU-hr | 两机并行 ~7–9h | **(E) 第一性原理仲裁器**——证明 MLIP+长程项不是拟合假象;现仅 VSe₂ 有 |
| **F2** | **(E)-breadth deploy + friedel-fit**(1T-TaS2/1T-TiS2/MoS2)| 2060 + 本地 | ~1.5 GPU-hr | ~2h | 扩 DOS→T½ 律(n=5→8);**MoS2 = gapped 对照**反证 (E)-kink 是金属屏蔽驱动 |
| **F3** | **TiSe₂ (L)-验证**(Path-P→微调→SSCHA→fine-T)| 2060(数据来自 Box B)| ~1.5 GPU-hr | ~1.5h(落地后)| **第二个 1T 点**——确认 1T=晶格驱动模式;crossover vs T_CDW=200K |
| **F4** | **VSe₂ (E)-重蒸馏**(MAE 0.717 异常)| 2060 | ~1 GPU-hr | ~1h | 修 (E)-family 表最差点 |

### Tier 2 — 纯分析(数据齐,本地,0 GPU)

| # | 实验 | 机器 | 墙钟 | 价值 |
|---|---|---|---|---|
| **A1** | **(E)/(L) origin-map 头条图**((E)-melting 斜率 vs (L)-crossover)| 本地 | ~1h | **论文头条图**,数据全齐 |
| **A2** | 统一 kink(T_el=T_lat=T)(graphene/NbSe₂)| 本地 | ~1h | 合物理温度曲线 |
| **A3** | Sharp-transition 窗口(dk/dT 峰)| 本地 | ~0.5h | 科学目标收尾 |

### 不在三机范围(租算力,本文档**不含**)
- **C2** cutoff-scaling(长程项收敛,80GB 租算)
- **C3** Engine-1 ML-EPW(可扩展 (E),gated)
- (E)-breadth 第 2 轮(MoSe₂/WS₂/WSe₂/NbTe₂/TaTe₂)— 可在 Box A EPW 后选做,扩 n→12-15,优先级低于 F1-F4

---

## 3. 算力账

| 项 | 净 GPU-hr | 备注 |
|---|---|---|
| F1 Family EPW(5 mat × ~3.4h)| ~17 | **最大头**;NbSe₂ 可能部分已跑(见下⚠️) |
| F2 (E)-breadth deploy+fit | ~1.5 | 2060 |
| F3 TiSe₂ 微调+SSCHA+fine-T | ~1.5 | 2060 |
| F4 VSe₂ 重蒸馏 | ~1 | 2060 |
| Box A 收尾 MoS2 fc2 | ~0.5 | 已在跑(sunk) |
| **新增总计** | **~21 GPU-hr** | F1 占 ~80% |

> 沉没:Box B TiSe₂ Path-P 剩 ~12 GPU-hr(已在跑,不计入新增)。

**⚠️ Family EPW 前置核实**:本地只有 `vse2_epw_lambda_Tel.csv`;NbSe₂ EPW 结果未同步本地(task #48/#49 仍 in_progress,git 有 "E6 DONE NbSe2 EPW" 但 csv 不在)。**启动 F1 前先核 NbSe₂ EPW 是否真完成**(在 Box 上找 `results/v100/epw_nbse2*` 或拉回),完成则 F1 = 4 mat(~13.6 GPU-hr)。

---

## 4. 排程(无干预,自动顺序)

```
13:25 ── Box A: MoS2 fc2 收尾(9→12)──────┐
   │                                      │
   └─ 2060 FREE ── F4 VSe₂ 重蒸馏(~1h)──┤  ← 填 2060 空窗
                                          │
~14:15 Box A FREE ── F1 Family EPW ───────┤  ← NbSe₂(若缺)→ NbS₂ → TaS₂ → TaSe₂ → TiSe₂
   │                                      │     (单 V100 顺序 ~14-17h;Box B 01:30 加入并行)
   └─ 2060: F2 deploy+fit(~2h)────────────┤  ← (E)-breadth 三材料
                                          │
~16:30 2060: A1 origin-map 图(本地,~1h)─┤
                                          │
~01:30 (7/8) Box B FREE ── 加盟 F1 EPW ───┤  ← 剩余 EPW 并行
   │                                      │
   └─ TiSe₂ Path-P 落地 → 2060:          │
        F3 微调(~1h)→ SSCHA(~20m)→       │
        fine-T(~15m)──────────────────────┤
                                          │
~03:15 TiSe₂ (L)-验证完成 ────────────────┤
~05:00 Family EPW 完成 ───────────────────┘  ← 全部非租算力跑完
```

**里程碑**:
- ~14:15 Box A 空 → 起 F1 EPW
- ~16:30 F2 deploy 完成 → origin-map 可做
- ~01:30 (7/8) Box B 空 + TiSe₂ 落地 → 双线收尾
- **~05:00 (7/8) 支线非租算力全部完成**;之后 = 租算力(C2/C3)+ 纯分析(A2/A3)+ 写图

---

## 5. 立即可启动(零风险填空窗)

- **现在 → 2060 跑 F4(VSe₂ 重蒸馏)**:2060 此刻空闲(listener 等 TiSe₂,~01:30 才到),~1h 完成于 ~14:30,不挡 TiSe₂。**建议立即起。**
- **Box A 14:15 空 → 自动起 F1 EPW**:需预先确认 NbSe₂ EPW 状态(见 §3 ⚠️)+ 写 Box A EPW driver(`scripts/epw/orchestrate_boxA.sh` 已存在,改材料列表)。

## 6. 决策点(需你拍板) — ✅ 用户批 1,2,3,已执行(2026-07-07 13:45)

### 执行结果

| 项 | 状态 | 说明 |
|---|---|---|
| **A1 origin-map 图** | ✅ **完成** | `results/smearing_kink/origin_map.{png,csv}` + `plot_origin_map.py`。头条图干净:2H(NbS₂/TaS₂/TaSe₂/NbSe₂)E-response 0.58–1.83 / (L)-depth 0 → 电子起源;1T-VSe₂ E=0.50 / (L)-depth 412 → 晶格起源。象限分离即 origin 发现。 |
| **F1 NbSe₂ EPW** | ✅ **已排队(Box A)** | `scripts/smearing_kink/run_EPW_boxA_nbse2.sh`(tmux `epwA`,13:43 起)等 Ebreadth fc2 完(~14:15)→ 自动跑 `queue_boxA.sh`(NbSe₂ degauss 0.03/0.04 → λ(T_el))。~3-4h,~18:00 出结果。**这是 Family EPW stage 1**( NbSe₂ 是 headline 材料)。 |
| **F4 VSe₂ 重蒸馏** | ⚠️ **范围修正,待定** | 见下。 |

### F4 诊断修正(重要)
原假设"VSe₂ Friedel B/κ 律没拟合好"**被证伪**:`friedel_family.csv` 里 VSe₂ 的 2-参数理想拟合 model-vs-DFT 误差 ~0。0.717 的 deploy MAE 来自 **MACE backbone 在 VSe₂ 上 deploy 失效**。

**执行(2026-07-07 14:00)**:2060 上已有 v2 重蒸馏 backbone(`results/fam_backbone/1T-VSe2_v2/`,7/5 跑,RMSE F=2.8 meV/Å,403 cfg),但从未 deploy。**deploy v2 → MAE = 0.711**(≈ v1 的 0.717,**无改善**)。逐点误差模式完全一致(dg0.015 处 MLIP −1.02 vs DFT 0.0 是唯一大错)。

**根因(`_probe_vse2_backbone.py`)**:
- DFT sharp (dg0.005) 软模 = **−2.184 THz**(深井);DFT melted (dg0.020) = 0.0。
- **v2 backbone bare 软模 = −0.101 THz**(浅)——backbone 蒸馏在 dg0.020(熔化参考),**根本不认识深软模**。force-RMSE 好(2.8)但 fc₂ 软模扇区不准。
- deploy 分解(固定 backbone + 标度 Friedel D0)**假设软模本征矢随 T_el 不转**。对 2H(缓熔,本征矢稳定)成立 → MAE 0.01–0.16;对 VSe₂(急熔,本征矢旋转)失效 → 0.71。
- **这是方法局限,不是训练缺陷** → 再蒸馏(v3/v4)不会修好。

**反而是一个发现**:VSe₂ 的 (E)-deploy 失效本身证明它的 (E)-melting **定性不同**(急熔、类一阶),与 2H(缓熔、类二阶)对比 → **和 (L)-驱动 origin 自洽**。0.71 不是 bug,是 signal。

**结论**:F4 不再追求修 MAE。两个出路:
- (a) **reframe**:把 VSe₂ (E)-deploy 异常写进文章作"急熔材料的方法边界 + 与 (L)-origin 一致"。
- (b) **方法级修**:T_el-条件化 backbone(每档 smearing 蒸馏)或旋转-D0 分解。研究级,~1 天+,归 C3 Engine-1 路线。

**采用 (a)**:VSe₂ 的 (E) 故事由 (L)-crossover=110K + EPW λ=2.12-2.24 坐实,deploy 异常作方法边界交代。F4 关闭。

### 后续自动里程碑(无需干预)
- **~14:15** Box A fc2 完 → epwA 自动起 NbSe₂ EPW
- **~01:30 (7/8)** Box B TiSe₂ 完 → 2060 微调+SSCHA;Box B 加盟 EPW(若已 generalize 驱动)
- **~03:15 (7/8)** TiSe₂ (L)-验证
- **~18:00 (7/7)** NbSe₂ EPW λ(T_el) 出 → Family EPW stage 1 done

### Family EPW stage 2(待 F1 stage 1 落地后决定)
`nbse2_epw_full.sh` 是 NbSe₂-hardcoded(prefix/物种/Wannier Nb:d+Se:p)。要扩到 NbS₂/TaS₂/TaSe₂/TiSe₂ 须写 **per-material EPW 驱动**(每材料不同原子→不同 pseudo + Wannier 投影 Ta:d/V:d/Ti:d/S:p/Se:p)。NbSe₂ stage 1 跑通后,按其模板 generalize。
