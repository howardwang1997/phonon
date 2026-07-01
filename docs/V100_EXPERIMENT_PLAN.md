# V100 单卡实验计划与运行手册（FP64 半边）

> **Strategy update (2026-07-01):** merged to ONE flagship paper (Part I method + Part II discovery); experiments unchanged — see docs/NCS_ROADMAP.md §0.

> 两台单卡 V100 上的 **FP64 DFT/DFPT/EPW** —— H20 那套纯 MLIP 的**互补半边**。
> H20 出 (L)-通道（MLIP+SSCHA），这里出 **DFT fc₂ 真值** + **(E)-通道**（DFPT-with-
> smearing + EPW γ_qν + nesting），两半合成单篇旗舰论文 **Part II — the discovery** 的 (E)/(L) origin-map。
>
> 这两台**可以 SSH**（和 H20 不同），但脚本仍做成幂等可续跑、一条命令一台。

对应 NCS roadmap §0b·2(b) "2×V100 NOW" 菜单。范围（已定）：**全家族 11 个 DFT fc₂ +
电子锚点**，**全部 7 个已知-CDW 做 (E)-通道 EPW**。家族清单与 `configs/h20_campaign.yaml`
一一对应，origin-map 直接拼。

---

## 0. 双车道架构（"GPU 一直跑"怎么实现）

每台 V100 有两条**互不抢资源**的车道（GPU vs CPU）：

| 车道 | 跑什么 | 资源 | 节奏 |
|---|---|---|---|
| **GPU 车道** | relax→fc₂→bands→χ(q)→Path-P | V100 GPU（pw.x，`--nproc 1` 限核） | **现在起跑、连续吃满 GPU** |
| **CPU 车道** | (E)-通道：scf→DFPT→dvscf→nscf→EPW | 16 核 CPU（GPU 全程闲） | 等现有 λ(T_el) campaign 清完 CPU 再排队上 |

关键：(E)-通道是 **CPU-bound**（DFPT 6h + EPW 2.5h，GPU 0%），所以 GPU 车道的 fc₂/Path-P
**搭在空闲 GPU 上免费跑**，两条车道时间重叠。Path-P 的几百个 GPU 单点是让 GPU "一直跑"的
主力（每个 CDW 材料 ~150 个单点）。

---

## 1. 怎么跑

### 1.1 每台一条命令

```bash
# 在 Box A 上：
cd ~/phonon && git pull
bash scripts/v100/run_box.sh A
# 在 Box B 上：
bash scripts/v100/run_box.sh B
```

`run_box.sh` 会：① 跑 `fetch_pseudos.sh` 补 Ta/Ti/V/S 赝势；② 开 tmux `v100gpuA` 跑 GPU 车道
（**立即开始**，吃空闲 GPU）；③ 开 tmux `v100cpuB` 跑 CPU 车道（**先等**现有 λ(T_el) campaign
的 tmux `j0*/jconv/jlq6` 消失，再开 (E)-通道队列）。

### 1.2 看进度

```bash
bash scripts/v100/status.sh         # tmux/GPU/load + fc2/bands/pathP 计数 + EPW 结果表 + 日志尾巴
tmux attach -t v100gpuA             # 实时看 GPU 车道
cat results/v100/SUMMARY.md         # 跑 aggregate.py 后的总表（含 (E) origin-map 列）
```

### 1.3 续跑 / 重试

全幂等：再跑一次 `run_box.sh A` 即可。每步有产出标记：fc₂ 的 `*_phonopy.yaml`、bands 的
`*_ebands.npz`、Path-P 的 `train.xyz`、EPW 的 `EPW_DONE` + step guards（scf/ph/nscf 看 `JOB DONE`）。
已完成的跳过，崩的重来。一步失败不拖垮整条车道（GPU 车道 `continue`，CPU 车道继续下一材料）。

### 1.4 旋钮（都在 `configs/v100_campaign.yaml`）

| 想改 | 改哪 |
|---|---|
| DFT 精度（ecut/k/degauss/超胞） | `dft:` |
| EPW 网格 / Wannier 窗口 | `epw:`（`nq/nk/nkf/nbndsub/dis_win_max`） |
| 家族成员 / 哪些是 CDW | `materials:`（`cdw: true` → 进 EPW + Path-P） |
| 两台分工 | `boxes.A/B.{epw,gpu}` |
| Path-P 采样量 | `path_p:` |

### 1.5 前提：赝势（唯一可能要手动的）

`fetch_pseudos.sh` 自动下 Ta/Ti/V/S/Mo/W 的 ONCV-PBE（Nb/Se 已有）。若机器下不到，
按日志提示手动把 `<元素>_ONCV_PBE-*.upf` 放进 `pseudo_dir`（PseudoDojo ONCV-PBE standard）。
**没下到 Ta/Ti/V 的话，那几个材料会失败**，但 Nb/Se 系（NbSe₂/NbS₂…只缺 S）照跑。

### 1.6 想提前上 EPW？

CPU 车道默认**等**现有 λ(T_el) campaign（+ Box B 的 phase 2）跑完才上，避免 16 核被抢。
若想立刻上 (E)-通道，砍掉现有 campaign 的 tmux 即可（`tmux kill-session -t jconv` 等），
CPU 车道的 `while` 等待条件一满足就自动开始。

---

## 2. 实验逻辑与关系

### 2.1 每个材料的 DAG

```
            ┌──────────── GPU 车道（连续，每材料）────────────┐
build TMD → │  DFT fc₂ ──┬─→ bands + χ(q) nesting             │
(2H/1T)     │  (有限位移) │   (2k_F? nesting peak?)            │
            │            └─→ Path-P（仅 CDW，需 fc₂ 软模本征矢）│
            └────────────────────┬───────────────────────────┘
                                 │ fc₂ = 真值/蒸馏标签
                                 ▼
            ┌──────────── CPU 车道（(E)-通道，仅 7 个已知-CDW）─┐
            │  scf → DFPT-smearing → dvscf → nscf → EPW γ_qν   │
            └─────────────────────────────────────────────────┘
```

### 2.2 每块喂给谁

1. **DFT fc₂ → 校准 H20 (L)-筛选**：H20 的 MLIP+SSCHA 会漏掉它看不见的软模（NbSe₂ 就是
   基础 MACE 判稳定）。**DFT fc₂ 是真值** —— 拿来判 H20 (L)-筛选哪儿对、哪儿假阴性。
2. **DFT fc₂ → 材料专属蒸馏标签**：V-Q3 把 NbSe₂ 的 CDW 灌进 MLIP 那招，对整个家族重做。
3. **bands + χ(q) → (E)-通道机制判定**：`2k_F≈q_CDW`?（nesting 候选）+ nesting 函数
   ξ(q) 在不在 q_CDW 峰?（峰 = nesting 驱动；不峰 = EPC 驱动，如 NbSe₂）。
4. **EPW γ_qν → (E)-通道定量**：γ_qν 在不在 q_CDW 宽峰 = EPC 驱动的指纹。
5. **Path-P → (L) 非简谐通道**：CDW 双势阱 + 热涨落的 DFT 力标签（喂 H20 的非简谐蒸馏）。

### 2.3 合成 origin-map（两半拼一张）

| 来源 | 列 |
|---|---|
| **H20**（results/h20/SUMMARY.md） | (L)：MLIP triage 软不软 + SSCHA 是否稳住 |
| **V100**（results/v100/SUMMARY.md） | DFT fc₂ 真值软模 + 2k_F/nesting + EPW γ_qν |

每个 TMD 一行，(L)+(E) 一起读 → 判它是晶格非简谐驱动还是电子驱动 = **Part II — the discovery（单篇旗舰论文的发现半边）**。

---

## 3. 怎么按结果调计划

| 看到（哪张表） | 含义 | 下一步 |
|---|---|---|
| **DFT fc₂ 有软模、H20 (L)-triage 却判稳定** | MLIP 漏掉了这个不稳定 | 用这材料的 DFT fc₂ 做**材料专属蒸馏**（Path-P/V-Q3），把不稳定灌进 MLIP；H20 (L)-列据此修正 |
| **DFT fc₂ 无软模、实验却是 CDW** | 这是 1×1 cell 的局限（CDW 在更大超胞）或电子驱动 | 检查更大超胞 / 看 (E)-通道 γ_qν |
| **2k_F≈q_CDW 且 nesting ξ(q) 在 q_CDW 峰** | nesting 驱动的 Kohn 反常 | 归类"nesting-driven"，origin-map 标注 |
| **nesting 不峰、但 EPW γ_qν 在 q_CDW 宽峰**（NbSe₂ 模式） | **EPC 驱动**（动量依赖电声耦合），不是 nesting | 归类"EPC-driven" = 与文献一致的非平庸结论，Part II — the discovery 主线 |
| **λ 发散 / minfreq=-1 常数** | 软模处 1/ω² 发散 + Γ-声学 ASR 残差 | λ 绝对值不可信；只报 **γ_qν** 和它的 q-依赖（已知坑） |
| **某材料 EPW Wannier 不收敛 / 虚频太多** | `dis_win_max`/proj 窗口对该材料不合适 | 调 `epw.dis_win_max`（per-material），step guard 让重跑只重 EPW 段 |
| **GPU 车道早早跑完、GPU 又闲** | fc₂/bands 是有限的 | 把更多 CDW 材料的 Path-P 加密（`path_p.n_therm/n_well`），或给非-CDW 材料也加 Path-P 风格热数据 |

---

## 4. 附录

### 4.1 文件地图

```
configs/v100_campaign.yaml         # 唯一配置（材料/DFT/EPW/Path-P/分工）
scripts/v100/
├── run_box.sh                     # ⭐ 每台入口：Phase0 赝势 + 起两条车道
├── fetch_pseudos.sh               # Phase 0：下 Ta/Ti/V/S/Mo/W ONCV-PBE
├── gpu_lane.sh                    # GPU 车道：fc2→bands→Path-P（连续、限核）
├── cpu_lane.sh                    # CPU 车道：(E)-通道 EPW（等现有 campaign 后顺序）
├── tmd_common.py                  # 通用 TMD builder + 配置/赝势解析 + bash CLI
├── tmd_dft_fc2.py                 # 泛化 vq3_nbse2_dft.py（DFT fc₂ 真值）
├── tmd_dft_bands.py               # 泛化 vq3c（bands k_F + χ(q) nesting）
├── tmd_path_p.py                  # 泛化 path_p（CDW 双势阱 DFT 标签）
├── tmd_epw_full.sh                # 泛化 nbse2_epw_full.sh（(E)-通道 EPW）
├── aggregate.py                   # 汇总 → SUMMARY.md（(E)+fc₂ origin-map 列）
└── status.sh                      # 进度速览
docs/V100_EXPERIMENT_PLAN.md       # 本文件
```
复用：`scripts/td_common.py`、`vq3_nbse2_dft.py`、`vq3c_nbse2_bands.py`、
`path_p_nbse2_make_data.py`、`epw/nbse2_epw_full.sh`、`src/phonon_accel/*`。

### 4.2 算力 / 墙钟（全 CDW 方案，两台并行）

| 部分 | 资源 | 量 |
|---|---|---|
| 全家族 fc₂ + 锚点（10 新） | GPU | ~30–45 GPU-h（**搭空闲 GPU 免费**） |
| Path-P（7 CDW × ~150 单点） | GPU | ~5–8 GPU-h |
| (E)-通道 EPW（6 新，NbSe₂ 在跑） | CPU | ~51 CPU-box-h |

两台并行、EPW 是瓶颈：6 新 ÷ 2 = 3 轮 × ~8.5h ≈ **~26h ≈ 1.1 天**（误差带 1.1–2.2 天，
看重 d-电子材料的 DFPT 变慢多少 + EPW 网格收敛）。GPU 车道全程免费搭车。接在现有
campaign 之后（Box B phase 2 ~10–20h），或砍 phase-2 提前。

### 4.3 已知坑（已写进脚本）

- **EPW 在 CPU 上跑、GPU 0% 是正常的**（只有 scf/nscf 用 GPU）。
- **EPW 修复链**（全保留）：去掉 `dis_froz_max`、`efermi_read=scf E_F`、awk 生成 kpts、
  epw.x 串行在 pty、step guards 可重跑。
- **2H vs 1T 几何**：脚本按 polytype 切换 X 的面内位置（2H 两个 X 都在 1/3,2/3；1T 一个在
  1/3,2/3 一个在 2/3,1/3）。
- **Wannier 窗口 per-material**：`dis_win_max=8`、`nbndsub=11`、proj `M:d`+`X:p` 是模板，
  个别材料可能要调 `dis_win_max`（EPW 老问题），step guard 让重跑只重 EPW。
- **λ 发散**：软模处 1/ω² → 报 γ_qν 不报 λ 绝对值。
- **赝势版本后缀**：脚本用 glob `{El}_ONCV_PBE*.upf` 解析，不写死版本号。
```
