# 8×H20 实验计划与运行手册

> 一台 8 卡 H20 上、**纯 MLIP / FP32**、一口气跑完的全套实验。H20 的 FP64 被阉割，
> **这台机器不跑任何新的 DFT/DFPT/EPW**——那些 FP64 工作在 V100/租机上做（见
> `docs/NCS_ROADMAP.md §0b`）。这里只跑 benchmark / 蒸馏 / (L)-通道 SSCHA / κ。
>
> **交付方式（重要）**：作者无法 SSH 进这台机器。所有脚本+文档走 GitHub，你
> `git pull` 后跑**一条命令**，脚本自愈式装环境、自己排队、自己续跑。

对应 NCS roadmap §0b·2(a) 的 4 个组：

| 组 | roadmap | 这台机器产出 | 服务于 |
|---|---|---|---|
| **E1** benchmark atlas | E1 | ≥6 基础 MLIP × MDR 分层子集的声子精度主表 + 失效模式统计 | Paper-1 |
| **E3/E4** FC-蒸馏 + 消融 | E3/E4 | 蒸馏前后精度 + breadth/anti-forgetting/LoRA 数据效率曲线 | Paper-1 |
| **(L)-通道** TMD 家族 SSCHA | Paper-2 种子 | TMD 家族 (L)-通道 origin-map（晶格非简谐 vs 电子驱动的分流） | Paper-2 |
| **E9** κ | E9 | κ(MLIP) vs 实验（impact line） | Paper-1 |

---

## 1. 怎么跑

### 1.1 一条命令

```bash
cd ~/phonon
git pull
nohup bash scripts/h20/run_campaign.sh > results/h20/campaign.log 2>&1 & disown
```

就这样。`run_campaign.sh` 会依次：

1. **自愈式 bootstrap**（`bootstrap_h20_full.sh`）：检测缺什么装什么——
   6 个 conda 环境（`phonon`/`sevennet`/`mattersim`/`orb`/`chgnet`/`sscha14`）+
   E1 的 MDR 分层数据池。装好的跳过，没装的补；**单个环境装失败不会中断全局**，
   只是它那部分 job 被跳过。结果写进 `results/h20/BOOTSTRAP_STATUS.txt`。
2. **生成任务清单**（`gen_h20_manifest.py` → `results/h20/jobs.jsonl`，132 个 job）。
3. **释放上次崩溃残留的认领锁**（`queue.py release-stale`）。
4. **Wave A**：8 个 worker（一卡一个）抢占式跑完所有「与模型无关」的活
   （E1 + 微调 + 基础模型 L-通道 + κ）。
5. **Wave B**：等 Wave A 全绿后，用蒸馏出的 canon 模型重跑 L-通道 + 全池 E1-after。
6. **聚合**（`aggregate.py` → `results/h20/SUMMARY.md` + 主表 + origin-map）。

### 1.2 看进度（不需要作者帮忙）

```bash
bash scripts/h20/status.sh      # 各组 done/total、GPU 利用率、最近 job 日志、环境就绪情况
tail -f results/h20/campaign.log
cat results/h20/SUMMARY.md      # 跑到一半也能看（聚合是容错的）
```

### 1.3 续跑 / 中断 / 重试（全幂等）

- **续跑**：再跑一次 `bash scripts/h20/run_campaign.sh` 即可。每个 job 有一个
  *done 标记文件*，存在就跳过——已完成的零重算。
- **中断**：`Ctrl-C`（或 `kill` 那个 nohup）随时安全。下次启动 `release-stale`
  会把「认领了但没出 done 标记」的崩溃 job 重新放回队列。
- **失败的 job**：本轮保留认领锁、不无限重试；**重跑 `run_campaign.sh` 会重试它们**。
  失败原因看 `results/h20/joblogs/<job_id>.log`。

### 1.4 旋钮（都在 `configs/h20_campaign.yaml`）

| 想改 | 改哪 |
|---|---|
| 用几张卡 | `runtime.ngpu`（或 `NGPU=4 bash run_campaign.sh`） |
| E1 材料数 / 分片数 | `e1.pool_size` / `e1.nshards` |
| 装哪些基础模型 | `models:` 列表（删一行 = 不跑那个模型） |
| 微调矩阵 / canon 模型 | `finetune.jobs` / `finetune.canon` |
| TMD 家族成员 / 哪些做 SSCHA / 温度网格 | `lchannel.materials`（`sscha: true/false`）/ `lchannel.sscha_temps` |
| κ 的材料 | `e9_kappa.materials` |
| Wave B 要不要做 | `waveB.*` |

改完不用动脚本，`run_campaign.sh` 每次都会用最新 config 重新生成清单。

### 1.5 你可能需要手动中转的东西（no-SSH 现实）

bootstrap 会尽力自动拉，但有一样可能拉不到，看 `BOOTSTRAP_STATUS.txt` 的提示：

- **`~/.cache/mace/mp_traj_combinedxyz`（~56 MB，anti-forgetting replay，gitignore）**：
  `pt*`（多头 replay）微调需要它。bootstrap 会让 MACE 试着自动下载；如果机器
  下不到，请手动把这个文件传到 `~/.cache/mace/`。
  - ⚠️ **canon 模型是 `pt1000` 模式**，它依赖这个文件。若实在拿不到，把
    `configs/h20_campaign.yaml` 里 `finetune.canon` 改成 `b16single`（单头、不需要
    replay），否则 Wave B 会因为 canon 模型缺失而不启动。

其它（MDR DFPT 数据、各基础模型权重）都能自动下载/复现，不用你管。

### 1.6 产出在哪

```
results/h20/
├── SUMMARY.md              # 一页纸总表（E1 / 微调 / origin-map / κ）+ 行动建议
├── e1_master.csv           # E1 全部 (model×material×metric) 行
├── e1/*.csv                # 每个 (模型,分片) 的原始行
├── lchannel/*.csv          # 每个 (材料, triage|sscha, 模型) 的结果
├── e9/*.json               # 每个材料的 κ(T)
├── jobs.jsonl              # 任务清单（队列）
├── joblogs/<id>.log        # 每个 job 的完整日志
├── campaign.log            # 主驱动日志
└── BOOTSTRAP_STATUS.txt    # 环境/数据就绪情况 + 手动中转清单
results/ablation/eval_*.csv # E3/E4 微调的留出集评测（沿用既有路径）
```

---

## 2. 实验之间的逻辑与关系

### 2.1 依赖图（为什么分两波）

```
                 ┌───────────────────────── Wave A（与模型无关，8 卡并行抢占）─────────────────────────┐
                 │                                                                                     │
  E1 benchmark   │   E3/E4 微调            (L)-通道（基础模型）            E9 κ（基础模型）             │
  6 模型 × MDR   │   breadth/anti-forget   triage(全家族) + SSCHA(CDW子集)  Si/Ge/C/GaAs/BAs           │
  → 失效模式表   │   → canon 蒸馏模型 ─┐   → 谁有软化倾向                  → κ(MLIP) vs 实验            │
                 │                     │                                                               │
                 └─────────────────────┼───────────────────────────────────────────────────────────-─┘
                                       │ canon = results/ablation/b64_s1/ft.model
                                       ▼
                 ┌───────────────────── Wave B（依赖 canon 蒸馏模型）─────────────────────┐
                 │  (L)-通道（蒸馏模型）重筛           E1-after（蒸馏模型全池 benchmark）  │
                 │  → 蒸馏有没有改变 origin 判定        → 蒸馏把精度/虚频改善了多少（headline）│
                 └────────────────────────────────────────────────────────────────────────┘
```

- **为什么 Wave A 里 E1/微调/L-通道/κ 全部并行**：它们彼此没有依赖，都只需要
  「公开数据 + 一个基础模型」。抢占式队列把它们塞满 8 张卡，**只要还有活、没有
  卡空转**。长 job（SSCHA ~90min、微调 ~50min）排在短 job（E1 分片 ~15min、triage
  ~8min）前面，让长尾和短活重叠，避免「7 张卡等 1 张」。
- **为什么 SSCHA 单列 `sscha14` 环境**：cellconstructor/python-sscha 需要 numpy 1.23 +
  老 phonopy，和 MACE 主环境冲突（见 §4 gotchas）。triage 只用 phonopy+MLIP，留在
  `phonon` 环境。
- **为什么有 Wave B**：蒸馏模型是 Wave A 的*产物*，用它重跑 L-通道和全池 benchmark
  才能给出「蒸馏前 vs 蒸馏后」的对照——这是 Paper-1 的 headline 和 Paper-2 origin-map
  的第二列。Wave B 在 Wave A 全绿后才放闸（canon 模型必须先存在）。

### 2.2 每组怎么喂给下一步

1. **E1 → E3/E4**：E1 主表告诉你*哪个基础模型最准、失效模式最轻*。它就是该蒸馏的
   `distill_base`（默认 MACE）。E1 同时量化「blind spot」——Paper-1 的 Act 1。
2. **E3/E4 → (L)-通道 / κ**：蒸馏出的 canon 模型是 Wave B 的输入。蒸馏数据效率曲线
   （breadth B16/B32/B64 + anti-forgetting pt vs single vs LoRA）= Paper-1 的 Act 3
   「便宜地修好它」。
3. **(L)-通道 + (E)-通道（链下）→ origin-map**：这台机器只产**(L) 半边**（晶格非简谐，
   MLIP+SSCHA）。**(E) 半边**（电子 Fermi-面/EPC，DFPT-smearing+EPW）是 FP64，在
   V100/租机上做。两半合起来才是 Paper-2 的 (E)/(L) 分解 origin-map。**这台机器的
   L-通道筛选，正是用来决定哪些材料值得花钱去做 (E) 半边**（见 §3）。
4. **κ**：独立的 impact line（E9），证明蒸馏模型在下游热输运上也接近 DFT。

### 2.3 映射到两篇论文（track 不 drift）

- **Paper 1（npj 保底）**：E1（blind spot）+ E3/E4（FC-蒸馏修复 + 数据效率）+ E9（κ impact）。
  **这台 H20 基本能把 Paper-1 跑全**（DFT 参考数据用既有的 + 公开 MDR）。
- **Paper 2（NCS 冲刺）**：(L)-通道 TMD 家族 origin-map 是**种子**；headline 发现需要
  (E) 半边（链下 FP64）合体。NbSe₂ 已是第一个被判定的家族成员
  （见 `docs/NCS_ROADMAP.md §0` 和 `[[td-phonon-m1-status]]`）。

---

## 3. 怎么根据实验结果调整后面的计划

把每张表读成一个**闸门**，每个闸门直接接一个**下一步算力决策**——这也是「少浪费」
在科学层面的延伸：**不到发现闸门给信号，不租 FP64 机器**。

### 3.1 E1（benchmark atlas）

| 看到 | 含义 | 下一步 |
|---|---|---|
| 某模型 freq-MAE 明显最低、虚频最少 | 它是最佳蒸馏底座 | 把 `finetune.distill_base` / `canon` 指到它；若不是 MACE，改 `lchannel`/`e9` 的 `model_type` |
| 所有模型在某类材料（极性/重元素/层状）系统性 softening | blind spot 的结构 | Paper-1 Act 1 的核心图；这些类别优先进蒸馏训练集 |
| 虚频集中在某些空间群 | 失效模式可定位 | E3/E4 训练集按这些空间群分层补样 |

### 3.2 E3/E4（FC-蒸馏）

| 看到 | 含义 | 下一步 |
|---|---|---|
| breadth 曲线（B16→B32→B64）仍在降、未到膝点 | 数据还不够广 | 加 `finetune.jobs` 里更大的 train 集（B79…），或扩 MDR 池 |
| `pt*`(anti-forgetting) ≫ `single`/`lora` | 多头 replay 是关键 | canon 用 pt；写进 Paper-1 differentiation |
| 留出集 MAE 没降 / 反升 | 过拟合或灾难遗忘 | 调 `energy_weight`/replay 强度；或 LoRA |
| 蒸馏后 E1-after 全池虚频大幅下降 | headline 成立 | 锁 Paper-1，进入 §7b 的「立即可做」清单 |

### 3.3 (L)-通道 origin-map（**Paper-2 的发现闸门 / 决定租不租 FP64**）

读法：`triage 谐波 min-freq < 0` ⇒ MLIP 看到谐波软化；`SSCHA label` 给出量子/热修
正后的 (L) 判定。对照实验 `T_CDW`：

| (L)-通道结果 | 物理含义 | 下一步算力决策 |
|---|---|---|
| **`L-stable` 但实验是 CDW** | MLIP+SSCHA 看不到这个不稳定 ⇒ 它**不是晶格非简谐驱动** | **➜ 电子驱动候选：把这个材料放进 V100/租机的 (E)-通道清单**（DFPT-smearing + EPW γ_qν）。SUMMARY.md 会自动把这类材料列成 shortlist。**这是最该花 FP64 的地方。** |
| **`L-unstable`（SSCHA 到顶温仍软）** | 强晶格不稳定，MLIP 自己就抓到 | (E)-通道**降优先级**（先不租）；用 Path-P 风格的非简谐蒸馏深挖 (L) 即可 |
| **`L-crossover@T`** | (L) 通道给出一个晶格驱动的 T_CDW 估计 | 和实验 T_CDW 比；接近 ⇒ (L) 主导，远低 ⇒ (E) 有贡献，进 (E) 清单 |
| **gapped 对照（MoS₂/WSe₂）= `stable`** | 负对照成立，方法不假阳性 | 无；作为 origin-map 的对照列 |
| triage 全家族都偏硬、几乎无软化 | 基础模型把 TMD 软模都抹平了（已知失效） | 优先做 Wave B 蒸馏重筛；若蒸馏后仍抹平 ⇒ 必须靠材料专属 DFT fc₂（FP64） |

> **黄金法则**：FP64 租机只投给「L-通道判定为电子驱动候选」的材料。origin-map 把
> 几十个候选收敛成少数几个值得做 EPW 的——这就是 engine「让发现变得 tractable」的地方。

> **2026-07-05 update — (L)-screen executed + rental items logged.** The (L)-triage (T0) +
> SSCHA (T1) + TDEP (T2) family screen has RUN (on 2060 + V100, no rental); see
> `docs/LCHANNEL_FAMILY_PLAN.md`. **Result**: 2H CDW family is (L)-stable under foundation MACE
> (→ electronic-origin candidates, as expected); **1T-VSe₂ shows a real (L) soft mode (−0.96 THz @ M)**
> = the non-trivial-finding candidate → gated to **C1 (Path-P T3, rental)** for DFT validation.
> The rigorous (L) curves (C1), long-range cutoff ceiling (C2, needs 80 GB card), and Engine-1
> ML-EPW (C3, **H20 8-card train** + V100 H(R) dump) are now registered in `RENTAL_EXPERIMENTS.md`
> C-series. H20's role: C3's DeepH/HamGNN fit (GPU-weeks, FP32 — H20's strength) + any R1
> multi-material long-range-term BAMBOO train (`COMPUTE_DATA_MASTERPLAN.md` §4).

### 3.4 E9（κ）

| 看到 | 含义 | 下一步 |
|---|---|---|
| κ(MLIP) 在共价材料上 vs 实验误差小（<~20%） | 蒸馏模型下游可用 | 直接进 Paper-1 impact 图 |
| 某材料 κ 误差大 | 三阶 FC（非简谐）不准 | 给该材料加 fc₃ 蒸馏（Path-P 风格，链下 DFT）；或检查 mesh/supercell 收敛 |

### 3.5 闸门串起来（一句话）

E1 选底座 → E3/E4 蒸出 canon 并量化「修好了多少」→ (L)-通道把 TMD 家族分流成
「晶格驱动（深挖即可）」和「电子驱动（值得租 FP64 做 (E) 半边）」→ 只对后者租机做
EPW → (E)+(L) 合体成 origin-map = Paper-2 的发现。κ 是平行的 impact 证据。

---

## 4. 附录

### 4.1 文件地图

```
configs/h20_campaign.yaml          # 唯一配置（模型/材料/温度/seeds/分组开关）
scripts/h20/
├── run_campaign.sh                # ⭐ 主入口：bootstrap → Wave A → Wave B → 聚合
├── bootstrap_h20_full.sh          # 自愈式多环境 + 数据 bootstrap
├── gen_h20_manifest.py            # config → jobs.jsonl
├── queue.py                       # 抢占式队列引擎（原子 mkdir 认领 + 波次闸门）
├── queue_worker.sh                # 单卡 worker：领单→跑→再领，直到队空
├── run_job.sh                     # 派发器：激活对应 conda env，跑命令，校验 done 标记
├── e1_benchmark.py                # E1：一个 (模型,分片) 的 MLIP-vs-DFPT
├── lchannel_sscha.py              # (L)-通道：通用 TMD 的 triage / SSCHA
├── e9_kappa_mlip.py               # E9：MLIP 力 + phono3py RTA 的 κ
├── aggregate.py                   # 汇总 → SUMMARY.md + 主表 + origin-map
└── status.sh                      # 进度速览（无需 SSH）
docs/H20_EXPERIMENT_PLAN.md        # 本文件
```
复用既有：`src/phonon_accel/*`（模型工厂/MDR 参考/声子流水线）、
`scripts/run_one_job.sh` + `make_finetune_data.py` + `eval_finetune.py`（微调）、
`scripts/td_common.py` + `vq3e_nbse2_sscha.py`（SSCHA 配方）、
`scripts/campaign_materials.sh`（breadth 集 B16/B32/B64…）。

### 4.2 job 清单 schema（`results/h20/jobs.jsonl`，每行一个 job）

```json
{"id":"lch_sscha_NbSe2_found","group":"lchannel","wave":"A","env":"sscha14",
 "gpu":true,"cmd":"python scripts/h20/lchannel_sscha.py --mode sscha ...",
 "done":"results/h20/lchannel/NbSe2_sscha_found.csv","est_min":90}
```
`done` 是标记文件路径，存在即完成；`wave` 决定波次；清单按 `est_min` 降序（波内长在前）。

### 4.3 「少浪费 GPU」是怎么做到的

1. **抢占式拉取队列**：只要有未认领 job，没有 worker 空闲（vs 静态轮询会让长 job
   卡住某卡时其他卡空转）。
2. **幂等跳过**：续跑只补缺，零重算。
3. **长 job 先入队**：长尾和大量短活重叠，消除「7 卡等 1 卡」的尾部空转。
4. **CPU 旁路**：κ 的 phono3py RTA 是 CPU-bound，`run_job.sh` 给它 16 线程而不是占额外卡。
5. **波次闸门**：Wave B 只在 canon 模型就绪后启动，不空等。
6. **科学层面少浪费**：origin-map 把 FP64 租机只导向电子驱动候选（§3.3）。

### 4.4 已知坑（来自前几轮，已写进脚本/bootstrap）

- **8 GB ≠ 96 GB**：H20 显存大，微调可用大 batch；但 **FP64 被阉割 → 不在这跑 DFT**。
- **每个模型一个 env**：MACE(e3nn 0.4.4) 和 MatterSim(e3nn 0.6.0) 冲突，必须分环境。
- **`sevennet`/`sscha14` 需要各自的 `LD_LIBRARY_PATH=$env/lib`**（libstdc++/lapack）；
  `run_job.sh` 已自动按激活的 env 设置，不要把别的 env 的 LD 泄漏进来。
- **`sscha14` 是最脆的环境**：cellconstructor 1.4.1 + numpy 1.23 + spglib 1.16 +
  phonopy 2.18 + mace 0.3.16（精确 pin）。装不上时 SSCHA job 被跳过，**triage 仍照跑**，
  origin-map 仍有谐波列——不会拖垮全局。
- **SSCHA 关键修正**：传给 CellConstructor 的是对角 `[nx,ny,nz]` 而非 3×3 矩阵，FC 要
  转成 Ry/Bohr²（`vq3e_nbse2_sscha.py` 验证过的配方，`lchannel_sscha.py` 直接复用）。
- **蒸馏模型跨 mace 版本**：canon 由 `phonon` 环境的新 mace 训练，被 `sscha14` 的
  mace 0.3.16 加载（Wave B 蒸馏 SSCHA）。若报模型格式不兼容，job 会失败但基础模型
  SSCHA 已给出 origin-map；如需蒸馏列，可在 sscha14 内重训 canon。
- **anti-forgetting replay 文件**：见 §1.5，是唯一可能要手动中转的东西。

### 4.5 粗略算力 / 墙钟（8 卡，理想抢占）

132 个 job ≈ 8×SSCHA(90) + 7×微调(50) + 18×L-triage/其它 + 5×κ(30) + 72×E1(15) +
Wave B 30 个 ≈ **~45–60 GPU-小时 ÷ 8 ≈ 6–9 小时墙钟**，外加首跑 bootstrap（建 6 个
环境 + 下载 MDR 池）约 **1–3 小时**。SSCHA 真实耗时方差最大（近不稳定时迭代慢），
可能把墙钟拉到 ~1 天。和 `NCS_ROADMAP.md §0b·2(a)` 的 ~900–1800 GPU-h 量级一致
（那是含更大 dataset/active-learning 的上限；本清单是其中能一口气跑的核心子集）。
跑完一轮即可用真实耗时重标后续规模。
```
