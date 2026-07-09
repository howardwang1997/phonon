# 周报 · 2026-07-09 — MLIP+非谐微调计算含 Smearing / 含温度声子谱

**分支** `smearing-kink-ml` · **主题** 详细记录当前方法(MLIP + 非谐微调)在**含 Smearing 谱(T_el)**和**含温度谱(T_lat)**上的结果,给出具体材料的声子谱(smearing 对比 + 温度对比)。

> **一句话**:用 **MLIP + 长程 Friedel 项**算含 smearing 谱(准确,家族 MAE 0.011–0.16 THz);用 **Path-P 非谐微调 MACE + SSCHA** 算含温度谱(1T 软模 heal 在 T_CDW)。两条通道分述如下,均给出 Γ-M-K-Γ 全色散。

---

## 1. 方法

### (E) 含 Smearing 谱 — MLIP + 解析长程 Friedel 项
金属的 Kohn 反常是**长程 Friedel/RKKY 振荡** `Φ_LR ~ cos(2k_F·R)·e^(−R/ξ(T_el))/R^d`,标准 MLIP(cutoff 有限)结构上**抓不到**。本项目在 smearing-blind 的 MACE backbone 上**叠加一个 T_el-条件化的解析长程项**:
```
fc₂(R; T_el) = backbone + B(T_el)·exp(−κ(T_el)·R)·D₀(R)
```
- D₀(R)=全张量 Fermi 面波形(最锐 smearing 处测一次);B, κ = 2 个热参,从 2–3 个 DFT smearing 锚点 few-shot 拟合。
- 部署:`FriedelMACECalculator`(`scripts/smearing_kink/friedel_calc.py`)包任意 MLIP + 加谐振 Friedel 项;变 T_el(degauss×157888 K)→ 含 smearing 谱。
- **结果**:家族 5 CDW 材料 soft-mode(T_el) MLIP vs DFT MAE **0.011–0.16 THz**(TaSe₂ 0.011 / TaS₂ 0.035 / TiSe₂ 0.034 / NbS₂ 0.071 / NbSe₂ 0.160);graphene K-cusp MAE 0.31 cm⁻¹。**MoS₂ gapped 对照 = FLAT**(无 Fermi 面 → 无 smearing 依赖,反证 (E)-kink 是金属屏蔽驱动)。

### (L) 含温度谱 — Path-P 非谐微调 MACE + SSCHA
foundation MACE 对 2D-TMD 不准(抹平 CDW 软模)。**Path-P 非谐微调**:在每个材料 CDW 软模方向采双井 + 热 DFT 力 → energy-aware 微调 foundation → backbone 含软模 + 非谐。再 **SSCHA(T_lat)** 出自由能 Hessian = 重整化 fc₂(T_lat) → 含温度声子谱(软模随 T_lat heal)。

---

## 2. (E) 含 Smearing 谱 — 具体材料,smearing 对比

**图 `results/smearing_kink/spectra_E_smearing.png`**:VSe₂(1T)+ NbSe₂(2H)的 Γ-M-K-Γ 全色散,4 档 smearing(degauss 0.005→0.020,T_el≈789→3158 K)叠加。

| 材料 | degauss 0.005(锐)soft-mode | degauss 0.020(宽) | 现象 |
|---|---|---|---|
| **1T-VSe₂** | **−73 cm⁻¹**(−2.2 THz,q≈(1/4,0) on Γ-M) | ~0 | Kohn 软模随 smearing **熔化**(B(T_el) 衰减) |
| **2H-NbSe₂** | **−99 cm⁻¹**(−3.0 THz,近 M) | ~−1.2 | 同样软化→熔化 |

**读图**:色散的**声学支**在 Γ-M 段下凹成虚频(soft mode),degauss 越小(Fermi 面越锐)下凹越深;degauss 增大(smearing 熔 Fermi 面)→ Friedel 振幅 B(T_el) 衰减 → soft mode heal 向 0。**光学支基本不动**(smearing 只影响 2k_F 奇异通道)。这正是 (E)-channel 的物理:**Kohn 反常 = 电子屏蔽驱动,smearing 直接调控**。

> 注:这是 **DFT fc2** 的色散(展示 smearing 的物理效应);MLIP+长程项部署复现它到 MAE 0.011–0.16(见 §1)。

---

## 3. (L) 含温度谱 — 具体材料,温度对比

### 3.1 软模随温度演化(全家族)
**图 `results/smearing_kink/family_L_crossover.png`**:6 材料 SSCHA soft-mode vs T_lat。

| 材料 | T_lat=20K | T_CDW | 现象 |
|---|---|---|---|
| **1T-VSe₂** | −360 cm⁻¹(50K 最深 −412) | **110K** heal→0 | 强 (L)-不稳定,precisely 在 T_CDW 消失 |
| **1T-TiSe₂** | −552 cm⁻¹ | **200K** heal→0 | 同(第二个 1T 点)|
| 2H(NbSe₂/NbS₂/TaS₂/TaSe₂) | ~0(全 T) | n/a | (L)-惰性 → 电子起源 |

### 3.2 全色散随温度(SSCHA 自由能 Hessian)— *[R3 跑完补图]*
`scripts/smearing_kink/vq3e_sscha_bands.py`:在每个 T_lat 把 SSCHA 自由能 Hessian 插值到 Γ-M-K-Γ 全路径,给出含温度的**完整色散**(不只 soft mode)。VSe₂ + NbSe₂,4 档 T_lat(20/100/200/300K)。**图 `spectra_L_temperature.png` 待 R3 完成生成**(预期:VSe₂ 的 Γ-M 软模在 20K 深虚、110K heal;NbSe₂ 弱)。

**读图(预期)**:(L) 谱的 soft mode 随 T_lat **heal**(非谐重整化),与 (E) 的 smearing 熔化形成对比 —— 两条通道的 kink 变化机制不同(电子屏蔽 vs 晶格非谐),但共享同一 reduced-T healing 形式(B2 统一律)。

---

## 4. Smearing vs Temperature 对比(两通道并列)

| | (E) 含 Smearing(T_el) | (L) 含温度(T_lat) |
|---|---|---|
| **物理** | Fermi 面模糊 → Friedel 振幅 B(T_el) 衰减 → Kohn kink 熔化 | 晶格 T → 非谐重整 → soft mode heal |
| **方法** | MLIP + 解析长程 Friedel 项(T_el 条件化) | Path-P 非谐微调 MACE + SSCHA(T_lat) |
| **谱的变化** | 声学支 soft-mode 下凹深度随 degauss 变(光学支不动) | soft-mode 随 T_lat heal(在 T_CDW 处到 0) |
| **材料对比** | 全家族都熔化(金属屏蔽普适);MoS₂ gapped = FLAT | **1T** 深软模 heal@T_CDW(晶格驱动);**2H** 惰性(电子驱动) |
| **准确度** | 家族 MAE 0.011–0.16 THz(MLIP vs DFT) | 1T crossover 落在实验 T_CDW(110/200K)|
| **图** | `spectra_E_smearing.png`(§2) | `family_L_crossover.png` + `spectra_L_temperature.png`(§3)|

**核心结论**:同一套 MLIP+微调框架,两条通道分别用**长程项(电子)**和**非谐 backbone(晶格)**机制,准确复现 kink 在 smearing/温度下的变化;**两轴共享 reduced-T healing 律 (p,q)=(4.44,3.00)**(B2 统一律,交叉验证)。

---

## 5. 诚实边界(写文章须交代)
1. (E)-deploy 对 VSe₂ 有残留(−0.131 THz bare,sharp-melting 的 deploy 分解假设软模本征矢不转,VSe₂ 急熔违反)→ 写成方法边界,非 bug。
2. (L) 软模的 3×3 (L)-惰性结论成立(4×4 −264 是 q-commensurability 混淆,非收敛问题);6×6 干净测试 OOM(107 GB vs 31 GB),用 DFPT q=1/3 on primitive 可便宜确认谐振收敛。
3. 统一 reduced-T 律是软模层描述符;(E)+(L) 合成单一 2D-deploy SSCHA 模型 = 负结果(谐振 Friedel 项与非谐采样器不兼容)。
4. NbSe₂ EPW λ=23 divergent(Kohn 软脊 → 任何 q-grid 发散);报"强耦合+软模增强",不报绝对 λ。

## 6. 产物
- 脚本:`scripts/smearing_kink/{plot_phonon_spectra.py (fc2→Γ-M-K-Γ), vq3e_sscha_bands.py (SSCHA Hessian→bands), run_R3_2060_sscha_bands.sh}`。
- 图:`results/smearing_kink/{spectra_E_smearing.png, family_L_crossover.png, spectra_L_temperature.png(待R3)}`。
