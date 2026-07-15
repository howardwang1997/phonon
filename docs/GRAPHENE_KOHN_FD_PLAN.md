# 实验计划 · graphene Kohn 反常:(E)-channel smearing 扫描,DFT vs MLIP+微调+长程项(+可学习 D₀)

**分支** `smearing-kink-ml` · **日期** 2026-07-16 · **状态** DFT 前置数据采集中(运行中)

## 目标
在 graphene 上复现 **Kohn 反常**(Γ G 带 + K iTO)及其随**电子温度 smearing**(0.01→0.2)**熔化**的行为,对比 **DFT** 与 **MLIP+微调+长程项**,均用 **Fermi-Dirac**(degauss=k_B·T_el 严格),对照 **Piscanec et al. 2004**(PRL 93, 185503)。

## 关键设计决策(已锁定)
1. **Fermi-Dirac smearing**:`degauss=k_B·T_el` 严格(cold 不是)。已验证 graphene 上 fd≠cold(dg0.040: K-kink fd 3.9 vs cold 12.5)。
2. **结构弛豫**:graphene PBE vc-relax 面内 a(`cell_dofree=2Dxy`,fd)→ **a=2.4576 Å**(≈实验 2.46);DFT 与 MLIP **都用这个 a**(Kohn 反常深度对 a 敏感)。
3. **smearing**:10 点覆盖 0.01–0.2(0.01/0.015/0.02/0.03/0.04/0.06/0.08/0.10/0.14/0.20),T_el 1579→31577 K。
4. **backbone**:复用 v11 + 重新蒸馏/微调(fd 0.01–0.2 数据)**两条都做**。
5. **长程项 + 可学习 D₀**:主框架 = 短程 NN backbone + 解析 Friedel 长程项 `B(T)·exp(−κ(T)R)·D₀`。**D₀ 做消融**:
   - **变体 A(固定 D₀)**:D₀ = 最锐 smearing 实测的全张量 fc₂ 差(现有方法)。
   - **变体 B(可学习 D₀)**:D₀_fit(R)=α(R)·D₀_meas(R),α(R)=每距离壳可学习标量(~12 壳,init 1),与 B、κ **联合最小二乘**拟合到全 fd 扫描。保留测量张量的方向性,让拟合重塑径向波形——解析项不准确时的"可学习参数"修正。
6. **磁性**:graphene 非磁,自旋非极化(nspin=1)正确,无需处理。

## 流水线(每阶段完成自动推进下一阶段)
1. **[DFT] 采数**(运行中):`run_graphene_kohn_fd.sh` — 弛豫 a + 10 点 fd fc₂(6×6,PBE,ecutwfc 60,kpts 6)。两台 V100 并行,A=锐 5 点、B=宽 5 点。~40 min。产物 `results/graphene_kohn_fd/graphene_sc6_dg{dg}_phonopy.yaml`。
2. **[MLIP backbone] 复用 + 重训**:
   - 复用 v11(`scripts/smearing_kink/` 现有 graphene backbone)→ 直接进 deploy。
   - 重训:在 fd 0.01–0.2 fc₂ 上蒸馏/微调一个 backbone(FC-distill recipe)。
3. **[Friedel 拟合]**(`fit_friedel.py` 改 fd 数据 + 新 `fit_friedel_learnableD0.py`):
   - 变体 A:D₀ 固定,few-shot 拟合 B(T)、κ(T)。
   - 变体 B:D₀_fit(R)=α(R)·D₀_meas(R),联合拟合 {α, B-law, κ-law} 到 10 点 DFT fc₂。
4. **[Deploy]** `FriedelMACECalculator`,10 个 T_el × {复用-v11, 重训} × {固定-D₀, 可学习-D₀} → fc₂(T_el) → Γ-M-K-Γ 色散 + kink_K/kink_G/Γ-E2g/K-iTO。
5. **[对比]** overlay:DFT vs MLIP+LR(固定 D₀)vs MLIP+LR(可学习 D₀)vs Piscanec 2004(Γ~1580、K~1350、反常随电子参量熔化)。产物:对照图 + 表 + 写进周报。

## 对比指标(每个 T_el)
- `kink_K(T_el)`、`kink_G(T_el)`(Kohn 反常深度)
- Γ E₂g(G 带,~1580)、K iTO(~1350)频率
- DFT vs MLIP+LR 全色散 MAE(cm⁻¹)

## 时间预算
- DFT 10 点:~40 min(并行)。
- backbone 复用 deploy 10 T_el:~15–30 min;重训:~4 h(一次性)。
- Friedel 拟合(固定+可学习):秒–分钟。
- 全流程:DFT 数据齐后 ~1 h(复用线)/ +4 h(重训线)。

## 诚实边界(写文章交代)
- D₀ 固定假设在 Fermi-面波形随 T_el 演化的材料(VSe₂ 急熔)失效;graphene 单一干净反常,固定 D₀ 预期已够,可学习 D₀ 作增益量化(预期边际)。
- smearing 0.2(k_BT≈2.7 eV)极宽,反常完全熔化,可能带轻微色散畸变——作为"纯 backbone"锚点。
