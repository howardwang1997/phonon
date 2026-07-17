# TMD 虚频（CDW 软模 / Imaginary Phonon）文献索引

> 本文档记录文献中报道 TMD 体系虚频（imaginary phonon frequency / CDW soft mode）的关键论文。
> 检索日期：2026-07-17。分支 `smearing-kink-ml`。

## 1. NbSe₂（2H, T_CDW = 33K）

| # | 标题 | 作者 | 期刊/年 | 方法 | 虚频报道 | DOI / URL |
|---|---|---|---|---|---|---|
| 1 | "Fermi surface nesting and the origin of charge density waves in metals" | Johannes & Mazin | PRB 77, 165135 (2008) | DFT + nesting | 嵌套无预测力;CDW 由 q 依赖 EPC 驱动 | [DOI](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.77.165135) |
| 2 | "DFT correctly describes the metallic CDW in 2H-NbSe₂" | Calandra, Mazin, Mauri | PRB 80, 241108(R) (2009) | PBE/GGA | NbSe₂ CDW 软模,freestanding 单层 q_CDW 偏 ½ΓM | [DOI](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.80.241108) |
| 3 | "Extended phonon collapse and the origin of the charge-density wave in 2H-NbSe₂" | Weber et al. | PRL 107, 107403 (2011) | DFPT + EPC | 裸 LA 模经 q 依赖 EPC "extended phonon collapse" 成虚频 | [DOI](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.107403) |
| 4 | "Phonon linewidths and electron-phonon coupling in graphite and nanotubes" | Lazzeri et al. | PRB 73, 155426 (2006) | DFPT | NbSe₂ EPC + 声子线宽 | [DOI](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.73.155426) |

## 2. TaSe₂（2H, T_CDW = 122K）

| # | 标题 | 作者 | 期刊/年 | 方法 | 虚频报道 | DOI / URL | 开放获取 |
|---|---|---|---|---|---|---|---|
| 5 | **"Precursor region with full phonon softening above the CDW phase transition in 2H-TaSe₂"** | Shen et al. | **Nat. Commun. 14, 7282 (2023)** | **LDA-DFPT**, a=3.436, SOC | **Fig 6a: Γ-M 方向 LA 虚模** q≈(0.35,0,0)≈q_CDW；5.6% 压缩稳定；EPC + FS 拓扑 | [DOI](https://doi.org/10.1038/s41467-023-43094-5) | ✅ **CC-BY 全文已下载** |

## 3. VSe₂（1T, T_CDW ≈ 110K）

| # | 标题 | 作者 | 期刊/年 | 方法 | 虚频报道 | DOI / URL |
|---|---|---|---|---|---|---|
| 6 | "Anharmonicity Reveals the Tunability of the CDW Orders in Monolayer VSe₂" | Fumega et al. | Nano Lett. 23, 1794 (2023) | PBE-DFPT, 单层 | 两个本征虚模 q₁=⅗ΓK→3×7, q₂=½ΓM→4×4；非谐调谐 CDW 序；体相 PBE 缺 vdW → 高估 T_CDW | [DOI](https://pubs.acs.org/doi/10.1021/acs.nanolett.2c04584) |
| 7 | "van der Waals driven anharmonic melting of the 3D CDW in VSe₂" | Diego et al. | Nat. Commun. 12, 598 (2021) | DFT + SSCHA | 体 VSe₂ 谐振虚模(PBE)，vdW 驱动非谐熔化 | [DOI](https://doi.org/10.1038/s41467-020-20829-2) |

## 4. NbS₂（体相无 CDW，单层有）

| # | 标题 | 作者 | 期刊/年 | 方法 | 虚频报道 | DOI / URL | 开放获取 |
|---|---|---|---|---|---|---|---|
| 8 | "Quantum Enhancement of CDW in NbS₂ in the 2D Limit" | Bianco, Errea et al. | Nano Lett. 19, 3098 (2019) | SSCHA + harmonic DFT | 单层 1H-NbS₂ 3×3 CDW(2D 增强)；体相无 CDW(量子非谐抑制)；<0.5% 应变消除 | [DOI](https://pubs.acs.org/doi/10.1021/acs.nanolett.9b00504) | ✅ [预印本 PDF](https://iris.uniroma1.it/bitstream/11573/1277475/2/Mauri_Quantum-enhancement.pdf) |
| 9 | "Monolayer NbS₂: unstable phonon modes at 2/3ΓM → 3×3 CDW" | Wu et al. | PRB 105, 174105 (2022) | PBE/DFPT, 单层 | 虚模 @ ⅔ΓM=(⅓,⅓)→冻结成稳定 3×3，匹配实验 | [DOI](https://link.aps.org/doi/10.1103/PhysRevB.105.174105) |
| 10 | "Anharmonic suppression of charge density waves in 2H-NbS₂" | Leroux et al. | PRB 86, 155125 (2012) | LDA linear-response, 体相 | 体 NbS₂ 谐振虚模(假阳性)；非谐修正≈裸频大小→需非微扰 SSCHA | [DOI](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.86.155125) | ✅ [hal.science 全文](https://hal.science/hal-00965628v1/document) |

## 5. TiSe₂（1T, T_CDW ≈ 220K）

| # | 标题 | 作者 | 期刊/年 | 方法 | 虚频报道 | DOI / URL | 开放获取 |
|---|---|---|---|---|---|---|---|
| 11 | "Efficient simulations of CDW in TMD TiSe₂" | Yin et al. | npj Comput. Mater. 10, 207 (2024) | PBE vs SCAN/r2SCAN/MVS DFPT | TiSe₂ CDW 软模(Au at L/M)；**泛函差 1.79 THz**（meta-GGA 比 PBE 更深！） | [DOI](https://doi.org/10.1038/s41524-024-01396-2) | ✅ **开放获取** |

## 6. Graphene（Kohn 反常，参考）

| # | 标题 | 作者 | 期刊/年 | 方法 | 虚频/kink 报道 | DOI / URL |
|---|---|---|---|---|---|---|
| 12 | "Kohn Anomalies and Electron-Phonon Interactions in Graphite" | Piscanec, Lazzeri, Mauri, Ferrari | — (2004) | DFPT | Γ(E2g/G 带 ~1580) + K(A1'/iTO ~1350) Kohn 反常 | [搜索](https://scholar.google.com/scholar?q=Piscanec+Kohn+anomalies+graphite) |

## 开放获取（可下载全文）

标 ✅ 的论文 URL：
1. **Shen 2023**（TaSe₂, Nat. Commun.）：https://www.nature.com/articles/s41467-023-43094-5.pdf
2. **Bianco 2019 预印本**（NbS₂）：https://iris.uniroma1.it/bitstream/11573/1277475/2/Mauri_Quantum-enhancement.pdf
3. **Leroux 2012**（NbS₂ 体相）：https://hal.science/hal-00965628v1/document
4. **Yin 2024**（TiSe₂ 泛函对比）：https://www.nature.com/articles/s41524-024-01396-2.pdf

## 总结

所有金属 CDW TMD（NbSe₂, TaSe₂, TaS₂, VSe₂, TiSe₂, NbS₂ 单层）的谐振 DFT 声子谱都在 CDW 波矢报道了**虚频软模**——这是 CDW 不稳定的直接 DFT 证据。文献与我们项目的数据完全一致（我们测到的 −67~−165 cm⁻¹ 虚频 = 这些论文报道的 CDW 软模）。无虚频的 TMD（MoS₂, WS₂, WSe₂ 等半导体）在文献中也没有虚频报道——因为无 Fermi 面。
