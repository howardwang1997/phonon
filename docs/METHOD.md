# 加速方法与原理 (Method & Why It Works)

## 0. 慢在哪 —— 必须先认清瓶颈

有限位移法算声子谱 = 把超胞里每个原子沿 ±x/±y/±z 各推一下，每推一次做一次完整的
**SCF 自洽场 DFT 计算** 读受力。一个超胞的任务量 = `6N` 次 SCF（N = 超胞原子数）。

- 单次 SCF 才是真正的瓶颈（平面波 DFT 里是 `O(N³)` 对角化 + 反复 FFT，且迭代多步）。
- 力常数装配、对角化求频率这些"后处理"相对极便宜。

**所以加速的三条路只有三条：减少 SCF 次数、降低单次 SCF 代价、让多个 SCF 并行。**
本项目三件事都做了，且作用在管线不同环节，因此**相乘**。

---

## 1. 对称性约化 —— 减少 SCF 次数（贡献 16–64×，最大头）

**实现**：`phonongpu/symmetry.py`、`displacements.py`、`force_constants.py`

- 自写空间群 finder：枚举整数旋转 R（det ±1）使 `RᵀGR=G`（晶格点群）→ 配平移 τ
  搜索完整空间群操作（也支持 spglib 加速）。
- 算原子轨道（orbit），只对**不等价原子**做位移。
- 只算不等价原子的力常数块，其余用对称操作补全（关键公式，必须把旋转转到笛卡尔）：
  $$\Phi_{g(l),\,g(i)} = R_{cart}\,\Phi_{l,i}\,R_{cart}^{T},\qquad R_{cart}=A\,R_{frac}\,A^{-1}$$

**为什么能加速**：金刚石 Si 的 64 原子超胞只有 **1 个不等价原子** → 只需 **6 次 SCF**，
而非 naive 的 384 次。SCF 是瓶颈，次数砍 64×，墙钟就砍 ~64×。这是"免费"的提速
——不损失精度（FC 恢复误差 `1e-16`），是单点收益最大的杠杆。

---

## 2. 单次 SCF 的 GPU 卸载 —— 降低单次代价（1.5×→4.75×，随体系变大）

**实现**：在主机上从零编译 **GPU 版 Quantum ESPRESSO 7.4**（NVIDIA HPC SDK 的
`nvfortran` + OpenACC + CUDA 12.5，Hopper cc90），由 `phonongpu/backends/qe_gpu.py`
驱动。

**为什么能加速**：单次 SCF 的计算热点恰好是 GPU 最擅长的：

- **FFT**（密度/势在实空间⇄倒易空间）→ cuFFT。H20 显存带宽 ~4 TB/s，CPU ~50 GB/s。
- **子空间对角化 / Davidson 迭代**（`O(N_bands³)` 密集线性代数）→ cuBLAS / OpenACC kernel。

这些都是大规模可并行、访存密集的算子，GPU 并行度远高于 CPU。实测单次 SCF：
64 原子 GPU `93.8 s` vs CPU `445.5 s` = **4.75×**；16 原子只有 1.5×（体系太小，
kernel 启动/数据搬运开销还没被计算量摊薄）。**体系越大 GPU 优势越大。**

---

## 3. 多 GPU 并发 —— 同时跑多个 SCF（~N×）

**实现**：`phonongpu/scheduler.py` 的 `MultiGPUExecutor`。不等价原子的各次 SCF
**彼此完全独立**（embarrassingly parallel）。用线程池把多个 `pw.x` 子进程同时丢到
多张 GPU 上，每个用 `CUDA_VISIBLE_DEVICES` 绑定一张卡。

**为什么能加速**：6 个 SCF + 2 张卡 → 3 批 × 单次时间，而非 6 个串行。纯并发，
2 张卡 ≈ 2× 吞吐。（顺带绕开本机一个坑：隔离 0/1 号卡会让 `cudaDeviceSynchronize`
报错 46，故用 `gpu_ids` 绑到 ≥2 号卡。）

---

## 4. 后处理也搬上 GPU（~39×，但不是瓶颈）

**实现**：`phonongpu/dynamical.py`。把所有 q 点的动力矩阵堆成 `(nq, 3N, 3N)` 批张量，
一次 batched Hermitian `eigh` 解出全部频率：

$$D_{\kappa\alpha,\kappa'\beta}(q)=\frac{1}{\sqrt{m_\kappa m_{\kappa'}}}\sum_R \Phi_{0\kappa,R\kappa'}\,e^{iq\cdot(R+r_{\kappa'}-r_\kappa)}$$

（Born–von Karman 折叠超胞 FC → 初胞动力矩阵。）实测 8000 q 点 / 24 支：
CPU `10.5 s` → GPU `0.27 s` = **39×**。诚实说这步本就便宜，对端到端影响很小，
只是确保后处理不成为新瓶颈。

---

## 5. 三者为什么能相乘 → ~400×

它们作用在管线**不同环节**，彼此独立：

| 环节 | 手段 | 实测 |
|---|---|---|
| SCF 次数 | 对称性约化 | 64× 更少 |
| 单次 SCF 代价 | GPU 卸载 | 4.75× |
| 同时跑几个 SCF | 多卡并发 | 2× |

64 原子 Si 声子端到端：GPU **421 s** vs CPU 单核 naive **171 072 s ≈ 47.5 h** →
**~406×**（GPU 实测；CPU 由实测单次 × naive 次数推算）。

---

## 6. 正确性的关键实现细节

- FC 对称填充用**笛卡尔旋转** `R_cart = A R_frac A⁻¹`（不是分数旋转）。
- **声学求和规则**（acoustic sum rule）投影：保证 Γ 有 3 个零模（平移不变性）。
- 中心差分求力 → FC，恢复精度 `1e-16`（解析 backend 验证）。
- 单位约定：力常数 → 动力矩阵 → 频率，统一用 eV/Å²/amu（QE 力由 Ry/Bohr 换算进来），
  频率转 THz 用因子 `15.633`。

## 7. 诚实边界

- **单次 SCF 的 GPU 提速只有 ~4.75×，不是 100×。** 100×+ 来自"GPU × 对称性 × 多卡"相乘。
- 体系越小 GPU 越没优势（16 原子端到端仅 ~49×）；**体系越大越划算**。
- 对称性约化是最"值"的——零精度损失、最大单点收益。
- demo 用了低 ecut + 小超胞（速度优先），频率数值未收敛；生产设置下管线完全相同。
