# Graphene degauss 与晶格温度物理化引入方案

**实验队列与算力排期：**[`GRAPHENE_PHYSICS_TEMPERATURE_EXPERIMENT_COMPUTE_PLAN.md`](GRAPHENE_PHYSICS_TEMPERATURE_EXPERIMENT_COMPUTE_PLAN.md)

## 1. 目标与当前状态

本方案用于替代当前 300/600 K 端点模型的线性温度混合。目标是在 graphene 声子计算中分别用物理公式引入：

- Fermi–Dirac `degauss (Ry)` 对电子自由能、电子极化和 Kohn anomaly 的影响；
- lattice temperature 对原子构型分布、热膨胀和非谐声子自能的影响。

最终模型需要在未参与建模的温度和 `degauss (Ry)` 组合上，定量复现 Γ、K 附近的最高光学支及 Kohn anomaly。只确认结构稳定或没有软模不作为完成标准。

当前正在运行的 450 K 冻结实验保持不变。它仍用于检验现有线性条件化模型，不能在结果完成前改公式或回写系数。本方案从该实验完成后开始实施，目前不启动新计算。

## 2. 记号与计算范围

- \(T_{\mathrm{lat}}\)：晶格温度，单位 K。
- \(s\)：Quantum ESPRESSO Fermi–Dirac `degauss`，单位 Ry。
- \(s_0=0.00285013035\ \mathrm{Ry}\)：第一版实现的参考展宽，取开发区间中点以减小两侧修正幅度；参考点只定义分解方式，不应改变最终可观测量。
- \(\mathbf R\)：原子坐标；\(\mathbf u=\mathbf R-\mathbf R_0\) 为相对参考结构的位移。
- \(A_{\mathrm{el}}(\mathbf R;s)\)：给定 `degauss (Ry)` 下的 Mermin 电子自由能。
- \(D(\mathbf q;T_{\mathrm{lat}},s)\)：质量加权的有限温动力学矩阵。

文档和图表直接报告 `degauss (Ry)`，不把它换算成电子温度 K。晶格温度始终用 K 表示。

只有 Quantum ESPRESSO 使用 Fermi–Dirac occupations 时，\(s\) 才按上述物理电子自由能解释。cold、Gaussian 等 smearing 的 `degauss` 只作为数值积分参数，不能混入本方案的物理参数拟合。

第一阶段范围固定为：

- graphene，电中性；
- 与当前实验相同的赝势、截断能、晶格几何约定和 q 路径；
- static/adiabatic DFPT 电子响应；
- 重点评价 Γ 和 K 附近最高光学支及 K 点 kink；
- 先做固定晶格常数的机制检验，再单独加入零面内应力下的热膨胀。

## 3. 当前方法与目标方法的差别

当前冻结模型使用

\[
E_{\mathrm{SR}}(T)=E_{v11}+[1-w(T)]E_{\Delta,300}+w(T)E_{\Delta,600},
\qquad
w(T)=\frac{T-300}{300},
\]

并把 q 空间 `rounded cusp + q²` 的系数在 300/600 K 之间线性插值。晶格温度与 `degauss (Ry)` 沿 physical-FD 路径同步变化，因此当前实验不能独立识别两种效应。

目标方法取消以下两类温度插值：

1. 不再按 \(T_{\mathrm{lat}}\) 混合两个短程 MLIP；
2. 不再按 \(T_{\mathrm{lat}}\) 或 \(s\) 线性插值 Kohn-cusp 系数。

线性模型保留为对照，不能作为物理模型的组成部分。

## 4. 目标理论框架

### 4.1 总自由能分解

给定 `degauss (Ry)`，用于原子采样的自由能面写为

\[
A_{\mathrm{tot}}(\mathbf R;s)
=U_{\mathrm{SR}}(\mathbf R)
+\Delta A_{\mathrm{loc}}(\mathbf R;s)
+\frac{1}{2}\mathbf u^{\mathrm T}\Phi_{\mathrm{KA}}(s)\mathbf u.
\]

三项分别表示：

- \(U_{\mathrm{SR}}\)：统一的、无显式晶格温度输入的非谐短程势；
- \(\Delta A_{\mathrm{loc}}\)：`degauss (Ry)` 引起的局域电子自由能变化；
- \(\Phi_{\mathrm{KA}}\)：由有限展宽电子极化产生的 Γ/K 非解析谐波项。

原子力统一从同一个标量自由能求导：

\[
\mathbf F(\mathbf R;s)=-\nabla_{\mathbf R}A_{\mathrm{tot}}(\mathbf R;s).
\]

这样可以保持能量—力一致性，并避免分别拟合互不相容的温度相关力。

### 4.2 `degauss (Ry)` 的局域自由能项

Fermi–Dirac 展宽通过 Mermin 自由能进入 DFT。定义

\[
\sigma_{\mathrm e}(\mathbf R;s)
=-\frac{\partial A_{\mathrm{el}}(\mathbf R;s)}{\partial s},
\]

则相对参考展宽 \(s_0\) 的局域修正为

\[
\Delta A_{\mathrm{loc}}(\mathbf R;s)
=-\int_{s_0}^{s}\sigma_{\mathrm e,loc}(\mathbf R;s')\,\mathrm ds'.
\]

实现时优先从同一构型、不同 `degauss (Ry)` 的 DFT Mermin 自由能、电子熵信息和力差构造训练数据。模型预测的是标量 \(\sigma_{\mathrm e,loc}\) 或其低参数表示，再进行积分；不直接为每个 \(s\) 拟合彼此独立的力模型。

是否需要该项由数据决定：先扣除 Kohn-anomaly 谐波响应，再检查剩余的 `degauss` 力差。如果剩余量低于第 8 节的门槛，则在当前精度范围内令 \(\Delta A_{\mathrm{loc}}=0\)，避免引入没有数据支持的自由度。

### 4.3 `degauss (Ry)` 的 Kohn-anomaly 项

主模型使用有限展宽静态电子极化，而不是温度多项式。模分辨的电子声子自能写为

\[
\Pi_{\nu\nu'}(\mathbf q;s)
=\frac{2}{N_k}
\sum_{mn\mathbf k}
\frac{
g_{mn\nu}(\mathbf k,\mathbf q)
g^{*}_{mn\nu'}(\mathbf k,\mathbf q)
\left[f(\epsilon_{n\mathbf k};s)-f(\epsilon_{m,\mathbf k+\mathbf q};s)\right]
}{
\epsilon_{n\mathbf k}-\epsilon_{m,\mathbf k+\mathbf q}+i\eta
}.
\]

上式用于固定展宽依赖和参数共享方式；实际实现中的 k 点权重、质量归一化、简并因子和矩阵单位统一遵循 QE dynamical matrix 约定，并通过 \(s_0\) 的 DFPT 矩阵重放确定。

这里 `degauss (Ry)` 只通过 Fermi–Dirac 占据和自洽电子结构进入。相对参考展宽的非解析修正为

\[
\Delta D_{\mathrm{KA}}(\mathbf q;s,s_0)
=\mathcal P_{\Gamma,K}
\left[\Pi(\mathbf q;s)-\Pi(\mathbf q;s_0)\right]
\mathcal P_{\Gamma,K}^{\dagger},
\]

其中 \(\mathcal P_{\Gamma,K}\) 是冻结的 Γ/K 模子空间投影，避免支序交换造成伪差异。

实现路线按以下优先级推进：

1. 从统一的 DFT/Wannier 电子结构和电子—声子矩阵元重算不同 \(s\) 的占据求和；
2. 如果完整 band-sum 暂时不可用，在 Γ/K 邻域使用 graphene Dirac 低能模型：

   \[
   q_s=\frac{s}{\hbar v_F},
   \qquad
   \Delta D_Q(x;s)
   =A_Q q_s\,
   \mathcal F_Q\!\left(\frac{x}{q_s},\frac{\mu}{s}\right)
   +D_{Q,\mathrm{bg}}(x),
   \quad Q\in\{\Gamma,K\}.
   \]

   \(v_F\)、\(\mu\) 和必要的电子—声子耦合振幅由电子结构或独立计算确定。只允许拟合少量、在所有 \(s\) 间共享的参数。

3. 当前 `rounded cusp + q²` 模型作为消融对照，不再允许三个系数随温度自由线性变化。

所有 q 空间修正需要满足晶体对称性、Hermiticity 和 acoustic sum rule。Γ/K 星上的等价点使用同一组物理参数。

### 4.4 晶格温度的统计力学引入

短程势不接收 \(T_{\mathrm{lat}}\) 作为特征。晶格温度通过原子统计分布进入。经典极限下

\[
P(\mathbf R\mid T_{\mathrm{lat}},s)
\propto
\exp\!\left[-\frac{A_{\mathrm{tot}}(\mathbf R;s)}
{k_{\mathrm B}T_{\mathrm{lat}}}\right].
\]

与当前结果连续的低成本对照使用 TDEP：

\[
\Phi_{\mathrm{TDEP}}(T_{\mathrm{lat}},s)
=\underset{\Phi}{\arg\min}\;
\left\langle
\left\|\mathbf F(\mathbf R;s)+\Phi\mathbf u\right\|^2
\right\rangle_{P(T_{\mathrm{lat}},s)}.
\]

最终物理结果优先使用 quantum SSCHA 或 SCPH。温度通过 Bose–Einstein 占据

\[
n_B(\omega,T_{\mathrm{lat}})
=\frac{1}{\exp(\hbar\omega/k_{\mathrm B}T_{\mathrm{lat}})-1}
\]

以及自洽自由能最小化进入，不使用关于 \(T_{\mathrm{lat}}\) 的经验多项式。可采用的等价表述为

\[
\Omega_{\nu\mathbf q}^2(T_{\mathrm{lat}},s)
=\omega_{\nu\mathbf q}^2(s)
+2\omega_{\nu\mathbf q}(s)
\operatorname{Re}
\left[\Sigma^{(3)}_{\nu\mathbf q}
+\Sigma^{(4)}_{\nu\mathbf q}\right],
\]

其中三阶、四阶声子自能的温度依赖由 \(n_B\) 和自洽频率决定。

经典 TDEP 和 quantum SSCHA/SCPH 的结果分别保存和标注。最终不能把经典核采样结果直接表述为已经包含量子晶格温度效应。

### 4.5 热膨胀

先在固定晶格常数下验证电子响应和非谐自能，避免晶格变化与 Kohn anomaly 混在一起。通过后增加零面内应力分支：

\[
a(T_{\mathrm{lat}},s)
=\underset{a}{\arg\min}\;
\left[A_{\mathrm{el}}(a;s)+F_{\mathrm{vib}}(a;T_{\mathrm{lat}},s)\right].
\]

最终同时报告：

- fixed-cell spectrum：用于方法验收和不同温度的直接比较；
- zero-stress spectrum：包含热膨胀的物理预测。

两条结果不能混用同一组验收误差。

### 4.6 有限超胞采样与 q 空间重建

在有限超胞采样时，将物理电子极化公式投影成对应超胞的实空间算子 \(\Phi_{\mathrm{KA}}^{\mathrm{SC}}(s)\)。保存轨迹后，从每帧总力中扣除完全相同的算子，再拟合 short-range TDEP/SSCHA 响应。最终在细 q 网格上加回由同一电子极化公式生成的 \(D_{\mathrm{KA}}(\mathbf q;s)\)：

\[
D_{\mathrm{tot}}(\mathbf q;T_{\mathrm{lat}},s)
=D_{\mathrm{SR,eff}}(\mathbf q;T_{\mathrm{lat}},s)
+D_{\mathrm{KA}}(\mathbf q;s).
\]

实空间算子和 q 空间修正必须来自同一个参数清单和代码版本。矩阵重放误差、扣除误差及重复计数检查均写入冻结清单。

## 5. 参数可辨识性与数据设计

只沿 \(T_{\mathrm{lat}}\) 与 \(s\) 同步变化的对角线取点，无法区分晶格和电子效应。新方案使用交叉设计。

### 5.1 开发点

当前 450 K 实验结束后，开发集合使用

\[
T_{\mathrm{lat}}\in\{300,450,600\}\ \mathrm K,
\]

\[
s\in
\{0.0019000869,\ 0.00285013035,\ 0.0038001738\}\ \mathrm{Ry}.
\]

不需要立即运行九条 DFT-MD。先复用三个晶格温度已有的热构型，并对相同构型使用三个 \(s\) 重新计算单点自由能和力。这样可以在固定构型下直接测量电子展宽响应。

第一轮 pilot：

- 每个 \(T_{\mathrm{lat}}\) 固定选择 5 个已有构型；
- 每个构型补齐三个 \(s\) 的 Mermin 自由能、电子熵信息和力；
- 已存在的同条件标签直接复用，只补缺失的交叉标签；
- 先判断电子响应是否主要由谐波 Kohn 项解释。

若 pilot 表明局域 `degauss` 力差不可忽略，扩展到每个晶格温度 20 个固定索引构型。扩展前固定索引，不按中间误差选择新增构型。

### 5.2 静态电子响应数据

对固定参考几何执行：

- Γ/K 固定点的 \(k=120\rightarrow144\) 收敛；
- Γ/K 邻域 dense line；
- 所有开发 \(s\) 使用相同 k 网格、赝势、截断能和收敛阈值；
- 保存 dynamical matrix，不只保存排序后的频率；
- 记录模投影、支追踪、acoustic sum rule 和对称性修复前后的差异。

band-sum/Wannier 方案还需保存统一的电子能带、Fermi level、\(v_F\)、电子—声子矩阵元及 k 网格收敛信息。

### 5.3 前瞻验证点

开发完成后，在读取验证标签前固定模型、参数、构型索引和验收标准。建议验证点为

\[
T_{\mathrm{lat}}\in\{375,525\}\ \mathrm K,
\]

\[
s\in
\{0.002375108625,\ 0.003325152075\}\ \mathrm{Ry}.
\]

至少包含：

- 两个 physical-FD 对角点：
  \((375\ \mathrm K,0.002375108625\ \mathrm{Ry})\) 和
  \((525\ \mathrm K,0.003325152075\ \mathrm{Ry})\)；
- 两个交换后的非对角点，用于检验电子与晶格效应是否真正解耦；
- 一个独立 DFT-MD 轨迹，用于排除只对 MLIP on-policy 构型有效的情况。优先选择
  \((450\ \mathrm K,0.0019000869\ \mathrm{Ry})\) 这一非对角条件；当前方案若已完成
  \((450\ \mathrm K,s_0)\) 的独立轨迹，则作为额外的 physical-FD 对角验证。

验证数据一旦用于修改公式或参数，即转为开发数据；随后必须另选未见条件做最终验收。

## 6. 实施阶段

### P0：完成并封存当前 450 K 实验

- 不改变当前线性温度模型、轨迹、标签索引和验收门槛；
- 保存最终 sampling bootstrap、DFT force、DFT-TDEP 和 DFPT line 结果；
- 明确其结论只适用于现有 physical-FD 对角路径。

### P1：统一数据约定与参考分解

- 固定参考几何、模规范、\(s_0\)、q 路径和矩阵单位；
- 审计 QE 输出中的 Mermin 自由能、电子熵项和用于原子力的能量定义；
- 对已有 300/450/600 K 标签建立可追踪 inventory；
- 建立 `development`、`validation_locked`、`independent_dftmd` 三个数据域；
- 输出第一版 freeze manifest。

### P2：构建有限展宽电子极化模型

- 先完成 static DFPT 的 \(s\) 与 k 网格收敛；
- 实现 band-sum/Wannier 极化，或冻结 Dirac 低能模型；
- 在 dynamical-matrix 层面拟合共享参数；
- 生成 Γ/K 星上对称化的 q 空间修正；
- 与当前 `rounded cusp + q²` 线性模型做相同数据上的消融比较；
- 通过电子通道 gate 后才进入热构型训练。

### P3：判断并构建局域 `degauss` 自由能修正

- 对 pilot 构型做交叉 \(s\) 单点标签；
- 从 DFT 力差中扣除 \(\Phi_{\mathrm{KA}}(s)\) 的贡献；
- 若残差可忽略，冻结 \(\Delta A_{\mathrm{loc}}=0\)；
- 若残差显著，拟合 \(\sigma_{\mathrm e,loc}\) 并对 \(s\) 积分；
- 检查能量—力一致性及 \(s\) 往返积分闭合误差。

### P4：训练统一短程非谐势

- 合并 300/450/600 K 构型覆盖范围，但不给模型输入 \(T_{\mathrm{lat}}\)；
- 从训练标签中扣除同一物理 \(\Phi_{\mathrm{KA}}(s)\)；
- 需要时加入 P3 的局域电子自由能项；
- 使用按温度、按轨迹、按构型同时分组的训练/验证切分；
- 进行端点重放、未见构型 force gate 和 Hessian replay。

### P5：晶格温度重整化

- 使用三个 seed 做经典 TDEP，验证与当前管线的连续性；
- 对同一势能面运行 quantum SSCHA 或 SCPH；
- 逐温度检查自由能收敛、有效样本数、频率自洽和声学支；
- 分别保存 classical 与 quantum 结果；
- 在固定晶格常数下先通过 gate，再计算热膨胀分支。

### P6：二维组合与交叉项检验

在开发网格计算完整的 \(D(T_{\mathrm{lat}},s)\)，并定义有限差分交叉项

\[
\Delta_{\times}D
=D(T_2,s_2)-D(T_2,s_1)-D(T_1,s_2)+D(T_1,s_1).
\]

如果交叉项低于第 8 节门槛，可以报告电子与晶格贡献在当前精度下近似可加。如果超过门槛，最终模型必须在目标 \((T_{\mathrm{lat}},s)\) 上进行耦合采样，不能用两个一维结果相加。

### P7：前瞻验证与独立轨迹

- 冻结所有模型和门槛后生成 375/525 K 标签；
- 先评价对角点，再评价非对角点；
- 所有验证点均通过后启动独立 DFT-MD；
- 独立轨迹通过后，才能声称模型同时描述晶格温度和电子展宽的可迁移效应。

## 7. 机器分工

当前队列完成后再分配新任务：

- V100-A：低 `degauss (Ry)` 的 DFPT k 网格收敛、Γ/K dense line 和电子极化基准；
- V100-B：交叉 `degauss (Ry)` 的热构型 DFT 单点标签及验证标签；
- RTX 2060：统一 MLIP、交叉组合采样、TDEP、bootstrap，以及可承载的 SSCHA/SCPH 任务；
- 本地：数据 inventory、矩阵审计、拟合、冻结清单和结果汇总。

远程提交、同步和取回全部使用 Tailscale 通道。新任务不得抢占或修改当前 450 K 冻结实验目录。

## 8. 固定验收标准

以下是新方案的最低 release gate；更严格的数值可以在验证标签生成前根据开发集固定，但不能在看到验证结果后放宽。

| 层级 | 指标 | 门槛 |
|---|---|---:|
| DFPT 收敛 | \(k=120\rightarrow144\) 的 Γ/K 顶支最大变化 | `<1 cm⁻¹` |
| 电子公式 | 未见 `degauss (Ry)` 的 Γ/K line MAE | `<10 cm⁻¹` |
| 电子公式 | 未见 `degauss (Ry)` 的 Γ/K 顶支绝对误差 | `<15 cm⁻¹` |
| 电子公式 | 未见 `degauss (Ry)` 的 K kink 相对误差 | `<20%` |
| 总力 | 未见构型 force RMSE/max | `≤50/250 meV/Å` |
| 采样 | 三 seed 全谱两两 MAE 的 point estimate 和 95% block-bootstrap 上界 | `<5 cm⁻¹` |
| 温控 | mean lattice temperature 相对目标值误差的 point estimate 和 95% 上界 | `≤5%` |
| 最终声子 | Γ/K line MAE | `<10 cm⁻¹` |
| 最终声子 | Γ/K 顶支绝对误差 | `<15 cm⁻¹` |
| 最终声子 | K kink 相对误差 | `<20%` |
| 重放 | 实空间算子与 q 空间矩阵投影的最大频率差 | `<1e-5 cm⁻¹` |

局域 `degauss` 分支的判定标准：扣除物理 Kohn 项后，交叉标签的剩余力差 RMSE
`≤10 meV/Å`，并且满足以下任一条件：

- 原始 `degauss` 力差 RMSE 本身 `≤10 meV/Å`，整个局域响应低于目标分辨率；
- 剩余力差不超过原始 `degauss` 力差 RMSE 的 `20%`。

满足时允许令 \(\Delta A_{\mathrm{loc}}=0\)。否则必须实现局域 Mermin 自由能修正。

电子—晶格可加性的判定标准：忽略 \(\Delta_{\times}D\) 后，Γ/K line 的附加误差必须 `<2 cm⁻¹`，且不能使顶支或 kink 越过最终验收门槛。未满足时使用耦合采样结果，不作可加性结论。

bootstrap 固定使用 1000 次重采样；block 长度继续采用 `max(5, ceil(2*tau_int))`。不根据中间结果提前停止或反复追加直到通过。

## 9. 结果文件与可复现性

建议使用独立目录，避免覆盖当前实验：

```text
results/graphene_physics_temperature/
  inventory/
  electronic_response/
  degauss_cross_labels/
  local_mermin_model/
  unified_short_model/
  lattice_tdep_classical/
  lattice_sscha_quantum/
  cross_grid/
  validation_locked/
  independent_dftmd/
  freeze_manifest.json
```

对应代码统一放在：

```text
scripts/smearing_kink/physics_temperature/
```

每个可发布结果必须记录：

- Git commit 和 dirty-worktree inventory；
- 模型、DFT 输入、赝势、轨迹、构型索引和矩阵文件的 SHA-256；
- \(T_{\mathrm{lat}}\)、`degauss_Ry`、晶格常数、采样方法及 classical/quantum 标记；
- 电子响应参数、参考 \(s_0\)、投影子空间和支追踪规则；
- 开发数据与未见验证数据的边界；
- 所有 gate 的 point estimate、置信区间和失败项。

## 10. 分层结论范围

- **E1：电子通道通过。** 物理电子极化公式预测未见 `degauss (Ry)` 的静态 DFPT Γ/K 线。
- **L1：晶格通道通过。** 无显式晶格温度输入的势能面结合 TDEP/SSCHA，预测未见 \(T_{\mathrm{lat}}\) 的 DFT-TDEP 或相应量子基准。
- **X1：二维交叉通过。** 对角和非对角 \((T_{\mathrm{lat}},s)\) 条件同时通过，能够区分电子与晶格效应。
- **C2：独立轨迹通过。** 独立 physical-FD DFT-MD 验证通过，可以报告对独立构型分布的迁移能力。

只通过 E1 或 L1 时，不能把结果表述为已经完成有限晶格温度 Kohn anomaly 的整体定量复现。只通过 on-policy 验证时，也不能声称对独立 DFT-MD 轨迹成立。

## 11. 失败分支

- band-sum/Dirac 公式不能通过静态 DFPT gate：检查 k 网格、掺杂/化学势、电子—声子矩阵元和动态效应；不回退为任意温度多项式并继续称为物理模型。
- 局域 `degauss` 力残差显著：增加交叉单点标签，使用可积分的 Mermin 自由能修正。
- 统一短程势不能覆盖 300–600 K 构型：增加主动学习构型，但保持 \(T_{\mathrm{lat}}\) 不作为模型输入。
- classical TDEP 与 quantum SSCHA/SCPH 差异显著：分别报告，并以预先选定的 quantum 结果作为物理晶格温度结论。
- 电子—晶格交叉项显著：在每个目标组合上运行耦合采样，停止使用一维贡献相加。
- 375/525 K 验证失败：将失败点转为开发数据，修改后选择新的未见温度和非对角组合重新验证。

## 12. 最近的下一步

1. 等当前 450 K 线性条件化实验按原冻结方案完成；
2. 封存其结果，不用本方案重新解释为物理公式验证；
3. 审计 QE Mermin 自由能、电子熵和力输出是否足以构造 P3 数据；
4. 用每个晶格温度 5 个构型启动交叉 `degauss (Ry)` pilot；
5. 同时实现有限展宽电子极化的最小原型，并在已有静态 DFPT 数据上做开发集重放；
6. 在读取 375/525 K 标签前，固定模型、验证点和验收标准。
