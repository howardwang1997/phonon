# Graphene R2R-1 条件线性 readout attempt2_retry1 实现合同

## 1. 目标和授权边界

R2R-0 attempt3 已得到 `R2R0_FORMAL_REPRESENTATION_PRECHECK_PASSED`。R2R-1 只在冻结的 R2Q encoder 和冻结的 R2R 多极化表示上拟合 65 个线性 residual coefficients。它不更新 encoder，不增加或删除特征，不读取 energy label，也不把温度、electronic smearing 或数据来源作为模型输入。

R2R-1 的四折结果只能解释为 conditional linear-readout OOF。R2Q encoder 的上游训练见过相关 thermal corpus，因此这不是 encoder-level 独立验证。R2R-1 通过也不直接授权 SSCHA、正式声子谱或 Kohn anomaly 结论。

在合同、代码和 tests 就绪但外部 GO 尚未给出时，程序只允许执行 label-blind preflight 和 label-blind manifest metadata binding。manifest 生成器不得创建 marker。preflight 对 thermal92 只检查冻结的 lexical path，不 `stat` 最终文件，也不打开或读取任何字节；metadata-only binding 是独立 manifest 生成步骤，由生成器用 no-follow `stat` 记录 parent chain、regular-file identity、单硬链接和文件大小。manifest 记录既有 whole-file SHA256 常量，但 live whole-file hash、header/Properties 检查和标签解析全部推迟到 GO 通过之后。fit 入口必须先验证 marker 的精确内容等于 manifest SHA256 加一个换行；parser 随后打开 marker 所绑定的 exact thermal fd，并在读取第一个 payload byte 前再次重验授权。通过后才可在该 fd 上先 hash、再 rewind/stream parse；解析后的 unchanged-file SHA 复核也必须以固定大小 chunk 流式更新 digest，不能重新聚合整份 extxyz bytes。

正式 fit 全程禁止打开名称或路径含 `seed1|seed2|small|small12|small_gate|support|reserved|outer_fold|held|holdout|valid_e50|full_report` 的输入；分隔符和 `seed01/seed0_1` 等别名先归一化再检查。actual small-H、seed1、seed2、support 和任何 held 数据都不在 R2R-1 fit 程序输入中。

### 1.1 attempt1、失败 attempt2 的只读封存与 attempt2_retry1 路径

2026-08-26 的第一次物化使用以下三个 sibling：

- fit：`R2R1_conditional_linear_readout_fit_seed83_20260826`；
- control：`R2R1_conditional_linear_readout_control_20260826`；
- candidate：`R2R1_conditional_linear_readout_candidate_20260826`。

该次计算只留下 `RUNNING + fit_manifest.json + fit_arrays.npz + fit_receipt.json + completion.json`；没有 `DONE/FAILED/EXIT_CODE`，candidate 不存在。长 recovery 期间 macOS 在 owned tree 上异步加入 `com.apple.provenance`，使旧实现把目录 `mtime/ctime` 漂移误判为 release root 被替换。因此它是不可部署的 forensic partial，禁止 cleanup、adopt、补写 terminal 或从中恢复发布。其冻结只读记录为：

| artifact | SHA256 |
|---|---|
| `RUNNING` | `2f232f1bed321fa3c4f45ffca89cbf66aaad3cc6a00c0ff7db5f2d3fa921a9c5` |
| `fit_manifest.json` | `264596e7564220a75a8ace1e0b9a79ab08b2b1e0950d4986a5cdc8ad9cdcfeb1` |
| `fit_arrays.npz` | `3c79fb6215f54d48e8ac12cbd9f9cfa37c95dfc10c155f25da92471d99c8e17e` |
| `fit_receipt.json` | `dc87a17a2ead76b7e180daaec45f4deeeeb13a920f8c9d2f4dc4a2af8e5e15fb` |
| `completion.json` | `13be6d87891aa2a976f8b0950db09b8a2b034b9ef38e686a81a08be049efe7be` |
| control `freeze_manifest.json` | `264596e7564220a75a8ace1e0b9a79ab08b2b1e0950d4986a5cdc8ad9cdcfeb1` |
| control `R2R1_FORMAL_GO` | `f48819a09bfcaf2772cf4d7316b9da58fd68ad12b3c79d4bf3eeadb3f208438b` |
| OOF FP64 raw array | `7423ffdb60eba487ca00f75f557639fa3bdf4bfe9e1465d26c4378856b488657` |

GO marker 恰为 65 bytes：ASCII manifest SHA `264596e7…cfeb1` 后接一个 `LF`，无其他字节。partial 中的科学 receipt 已完成唯一一次 nested OOF，状态为 `R2R1_CONDITIONAL_OOF_FAILED`：E50 A′ RMS `19.0399904293643 meV/Å`，高于 `15 meV/Å` 门；E50/T300/T600 force RMSE 分别为 `15.8971402310/16.5000452910/21.7770818781 meV/Å`，最大误差分别为 `63.2339838565/93.8983675877/120.3979163662 meV/Å`，E50 slope relative error 为 `0.0064464230`，这些门均通过。该 receipt 只用于 forensic 说明，不能替代 attempt2_retry1 的独立事务终态。

第二次尝试使用 basename 含 `attempt2` 的三路径。它在生成 manifest 时因旧的 macOS owned-inode provenance settlement 规则 fail closed；只创建了空 control directory，未留下 manifest、GO、fit root 或 candidate。该 control directory 与 attempt1 一样永久只读，不能 cleanup、复用、adopt 或作为 retry 的 control root。

本合同唯一可授权的新事务显式命名 `attempt2_retry1`：

- fit：`R2R1_conditional_linear_readout_fit_seed83_attempt2_retry1_20260826`；
- control：`R2R1_conditional_linear_readout_control_attempt2_retry1_20260826`；
- candidate：`R2R1_conditional_linear_readout_candidate_attempt2_retry1_20260826`。

三者必须彼此独立，并与 attempt1 三路径、失败 attempt2 三路径、attempt3、thermal/source/dependency 输入均不重叠。attempt1 与失败 attempt2 的路径只参与 lexical overlap 保护；任何 attempt2_retry1 preflight、manifest、authorization、launcher、recovery 或 production test 都不得对它们执行 `stat/open/read/scandir/write/cleanup`。科学合同、fold、alpha grid、threshold、labels 和模型结构在 attempt2_retry1 中保持不变。

## 2. 冻结输入

唯一 R2R-0 来源为 attempt3：

| artifact | SHA256 |
|---|---|
| aggregate receipt | `535ab43e1771523dbee0c949caa747f8897d89ba38469d8ef1f9d4fef716a205` |
| aggregate arrays | `caa344eda65448cea0bc0c48c307b8777c0062243777927f3a34c22c3dd6af45` |
| launch receipt | `cadc0616a5426c84e27ae13b3fc1acf90aa11e76225a68b92ec58299a682d675` |
| freeze manifest | `a9a1732af38c4513242fbbe7d23c6e5fd6fdd091ee27cf62a8600544bb8cf73f` |
| R2R canonical | `e589497c7b9f6a9d4ff0cdcc434cf07804fa0fd732d791dba811d4c7e1c5aecc` |
| R2R-0 formal contract | `cc407b3e1b18ad00097fc433fbe9458100790b8150dac599908cfcf7d2f4ddea` |

aggregate receipt 必须继续声明：representation precheck pass，非 borderline、非 numerically inconclusive，fit/force-label/energy-label/seed1/seed2/held/support access 全为 JSON Boolean `false`。

R2R-1 只读取 aggregate 中五个 thermal 数组：

| array | shape | C-order raw-byte SHA256 |
|---|---:|---|
| `thermal_global_index` | `92` | `d55134adfc6b7022c523e96db6e8645053495abd0d09500d45ae4657ac3b64f1` |
| `thermal_fixed_energy_eV` | `92×1` | `68f12611817fd8b427b05d7e98d58c5ec3d6cc97845e5adb7bd29a7801eda2da` |
| `thermal_fixed_force_eV_A` | `92×72×3` | `c6459749ba0eea8c6a1e5cae963fd3cea8c8023fee65a5c3a1fb3f9a79726d7a` |
| `thermal_parameter_energy_design_eV` | `92×65` | `223894c3749a6ec15d000d0ca6253c58c541143b8f6af8cbd6d490a2c59b29f7` |
| `thermal_parameter_force_design_eV_A` | `92×72×3×65` | `76278c36f35b269873140bf750177eb2e52513800ca9970817492dae4c6831e2` |

`thermal_global_index` 必须逐元素等于 `0..91`。launcher/aggregate 不能只信 aggregate NPZ；三台 shard 必须分别从 attempt3 的 modulo-3 shard artifacts 重建这五个数组，aggregate 再验证与冻结 aggregate bit-exact。

## 3. thermal92 标签白名单

唯一标签文件 SHA256 为 `cc2e9c68d418fba996e8ac89c8156c8adf00768406f6171aaa57f0a9bd2720c1`。配置顺序固定为：

- global `0:20`：`r2o_exact_e50_seed0_train`；
- global `20:56`：`r2o_auxiliary_T300_train`；
- global `56:92`：`r2o_auxiliary_T600_train`。

label loader 必须逐行 streaming 解析。禁止 `ase.io.read` 或把 extxyz 的全部数值列加载到内存。所有未列入白名单的原子列和 energy header 保持 opaque string，不能转换为数值。

全部 92 构型唯一 fit target 是 `REF_forces`。其 little-endian FP64 C-order hashes 为：

| scope | SHA256 |
|---|---|
| all92 | `75da2423b0584112c2fc85c0788eefae4ba6263d0e712a719491cad68439c65b` |
| E50 | `6b418481503ae5457c7755b717283c923f2369a30e67c1982d9b917396d1d7b0` |
| T300 | `f5818695082b9af9d8b271953d1cacfd3d51c41db036de26b6deba5fed6e64b0` |
| T600 | `18e25a9b2499d86c780e1bfe5f46b38b5a43d9ee9e645ef180e9a396f72a19ef` |

E50 OOF metric 额外允许：

| field | required shape | raw SHA256 |
|---|---:|---|
| `APRIME_mode_real` | `20×72×3` | `8a94f82d559da0e41dfcd8834140f9937e6076c3b7a024d9ae1305dccc632bbb` |
| `APRIME_mode_imag` | `20×72×3` | `4514e35b7a98d710ffdb66766240c9e3d29a946c63cd6a62a09e0acee91fb312` |
| complex coordinates `[real,imag]` | `20×2` | `d02189db4494c4021a3b44c632655db4a096abcd640ad0b280f233d9f34d0807` |
| `FOUNDATION_BASE_forces` | `20×72×3` | `525265d529ed19e7222ac692d924a34d7dd665f36f44c72c0585adee22332112` |
| `FROZEN_Q6_forces` | `20×72×3` | `fc518fc5f187bea6b5cbf7149bcaa5eec5977e12eb74c296df1c97c72a0372ee` |

`DFT_TOTAL_forces` 不进入 fit 或 alpha selection。所有 `*_energy`、`REF_energy=0`、TOTAL/LONG_RANGE/SHORT_RANGE/CORE_TARGET/CURRENT_S0 数值列均禁止使用。temperature、degauss、split、source 只可作 schema/provenance 检查，不能进入 feature、scaler、weight 或 selection。

## 4. 唯一模型、目标和 energy gauge

对构型 `i`：

`F_hat_i(p) = F_fixed_i + X_F_i p`

`E_hat_i(p) = E_fixed_i + X_E_i p`

`y_i = REF_forces_i - F_fixed_i`

其中 `p∈R^65`。列顺序固定为 `w0[0:32], b1, w1[0:32]`。不允许 global intercept、目标均值消去、删列、PCA 或额外非线性映射。部署组合仍是 `frozen foundation + G_R2R + frozen q6`，不能再次加旧 R2Q tail。

R2R-1 是 force-only fit。`REF_energy=0` 不是物理能量标签。R2R Taylor-null 构造固定 `G_R2R(x0)=0`，所以不拟合 energy offset。`E_fixed+X_E p` 只用于输出与力一致的保守能量和 mechanics。

## 5. RMS scaler、group weight 和 ridge

冻结数值为：

- `alpha_grid = [1e-10,1e-9,...,1e2]`，共 13 个十进幂；
- `force_scale = 0.030 eV/Å`；
- group masses `E50/T300/T600 = 0.50/0.25/0.25`；
- zero-column relative RMS `1e-12`；
- scaled condition limit `1e8`。

对任一训练子集 `T`：

`s_k = sqrt(mean_{i∈T,atom,cart}(X_iatomcartk²))`

只除 train-force RMS，不减列均值。若任一列 `s_k≤max(s)×1e-12`、rank 不为 65 或 condition 超过 `1e8`，fail closed。

令 `D=X/s`、`q=s*p/0.030`、`t=(REF-F_fixed)/0.030`。每个力分量的权重为

`w_iatomcart = M_group / [N_group(T) × N_force_components_per_config]`。

权重总和必须在绝对 `2e-15` 内等于 1。用 FP64 thin SVD 唯一求解：

`min_q Σ w(Dq-t)² + alpha ||q||²`

SVD filter 固定为 `sigma/(sigma²+alpha)`；最终 `p=0.030*q/s`。ridge 只作用于 65 个新增系数。

attempt3 design-only 固定预期为：outer train69 condition `[17597.3386272,17562.7436723,17758.5455550,18056.6625238]`；12 个 inner train46 的 condition 范围 `17308.2376174–18328.3568900`；所有 rank 65，所有 split 的 `min(s)/max(s)` 范围 `0.0466546091–0.0481074826`。这组量不读取标签。

## 6. Nested conditional OOF

四个 outer fold 是连续分层切片。每折 global indices 分别为：

- fold0：E50 `0:5`、T300 `20:29`、T600 `56:65`；SHA `b24b8a36...ffef`；
- fold1：E50 `5:10`、T300 `29:38`、T600 `65:74`；SHA `351db856...96f1`；
- fold2：E50 `10:15`、T300 `38:47`、T600 `74:83`；SHA `616490f0...f14b`；
- fold3：E50 `15:20`、T300 `47:56`、T600 `83:92`；SHA `14668a31...a920`。

完整 fold JSON semantic SHA 为 `f2dd459f6d2ba9633843ba68ebddf09ee1dac0b74e78bf9756ac8183b5d52bc6`。

对 outer fold `f`，其 23 个构型在 alpha 选择、scaler 和 fit 中完全不可见。在剩余 69 个构型内，用另外三个 canonical blocks 做 inner 3-fold：每个 inner train 是 `10 E50 + 18 T300 + 18 T600`，inner validation 是 `5+9+9`。每个 inner train 独立算 scaler。某个 alpha 的三份 inner predictions 拼成 outer-train69 的 selection-only prediction。

候选 score 固定为三组 force RMSE/max、E50 A′ RMS 和 slope error 的最坏归一化比：

`S=max(RMSE_g/30, max_g/200, Aprime_RMS/15, |slope_error|/0.05)`。

比较 `round(S,12)`；并列时取更大的 alpha。raw score 和全部 candidate metrics 都必须保存。选中 `alpha_f` 后，用完整 outer-train69 的 scaler/refit，产生该 fold 唯一 outer prediction。四份 prediction 按 global index 拼成 92 个 OOF predictions；只有 pooled metrics 决定通过，逐折指标仅报告。

A′ 使用 complex128 口径。对每个 E50：

- `mode = mode_real + i mode_imag`，shape `72×3`，flatten 后参与 `np.vdot`；
- `coord = coord_real + i coord_imag`；
- `F_target_total = FOUNDATION_BASE + FROZEN_Q6 + REF`；
- `F_pred_total = FOUNDATION_BASE + FROZEN_Q6 + F_fixed + Xp`；
- `z = np.vdot(mode.flatten(), F_total.flatten())`；
- A′ RMS 为 `1000*sqrt(mean(abs(z_pred-z_target)^2)) meV/Å`；
- `k=-Re(vdot(coord,z))/Re(vdot(coord,coord))`；
- 若 `|k_target|≤1e-14 eV/Å²`，数值状态 fail closed；
- slope relative error 为 `(k_pred-k_target)/k_target`。

Pooled OOF gate 为：

| gate | threshold |
|---|---:|
| E50 force RMSE/max | `≤30/200 meV/Å` |
| E50 A′ RMS | `≤15 meV/Å` |
| E50 slope relative absolute error | `≤0.05` |
| T300 force RMSE/max | `≤30/200 meV/Å` |
| T600 force RMSE/max | `≤30/200 meV/Å` |

OOF 未通过时状态为 `R2R1_CONDITIONAL_OOF_FAILED`，立即停止；不得运行 final alpha CV、全 92 fit 或读取 development。

## 7. Final alpha 和全 92 readout

只有 nested OOF PASS 后，才运行一套明确标记为 selection-only、不是 OOF gate evidence 的普通 canonical 4-fold CV。它对每个 alpha 在四个 train69 上独立算 scaler/refit，按同一 score/round/tie-break 选 `alpha_final`。

随后用全 92 train scaler 和 `alpha_final` 拟合唯一 65-vector。全 92 train 使用与 pooled OOF 相同的三组 force/A′ 门。必须原子冻结 coefficients、scaler、alpha、nested OOF、final selection、train metrics、代码/输入/环境 hash 和 label parser receipt。outer/inner coefficients 是临时 OOF 产物，禁止部署。

## 8. 冻结 readout mechanics

arbitrary-coefficient production 物理路径只允许由独立、hash-bound 的 `graphene_r2r1_frozen_readout.py` 提供。linear-readout core 不复制 private R2R production 公式。该模块必须用一个 scalar `E_fixed + X_E·p` 求 E/F/full-H；禁止保留 66 列 create-graph 后再求 Hessian。

本合同绑定的 frozen-readout module SHA256 为 `356c8506582a201be11dfd58fe3acd1c079e843dba568b73132f405519ff77bb`，其独立 tests SHA256 为 `2ff64cb565d5a6ce2550898f2b390573ede31cc6623b368765c84675af3056c0`。R2R-1 fit stage 只能按该模块的 `checkpoint_receipt_payload` 生成恰好三个 checkpoint 文件；不能复制 receipt schema 或生产 E/F/H 公式。

正式 mechanics 必须使用实际 final `p`，不能复用 R2R-0 的 `linspace(-0.2,0.3,65)` receipt，也不能手工把 harmonic design 置零。重验至少包括：

- harmonic-zero32：`max b≤1e-20 Å⁴`、`max a≤1e-12`、`max|E|≤1e-10 eV`、`max|F|≤1e-9 eV/Å`；
- reference E/F `≤1e-10/1e-9`，H/H-antisymmetry/ASR 各 `≤1e-7`，Γ/K drift `≤2 cm⁻¹`；
- O(3) E/F `≤1e-6/1e-5`；translation E/F `≤1e-9/1e-8`；permutation E/F `≤1e-6/1e-5`；
- native wrap/order-MIC E/F `≤1e-6/1e-5`；
- E→F 和 F→H finite difference 各 `≤1e-5`；
- nonreference H antisymmetry/ASR 各 `≤1e-7`；
- localized 6×6↔8×8 total E/max F `≤1e-7/1e-5`。

当前这组 CLI 不执行 mechanics，最高只能发布 `R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING`。pending checkpoint 只是下一阶段 actual-`p` mechanics 的输入，不能作为 `R2R1_FINAL_READOUT_FROZEN` 部署。下一阶段必须在独立 freeze manifest 中绑定 fit release、external anchor 和 checkpoint receipt SHA，并只通过 frozen-readout module 加载和调用 production E/F/H API。上述 mechanics 全部通过后，后续独立阶段才可发布 `R2R1_FINAL_READOUT_FROZEN`。

## 9. Development 和真正未见数据

严格顺序为：nested OOF PASS → final fit/hash 冻结 → final train/harmonic/reference/mechanics PASS → 才打开 E50 seed1 与 actual small-H，作为 development confirmation。

development 沿用 E50 `30/200/A′15/slope5%` 和 actual small-H `0.5/10 meV/Å`。它们已被 R2Q endpoint 打开过，不能称为 held。development 结果不得回调 alpha、scaler、权重、阈值或 coefficients。失败时停止，不打开真正未见数据。

旧项目记录还表明 seed2 已用于早期 descriptor screen，旧 expert 也看过 support；因此二者都不能称为项目级 blind holdout。真正未见验收必须另立 immutable manifest，使用此前未访问的 off-policy trajectory。候选可为预先冻结的 525 K DFT-MD 轨迹；其身份、seed、抽样规则和 `50/250 meV/Å` force gate 必须在首次计算标签前另行冻结。support 继续保留到 unseen trajectory 通过后的 routed-tail 阶段。

## 10. 状态、终态和停止规则

- `R2R1_LABEL_BLIND_PREFLIGHT_PASSED`：验证 R2R-0 dependency/provenance closure、attempt3 DONE recovery、fold/design 和 thermal 的冻结 lexical path；thermal 最终文件的 `stat/open/read` 均为零，未解析标签、未 fit。R2R-1 自身 source closure 与 thermal metadata binding 由另一步 label-blind manifest 生成器冻结，并在授权阶段重验，不属于该 preflight receipt。
- `R2R1_CONDITIONAL_OOF_FAILED`：数值有限但 pooled OOF gate 失败；这是 `DONE + EXIT_CODE=0` 的有限科学终态，停止且不做 final CV/fit/checkpoint。
- `R2R1_FINAL_READOUT_FAILED`：conditional OOF 已通过，但全 92 final-train gate 有至少一项失败；这是 `DONE + EXIT_CODE=0` 的有限科学终态，不发布 checkpoint。
- `R2R1_FINAL_READOUT_TRAIN_GATE_PASSED_MECHANICS_PENDING`：OOF、selection-only final CV 和全 92 train 通过；这是 `DONE + EXIT_CODE=0`，并发布 actual-`p` 三文件 checkpoint，但 mechanics 尚未执行。
- `R2R1_FINAL_READOUT_FROZEN`：仅保留给后续独立 mechanics 阶段；本轮 launcher 不得产生该状态。
- `R2R1_DEVELOPMENT_FAILED/PASSED`：仅保留给 mechanics 通过并冻结后的 development confirmation；本轮不得打开 development。
- 缺失或错误 manifest/GO 属于 transaction 之前的 authorization failure：fit root 必须保持 absent，label parser 和 fit SVD 调用数均为零。
- GO 已验证并建立 canonical root 后，若在任何 science artifact 写入前发生数值/I/O 异常，则可提交 `FAILED + EXIT_CODE=1`；若已有 artifact、出现未知/禁用 entry、RUNNING identity 漂移或 failure conversion 本身不安全，则保留 `RUNNING` incomplete，不删除、不自动 refit。
- `FAILED`、`RUNNING`、empty/partial 或多个 terminal 的已有 root 一律 fail closed；launcher 不运行 preflight、parser、SVD 或 materialize，进程返回非零。
- attempt1 的既有 `RUNNING + science artifacts` 永久按 forensic partial 分类；失败 attempt2 的空 control root 也永久保留。attempt2_retry1 代码不得把两者解释为可恢复状态或可复用路径。
- attempt2_retry1 fresh materialization 在同一进程内返回 sealed witness；它由 materializer lexical closure 的空状态 token 与 issuer-side identity registry 唯一签发，不能读取 seal/payload、不能复制、pickle、手工构造，也不能用 ledger 的公开字段重建。candidate 尚不存在时，anchor 必须先验证 exact token identity，再精确比较 witness 中包括 `DONE` owned identity 与 bytes/SHA 在内的整份 release snapshot。进程若在 `DONE` 后、anchor 前退出而丢失 witness，candidate-absent 分支拒绝 live-tree rebaseline，必须另立外部 pin/新授权；不能自动采纳未锚定 DONE。

任何有限科学 gate 失败都不得通过扩 alpha grid、改 group mass、放宽阈值、重分 folds 或使用 development/held 选择新 readout 来补救。

## 11. 数值出处和本轮新增冻结

- 四折 `5+9+9`、RMS-only/no-mean scaler、65 列、relative-zero `1e-12`、condition `1e8`、无 intercept 和 Taylor-null energy gauge：R2R canonical。
- E50 `30/200/A′15/slope5%`、small-H `0.5/10` 和 reference/mechanics 门：R2O/R2Q formal。
- `0.030 eV/Å`：R2O thermal force scale；group masses `0.50/0.25/0.25`：R2O Stage-1 group mass。
- T300/T600 `30/200`、alpha grid、nested outer4/inner3、round-12/larger-alpha tie-break 和 selection-only final CV：R2R-1 在首次 fit 前新增冻结。T300/T600 的 `30/200` 不是旧 formal acceptance 的直接沿用，必须明确记录为本轮新门。

## 12. 授权、发布和恢复事务

freeze manifest 在不读取 thermal payload 的条件下绑定 control root、唯一 shared result parent、fit/candidate 两个当时不存在的 child basename、thermal file metadata、attempt3 root metadata，以及 R2R-1 source/dependency 每个文件的 metadata identity 与 parent chain；fit 与 candidate 不再拥有可独立采样的 parent binding。manifest 生成器直接复用其已持有的 control-parent/result-parent fd，并在最终 parent-chain rebind 后再次从 held control fd 有界读取 canonical manifest bytes/SHA。全部 metadata-bound path 的共有 lexical prefix 必须映射到 type-exact 相同的目录 identity；混合 A/B 时相的自洽 JSON 也拒绝。外部 GO marker 对整份 canonical manifest bytes 签 SHA256。manifest 是 R2R1-owned regular file，始终使用下述 owned-eight identity 加 exact raw/SHA；GO 是 external regular file，使用包含 `ctime` 的 full-nine identity。manifest 与 marker 必须是同一 held control-root fd 下的两个唯一 sibling。正式授权在同一事务中持有 control/result/attempt3、manifest/marker 及全部 source/dependency fds，并把 control manifest/GO、release frozen snapshot、candidate absent/existing anchor 作为一个 authority set 做正向与反向 exact closure；最后一次 control closure 后还必须完整重验 release bytes/SHA 和 existing anchor raw，随后只做 stable binding、inventory 与 candidate absence 的短检查，才允许 thermal metadata/payload open。thermal open 后、首字节前以及 parser/ridge 边界仍重复联合闭合。正式授权还绑定 R2R-0 formal source `e1013323a9a9664e885a806362d0e4312fb0947b96a6ab3a44385d8a9f907769`、R2R-0 launcher provenance `32c6af5475954880b4bc9192763f948609421fded9cd9fb17f387fac55ccb358` 和 R2R primitive `4db3665f945783086de5c7a2f6262843bbe7d03df2a32f9153e450200b25064c`。

fit publication 使用 canonical-root reservation：授权前先持有 shared result-parent fd，并在同一 fd 上确认 fit 与 candidate basename 都 fresh。对 fresh-fit 分支，任何预存 candidate file/dir/symlink/partial 都必须在 preflight 以及 thermal 最终文件的 `stat/open/read`、label parser 和 fit SVD 之前拒绝。初次授权在 thermal 打开前后再次确认两 child absent；root reservation 后在 pipeline 前后及 terminal commit 边界继续要求 canonical fit basename 指向 held root 且 candidate absent。授权通过后在 held parent fd 下 `mkdir` 唯一 canonical root，立即建立并持有 `RUNNING` inode。所有 artifact 读写、checkpoint、inventory 和 readback 均相对 held root fd。materializer/recovery 用 closure-private opaque publication token 把 held parent/root/candidate policy 传入 linear pipeline；正式授权在 thermal metadata open 前后验证该 policy，label loader 在打开 payload fd 前和首字节读取前再次验证，每个 ridge SVD 紧前后也验证。若 publication change 发生在已经开始的 label-blind attempt3/design audit 中，既有 design-SVD 不追溯计为 label access，但 audit 结束后必须在 thermal metadata/payload、parser 和 fit-ridge SVD 之前拒绝；若 collision 在初始 authorization 前已存在，则 preflight/design SVD 也必须为零。

R2R1-owned regular file 的 authoritative record 固定为八个 type-exact integer 字段：`dev/ino/mode/nlink/uid/gid/size/mtime_ns`，并另行绑定 canonical bytes、size、SHA256 和 schema；必须是 single-link regular file。`ctime` 不属于 authority，因为 Desktop/APFS 会在 write/link/rename 数秒后由系统异步更新 provenance 与 `ctime`。Darwin 上 `com.apple.provenance` 和 `com.apple.decmpfs` 只作诊断，不读取其 value、不参与 identity；任何其他 xattr name 在 bounded one-shot `flistxattr` 检查中拒绝。非 Darwin 平台不把 xattr 纳入本合同。该口径允许同 inode、exact same bytes、恢复原 `mtime` 的历史 rewrite，因为当前可见状态与科学语义不变；不同 bytes 必须由 bounded raw/SHA closure 拒绝，inode/mode/nlink/owner/size/mtime 的变化仍由 owned identity 拒绝。目录不比较 `mtime/ctime`；root、checkpoint 和 candidate directory 只绑定 stable `dev/ino/mode/uid/gid`、held parent entry 与 exact recursive inventory。atomic helper 仍使用同目录 private temp inode、写入/fsync、hardlink final basename、unlink temp 和 parent fsync，但不再等待或采纳 postpublication `ctime/xattr` settlement；helper 返回的 owned identity直接进入 science/terminal/candidate snapshot，caller 不得从 live tree静默重置 baseline。

captured attempt3 的 legacy R2R-0 recovery 只在本进程创建、随机命名且 mode `0700` 的 opaque `TemporaryDirectory` 副本中运行。副本 regular files 用同一 owned-eight+raw/SHA primitive 物化，目录只比较 stable binding，并在 legacy recovery 前后核 exact recursive inventory、owned identities 与 bytes。冻结的旧 R2R-0 validator 内部仍可能因 managed `ctime` 产生一次瞬时 false failure；仅当失败后整份 private tree 的 owned-eight+raw/SHA snapshot 与调用前 baseline type-exact 相同，才允许一次纯读 retry，第二次失败直接传播；任一 content/inode/inventory 变化都禁止 retry。checkpoint frozen-loader 私有副本同样使用 held mode-`0700` directory、三个 owned files，并在 loader 返回后核 exact inventory、directory stable binding、owned identity 与 raw bytes。该内部工作区不暴露给外部调用者，合同排除同 UID 主动枚举随机 private path 后对它执行 rename-swap；这项收窄不适用于任何 production-known control/result/attempt3/thermal/source/candidate 路径。

success/failure terminal 不再是单行 marker，而是有界 canonical JSON persistent ledger。`completion.json` 内嵌一个排除自身、`EXIT_CODE` 和 terminal 的 precompletion science/checkpoint snapshot：每个 regular artifact 绑定 owned identity、size 与 SHA，checkpoint directory 绑定 stable identity和三个 child owned identity/raw SHA，并写入 canonical snapshot digest与 release-root stable binding。`DONE` ledger 绑定 completion 的 SHA/owned identity、`EXIT_CODE` 的 SHA/owned identity、root stable binding、snapshot digest和原 `RUNNING` inode 的 stable fields；`FAILED` 对 `failure.json/EXIT_CODE/root/RUNNING` 做同构绑定。最终 canonical ledger bytes 必须在 rewrite 前满足 4096-byte terminal 上限。长 precommit validator 与 parent-chain walk 返回后，由 terminal helper 本体对同一 frozen snapshot做完整 bounded bytes/SHA closure，再验 RUNNING、canonical root、parent 与 candidate，随后才用 no-replace rename 原子改名为 `DONE` 或 `FAILED`。一旦进入 no-replace 调用，任何异常都按 committed/uncertain 处理；此后不得清理、降级、删除或改写 release tree。正常返回的 rename 成功是确定 commit point；之后的 fsync、stat、full release return closure 或 witness 签发错误都只报告并保留 terminal。commit helper 直接返回内部已验证的 post-rename owned identity，materializer 不得在 helper 返回后重新采样。

成功提交前必须做两次 authorized scientific re-solve，并逐 receipt canonical JSON bytes、逐 array C-order little-endian FP64 raw bytes、checkpoint 三文件、recursive key/type schema、runtime environment 和 exact inventory 对比。第一次 re-solve 使用本次 atomic publication identities 组装的 in-memory prepublish snapshot；不得在 internal recovery 入口从 live tree 创建 baseline。完成文件写入后，`completion → DONE ledger` 把同一 science baseline 持久化；后续 readback、terminal commit 和 return 只能与它比较。提交 DONE 后、函数返回前再做一次无 SVD 的 ledger-authoritative 全内容验证，对 `+0.0/-0.0` 也要求 bit-exact。

GO、freeze manifest 与 publication policy 必须持续有效到最后一次 thermal hash/parse/ridge SVD 及 final scientific authorization recheck 完成。其后 terminal 与 anchor 阶段只原子持久化前述已验证 bytes，并以 sealed in-memory receipt、witness、persistent ledger 和 external anchor为authority；control marker不是异步取消信号。最后scientific recheck完成后撤销GO，不要求回滚已验证science、阻止terminal/candidate commit或改写已提交terminal。

身份schema变化后的持久/返回format均显式使用v2：freeze manifest、execution authorization、nested OOF aggregate、local fit release、fit receipt、completion、terminal ledger和external release manifest不得以同一个v1名称同时承载full-nine与owned-eight两种语义。

所有 completed recovery 必须 ledger-first：先有界读取 `DONE` canonical bytes，从 ledger 取得 completion/EXIT 的 expected owned identities与 hashes；解析 completion 后，按 persisted precompletion snapshot 在读取 science child 前逐项 gate owned identity，随后以 frozen raw/SHA 做正反两遍闭合，禁止先 capture live science tree 再把它当 expected。public recovery 必须更早从 external `release_manifest.json` 的 SHA/owned-identity pin 约束 DONE；existing candidate recovery 也必须 anchor-first。每次 inner recovery 完成 authorization、artifact 和 initial-snapshot 长读后，必须在 scientific pipeline 紧邻边界由 helper 本体再验证完整 release bytes/SHA 及 candidate exact-absent/exact-existing anchor raw；materializer 两次 readback 使用 absent policy，anchor/public recovery使用前面冻结的 policy。candidate-absent fresh anchor 只能使用本次 materializer closure 签发的 sealed witness，并对包括 DONE owned identity/raw SHA 的整个 expected snapshot作 type-exact比较；无 witness时拒绝，不能把live DONE重新采成可发布baseline。

external anchor 单独采用 sibling temporary candidate directory：在 held shared result-parent fd 中先完整写入唯一 `release_manifest.json`、fsync，并在所有长 release closure 后再次有界读取 anchor raw/SHA，再核 release root/parent、temporary directory basename、owned anchor identity和exact inventory，最后以 no-replace directory rename发布canonical candidate root。manifest除science/completion pins外，还绑定`expected_done_sha256`、DONE owned identity和precompletion snapshot digest；external anchor后，DONE 的owned identity与raw/SHA均必须精确匹配，`ctime`仍不属于R2R1-owned authority。进入no-replace调用后的任何异常按committed/uncertain处理，不回滚或清理held candidate；precommit确定失败时canonical candidate保持absent。postcommit错误保留已发布目录，重试只能从已有anchor bytes出发约束release并补做parent fsync。existing empty、partial、mixed或不同内容candidate均为incomplete/collision。对已有`DONE`的anchor/recovery分支，有效GO允许先执行thermal文件的zero-byte metadata rebind；但错误、symlink、hardlink、special或超限candidate必须在thermal label payload、scientific re-solve或ridge SVD前拒绝。

正式 fit arrays 的状态相关 schema 固定为：OOF fail 仅 `OOF_predicted_force_eV_A (92,72,3)`；另两个 final 状态恰好再含 `final_predicted_force_eV_A (92,72,3)`、`final_coefficients (65,)`、`final_scale_eV_A (65,)`。全部为 finite little-endian FP64；shape 每一维必须是 JSON integer，不能用等值 float/bool。fit receipt 记录 hostname、platform、Python、NumPy/SciPy/ASE、BLAS/LAPACK canonical capture/hash 和 thread environment；authorized recovery 要求运行环境 exact。

路径安全的威胁模型覆盖 symlink/hardlink、非合作 collision、父目录或 basename 在每个可观测 phase boundary 的普通换位、一次离散的同 inode 内容篡改，以及 bounded-read/ZIP 资源攻击。authority set以正向/反向raw pass和最后stable closure抵抗一次可观测phase tamper；不声称抵抗一个与最终trusted pass逐项同步、持续改写多个authority成员的同UID主动进程。所有external输入用逐组件`openat(O_NOFOLLOW)`和held-fd full identity rebind；owned输出同时以owned identity和raw/SHA闭合。R2R-1 source/dependency每文件上限2 MiB，attempt3 capture上限710 items、64 MiB total、16 MiB single file，所有读循环均按`limit+1`即时硬停。thermal92 canonical大小固定为`2,634,387` bytes，首次SHA、stream parser和final rehash各自要求累计byte count exact。freeze manifest上限8 MiB。fit NPZ在`ZipFile`/`np.load`构造前先有界解析classic EOCD/central directory，拒ZIP64/multidisk，要求exact member count/set、central directory≤64 KiB和offset/EOCD闭合；随后逐member解析`.npy`header，要求shape、little-endian FP64、C order及header+payload size exact，全部通过后才允许NumPy分配。合同也不覆盖单个`mkdir→stat/open`或内核rename syscall内部不可观测间隙，以及具备进程内存/ptrace修改能力的主体。

## 13. 仍需后续阶段决定的事项

- actual-`p` mechanics runner 的独立 manifest、执行位置和最终 receipt SHA；本轮 fit launcher 明确 `mechanics_execution_in_this_launcher=false`。
- mechanics 通过后 development E50 seed1/actual small-H 的独立开启 marker。
- 真正未见 off-policy trajectory 的最终身份、seed、抽样规则、DFT 标签预算和首次标签计算前的 immutable manifest；525 K 仅为候选，尚未授权。
- unseen 通过后是否进入 routed-tail/support 阶段；support 在此之前继续关闭。
