# E51：固定 smearing 的 300/450 K DFT 对比诊断

状态：`450K_off_policy_DFT_diagnostic_failed_as_thermodynamic_validation`

## 结果

- 300 K K 点：MLIP `1281.522`，DFT-TDEP+同一 full-EPC `1280.092 cm^-1`；近 K 绝对/形状 RMSE `1.731/0.378 cm^-1`。
- 450 K K 点：MLIP `1263.173`，DFT-TDEP+同一 full-EPC `1303.347 cm^-1`；近 K 绝对/形状 RMSE `39.714/0.588 cm^-1`。
- 450 K 原始 DFT-TDEP 从 `0.00285013035` 改到 `0.0019000869 Ry` 后，K 点只改变 `-1.135 cm^-1`。

## 结论

450 K 的约 40 cm^-1 差异主要来自采样分布和短程模型，不是 electronic smearing。该批构型来自此前已经 failed-C1 的 MLIP on-policy 轨迹，并非目标 DFT Hamiltonian 的平衡轨迹；三个 seed 的 K 点 spread 也较大。因此 450 K 曲线只能作为模型失配诊断，不能与 300 K DFT-MD 重标注结果组成受控的 DFT 晶格温度趋势。下一步需要调整短程模型/有限温自由能模型，并用少量 DFT 力做 on-policy overlap 检查；在 overlap 通过前不继续堆叠同分布标签。
