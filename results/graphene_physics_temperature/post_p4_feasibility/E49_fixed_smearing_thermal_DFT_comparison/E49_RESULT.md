# Graphene 300 K：MLIP+长程项与 DFT-TDEP 对比

状态：`provisional_300K_matched_smearing_DFT_comparison_450K_pending`

电子设置统一为 Fermi–Dirac `smearing/degauss = 0.0019000869 Ry`，晶格温度为 300 K。DFT-TDEP 使用累计 15、30、45 个 72 原子构型；主曲线采用 45 标签结果。

## 数值结果

- MLIP+full-EPC 的 K 点频率：`1281.522 cm^-1`。
- 45 标签 DFT-TDEP+同一 full-EPC 的 K 点频率：`1280.092 cm^-1`。
- K 点差值（MLIP−DFT）：`+1.431 cm^-1`。
- `|q-K|/(2π/a) ≤ 0.025` 内绝对 RMSE：`1.731 cm^-1`。
- 同一区间 K-referenced shape RMSE：`0.378 cm^-1`。
- 原始 DFT-TDEP K 点在 15/30/45 标签时分别为 `1284.733`、`1296.396`、`1280.670 cm^-1`。

## 适用范围

DFT 混合参考与 MLIP 曲线共享同一个 dense full-EPC 长程响应，因此这张图主要检验有限晶格温度的短程背景和绝对频率，不能作为 full-EPC cusp 形状的独立 DFT 验证。现有 300 K 构型来自旧 cold-smearing DFT-MD 轨迹，再按目标 FD 设置重算力，属于 off-policy 重标注；wave2 到 wave3 仍有可见变化。450 K 同 smearing 重标注完成后，才能形成两温度的完整受控对比。
