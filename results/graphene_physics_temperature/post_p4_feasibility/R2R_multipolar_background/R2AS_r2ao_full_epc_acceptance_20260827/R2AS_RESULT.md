# Graphene 有限晶格温度 full-EPC 验收

状态：`R2AS_KOHN_CUSP_PASSED_MODEL_SENSITIVITY_FAILED`

电子展宽固定为 `smearing/degauss = 0.0019000869 Ry`；晶格温度为 300/450/600 K。

R2AO 的一般力门槛仍未通过；本结果只用于声谱敏感性与 Kohn anomaly 定量验收。

## 核心门槛

- 矩阵重构：通过
- 静态 full-EPC 对直接 DFPT 的频率/形状/depth/width：通过
- R2AO 对独立 S0 的有限温模型敏感性：未通过

详细数值和逐项门槛见 `r2ao_full_epc_acceptance_summary.json`。
