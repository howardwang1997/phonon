# Graphene K cusp 无 degauss pilot

**状态：**`passed_pilot`  
**电子积分：**`tetrahedra_opt (no degauss)`  
**k 网格：**`192×192×1`

## 数值

- K 点跨机器六模最大差：0.000000 cm⁻¹；
- K→Γ，d=0.007 的 A1′ 上升量：8.082531 cm⁻¹；
- K→M，等距离 d=0.007 的 A1′ 上升量：8.141554 cm⁻¹；
- 最小本征矢重叠：0.999849。

## 检查

- `no_smearing_or_degauss`: PASS
- `same_kgrid_and_QE_tag`: PASS
- `same_QE_binary_hashes`: PASS
- `same_pseudopotential_hash`: PASS
- `Hermiticity_le_5e-7`: PASS
- `matrix_replay_le_0p1_cm-1`: PASS
- `cross_host_K_le_0p2_cm-1`: PASS
- `minimum_mode_overlap_ge_0p95`: PASS
- `KG_neighbor_above_K`: PASS
- `KM_neighbor_above_K`: PASS

该阶段只判断四面体积分和 cusp 方向是否可用；k 网格收敛、形状拟合和 blind holdout 仍需后续阶段完成。
