"""STEP 4: deploy the long-range term on the REAL gr_backbone MACE, at the CONVERGED
8x8 supercell. gr_backbone (short-range, smearing-blind) + T_el-conditioned Friedel
term -> reproduce the converged 8x8 kink(T_el). Laws from step3 (8x8 fit)."""
import sys, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
import friedel_module as fm
from friedel_calc import FriedelCorrection, FriedelMACECalculator, fc2_from_calc
from phonon_accel.phonons import phonopy_to_ase
SC=str(ROOT/"results/sc_conv/graphene_sc8_dg{}_phonopy.yaml")
BB=ROOT/"results"/"gr_backbone_v11"/"ft_graphene.model"  # v11 FC-distilled backbone (kink~0.24); finetune_mace/ft_phonon.model is the foundation-finetune ABLATION (kink~85)
def mace_calc(mp):
    from mace.calculators import MACECalculator
    import torch; dev="cuda" if torch.cuda.is_available() else "cpu"
    return MACECalculator(model_paths=str(mp),device=dev,default_dtype="float32")
Tp=[789,1579,3158,6316,12631]; Bp=[1.0,1.0,0.982,0.874,0.0]; Kp=[0,0,0,0.05,0]
Blaw=lambda T:float(np.interp(T,Tp,Bp)); Klaw=lambda T:float(np.interp(T,Tp,Kp))
ph=fm.load_ph(SC.format("0.08")); fc_bg=ph.force_constants
fc_ref=fm.load_ph(SC.format("0.005")).force_constants
corr=FriedelCorrection(ph,fc_bg,fc_ref,rmin=1.0,rmax=9.8,B_law=Blaw,kappa_law=Klaw)
ref_atoms=phonopy_to_ase(ph.supercell); base=mace_calc(BB)
_,fcm=fc2_from_calc(ph,base,subtract_ref=True); kb=fm.kink_of(ph,fcm)[0]
print(f"# gr_backbone MACE alone (8x8) -> kink_K={kb:.2f}  (backbone DFT dg0.08 = 0.10)")
DFT={789:6.12,1579:6.35,3158:6.45,6316:3.40,12631:0.10}
print(f"# {'T_el':>6} {'MACE+Friedel':>12} {'8x8_DFT':>8} {'err':>6}")
errs=[]; save_T={789,1579,3158,12631}  # 1579=dg0.01 added so the MLIP@smearing-0.01 spectrum is saved
for T,kd in DFT.items():
    calc=FriedelMACECalculator(base,ref_atoms,corr,T)
    _,fct=fc2_from_calc(ph,calc,subtract_ref=True); kK=fm.kink_of(ph,fct)[0]
    errs.append(abs(kK-kd)); print(f"# {T:>6} {kK:>12.2f} {kd:>8.2f} {kK-kd:>+6.2f}")
    if T in save_T:
        np.savez(str(ROOT/f"results/smearing_kink/deploy_fc2_T{T}.npz"), fc2=fct)
print(f"# deployment kink MAE = {np.mean(errs):.2f} cm-1  (real MLIP backbone + long-range term vs converged DFT)")
# grid-independence: the deployed calculator is a force field -> vary displacement amplitude
print("# grid/robustness: deployed kink(T_el=789) vs finite-displacement amplitude")
for dist in (0.005,0.01,0.02):
    calc=FriedelMACECalculator(base,ref_atoms,corr,789)
    _,fcd=fc2_from_calc(ph,calc,distance=dist,subtract_ref=True)
    print(f"#   disp={dist:.3f} A -> kink_K={fm.kink_of(ph,fcd)[0]:.2f}")
