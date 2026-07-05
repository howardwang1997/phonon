"""Family STEP-4: deploy the T_el-conditioned long-range term on a REAL distilled MACE
backbone, to compute a CDW material's smearing-dependent soft-mode(T_el) via MLIP.
  usage: deploy_family.py <material> <backbone_yaml(dg0.020)> <sharp_yaml(dg0.005)> <mace.model> [mesh]
B(T_el),kappa(T_el) laws read from results/smearing_kink/friedel_family.csv; DFT min-freq
also from that CSV. Reports MLIP-deployment min-freq(T_el) vs DFT."""
import sys, csv, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
import friedel_module as fm
from friedel_calc import FriedelCorrection, FriedelMACECalculator, fc2_from_calc
from phonon_accel.phonons import phonopy_to_ase
mat=sys.argv[1]; bgY=sys.argv[2]; shY=sys.argv[3]; MODEL=sys.argv[4]
mesh=int(sys.argv[5]) if len(sys.argv)>5 else 12
rows=[r for r in csv.DictReader(open(ROOT/"results/smearing_kink/friedel_family.csv")) if r["material"]==mat]
Tb=np.array([float(r["T_el"]) for r in rows]); Bb=np.array([float(r["B"]) for r in rows])
Kb=np.array([float(r["kappa"]) for r in rows]); MFd={float(r["dg"]):float(r["minf_dft_THz"]) for r in rows}
o=np.argsort(Tb); Tb,Bb,Kb=Tb[o],Bb[o],Kb[o]
Blaw=lambda T:float(np.interp(T,Tb,Bb)); Klaw=lambda T:float(np.interp(T,Tb,Kb))
def mace_calc(mp):
    from mace.calculators import MACECalculator
    import torch; return MACECalculator(model_paths=str(mp),device="cuda" if torch.cuda.is_available() else "cpu",default_dtype="float32")
def minf(ph,fc):
    ph.force_constants=fc; ph.run_mesh([mesh,mesh,1],with_eigenvectors=False,is_gamma_center=True)
    return float(np.min(ph.get_mesh_dict()["frequencies"]))
ph=fm.load_ph(bgY); fc_bg=ph.force_constants; fc_sh=fm.load_ph(shY).force_constants
corr=FriedelCorrection(ph,fc_bg,fc_sh,rmin=1.0,rmax=14.0,B_law=Blaw,kappa_law=Klaw)
ref=phonopy_to_ase(ph.supercell); base=mace_calc(MODEL)
_,fcm=fc2_from_calc(ph,base,subtract_ref=True)
print(f"# {mat}: MACE backbone alone min-freq={minf(ph,fcm):.3f} THz  (backbone dg0.020 DFT={MFd.get(0.020,'?')})")
print(f"# {'dg':>6} {'T_el':>6} {'MLIP+term':>10} {'DFT':>8} {'err':>6}")
errs=[]
for dg in sorted(MFd):
    if dg>0.021: continue
    T=dg*157888; calc=FriedelMACECalculator(base,ref,corr,T)
    _,fct=fc2_from_calc(ph,calc,subtract_ref=True); mf=minf(ph,fct); md=MFd[dg]
    if 0.005<dg<0.020: errs.append(abs(mf-md))
    print(f"# {dg:>6.3f} {T:>6.0f} {mf:>10.3f} {md:>8.3f} {mf-md:>+6.3f}")
print(f"# {mat} deployment min-freq MAE (held-out) = {np.mean(errs):.3f} THz")
