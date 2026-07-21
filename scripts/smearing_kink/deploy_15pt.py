"""Deploy MLIP+healing Friedel on ALL 15 fd smearing points. Saves bands + K-iTO + all branches."""
import sys, warnings, json, time
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

ROOT = Path("/Users/howardwang/Desktop/playground/phonon")
FD = ROOT / "results" / "graphene_kohn_fd"
sys.path.insert(0, str(ROOT / "scripts" / "smearing_kink"))
sys.path.insert(0, str(ROOT / "src"))
import friedel_module as fm
from friedel_calc import FriedelCorrection, FriedelMACECalculator, fc2_from_calc
import phonopy

LOGF = open(FD / "deploy_15pt.log", "a")
def log(m): print(m, flush=True); LOGF.write(m+"\n"); LOGF.flush()

LABELS = ["Γ", "M", "K", "Γ"]
PTS = [np.array([0.,0,0]), np.array([.5,0,0]), np.array([1/3,1/3,0]), np.array([0.,0,0])]
CM = 33.35641

all_dgs = sorted([float(f.stem.split("_dg")[1].split("_")[0]) for f in FD.glob("graphene_sc6_dg*_phonopy.yaml")])
log(f"# deploy 15pt: {len(all_dgs)} smearings: {all_dgs}")

hl = json.loads((FD / "healing_law.json").read_text())
BB_MODEL = ROOT / "results" / "gr_backbone_fd" / "graphene_backbone_fd.model"
from mace.calculators import MACECalculator
import torch
dev = "cpu"
mace = MACECalculator(model_paths=str(BB_MODEL), device="cpu", default_dtype="float64")
log(f"# backbone {BB_MODEL.name}, dev={dev}")

ph_ref = fm.load_ph(str(FD/"graphene_sc6_dg0.08_phonopy.yaml"))
from phonon_accel.phonons import phonopy_to_ase
ref_atoms = phonopy_to_ase(ph_ref.supercell)
fc0_08 = fm.load_ph(str(FD/"graphene_sc6_dg0.08_phonopy.yaml")).force_constants
fc0_01 = fm.load_ph(str(FD/"graphene_sc6_dg0.01_phonopy.yaml")).force_constants
_, fc_bb = fc2_from_calc(ph_ref, mace, distance=0.03, subtract_ref=True)
corr = FriedelCorrection(ph_ref, fc0_08, fc0_01,
    B_law=lambda T: hl["B0"]*max(0.0,1.0-(T/hl["Tstar"])**hl["p"])**hl["q"],
    kappa_law=lambda T: hl["kappa"])

R = {"dgs":[], "dft_kK":[], "mlip_kK":[], "dft_KiTO":[], "mlip_KiTO":[],
     "dft_G2g":[], "mlip_G2g":[], "dft_Kbr":[], "mlip_Kbr":[],
     "dft_bands":[], "mlip_bands":[], "x":None, "tick":None}
qs = []
for i in range(len(PTS)-1):
    for j in range(1,41): qs.append(PTS[i]+(PTS[i+1]-PTS[i])*j/40)
qs = np.array(qs)

t0=time.perf_counter()
for dg in all_dgs:
    dg_str=f"{dg}"; yml=FD/f"graphene_sc6_dg{dg_str}_phonopy.yaml"
    if not yml.exists():
        for alt in [f"{dg:.3f}"]:
            yml=FD/f"graphene_sc6_dg{alt}_phonopy.yaml"
            if yml.exists(): break
    if not yml.exists(): log(f"  skip dg{dg}"); continue
    T=dg*157887
    # DFT
    ph_d=fm.load_ph(str(yml)); fc_d=ph_d.force_constants
    kK_d=fm.kink_of(ph_d,fc_d)[0]
    ph2=phonopy.load(str(yml),is_compact_fc=False)
    ph2.run_qpoints(qs,with_dynamical_matrices=False)
    fb_d=np.array(ph2.get_qpoints_dict()["frequencies"])*CM
    ph2.run_qpoints([np.array([1/3,1/3,0.])],with_dynamical_matrices=False)
    Kbr_d=np.sort(np.array(ph2.get_qpoints_dict()["frequencies"])[0])*CM
    KiTO_d=float(Kbr_d[-1])
    ph2.run_qpoints([np.array([0.,0.,0.])],with_dynamical_matrices=False)
    G2g_d=float(np.array(ph2.get_qpoints_dict()["frequencies"])[0].max()*CM)
    # MLIP
    if dg==0.01:
        kK_m=kK_d; fb_m=fb_d; Kbr_m=Kbr_d; KiTO_m=KiTO_d; G2g_m=G2g_d
    else:
        calc=FriedelMACECalculator(mace,ref_atoms,corr,T)
        _,fc_m=fc2_from_calc(ph_ref,calc,distance=0.03,subtract_ref=True)
        kK_m=fm.kink_of(ph_ref,fc_m)[0]
        ph3=phonopy.load(str(FD/"graphene_sc6_dg0.08_phonopy.yaml"),is_compact_fc=False)
        ph3.force_constants=fc_m
        ph3.run_qpoints(qs,with_dynamical_matrices=False)
        fb_m=np.array(ph3.get_qpoints_dict()["frequencies"])*CM
        ph3.run_qpoints([np.array([1/3,1/3,0.])],with_dynamical_matrices=False)
        Kbr_m=np.sort(np.array(ph3.get_qpoints_dict()["frequencies"])[0])*CM
        KiTO_m=float(Kbr_m[-1])
        ph3.run_qpoints([np.array([0.,0.,0.])],with_dynamical_matrices=False)
        G2g_m=float(np.array(ph3.get_qpoints_dict()["frequencies"])[0].max()*CM)
    R["dgs"].append(dg_str)
    R["dft_kK"].append(float(kK_d));R["mlip_kK"].append(float(kK_m))
    R["dft_KiTO"].append(KiTO_d);R["mlip_KiTO"].append(KiTO_m)
    R["dft_G2g"].append(G2g_d);R["mlip_G2g"].append(G2g_m)
    R["dft_Kbr"].append(Kbr_d.tolist());R["mlip_Kbr"].append(Kbr_m.tolist())
    R["dft_bands"].append(fb_d.tolist());R["mlip_bands"].append(fb_m.tolist())
    if R["x"] is None:
        sl=np.linalg.norm(np.diff(qs,axis=0),axis=1)
        x=np.concatenate([[0.],np.cumsum(sl)]);R["x"]=x.tolist()
        sg=len(x)//3;R["tick"]=[float(x[0]),float(x[sg-1]),float(x[2*sg-1]),float(x[-1])]
    log(f"  dg{dg} ({time.perf_counter()-t0:.0f}s) DFT kK={kK_d:.2f} KiTO={KiTO_d:.0f} | MLIP kK={kK_m:.2f} KiTO={KiTO_m:.0f}")

np.savez(FD/"deploy_15pt.npz",
    dgs=np.array(R["dgs"]),dft_kink_K=np.array(R["dft_kK"]),mlip_kink_K=np.array(R["mlip_kK"]),
    dft_K_iTO=np.array(R["dft_KiTO"]),mlip_K_iTO=np.array(R["mlip_KiTO"]),
    dft_G_E2g=np.array(R["dft_G2g"]),mlip_G_E2g=np.array(R["mlip_G2g"]),
    dft_K_branches=np.array(R["dft_Kbr"]),mlip_K_branches=np.array(R["mlip_Kbr"]),
    dft_bands=np.array(R["dft_bands"]),mlip_bands=np.array(R["mlip_bands"]),
    x=np.array(R["x"]),tick=np.array(R["tick"]))
log(f"# DONE. kink_K MAE={np.mean(np.abs(np.array(R['mlip_kK'])-np.array(R['dft_kK']))):.2f} KiTO MAE={np.mean(np.abs(np.array(R['mlip_KiTO'])-np.array(R['dft_KiTO']))):.1f}")
