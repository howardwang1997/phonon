#!/usr/bin/env bash
# R2N: one loss-only correction after the support-free R2M core gate fails.
set -euo pipefail

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon-mlip}"
DATA="${DATA:-$ROOT/data/graphene_r2m_support_free_core}"
PREDECESSOR="${PREDECESSOR:-/data/graphene_r2m_core/support_free_r2_h16_l2_n24_seed83}"
OUT="${OUT:-/data/graphene_r2n_core/gate_normalized_r2_h16_l2_n24_seed83}"
NAME="gr_r2n_gate_normalized_r2_h16_l2_n24_seed83"
MODEL="$OUT/$NAME.model"
LOG="$OUT/run.log"
EVALUATOR_SOURCE="$ROOT/scripts/smearing_kink/evaluate_graphene_r2m_core_checkpoints.py"
EVALUATOR_SNAPSHOT="$OUT/code_snapshots/evaluate_graphene_r2m_core_checkpoints.py"
LAUNCHER_SOURCE="$ROOT/scripts/v100/run_graphene_r2n_gate_normalized_core.sh"
CONTRACT_VERSION="R2N_GATE_NORMALIZED_V1"

cd "$ROOT"

# A read-only local/remote smoke mode.  It never inspects predecessor results,
# creates an output directory, checks a GPU, or starts MACE.
if [[ "${R2N_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,inspect,json,pathlib,sys
import mace
import torch
from mace.data.utils import config_from_atoms
from mace.modules.loss import WeightedForcesLoss,mean_squared_error_forces,reduce_loss
data,evaluator,launcher=map(pathlib.Path,sys.argv[1:])
manifest_path=data/"manifest.json"
manifest=json.loads(manifest_path.read_text())
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected={
 "manifest":"bb20a86af073e344c38e39ab14d3a80af9e39fa051591bed2b039ae643ffe31e",
 "train.xyz":"789b65e1a2c3b260e24d80f008407c7ee55e02676ff3bcf9a875fe9de294519c",
 "valid.xyz":"92ad1508e1067265445b79679cfe5142d23b2507dab2aa44397578cee8c7a6da"}
assert digest(manifest_path)==expected["manifest"]
assert manifest["status"]=="frozen_before_R2M_support_free_core_training"
assert manifest["energy_training_enabled"] is False
assert manifest["counts"]["support_used"]==0
assert manifest["leakage_control"]["seed2_in_gradients_scales_or_checkpoint_selection"] is False
assert getattr(mace,"__version__","unknown")=="0.3.16"
semantic_hashes={
 "mean_squared_error_forces":"1a3b1d798f99e661e7a2384e863da32dd72b6f2bc4dbc5de643bc5136d9e1602",
 "reduce_loss":"3ed039b785f7ec4064a8687386bf6245ece2093c6be4d36085daf9cf7ca56aa9",
 "WeightedForcesLoss":"2712d2cfa03ce4918edb5f51a9991e84806e61aa2173b8c8616f9ae2ec5eeaad",
 "config_from_atoms":"d6a5e818c214af5779809094c1fee0f74451395ae9f87596fd92b2e5ce17adcb"}
objects={"mean_squared_error_forces":mean_squared_error_forces,"reduce_loss":reduce_loss,
 "WeightedForcesLoss":WeightedForcesLoss,"config_from_atoms":config_from_atoms}
assert {name:hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
        for name,obj in objects.items()}==semantic_hashes
for name in ("train.xyz","valid.xyz"):
 assert digest(data/name)==expected[name]
 assert manifest["outputs"][name]["sha256"]==expected[name]
assert manifest["outputs"]["reserved_e50_seed2.xyz"]["sha256"]=="ab8fb629501134ba9e94d11fcc67a6363971c2ccbf2bbae0e2d236a8c23988b1"
weights={"r2m_exact_e50_train":0.3272244428,
 "r2m_exact_e50_validation":0.3272244428,
 "r2m_harmonic_train":1.0,"r2m_harmonic_validation":1.0,
 "r2m_auxiliary_T300":0.04544783928,
 "r2m_auxiliary_T600":0.04544783928}
assert evaluator.is_file() and launcher.is_file()
print(json.dumps({"status":"R2N_preflight_passed","contract":"R2N_GATE_NORMALIZED_V1",
 "weights":weights,"mace_version":"0.3.16","semantic_hashes":semantic_hashes,
 "hashes":{"evaluator":digest(evaluator),"launcher":digest(launcher)}},indent=2))
' "$DATA" "$EVALUATOR_SOURCE" "$LAUNCHER_SOURCE"
    exit 0
fi

# Formal R2N is causally downstream of a completed, failed R2M gate.  Merely
# seeing a training checkpoint or a diagnostic trend is not sufficient.
for path in "$PREDECESSOR/DONE" "$PREDECESSOR/TRAINING_DONE" \
    "$PREDECESSOR/COMPLETED_AT" "$PREDECESSOR/EXIT_CODE" \
    "$PREDECESSOR/CORE_GATE_FAILED" \
    "$PREDECESSOR/core_checkpoint_gate.json" \
    "$PREDECESSOR/best_diagnostic_core.model"; do
    [[ -e "$path" ]] || {
        echo "R2N blocked: required failed-predecessor artifact absent: $path" >&2
        exit 4
    }
done
if [[ -e "$PREDECESSOR/FAILED" ]] || \
   [[ "$(tr -d '[:space:]' < "$PREDECESSOR/EXIT_CODE")" != "0" ]] || \
   [[ -e "$PREDECESSOR/CORE_GATE_PASSED" ]] || \
   [[ -e "$PREDECESSOR/selected_core.model" ]]; then
    echo "R2N blocked: predecessor completion/gate artifacts are inconsistent" >&2
    exit 4
fi
"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); gate=json.loads((root/"core_checkpoint_gate.json").read_text())
assert gate["status"]=="R2M_core_checkpoint_gate_failed"
assert gate["force_gate_allows_routed_tail_stage"] is False
assert gate["seed2_or_support_read_for_selection"] is False
artifact=gate["checkpoint_artifact"]
path=pathlib.Path(artifact["path"])
assert path.resolve()==(root/"best_diagnostic_core.model").resolve()
assert artifact["kind"]=="best_diagnostic_core_not_authorized_for_routed_tail"
assert artifact["sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
' "$PREDECESSOR"

mkdir -p "$OUT/checkpoints" "$OUT/logs" "$OUT/results" "$OUT/code_snapshots"
exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r2n-core}"

if [[ -e "$OUT/DONE" ]]; then
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,inspect,json,pathlib,sys
import mace
import torch
from mace.data.utils import config_from_atoms
from mace.modules.loss import WeightedForcesLoss,mean_squared_error_forces,reduce_loss
out,predecessor,launcher_source=map(pathlib.Path,sys.argv[1:4]); environment=sys.argv[4]
freeze=json.loads((out/"training_freeze.json").read_text())
gate=json.loads((out/"core_checkpoint_gate.json").read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (out/"FAILED").exists()
assert (out/"TRAINING_DONE").exists()
assert (out/"COMPLETED_AT").is_file()
assert (out/"EXIT_CODE").read_text().strip()=="0"
expected={"r2m_exact_e50_train":0.3272244428,
 "r2m_exact_e50_validation":0.3272244428,
 "r2m_harmonic_train":1.0,"r2m_harmonic_validation":1.0,
 "r2m_auxiliary_T300":0.04544783928,
 "r2m_auxiliary_T600":0.04544783928}
assert freeze["r2n_status"]=="frozen_before_R2N_gate_normalized_training"
assert freeze["experiment_stage"]=="R2N_loss_only_correction_after_failed_R2M_core_gate"
assert freeze["loss"]["contract_version"]=="R2N_GATE_NORMALIZED_V1"
assert freeze["loss"]["only_change_from_R2M"]=="config_type_weights"
assert freeze["loss"]["config_type_weights"]==expected
assert freeze["loss"]["dynamic_gradient_balancing"] is False
assert freeze["leakage"]["support_in_gradients"] is False
assert freeze["leakage"]["E50_seed2_read_by_training"] is False
assert freeze["leakage"]["E50_seed2_file_opened_by_launcher_or_training"] is False
assert freeze["leakage"]["support_file_opened_by_launcher_or_training"] is False
launcher_snapshot=out/"code_snapshots"/"run_graphene_r2n_gate_normalized_core.sh"
assert pathlib.Path(freeze["launcher"]["path"]).resolve()==launcher_snapshot.resolve()
assert freeze["launcher"]["sha256"]==digest(launcher_snapshot)==digest(launcher_source)
pre_gate=predecessor/"core_checkpoint_gate.json"
pre_model=predecessor/"best_diagnostic_core.model"
assert pathlib.Path(freeze["predecessor"]["gate_path"]).resolve()==pre_gate.resolve()
assert freeze["predecessor"]["required_status"]=="R2M_core_checkpoint_gate_failed"
assert freeze["predecessor"]["gate_sha256"]==digest(pre_gate)
assert freeze["predecessor"]["diagnostic_model_sha256"]==digest(pre_model)
assert freeze["predecessor"]["diagnostic_model_initialized_or_deployed"] is False
assert freeze["loss"]["mace_version"]=="0.3.16"==getattr(mace,"__version__","unknown")
assert torch.cuda.is_available() and torch.cuda.device_count()>=1
current_probe={"conda_environment":environment,
 "mace_version":"0.3.16","torch_version":str(torch.__version__),
 "torch_cuda_version":str(torch.version.cuda),"cuda_available":True,
 "cuda_device_count":int(torch.cuda.device_count()),
 "cuda_device0_name":str(torch.cuda.get_device_name(0)),
 "cuda_device0_capability":list(torch.cuda.get_device_capability(0))}
assert freeze["conda_environment"]==environment
assert freeze["environment_probe"]==current_probe
semantic_hashes={
 "mean_squared_error_forces":"1a3b1d798f99e661e7a2384e863da32dd72b6f2bc4dbc5de643bc5136d9e1602",
 "reduce_loss":"3ed039b785f7ec4064a8687386bf6245ece2093c6be4d36085daf9cf7ca56aa9",
 "WeightedForcesLoss":"2712d2cfa03ce4918edb5f51a9991e84806e61aa2173b8c8616f9ae2ec5eeaad",
 "config_from_atoms":"d6a5e818c214af5779809094c1fee0f74451395ae9f87596fd92b2e5ce17adcb"}
objects={"mean_squared_error_forces":mean_squared_error_forces,"reduce_loss":reduce_loss,
 "WeightedForcesLoss":WeightedForcesLoss,"config_from_atoms":config_from_atoms}
observed={name:hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
          for name,obj in objects.items()}
assert observed==semantic_hashes==freeze["loss"]["semantic_source_sha256"]
assert gate["inputs"]["training_freeze"]["sha256"]==digest(out/"training_freeze.json")
assert gate["seed2_or_support_read_for_selection"] is False
passed=gate["status"]=="R2M_core_checkpoint_gate_passed"
assert passed or gate["status"]=="R2M_core_checkpoint_gate_failed"
assert gate["force_gate_allows_routed_tail_stage"] is passed
expected_path=out/("selected_core.model" if passed else "best_diagnostic_core.model")
opposite_path=out/("best_diagnostic_core.model" if passed else "selected_core.model")
artifact=gate["checkpoint_artifact"]
assert pathlib.Path(artifact["path"]).resolve()==expected_path.resolve()
assert artifact["kind"]==(
 "selected_core_for_routed_tail_force_gate" if passed
 else "best_diagnostic_core_not_authorized_for_routed_tail")
assert expected_path.is_file() and not opposite_path.exists()
assert artifact["sha256"]==digest(expected_path)
assert (out/"CORE_GATE_PASSED").exists() is passed
assert (out/"CORE_GATE_FAILED").exists() is (not passed)
assert (out/"R2N_CORE_GATE_PASSED").exists() is passed
assert (out/"R2N_CORE_GATE_FAILED").exists() is (not passed)
print("R2N gate-normalized core already complete and contract-valid")
' "$OUT" "$PREDECESSOR" "$LAUNCHER_SOURCE" "$CONDA_ENV"
    exit 0
fi

RECOVER_AFTER_TRAINING=0
if [[ -e "$OUT/TRAINING_DONE" ]]; then
    for path in "$MODEL" "$OUT/training_runtime.json" "$OUT/training_freeze.json" \
        "$EVALUATOR_SNAPSHOT"; do
        [[ -s "$path" ]]
    done
    RECOVER_AFTER_TRAINING=1
elif compgen -G "$OUT/checkpoints/*_epoch-*.pt" >/dev/null || \
     [[ -e "$OUT/training_freeze.json" ]] || [[ -e "$OUT/RUNNING" ]]; then
    echo "refusing implicit R2N restart from an incomplete training state" >&2
    exit 2
fi

for path in "$DATA/manifest.json" "$DATA/train.xyz" "$DATA/valid.xyz" \
    "$EVALUATOR_SOURCE"; do
    [[ -s "$path" ]]
done

CONFIG_WEIGHTS="$($CONDA run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=json.loads((root/"manifest.json").read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(root/"manifest.json")=="bb20a86af073e344c38e39ab14d3a80af9e39fa051591bed2b039ae643ffe31e"
assert manifest["status"]=="frozen_before_R2M_support_free_core_training"
assert manifest["model_role"]=="replacement_short_delta_core"
assert manifest["energy_training_enabled"] is False
assert manifest["counts"]=={"train":164,"train_exact_e50":20,"train_harmonic":72,
 "train_auxiliary_T300":36,"train_auxiliary_T600":36,"valid":45,
 "valid_exact_e50_seed1":20,"valid_harmonic":25,
 "reserved_e50_seed2":20,"support_used":0}
assert manifest["leakage_control"]["support_labels_or_geometries_in_train_or_valid"] is False
assert manifest["leakage_control"]["seed2_in_gradients_scales_or_checkpoint_selection"] is False
for name,expected in {
 "train.xyz":"789b65e1a2c3b260e24d80f008407c7ee55e02676ff3bcf9a875fe9de294519c",
 "valid.xyz":"92ad1508e1067265445b79679cfe5142d23b2507dab2aa44397578cee8c7a6da"}.items():
 assert digest(root/name)==expected
 assert manifest["outputs"][name]["sha256"]==expected
assert manifest["outputs"]["reserved_e50_seed2.xyz"]["sha256"]=="ab8fb629501134ba9e94d11fcc67a6363971c2ccbf2bbae0e2d236a8c23988b1"
weights={"r2m_exact_e50_train":0.3272244428,
 "r2m_exact_e50_validation":0.3272244428,
 "r2m_harmonic_train":1.0,"r2m_harmonic_validation":1.0,
 "r2m_auxiliary_T300":0.04544783928,
 "r2m_auxiliary_T600":0.04544783928}
print(repr(weights))
' "$DATA" | tail -1)"

if (( RECOVER_AFTER_TRAINING == 0 )); then
    GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if (( GPU_USED_MIB > 1000 )); then
        echo "GPU is not idle: ${GPU_USED_MIB} MiB already used" >&2
        exit 3
    fi
    cp "$LAUNCHER_SOURCE" "$OUT/code_snapshots/run_graphene_r2n_gate_normalized_core.sh"
    cp "$DATA/prepare_script_snapshot.py" "$OUT/code_snapshots/prepare_script_snapshot.py"
    cp "$EVALUATOR_SOURCE" "$EVALUATOR_SNAPSHOT"
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,inspect,json,pathlib,sys
import mace
import torch
from mace.data.utils import config_from_atoms
from mace.modules.loss import WeightedForcesLoss,mean_squared_error_forces,reduce_loss
data,out,environment,launcher,evaluator,predecessor=map(pathlib.Path,sys.argv[1:7])
manifest=json.loads((data/"manifest.json").read_text())
pre_gate=predecessor/"core_checkpoint_gate.json"
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
loss_source=inspect.getsource(mean_squared_error_forces)
for required in ("torch.repeat_interleave","ref.weight","ref.forces_weight",
                 "torch.square","reduce_loss(raw_loss"):
 assert required in loss_source,required
assert getattr(mace,"__version__","unknown")=="0.3.16"
assert torch.cuda.is_available() and torch.cuda.device_count()>=1
semantic_hashes={
 "mean_squared_error_forces":"1a3b1d798f99e661e7a2384e863da32dd72b6f2bc4dbc5de643bc5136d9e1602",
 "reduce_loss":"3ed039b785f7ec4064a8687386bf6245ece2093c6be4d36085daf9cf7ca56aa9",
 "WeightedForcesLoss":"2712d2cfa03ce4918edb5f51a9991e84806e61aa2173b8c8616f9ae2ec5eeaad",
 "config_from_atoms":"d6a5e818c214af5779809094c1fee0f74451395ae9f87596fd92b2e5ce17adcb"}
objects={"mean_squared_error_forces":mean_squared_error_forces,"reduce_loss":reduce_loss,
 "WeightedForcesLoss":WeightedForcesLoss,"config_from_atoms":config_from_atoms}
assert {name:hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
        for name,obj in objects.items()}==semantic_hashes
weights={"r2m_exact_e50_train":0.3272244428,
 "r2m_exact_e50_validation":0.3272244428,
 "r2m_harmonic_train":1.0,"r2m_harmonic_validation":1.0,
 "r2m_auxiliary_T300":0.04544783928,
 "r2m_auxiliary_T600":0.04544783928}
environment_probe={"conda_environment":str(environment),
 "mace_version":"0.3.16","torch_version":str(torch.__version__),
 "torch_cuda_version":str(torch.version.cuda),"cuda_available":True,
 "cuda_device_count":int(torch.cuda.device_count()),
 "cuda_device0_name":str(torch.cuda.get_device_name(0)),
 "cuda_device0_capability":list(torch.cuda.get_device_capability(0))}
payload={
 "status":"frozen_before_R2M_core_training",
 "r2n_status":"frozen_before_R2N_gate_normalized_training",
 "experiment_stage":"R2N_loss_only_correction_after_failed_R2M_core_gate",
 "model_role":"replacement_short_delta_core",
 "combination":"frozen_v11_foundation + new_R2N_core + frozen_q6",
 "depth3_or_current_S0_initialized_or_deployed":False,
 "random_initialization":True,
 "architecture":{"model":"MACE","num_interactions":2,"r_max_A":2.0,
  "hidden_irreps":"16x0e+16x1o+16x2e","max_ell":2,
  "num_radial_basis":24,"num_cutoff_basis":5,"correlation":3,
  "interaction_diameter_bound_A":8.0},
 "loss":{"name":"forces_only","contract_version":"R2N_GATE_NORMALIZED_V1",
  "only_change_from_R2M":"config_type_weights",
  "formula":"raw_w_g=lambda_g/(N_g*L_g^2), normalized to harmonic=1",
  "group_mass":{"exact_E50":0.4,"harmonic":0.4,"auxiliary_T300":0.1,"auxiliary_T600":0.1},
  "gate_scale_eV_A":{"exact_E50":0.030,"harmonic":0.009044673057081592,
                    "auxiliary_T300":0.030,"auxiliary_T600":0.030},
  "config_type_weights":weights,"energy_weight":0.0,"forces_weight":100.0,
  "batch1_semantics":"one complete configuration per optimizer step; fixed weights normalize group MSE by the corresponding force-RMSE gate",
  "dynamic_gradient_balancing":False,
  "mace_version":"0.3.16","semantic_source_sha256":semantic_hashes,
  "mean_squared_error_forces_source_sha256":semantic_hashes["mean_squared_error_forces"]},
 "optimizer":{"lr":0.001,"weight_decay":1e-6,"ema":True,"ema_decay":0.99,
  "scheduler":"ExponentialLR","lr_scheduler_gamma":1.0,
  "validation_data_controls_lr":False},
 "training":{"seed":83,"batch_size":1,"max_epochs":240,"patience":300,
  "fixed_evaluation_epochs":[40,80,120,160,200,235],
  "final_is_also_evaluated":True,"validation_loss_cannot_stop_training":True},
 "leakage":{"support_in_gradients":False,"E50_seed1_in_gradients":False,
  "E50_seed2_read_by_training":False,
  "E50_seed2_file_opened_by_launcher_or_training":False,
  "support_file_opened_by_launcher_or_training":False,
  "atom_or_bond_random_split":False},
 "predecessor":{"required_status":"R2M_core_checkpoint_gate_failed",
  "gate_path":str(pre_gate),"gate_sha256":digest(pre_gate),
  "diagnostic_model_sha256":digest(predecessor/"best_diagnostic_core.model"),
  "diagnostic_model_initialized_or_deployed":False},
 "data_manifest":{"path":str(data/"manifest.json"),"sha256":digest(data/"manifest.json")},
 "launcher":{"path":str(launcher),"sha256":digest(launcher)},
 "evaluator":{"path":str(evaluator),"sha256":digest(evaluator),
  "compatibility_note":"unchanged R2M checkpoint evaluator; R2N launcher guards this exact freeze before and after evaluation"},
 "conda_environment":str(environment),"environment_probe":environment_probe}
(out/"training_freeze.json").write_text(json.dumps(payload,indent=2)+"\n")
' "$DATA" "$OUT" "$CONDA_ENV" \
        "$OUT/code_snapshots/run_graphene_r2n_gate_normalized_core.sh" \
        "$EVALUATOR_SNAPSHOT" "$PREDECESSOR"
fi

r2n_contract_guard() {
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,inspect,json,pathlib,sys
import mace
import torch
from mace.data.utils import config_from_atoms
from mace.modules.loss import WeightedForcesLoss,mean_squared_error_forces,reduce_loss
out,data,evaluator,launcher_source,predecessor=map(pathlib.Path,sys.argv[1:6]); environment=sys.argv[6]
freeze=json.loads((out/"training_freeze.json").read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={"r2m_exact_e50_train":0.3272244428,
 "r2m_exact_e50_validation":0.3272244428,
 "r2m_harmonic_train":1.0,"r2m_harmonic_validation":1.0,
 "r2m_auxiliary_T300":0.04544783928,
 "r2m_auxiliary_T600":0.04544783928}
assert freeze["status"]=="frozen_before_R2M_core_training"
assert freeze["r2n_status"]=="frozen_before_R2N_gate_normalized_training"
assert freeze["experiment_stage"]=="R2N_loss_only_correction_after_failed_R2M_core_gate"
assert freeze["loss"]["contract_version"]=="R2N_GATE_NORMALIZED_V1"
assert freeze["loss"]["only_change_from_R2M"]=="config_type_weights"
assert freeze["loss"]["config_type_weights"]==expected
assert freeze["loss"]["dynamic_gradient_balancing"] is False
assert freeze["training"]["fixed_evaluation_epochs"]==[40,80,120,160,200,235]
assert freeze["optimizer"]["scheduler"]=="ExponentialLR"
assert freeze["optimizer"]["lr_scheduler_gamma"]==1.0
assert freeze["optimizer"]["validation_data_controls_lr"] is False
assert freeze["leakage"]["support_in_gradients"] is False
assert freeze["leakage"]["E50_seed2_read_by_training"] is False
assert freeze["leakage"]["E50_seed2_file_opened_by_launcher_or_training"] is False
assert freeze["leakage"]["support_file_opened_by_launcher_or_training"] is False
assert freeze["data_manifest"]["sha256"]==digest(data/"manifest.json")
assert freeze["evaluator"]["sha256"]==digest(evaluator)
launcher_snapshot=out/"code_snapshots"/"run_graphene_r2n_gate_normalized_core.sh"
assert pathlib.Path(freeze["launcher"]["path"]).resolve()==launcher_snapshot.resolve()
assert freeze["launcher"]["sha256"]==digest(launcher_snapshot)==digest(launcher_source)
pre_gate=predecessor/"core_checkpoint_gate.json"
pre_model=predecessor/"best_diagnostic_core.model"
assert pathlib.Path(freeze["predecessor"]["gate_path"]).resolve()==pre_gate.resolve()
assert freeze["predecessor"]["required_status"]=="R2M_core_checkpoint_gate_failed"
assert freeze["predecessor"]["gate_sha256"]==digest(pre_gate)
assert freeze["predecessor"]["diagnostic_model_sha256"]==digest(pre_model)
assert freeze["predecessor"]["diagnostic_model_initialized_or_deployed"] is False
assert freeze["loss"]["mace_version"]=="0.3.16"==getattr(mace,"__version__","unknown")
assert torch.cuda.is_available() and torch.cuda.device_count()>=1
current_probe={"conda_environment":environment,
 "mace_version":"0.3.16","torch_version":str(torch.__version__),
 "torch_cuda_version":str(torch.version.cuda),"cuda_available":True,
 "cuda_device_count":int(torch.cuda.device_count()),
 "cuda_device0_name":str(torch.cuda.get_device_name(0)),
 "cuda_device0_capability":list(torch.cuda.get_device_capability(0))}
assert freeze["conda_environment"]==environment
assert freeze["environment_probe"]==current_probe
semantic_hashes={
 "mean_squared_error_forces":"1a3b1d798f99e661e7a2384e863da32dd72b6f2bc4dbc5de643bc5136d9e1602",
 "reduce_loss":"3ed039b785f7ec4064a8687386bf6245ece2093c6be4d36085daf9cf7ca56aa9",
 "WeightedForcesLoss":"2712d2cfa03ce4918edb5f51a9991e84806e61aa2173b8c8616f9ae2ec5eeaad",
 "config_from_atoms":"d6a5e818c214af5779809094c1fee0f74451395ae9f87596fd92b2e5ce17adcb"}
objects={"mean_squared_error_forces":mean_squared_error_forces,"reduce_loss":reduce_loss,
 "WeightedForcesLoss":WeightedForcesLoss,"config_from_atoms":config_from_atoms}
observed={name:hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
          for name,obj in objects.items()}
assert observed==semantic_hashes==freeze["loss"]["semantic_source_sha256"]
' "$OUT" "$DATA" "$EVALUATOR_SNAPSHOT" "$LAUNCHER_SOURCE" "$PREDECESSOR" "$CONDA_ENV"
}
r2n_contract_guard

if (( RECOVER_AFTER_TRAINING == 0 )); then
    date -Is > "$OUT/STARTED_AT"
else
    date -Is > "$OUT/EVALUATION_RECOVERY_STARTED_AT"
fi
touch "$OUT/RUNNING"
echo "timestamp_iso,gpu_util_percent,memory_used_MiB,memory_total_MiB" > "$OUT/gpu_samples.csv"
(
    while [[ -e "$OUT/RUNNING" ]]; do
        GPU_SAMPLE="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)"
        echo "$(date -Is),$GPU_SAMPLE" >> "$OUT/gpu_samples.csv"
        sleep 10
    done
) &
MONITOR_PID=$!

finish() {
    EXIT_STATUS=$?
    trap - EXIT
    rm -f "$OUT/RUNNING"
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
    echo "$EXIT_STATUS" > "$OUT/EXIT_CODE"
    if (( EXIT_STATUS != 0 )); then touch "$OUT/FAILED"; fi
    exit "$EXIT_STATUS"
}
trap finish EXIT

if (( RECOVER_AFTER_TRAINING == 0 )); then
    START_EPOCH="$(date +%s)"
    echo "=== R2N gate-normalized loss-only core TRAIN START $(date -Is) ==="
    echo "contract=$CONTRACT_VERSION config weights: $CONFIG_WEIGHTS"
    "$CONDA" run --no-capture-output -n "$CONDA_ENV" python -m mace.cli.run_train \
        --name "$NAME" --model MACE --num_interactions 2 \
        --hidden_irreps '16x0e+16x1o+16x2e' --r_max 2.0 \
        --num_radial_basis 24 --num_cutoff_basis 5 --max_ell 2 --correlation 3 \
        --train_file "$DATA/train.xyz" --valid_file "$DATA/valid.xyz" \
        --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
        --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
        --config_type_weights "$CONFIG_WEIGHTS" \
        --batch_size 1 --valid_batch_size 1 --max_num_epochs 240 \
        --patience 300 --eval_interval 5 --lr 0.001 --weight_decay 1.0e-6 \
        --scheduler ExponentialLR --lr_scheduler_gamma 1.0 \
        --ema --ema_decay 0.99 --default_dtype float32 --device cuda --seed 83 \
        --save_cpu --keep_checkpoints --save_all_checkpoints \
        --model_dir "$OUT" --checkpoints_dir "$OUT/checkpoints" \
        --log_dir "$OUT/logs" --results_dir "$OUT/results"
    [[ -s "$MODEL" ]]
    END_EPOCH="$(date +%s)"
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
start,end,model,out=sys.argv[1:]; model=pathlib.Path(model); out=pathlib.Path(out)
(out/"training_runtime.json").write_text(json.dumps({
 "status":"training_complete_pending_fixed_checkpoint_gate",
 "experiment_stage":"R2N_gate_normalized_loss_only_correction",
 "loss_contract":"R2N_GATE_NORMALIZED_V1",
 "wall_time_seconds":int(end)-int(start),"final_model":str(model),
 "final_model_sha256":hashlib.sha256(model.read_bytes()).hexdigest()},indent=2)+"\n")
' "$START_EPOCH" "$END_EPOCH" "$MODEL" "$OUT"
    date -Is > "$OUT/TRAINING_COMPLETED_AT"
    touch "$OUT/TRAINING_DONE"
else
    echo "=== R2N training complete; recovering fixed checkpoint gate $(date -Is) ==="
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
model,runtime_path=map(pathlib.Path,sys.argv[1:]); runtime=json.loads(runtime_path.read_text())
assert runtime["status"]=="training_complete_pending_fixed_checkpoint_gate"
assert runtime["experiment_stage"]=="R2N_gate_normalized_loss_only_correction"
assert runtime["loss_contract"]=="R2N_GATE_NORMALIZED_V1"
assert pathlib.Path(runtime["final_model"]).resolve()==model.resolve()
assert runtime["final_model_sha256"]==hashlib.sha256(model.read_bytes()).hexdigest()
' "$MODEL" "$OUT/training_runtime.json"
fi
r2n_contract_guard

"$CONDA" run --no-capture-output -n "$CONDA_ENV" python \
    "$EVALUATOR_SNAPSHOT" \
    --data "$DATA" --training-dir "$OUT" --template-model "$MODEL" \
    --device cuda --output "$OUT/core_checkpoint_gate.json" \
    --selected-model "$OUT/selected_core.model"

GATE_STATUS="$($CONDA run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
out=pathlib.Path(sys.argv[1]); gate=json.loads((out/"core_checkpoint_gate.json").read_text())
freeze=out/"training_freeze.json"; expected=hashlib.sha256(freeze.read_bytes()).hexdigest()
assert gate["inputs"]["training_freeze"]["sha256"]==expected
assert gate["seed2_or_support_read_for_selection"] is False
assert gate["status"] in {"R2M_core_checkpoint_gate_passed","R2M_core_checkpoint_gate_failed"}
print(gate["status"])
' "$OUT" | tail -1)"
if [[ "$GATE_STATUS" == "R2M_core_checkpoint_gate_passed" ]]; then
    rm -f "$OUT/CORE_GATE_FAILED" "$OUT/R2N_CORE_GATE_FAILED"
    touch "$OUT/CORE_GATE_PASSED" "$OUT/R2N_CORE_GATE_PASSED"
else
    rm -f "$OUT/CORE_GATE_PASSED" "$OUT/R2N_CORE_GATE_PASSED"
    touch "$OUT/CORE_GATE_FAILED" "$OUT/R2N_CORE_GATE_FAILED"
fi
date -Is > "$OUT/COMPLETED_AT"
rm -f "$OUT/FAILED"
touch "$OUT/DONE"
echo "=== R2N COMPLETE gate=$GATE_STATUS contract=$CONTRACT_VERSION $(date -Is) ==="
