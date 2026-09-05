#!/usr/bin/env bash
# R2O: FP64 whole-energy Cartesian Taylor-2 null core.
# Usage: bash run_graphene_r2o_taylor_null.sh smoke|formal
set -euo pipefail

MODE="${1:-${R2O_MODE:-}}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
    echo "usage: $0 smoke|formal" >&2
    exit 2
fi

ROOT="${ROOT:-/root/phonon}"
CONDA="${CONDA:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-phonon-mlip}"
DATA="${DATA:-$ROOT/data/graphene_r2o_taylor_null_core}"
FEASIBILITY="${FEASIBILITY:-$ROOT/data/graphene_r2o_taylor_null_feasibility}"
R2N_PREDECESSOR="${R2N_PREDECESSOR:-/data/graphene_r2n_core/gate_normalized_r2_h16_l2_n24_seed83}"
SMOKE_OUT="${SMOKE_OUT:-/data/graphene_r2o_core/fp64_smoke_2plus2_seed83}"
if [[ "$MODE" == "smoke" ]]; then
    OUT="${OUT:-$SMOKE_OUT}"
    PRETRAIN_EPOCHS=2
    STAGE2_EPOCHS=2
    STAGE2_CHECKPOINT_EPOCHS="1,2"
    NAME="gr_r2o_fp64_smoke_seed83"
else
    OUT="${OUT:-/data/graphene_r2o_core/formal_r3p2_h16_l2_n24_seed83}"
    PRETRAIN_EPOCHS=80
    STAGE2_EPOCHS=240
    STAGE2_CHECKPOINT_EPOCHS="40,80,120,160,200,235,240"
    NAME="gr_r2o_formal_r3p2_h16_l2_n24_seed83"
fi

EXPECTED_DATA_MANIFEST_SHA256="4d2b60f39537eccd3389bcc568c95d0bb61b7a1564a1c2f8e45786031eb92a0f"
EXPECTED_FEASIBILITY_SHA256="6528eda75fd8efd58134fa1451d14d8fb7959b98834ea3f410d3cc3df2523d07"
EXPECTED_FEASIBILITY_MARKER_SHA256="333ab24198785a219bd31a7fa085a3d85f426f07c00bfc531ea5e370864b89a6"
EXPECTED_R2N_GATE_SHA256="b2e00ba1efff0dee62167a76d98dac87798580c7e15c49ec47bf12ffab23487e"
EXPECTED_R2N_DIAGNOSTIC_MODEL_SHA256="d20b8281e18db3a196a4dcf2f5f2c185f02224d4f055d17d58655da63c1685e5"
EXPECTED_SMOKE_GATE_SHA256="${R2O_EXPECTED_SMOKE_GATE_SHA256:-}"

RUNNER_SOURCE="$ROOT/scripts/v100/run_graphene_r2o_taylor_null.sh"
WRAPPER_SOURCE="$ROOT/scripts/smearing_kink/graphene_r2o_taylor_null.py"
TRAINER_SOURCE="$ROOT/scripts/smearing_kink/train_graphene_r2o_taylor_null.py"
EVALUATOR_SOURCE="$ROOT/scripts/smearing_kink/evaluate_graphene_r2o_taylor_null.py"
SNAPSHOT_DIR="$OUT/code_snapshots"
RUNNER_SNAPSHOT="$SNAPSHOT_DIR/run_graphene_r2o_taylor_null.sh"
WRAPPER_SNAPSHOT="$SNAPSHOT_DIR/graphene_r2o_taylor_null.py"
TRAINER_SNAPSHOT="$SNAPSHOT_DIR/train_graphene_r2o_taylor_null.py"
EVALUATOR_SNAPSHOT="$SNAPSHOT_DIR/evaluate_graphene_r2o_taylor_null.py"
FREEZE="$OUT/training_freeze.json"
PRETRAIN_DIR="$OUT/pretrain"
STAGE2_DIR="$OUT/stage2"
PRETRAIN_MODEL="$PRETRAIN_DIR/$NAME.model"
LOG="$OUT/run.log"

# Recovery is governed by the already-created immutable snapshot, even if the
# working tree changes between invocations.
if [[ -f "$RUNNER_SNAPSHOT" ]]; then
    current_script="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"
    snapshot_script="$(cd "$(dirname "$RUNNER_SNAPSHOT")" && pwd -P)/$(basename "$RUNNER_SNAPSHOT")"
    if [[ "$current_script" != "$snapshot_script" ]]; then
        exec bash "$RUNNER_SNAPSHOT" "$MODE"
    else
        RUNNER_SOURCE="$RUNNER_SNAPSHOT"
        WRAPPER_SOURCE="$WRAPPER_SNAPSHOT"
        TRAINER_SOURCE="$TRAINER_SNAPSHOT"
        EVALUATOR_SOURCE="$EVALUATOR_SNAPSHOT"
    fi
fi

digest() {
    shasum -a 256 "$1" | awk '{print $1}'
}

for path in "$DATA" "$FEASIBILITY" "$OUT"; do
    lowered="$(printf '%s' "$path" | tr '[:upper:]' '[:lower:]')"
    if [[ "$lowered" == *seed2* || "$lowered" == *reserved* || "$lowered" == *support* ]]; then
        echo "R2O rejects forbidden data/output path: $path" >&2
        exit 3
    fi
done

for path in "$RUNNER_SOURCE" "$WRAPPER_SOURCE" "$TRAINER_SOURCE" \
    "$EVALUATOR_SOURCE" "$DATA/manifest.json" \
    "$FEASIBILITY/feasibility.json" "$FEASIBILITY/GO_FOR_TWO_EPOCH_SMOKE"; do
    [[ -f "$path" ]] || { echo "R2O required input absent: $path" >&2; exit 3; }
done
[[ "$(digest "$DATA/manifest.json")" == "$EXPECTED_DATA_MANIFEST_SHA256" ]] || {
    echo "R2O frozen data manifest hash mismatch" >&2
    exit 3
}
[[ "$(digest "$FEASIBILITY/feasibility.json")" == "$EXPECTED_FEASIBILITY_SHA256" ]] || {
    echo "R2O feasibility JSON hash mismatch" >&2
    exit 3
}
[[ "$(digest "$FEASIBILITY/GO_FOR_TWO_EPOCH_SMOKE")" == "$EXPECTED_FEASIBILITY_MARKER_SHA256" ]] || {
    echo "R2O feasibility marker hash mismatch" >&2
    exit 3
}

"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
data,feasibility,expected_manifest,expected_feasibility=map(str,sys.argv[1:5])
data=pathlib.Path(data); feasibility=pathlib.Path(feasibility)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((data/"manifest.json").read_text())
assert digest(data/"manifest.json")==expected_manifest
assert manifest["status"]=="frozen_before_R2O_training"
assert manifest["counts"]["gradient_train_thermal_only"]==92
assert manifest["counts"]["seed2_opened"]==0
assert manifest["counts"]["support_opened"]==0
assert manifest["training_data_contract"]["stage2_group_mass"]=={
 "E50_seed0":0.5,"T300":0.2,"T600":0.2,"harmonic_lambda1_small_zero":0.1}
assert manifest["architecture_freeze"]["r_max_A"]==3.2
assert manifest["architecture_freeze"]["default_dtype"]=="float64"
assert manifest["architecture_freeze"]["pair_repulsion"] is False
for name,record in manifest["outputs"].items():
 assert digest(data/name)==record["sha256"]
feas=json.loads((feasibility/"feasibility.json").read_text())
assert digest(feasibility/"feasibility.json")==expected_feasibility
assert feas["status"]=="GO_FOR_TWO_EPOCH_SMOKE_ONLY"
assert feas["authorization"]["two_epoch_smoke"] is True
assert feas["authorization"]["formal_80_plus_240_epoch_training"] is False
assert feas["data_isolation"]["seed2_opened"] is False
assert feas["data_isolation"]["support_opened"] is False
' "$DATA" "$FEASIBILITY" "$EXPECTED_DATA_MANIFEST_SHA256" "$EXPECTED_FEASIBILITY_SHA256"

if [[ "$MODE" == "formal" ]]; then
    for path in "$R2N_PREDECESSOR/DONE" "$R2N_PREDECESSOR/TRAINING_DONE" \
        "$R2N_PREDECESSOR/R2N_CORE_GATE_FAILED" "$R2N_PREDECESSOR/CORE_GATE_FAILED" \
        "$R2N_PREDECESSOR/EXIT_CODE" \
        "$R2N_PREDECESSOR/core_checkpoint_gate.json" \
        "$R2N_PREDECESSOR/best_diagnostic_core.model"; do
        [[ -e "$path" ]] || { echo "formal R2O blocked by missing R2N artifact: $path" >&2; exit 4; }
    done
    for forbidden_marker in RUNNING FAILED R2N_CORE_GATE_PASSED CORE_GATE_PASSED; do
        [[ ! -e "$R2N_PREDECESSOR/$forbidden_marker" ]] || {
            echo "formal R2O rejects inconsistent R2N marker: $forbidden_marker" >&2
            exit 4
        }
    done
    [[ "$(tr -d '[:space:]' < "$R2N_PREDECESSOR/EXIT_CODE")" == "0" ]] || { echo "formal R2O requires R2N EXIT_CODE=0" >&2; exit 4; }
    [[ "$(digest "$R2N_PREDECESSOR/core_checkpoint_gate.json")" == "$EXPECTED_R2N_GATE_SHA256" ]] || { echo "formal R2O R2N gate hash mismatch" >&2; exit 4; }
    [[ "$(digest "$R2N_PREDECESSOR/best_diagnostic_core.model")" == "$EXPECTED_R2N_DIAGNOSTIC_MODEL_SHA256" ]] || { echo "formal R2O R2N model hash mismatch" >&2; exit 4; }
    if [[ ! "$EXPECTED_SMOKE_GATE_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
        echo "formal R2O remains blocked until R2O_EXPECTED_SMOKE_GATE_SHA256 is independently frozen" >&2
        exit 4
    fi
    for path in "$SMOKE_OUT/DONE" "$SMOKE_OUT/SMOKE_PASS" "$SMOKE_OUT/EXIT_CODE" \
        "$SMOKE_OUT/smoke_gate.json" "$SMOKE_OUT/training_freeze.json"; do
        [[ -e "$path" ]] || { echo "formal R2O blocked by missing smoke artifact: $path" >&2; exit 4; }
    done
    for forbidden_marker in RUNNING FAILED SMOKE_FAILED CORE_GATE_PASSED CORE_GATE_FAILED; do
        [[ ! -e "$SMOKE_OUT/$forbidden_marker" ]] || {
            echo "formal R2O rejects inconsistent smoke marker: $forbidden_marker" >&2
            exit 4
        }
    done
    [[ "$(tr -d '[:space:]' < "$SMOKE_OUT/EXIT_CODE")" == "0" ]] || { echo "formal R2O requires smoke EXIT_CODE=0" >&2; exit 4; }
    [[ "$(digest "$SMOKE_OUT/smoke_gate.json")" == "$EXPECTED_SMOKE_GATE_SHA256" ]] || { echo "formal R2O smoke gate hash mismatch" >&2; exit 4; }
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
smoke,data_manifest,wrapper,trainer,evaluator,runner,feasibility=sys.argv[1:]
root=pathlib.Path(smoke); gate=json.loads((root/"smoke_gate.json").read_text())
freeze=json.loads((root/"training_freeze.json").read_text())
feasibility=pathlib.Path(feasibility)
digest=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert gate["status"]=="R2O_two_epoch_smoke_passed"
assert gate["smoke"] is True and gate["implementation_pass"] is True
assert gate["cutoff_C2_pass"] is True and gate["reference_null_pass"] is True
assert gate["postcore_or_deployment_authorized"] is False
assert gate["bundle"]["kind"]=="R2O_diagnostic_bundle_not_authorized_for_deployment"
assert freeze["mode"]=="smoke" and freeze["status"]=="frozen_before_training"
assert freeze["stage1"]["epochs"]==2 and freeze["stage1"]["dtype"]=="float64"
assert freeze["stage1"]["expected_optimizer_steps"]==184
assert freeze["stage2"]["epochs"]==2 and freeze["stage2"]["dtype"]=="float64"
assert freeze["feasibility"]=={
 "json_sha256":digest(feasibility/"feasibility.json"),
 "marker_sha256":digest(feasibility/"GO_FOR_TWO_EPOCH_SMOKE")}
assert gate["inputs"]["data_manifest_sha256"]==digest(data_manifest)
expected={"wrapper":digest(wrapper),"trainer":digest(trainer),
          "evaluator":digest(evaluator),"runner":digest(runner)}
assert freeze["source_sha256"]==expected
assert gate["inputs"]["training_freeze_sha256"]==digest(root/"training_freeze.json")
' "$SMOKE_OUT" "$DATA/manifest.json" "$WRAPPER_SOURCE" "$TRAINER_SOURCE" "$EVALUATOR_SOURCE" "$RUNNER_SOURCE" "$FEASIBILITY"
fi

if [[ "${R2O_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    "$CONDA" run -n "$CONDA_ENV" python -c '
import json,sys
print(json.dumps({"status":"R2O_preflight_passed","mode":sys.argv[1],
 "formal_training_started":False,"data_manifest_sha256":sys.argv[2],
 "feasibility_sha256":sys.argv[3]},indent=2))
' "$MODE" "$EXPECTED_DATA_MANIFEST_SHA256" "$EXPECTED_FEASIBILITY_SHA256"
    exit 0
fi

mkdir -p "$SNAPSHOT_DIR" "$PRETRAIN_DIR/checkpoints" "$PRETRAIN_DIR/logs" \
    "$PRETRAIN_DIR/results" "$STAGE2_DIR"
if [[ ! -f "$RUNNER_SNAPSHOT" ]]; then
    cp "$WRAPPER_SOURCE" "$WRAPPER_SNAPSHOT.tmp"
    mv "$WRAPPER_SNAPSHOT.tmp" "$WRAPPER_SNAPSHOT"
    cp "$TRAINER_SOURCE" "$TRAINER_SNAPSHOT.tmp"
    mv "$TRAINER_SNAPSHOT.tmp" "$TRAINER_SNAPSHOT"
    cp "$EVALUATOR_SOURCE" "$EVALUATOR_SNAPSHOT.tmp"
    mv "$EVALUATOR_SNAPSHOT.tmp" "$EVALUATOR_SNAPSHOT"
    # The runner is the publication marker and is written last.
    cp "$RUNNER_SOURCE" "$RUNNER_SNAPSHOT.tmp"
    mv "$RUNNER_SNAPSHOT.tmp" "$RUNNER_SNAPSHOT"
    exec bash "$RUNNER_SNAPSHOT" "$MODE"
fi

for path in "$RUNNER_SNAPSHOT" "$WRAPPER_SNAPSHOT" "$TRAINER_SNAPSHOT" "$EVALUATOR_SNAPSHOT"; do
    [[ -f "$path" ]] || { echo "incomplete R2O code snapshot set" >&2; exit 5; }
done

if [[ -e "$OUT/DONE" ]]; then
    [[ ! -e "$OUT/RUNNING" && ! -e "$OUT/FAILED" && -e "$OUT/TRAINING_DONE" ]] || { echo "inconsistent R2O DONE state" >&2; exit 5; }
    [[ "$(tr -d '[:space:]' < "$OUT/EXIT_CODE")" == "0" ]] || { echo "R2O DONE lacks EXIT_CODE=0" >&2; exit 5; }
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
out=pathlib.Path(sys.argv[1]); mode=sys.argv[2]; data=pathlib.Path(sys.argv[3])
digest=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
freeze_path=out/"training_freeze.json"; freeze=json.loads(freeze_path.read_text())
assert freeze["mode"]==mode and freeze["status"]=="frozen_before_training"
assert freeze["data_manifest_sha256"]==digest(data/"manifest.json")
snap=out/"code_snapshots"
assert freeze["source_sha256"]=={
 "runner":digest(snap/"run_graphene_r2o_taylor_null.sh"),
 "wrapper":digest(snap/"graphene_r2o_taylor_null.py"),
 "trainer":digest(snap/"train_graphene_r2o_taylor_null.py"),
 "evaluator":digest(snap/"evaluate_graphene_r2o_taylor_null.py")}
pre=json.loads((out/"pretrain_runtime.json").read_text())
assert pre["status"]=="R2O_stage1_exact_epoch_artifact"
assert digest(pre["model"]["path"])==pre["model"]["sha256"]
assert digest(pre["final_checkpoint"]["path"])==pre["final_checkpoint"]["sha256"]
stage=out/"stage2"; runtime=json.loads((stage/"stage2_runtime.json").read_text())
assert runtime["status"]=="R2O_stage2_training_complete_pending_fixed_gate"
assert runtime["stage2_contract_sha256"]==digest(stage/"stage2_contract.json")
for record in runtime["fixed_checkpoints"].values(): assert digest(record["path"])==record["sha256"]
gate_path=out/("smoke_gate.json" if mode=="smoke" else "core_checkpoint_gate.json")
gate=json.loads(gate_path.read_text())
assert gate["inputs"]["training_freeze_sha256"]==digest(freeze_path)
assert gate["inputs"]["data_manifest_sha256"]==digest(data/"manifest.json")
assert digest(gate["bundle"]["path"])==gate["bundle"]["sha256"]
if mode=="smoke":
 assert gate["status"]=="R2O_two_epoch_smoke_passed" and (out/"SMOKE_PASS").is_file()
 assert gate["postcore_or_deployment_authorized"] is False
 assert not (out/"SMOKE_FAILED").exists()
 assert not (out/"CORE_GATE_PASSED").exists() and not (out/"CORE_GATE_FAILED").exists()
else:
 assert gate["status"] in {"R2O_core_checkpoint_gate_passed","R2O_core_checkpoint_gate_failed"}
 passed=gate["status"].endswith("passed")
 assert (out/("CORE_GATE_PASSED" if passed else "CORE_GATE_FAILED")).is_file()
 assert not (out/("CORE_GATE_FAILED" if passed else "CORE_GATE_PASSED")).exists()
 assert not (out/"SMOKE_PASS").exists() and not (out/"SMOKE_FAILED").exists()
' "$OUT" "$MODE" "$DATA"
    echo "R2O $MODE already completed; immutable DONE+EXIT_CODE=0 no-op"
    exit 0
fi

exec > >(tee -a "$LOG") 2>&1
export LD_LIBRARY_PATH="${CONDA%/bin/conda}/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r2o}"

MAX_INITIAL_GPU_USED_MIB="${R2O_MAX_INITIAL_GPU_USED_MIB:-1000}"
INITIAL_GPU_USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
if [[ ! "$INITIAL_GPU_USED_MIB" =~ ^[0-9]+$ ]] || (( INITIAL_GPU_USED_MIB > MAX_INITIAL_GPU_USED_MIB )); then
    echo "R2O requires an isolated GPU: initial usage=${INITIAL_GPU_USED_MIB:-invalid} MiB, limit=$MAX_INITIAL_GPU_USED_MIB MiB" >&2
    exit 5
fi

"$CONDA" run -n "$CONDA_ENV" python -c '
import mace,torch
assert getattr(mace,"__version__","unknown")=="0.3.16"
assert torch.cuda.is_available() and torch.cuda.device_count()>=1
print({"mace":mace.__version__,"torch":torch.__version__,"gpu":torch.cuda.get_device_name(0)})
'

if [[ ! -f "$FREEZE" ]]; then
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,os,pathlib,sys,mace,torch
(out,mode,data,feas,r2n,smoke,runner,wrapper,trainer,evaluator,
 pretrain_epochs,stage2_epochs,checkpoint_epochs,initial_gpu,max_initial_gpu)=sys.argv[1:]
out=pathlib.Path(out); digest=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
payload={
 "format":"graphene_r2o_training_freeze_v1","status":"frozen_before_training",
 "mode":mode,"data_manifest_sha256":digest(pathlib.Path(data)/"manifest.json"),
 "feasibility":{"json_sha256":digest(pathlib.Path(feas)/"feasibility.json"),
   "marker_sha256":digest(pathlib.Path(feas)/"GO_FOR_TWO_EPOCH_SMOKE")},
 "source_sha256":{"runner":digest(runner),"wrapper":digest(wrapper),
   "trainer":digest(trainer),"evaluator":digest(evaluator)},
 "environment":{"conda_environment":os.environ.get("CONDA_DEFAULT_ENV"),
   "MACE":getattr(mace,"__version__","unknown"),"torch":str(torch.__version__),
   "CUDA":str(torch.version.cuda),"GPU":str(torch.cuda.get_device_name(0)),
   "initial_gpu_memory_used_MiB":int(initial_gpu),
   "maximum_allowed_initial_gpu_memory_used_MiB":int(max_initial_gpu)},
 "predecessor_R2N_gate_sha256":digest(pathlib.Path(r2n)/"core_checkpoint_gate.json") if mode=="formal" else None,
 "smoke_gate_sha256":digest(pathlib.Path(smoke)/"smoke_gate.json") if mode=="formal" else None,
 "stage1":{"epochs":int(pretrain_epochs),"force_only":True,"dtype":"float64",
   "zero_based_epoch_range":[0,int(pretrain_epochs)-1],
   "training_configurations_per_epoch":92,
   "expected_optimizer_steps":92*int(pretrain_epochs),
   "seed":83,"batch_size":1,"optimizer":"Adam","amsgrad":True,
   "gradient_clip":10.0,"lr":0.001,"CLI_weight_decay":1e-6,
   "MACE_0p3p16_expected_parameter_group_weight_decay":[0.0,1e-6,0.0,1e-6,0.0],
   "scheduler":"ExponentialLR","gamma":1.0,"EMA":False,"SWA":False,
   "early_stop_disabled_by_patience":True,"seed1_used_for_validation_or_LR":False,
   "validation_aliases_thermal_training_file":True,
   "group_mass":{"E50":0.5,"T300":0.25,"T600":0.25}},
 "stage2":{"epochs":int(stage2_epochs),"checkpoint_epochs":[int(x) for x in checkpoint_epochs.split(",")],
   "all_MACE_parameters_trainable":True,"dtype":"float64",
   "group_mass":{"E50":0.5,"T300":0.2,"T600":0.2,"harmonic_zero":0.1}},
 "forbidden":{"seed1_in_gradients_or_scales":False,"seed2_opened":False,"support_opened":False}}
tmp=out/"training_freeze.json.tmp"; tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
os.replace(tmp,out/"training_freeze.json")
' "$OUT" "$MODE" "$DATA" "$FEASIBILITY" "$R2N_PREDECESSOR" "$SMOKE_OUT" \
        "$RUNNER_SNAPSHOT" "$WRAPPER_SNAPSHOT" "$TRAINER_SNAPSHOT" "$EVALUATOR_SNAPSHOT" \
        "$PRETRAIN_EPOCHS" "$STAGE2_EPOCHS" "$STAGE2_CHECKPOINT_EPOCHS" \
        "$INITIAL_GPU_USED_MIB" "$MAX_INITIAL_GPU_USED_MIB"
fi

"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
freeze_path,mode,data,runner,wrapper,trainer,evaluator=sys.argv[1:]
freeze=json.loads(pathlib.Path(freeze_path).read_text()); digest=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert freeze["status"]=="frozen_before_training" and freeze["mode"]==mode
assert freeze["data_manifest_sha256"]==digest(pathlib.Path(data)/"manifest.json")
assert freeze["source_sha256"]=={"runner":digest(runner),"wrapper":digest(wrapper),
 "trainer":digest(trainer),"evaluator":digest(evaluator)}
' "$FREEZE" "$MODE" "$DATA" "$RUNNER_SNAPSHOT" "$WRAPPER_SNAPSHOT" "$TRAINER_SNAPSHOT" "$EVALUATOR_SNAPSHOT"

rm -f "$OUT/FAILED" "$OUT/EXIT_CODE" "$OUT/SMOKE_PASS" "$OUT/SMOKE_FAILED" \
    "$OUT/CORE_GATE_PASSED" "$OUT/CORE_GATE_FAILED"
date '+%Y-%m-%dT%H:%M:%S%z' > "$OUT/STARTED_OR_RESUMED_AT"
touch "$OUT/RUNNING"
STAGE1_MONITOR_PID=""
finish() {
    status=$?
    trap - EXIT
    if [[ -n "$STAGE1_MONITOR_PID" ]]; then
        kill "$STAGE1_MONITOR_PID" 2>/dev/null || true
        wait "$STAGE1_MONITOR_PID" 2>/dev/null || true
    fi
    rm -f "$OUT/RUNNING"
    printf '%s\n' "$status" > "$OUT/EXIT_CODE"
    if (( status != 0 )); then touch "$OUT/FAILED"; fi
    exit "$status"
}
trap finish EXIT

CONFIG_WEIGHTS='{"r2o_exact_e50_seed0_train":1.0,"r2o_auxiliary_T300_train":0.2777777777777778,"r2o_auxiliary_T600_train":0.2777777777777778}'
if [[ ! -e "$OUT/PRETRAIN_DONE" ]]; then
    if find "$PRETRAIN_DIR/checkpoints" -type f -name '*.pt' -print -quit | grep -q . \
        || [[ -e "$PRETRAIN_MODEL" ]]; then
        echo "partial stage1 is not resumed because MACE restart repeats its last zero-based epoch; use a new OUT for an exact seed83 run" >&2
        exit 6
    fi
    STAGE1_GPU_SAMPLES="$OUT/pretrain_gpu_samples.csv"
    printf '%s\n' 'timestamp_epoch_s,memory_used_MiB,utilization_percent' > "$STAGE1_GPU_SAMPLES"
    (
        while [[ -e "$OUT/RUNNING" ]]; do
            sample="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1)"
            printf '%s,%s\n' "$(date +%s)" "$sample" >> "$STAGE1_GPU_SAMPLES"
            sleep 2
        done
    ) &
    STAGE1_MONITOR_PID=$!
    STAGE1_STARTED_EPOCH_S="$(date +%s)"
    "$CONDA" run --no-capture-output -n "$CONDA_ENV" python -m mace.cli.run_train \
        --name "$NAME" --model MACE --num_interactions 2 \
        --hidden_irreps '16x0e+16x1o+16x2e' --r_max 3.2 \
        --num_radial_basis 24 --num_cutoff_basis 5 --max_ell 2 --correlation 3 \
        --train_file "$DATA/train_thermal.xyz" --valid_file "$DATA/train_thermal.xyz" \
        --energy_key REF_energy --forces_key REF_forces --E0s '{6:0.0}' \
        --loss forces_only --energy_weight 0.0 --forces_weight 100.0 \
        --config_type_weights "$CONFIG_WEIGHTS" \
        --batch_size 1 --valid_batch_size 1 --max_num_epochs "$PRETRAIN_EPOCHS" \
        --patience "$((PRETRAIN_EPOCHS + 20))" --eval_interval 1 \
        --optimizer adam --clip_grad 10.0 --lr 0.001 --weight_decay 1.0e-6 \
        --scheduler ExponentialLR --lr_scheduler_gamma 1.0 \
        --default_dtype float64 --device cuda --seed 83 \
        --save_cpu --keep_checkpoints --save_all_checkpoints \
        --model_dir "$PRETRAIN_DIR" --checkpoints_dir "$PRETRAIN_DIR/checkpoints" \
        --log_dir "$PRETRAIN_DIR/logs" --results_dir "$PRETRAIN_DIR/results"
    STAGE1_ENDED_EPOCH_S="$(date +%s)"
    kill "$STAGE1_MONITOR_PID" 2>/dev/null || true
    wait "$STAGE1_MONITOR_PID" 2>/dev/null || true
    STAGE1_MONITOR_PID=""
    [[ -s "$PRETRAIN_MODEL" ]] || { echo "exact pretrain artifact absent" >&2; exit 6; }
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,os,pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(sys.argv[2])))
from graphene_r2o_taylor_null import state_dict_sha256,torch_load,validate_mace_architecture
model_path=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[3]); epochs=int(sys.argv[4]); name=sys.argv[5]
started=int(sys.argv[6]); ended=int(sys.argv[7]); gpu_samples=pathlib.Path(sys.argv[8])
model=torch_load(model_path,"cpu"); architecture=validate_mace_architecture(model)
assert model.__class__.__name__=="ScaleShiftMACE"
checkpoint_path=out/"pretrain"/"checkpoints"/f"{name}_run-83_epoch-{epochs-1}.pt"
checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
steps={int(state["step"]) for state in checkpoint["optimizer"]["state"].values() if "step" in state}
expected_steps=92*epochs
assert steps=={expected_steps},(steps,expected_steps)
parameter_group_weight_decay=[float(group["weight_decay"]) for group in checkpoint["optimizer"]["param_groups"]]
assert parameter_group_weight_decay==[0.0,1.0e-6,0.0,1.0e-6,0.0]
parameter_group_amsgrad=[bool(group["amsgrad"]) for group in checkpoint["optimizer"]["param_groups"]]
assert parameter_group_amsgrad==[True,True,True,True,True]
model_state=model.state_dict(); checkpoint_state=checkpoint["model"]
assert set(model_state)==set(checkpoint_state)
assert all(torch.equal(model_state[key],checkpoint_state[key]) for key in model_state)
metric_path=out/"pretrain"/"results"/f"{name}_run-83_train.txt"
metrics=[json.loads(line) for line in metric_path.read_text().splitlines() if line.strip()]
opt=[item for item in metrics if item.get("mode")=="opt"]
by_epoch={epoch:[item for item in opt if int(item["epoch"])==epoch] for epoch in range(epochs)}
assert all(len(items)==92 for items in by_epoch.values())
epoch_compute_seconds={str(epoch):sum(float(item["time"]) for item in items)
                       for epoch,items in by_epoch.items()}
gpu_rows=[line.split(",") for line in gpu_samples.read_text().splitlines()[1:] if line.strip()]
peak_gpu_memory_MiB=max(float(row[1].strip()) for row in gpu_rows) if gpu_rows else 0.0
payload={"status":"R2O_stage1_exact_epoch_artifact","epochs":epochs,
 "zero_based_final_epoch":epochs-1,"training_configurations_per_epoch":92,
 "expected_and_observed_optimizer_steps":expected_steps,
 "optimizer":"Adam","CLI_weight_decay":1.0e-6,
 "observed_parameter_group_weight_decay":parameter_group_weight_decay,
 "observed_parameter_group_amsgrad":parameter_group_amsgrad,
 "final_checkpoint":{"path":str(checkpoint_path),
   "sha256":hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
   "zero_based_epoch":epochs-1},
 "wall_time_seconds":ended-started,"per_epoch_optimizer_compute_seconds":epoch_compute_seconds,
 "peak_nvidia_smi_memory_used_MiB":peak_gpu_memory_MiB,
 "gpu_samples":{"path":str(gpu_samples),"sha256":hashlib.sha256(gpu_samples.read_bytes()).hexdigest()},
 "model":{"path":str(model_path),"sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
          "state_sha256":state_dict_sha256(model)},"architecture":architecture,
 "scale":torch.atleast_1d(model.scale_shift.scale).tolist(),
 "shift":torch.atleast_1d(model.scale_shift.shift).tolist(),
 "seed1_used_for_validation_or_LR":False,"EMA":False,"SWA":False}
tmp=out/"pretrain_runtime.json.tmp"; tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
os.replace(tmp,out/"pretrain_runtime.json")
' "$PRETRAIN_MODEL" "$SNAPSHOT_DIR" "$OUT" "$PRETRAIN_EPOCHS" "$NAME" \
        "$STAGE1_STARTED_EPOCH_S" "$STAGE1_ENDED_EPOCH_S" "$STAGE1_GPU_SAMPLES"
    touch "$OUT/PRETRAIN_DONE"
fi

"$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys,torch
runtime_path=pathlib.Path(sys.argv[1]); model_path=pathlib.Path(sys.argv[2]); epochs=int(sys.argv[3])
runtime=json.loads(runtime_path.read_text()); digest=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert runtime["status"]=="R2O_stage1_exact_epoch_artifact"
assert runtime["epochs"]==epochs and runtime["zero_based_final_epoch"]==epochs-1
assert runtime["expected_and_observed_optimizer_steps"]==92*epochs
assert runtime["EMA"] is False and runtime["SWA"] is False
assert runtime["optimizer"]=="Adam" and runtime["CLI_weight_decay"]==1.0e-6
assert runtime["observed_parameter_group_weight_decay"]==[0.0,1.0e-6,0.0,1.0e-6,0.0]
assert runtime["observed_parameter_group_amsgrad"]==[True,True,True,True,True]
assert digest(model_path)==runtime["model"]["sha256"]
checkpoint_path=pathlib.Path(runtime["final_checkpoint"]["path"])
assert digest(checkpoint_path)==runtime["final_checkpoint"]["sha256"]
assert digest(runtime["gpu_samples"]["path"])==runtime["gpu_samples"]["sha256"]
checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
model=torch.load(model_path,map_location="cpu",weights_only=False)
steps={int(state["step"]) for state in checkpoint["optimizer"]["state"].values() if "step" in state}
assert steps=={92*epochs}
assert set(model.state_dict())==set(checkpoint["model"])
assert all(torch.equal(model.state_dict()[key],checkpoint["model"][key]) for key in model.state_dict())
' "$OUT/pretrain_runtime.json" "$PRETRAIN_MODEL" "$PRETRAIN_EPOCHS"

PRETRAIN_SHA256="$(digest "$PRETRAIN_MODEL")"
if [[ ! -e "$OUT/TRAINING_DONE" ]]; then
    resume=()
    if [[ -f "$STAGE2_DIR/checkpoints/latest.pt" ]]; then resume=(--resume); fi
    "$CONDA" run --no-capture-output -n "$CONDA_ENV" python "$TRAINER_SNAPSHOT" \
        --thermal-train "$DATA/train_thermal.xyz" \
        --harmonic-zero-train "$DATA/train_harmonic_lambda1_small_zero.xyz" \
        --reference-6x6 "$DATA/reference_6x6.xyz" --reference-8x8 "$DATA/reference_8x8.xyz" \
        --initial-model "$PRETRAIN_MODEL" --initial-model-sha256 "$PRETRAIN_SHA256" \
        --launcher-freeze "$FREEZE" --output-dir "$STAGE2_DIR" \
        --epochs "$STAGE2_EPOCHS" --checkpoint-epochs "$STAGE2_CHECKPOINT_EPOCHS" \
        --device cuda "${resume[@]}"
    "$CONDA" run -n "$CONDA_ENV" python -c '
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); expected=int(sys.argv[2]); runtime=json.loads((root/"stage2_runtime.json").read_text())
contract=json.loads((root/"stage2_contract.json").read_text())
assert runtime["status"]=="R2O_stage2_training_complete_pending_fixed_gate"
assert runtime["epochs"]==expected==contract["epochs"]
assert runtime["stage2_contract_sha256"]==hashlib.sha256((root/"stage2_contract.json").read_bytes()).hexdigest()
assert contract["selection_data_read"] is False and contract["seed1_seed2_support_read"] is False
' "$STAGE2_DIR" "$STAGE2_EPOCHS"
    touch "$OUT/TRAINING_DONE"
fi

if [[ "$MODE" == "smoke" ]]; then
    GATE="$OUT/smoke_gate.json"
    BUNDLE="$OUT/smoke_diagnostic_bundle.pt"
    smoke_arg=(--smoke)
else
    GATE="$OUT/core_checkpoint_gate.json"
    BUNDLE="$OUT/selected_or_diagnostic_bundle.pt"
    smoke_arg=()
fi
"$CONDA" run --no-capture-output -n "$CONDA_ENV" python "$EVALUATOR_SNAPSHOT" \
    --data "$DATA" --training-dir "$STAGE2_DIR" --device cuda \
    --output "$GATE" --selected-bundle "$BUNDLE" \
    --wrapper-snapshot "$WRAPPER_SNAPSHOT" "${smoke_arg[@]}"

GATE_STATUS="$("$CONDA" run -n "$CONDA_ENV" python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$GATE" | tail -1)"
if [[ "$MODE" == "smoke" ]]; then
    if [[ "$GATE_STATUS" != "R2O_two_epoch_smoke_passed" ]]; then
        touch "$OUT/SMOKE_FAILED"
        echo "R2O two-epoch smoke failed its implementation/null gate" >&2
        exit 7
    fi
    rm -f "$OUT/SMOKE_FAILED"
    touch "$OUT/SMOKE_PASS"
else
    if [[ "$GATE_STATUS" == "R2O_core_checkpoint_gate_passed" ]]; then
        rm -f "$OUT/CORE_GATE_FAILED"
        touch "$OUT/CORE_GATE_PASSED"
    elif [[ "$GATE_STATUS" == "R2O_core_checkpoint_gate_failed" ]]; then
        rm -f "$OUT/CORE_GATE_PASSED"
        touch "$OUT/CORE_GATE_FAILED"
    else
        echo "unexpected R2O formal gate status: $GATE_STATUS" >&2
        exit 7
    fi
fi

date '+%Y-%m-%dT%H:%M:%S%z' > "$OUT/COMPLETED_AT"
rm -f "$OUT/FAILED"
rm -f "$OUT/RUNNING"
printf '%s\n' '0' > "$OUT/EXIT_CODE.tmp"
mv "$OUT/EXIT_CODE.tmp" "$OUT/EXIT_CODE"
printf '%s\n' "$GATE_STATUS" > "$OUT/DONE.tmp"
mv "$OUT/DONE.tmp" "$OUT/DONE"
trap - EXIT
echo "R2O $MODE complete: $GATE_STATUS"
exit 0
