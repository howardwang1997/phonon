from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
from typing import Any


BUNDLE = Path(
    "/root/phonon/results/r2r0d_diagnostic_bundle_20260826/"
    "2335246e45c5/attempt_0001"
)
OUTPUT = BUNDLE / (
    "results/graphene_physics_temperature/post_p4_feasibility/"
    "R2R_multipolar_background/R2R0D_numerical_diagnostic/"
    "v100b_2335246e45c5_attempt1"
)
CONTROL = Path(
    "/root/phonon/results/r2r0d_diagnostic_exec_20260826/"
    "2335246e45c5_attempt_0001"
)
FREEZE = BUNDLE / (
    "results/graphene_physics_temperature/post_p4_feasibility/"
    "R2R0D_numerical_diagnostic_candidate_v2_20260826/freeze_manifest.json"
)
AUTH = FREEZE.with_name("R2R0D_DIAGNOSTIC_AUTH")
CLOSURE_MANIFEST = CONTROL / "science_closure_manifest.json"
SUPERVISOR = CONTROL / "external_supervisor.py"

CONDA = "/root/miniconda3/bin/conda"
CONDA_ENV = "phonon-mlip"
EXPECTED_MANIFEST_SHA256 = (
    "83f6e29e36220685bf6d6e59cd88fab1586bf59bfd98a231d9923688361dcb3e"
)
EXPECTED_AUTH_SHA256 = (
    "eaf26abacb7ea94e21dfb3c07a58be006f2e418d39eebc2e1a4d42f42c296b64"
)
EXPECTED_CLOSURE_MANIFEST_SHA256 = (
    "94a291753b0b3788eb176dc4e5620cf38c0f4bbaa204f654766b1ce2cd6ebad4"
)
EXPECTED_GPU_NAME = "Tesla V100-SXM2-32GB"
EXPECTED_GPU_UUID = "GPU-2006cd36-28af-c685-dae8-e3710431fd7d"
TERM_AFTER_SECONDS = 115.0
KILL_AFTER_SECONDS = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def command_output(argv: list[str]) -> str:
    return subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    ).stdout


def gpu_snapshot() -> dict[str, Any]:
    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu,compute_mode",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ]
    ).strip()
    apps_text = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ]
    ).strip()
    return {
        "gpu": gpu,
        "compute_apps": [] if not apps_text else apps_text.splitlines(),
    }


def process_group_rows(pgid: int) -> list[str]:
    text = command_output(
        ["/bin/ps", "-eo", "pid=,pgid=,rss=,etime=,stat=,args="]
    )
    rows = []
    for line in text.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) >= 2 and fields[1].isdigit() and int(fields[1]) == pgid:
            rows.append(line.strip())
    return rows


def verify_non_symlink_tree(path: Path) -> None:
    if path.resolve(strict=True) != path:
        raise RuntimeError(f"path resolution changed: {path}")
    probe = Path(path.anchor)
    for part in path.parts[1:]:
        probe /= part
        if probe.is_symlink():
            raise RuntimeError(f"symlink component: {probe}")


def closure_records() -> list[dict[str, Any]]:
    records = json.loads(CLOSURE_MANIFEST.read_text(encoding="ascii"))
    if not isinstance(records, list) or len(records) != 590:
        raise RuntimeError("science closure manifest count changed")
    expected_paths = set()
    observed = []
    inodes = set()
    for record in records:
        relative = record["path"]
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"unsafe relative closure path: {relative}")
        if relative in expected_paths:
            raise RuntimeError(f"duplicate closure path: {relative}")
        expected_paths.add(relative)
        path = BUNDLE / relative
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_nlink != 1:
            raise RuntimeError(f"closure file type/link changed: {relative}")
        digest = sha256(path)
        current = {"path": relative, "sha256": digest, "size": info.st_size}
        if current != record:
            raise RuntimeError(f"closure file changed: {relative}")
        inode = (info.st_dev, info.st_ino)
        if inode in inodes:
            raise RuntimeError(f"duplicate closure inode: {relative}")
        inodes.add(inode)
        observed.append(current)
    actual_paths = set()
    for path in BUNDLE.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"bundle contains symlink: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(BUNDLE).as_posix())
    controls = {
        FREEZE.relative_to(BUNDLE).as_posix(),
        AUTH.relative_to(BUNDLE).as_posix(),
    }
    output_prefix = OUTPUT.relative_to(BUNDLE).as_posix() + "/"
    nonoutput_paths = {
        path for path in actual_paths if not path.startswith(output_prefix)
    }
    if nonoutput_paths != expected_paths | controls:
        raise RuntimeError(
            "bundle inventory changed: "
            f"extra={sorted(nonoutput_paths - expected_paths - controls)} "
            f"missing={sorted(expected_paths + controls - nonoutput_paths)}"
        )
    return observed


def output_manifest() -> list[dict[str, Any]]:
    records = []
    if not OUTPUT.is_dir() or OUTPUT.is_symlink():
        return records
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"output contains symlink: {path}")
        if path.is_file():
            info = path.stat()
            if info.st_nlink != 1:
                raise RuntimeError(f"output contains hardlink: {path}")
            records.append(
                {
                    "path": path.relative_to(OUTPUT).as_posix(),
                    "sha256": sha256(path),
                    "size": info.st_size,
                }
            )
    return records


def terminal_audit(returncode: int, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    files = {record["path"] for record in manifest}
    markers = files & {"RUNNING", "DONE", "FAILED"}
    audit: dict[str, Any] = {
        "marker_xor": sorted(markers),
        "shell_returncode": returncode,
        "inventory": manifest,
        "valid_success": False,
    }
    if returncode != 0:
        return audit
    expected = {
        "DONE",
        "EXIT_CODE",
        "diagnostic_receipt.json",
        "diagnostic_arrays.npz",
    }
    if files != expected or markers != {"DONE"}:
        return audit
    if (OUTPUT / "EXIT_CODE").read_bytes() != b"0\n":
        return audit
    receipt_path = OUTPUT / "diagnostic_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_done = (
        receipt["status"] + "\n" + sha256(receipt_path) + "\n"
    ).encode("ascii")
    if (OUTPUT / "DONE").read_bytes() != expected_done:
        return audit
    runtime = receipt.get("runtime", {})
    scope_ok = (
        receipt.get("stopped_early") is not True
        and receipt.get("adjudicates_attempt2") is False
        and receipt.get("formal_authorization") is False
        and receipt.get("fit_or_training_performed") is False
        and receipt.get("deployment_performed") is False
        and receipt.get("labels_used") is False
        and receipt.get("held_or_support_access") is False
        and receipt.get("GO_marker_created") is False
        and receipt.get("attempt2_scientific_status_preserved")
        == "R2R0_FORMAL_REPRESENTATION_PRECHECK_FAILED"
        and runtime.get("cuda", {}).get("name") == EXPECTED_GPU_NAME
        and runtime.get("cuda", {}).get("exact_name_allowlist_pass") is True
    )
    audit.update(
        {
            "status": receipt.get("status"),
            "receipt_sha256": sha256(receipt_path),
            "scope_ok": scope_ok,
            "valid_success": bool(scope_ok),
        }
    )
    return audit


def main() -> int:
    if Path.cwd() != BUNDLE or BUNDLE.resolve(strict=True) != BUNDLE:
        raise RuntimeError("supervisor cwd/bundle identity changed")
    if CONTROL.resolve(strict=True) != CONTROL:
        raise RuntimeError("control path identity changed")
    verify_non_symlink_tree(BUNDLE)
    verify_non_symlink_tree(CONTROL)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("diagnostic output is not fresh")
    for path in (
        CONTROL / "command.json",
        CONTROL / "diagnostic.stdout",
        CONTROL / "diagnostic.stderr",
        CONTROL / "monitor.jsonl",
        CONTROL / "launcher_receipt.json",
        CONTROL / "LAUNCHER_DONE",
        CONTROL / "LAUNCHER_FAILED",
        CONTROL / "LAUNCHER_EXIT_CODE",
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"external control is not fresh: {path}")
    if sha256(FREEZE) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("freeze manifest hash changed before launch")
    if sha256(AUTH) != EXPECTED_AUTH_SHA256:
        raise RuntimeError("authorization marker hash changed before launch")
    if AUTH.read_bytes() != (EXPECTED_MANIFEST_SHA256 + "\n").encode("ascii"):
        raise RuntimeError("authorization marker bytes changed before launch")
    closure_before = closure_records()
    closure_manifest_hash = sha256(CLOSURE_MANIFEST)
    if closure_manifest_hash != EXPECTED_CLOSURE_MANIFEST_SHA256:
        raise RuntimeError("science closure manifest hash changed")
    supervisor_hash = sha256(SUPERVISOR)
    gpu_before = gpu_snapshot()
    gpu_fields = [item.strip() for item in gpu_before["gpu"].split(",")]
    if (
        len(gpu_fields) != 7
        or gpu_fields[1] != EXPECTED_GPU_NAME
        or gpu_fields[2] != EXPECTED_GPU_UUID
        or gpu_before["compute_apps"]
    ):
        raise RuntimeError(f"GPU preflight changed: {gpu_before}")

    argv = [
        CONDA,
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        "python",
        "-u",
        "scripts/smearing_kink/run_graphene_r2r0_diagnostic.py",
        "diagnostic",
        "--output",
        str(OUTPUT),
        "--device",
        "cuda:0",
        "--freeze-manifest",
        str(FREEZE),
        "--authorization-marker",
        str(AUTH),
    ]
    env_overrides = {
        "CUDA_VISIBLE_DEVICES": "0",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command_receipt = {
        "argv": argv,
        "cwd": str(BUNDLE),
        "env_overrides": env_overrides,
        "term_after_seconds": TERM_AFTER_SECONDS,
        "kill_after_seconds": KILL_AFTER_SECONDS,
        "stdin": "/dev/null",
        "stdout": str(CONTROL / "diagnostic.stdout"),
        "stderr": str(CONTROL / "diagnostic.stderr"),
    }
    atomic_json(CONTROL / "command.json", command_receipt)
    command_hash = sha256(CONTROL / "command.json")

    environment = os.environ.copy()
    environment.update(env_overrides)
    monitor_stop = threading.Event()
    monitor_path = CONTROL / "monitor.jsonl"
    monitor_handle = monitor_path.open("xb", buffering=0)
    stdout_handle = (CONTROL / "diagnostic.stdout").open("xb", buffering=0)
    stderr_handle = (CONTROL / "diagnostic.stderr").open("xb", buffering=0)
    proc: subprocess.Popen[bytes] | None = None
    monitor_thread: threading.Thread | None = None
    timed_out = False
    term_sent = False
    kill_sent = False
    cleanup_term_sent = False
    cleanup_kill_sent = False
    started_ns = time.time_ns()
    started_mono = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=BUNDLE,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        pgid = os.getpgid(proc.pid)
        if pgid != proc.pid:
            raise RuntimeError("diagnostic PGID is not its PID")

        def sample() -> None:
            while not monitor_stop.is_set():
                try:
                    payload = {
                        "time_ns": time.time_ns(),
                        "elapsed_seconds": time.monotonic() - started_mono,
                        "gpu": gpu_snapshot(),
                        "process_group": process_group_rows(pgid),
                    }
                except Exception as exception:
                    payload = {
                        "time_ns": time.time_ns(),
                        "elapsed_seconds": time.monotonic() - started_mono,
                        "sample_error": f"{type(exception).__name__}: {exception}",
                    }
                monitor_handle.write(
                    (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
                )
                monitor_stop.wait(1.0)

        monitor_thread = threading.Thread(target=sample, daemon=False)
        monitor_thread.start()
        try:
            returncode = proc.wait(timeout=TERM_AFTER_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            term_sent = True
            os.killpg(pgid, signal.SIGTERM)
            try:
                returncode = proc.wait(timeout=KILL_AFTER_SECONDS)
            except subprocess.TimeoutExpired:
                kill_sent = True
                os.killpg(pgid, signal.SIGKILL)
                returncode = proc.wait()
        diagnostic_wall_seconds = time.monotonic() - started_mono
        lingering = process_group_rows(pgid)
        if lingering:
            cleanup_term_sent = True
            os.killpg(pgid, signal.SIGTERM)
            deadline = time.monotonic() + KILL_AFTER_SECONDS
            while process_group_rows(pgid) and time.monotonic() < deadline:
                time.sleep(0.1)
            if process_group_rows(pgid):
                cleanup_kill_sent = True
                os.killpg(pgid, signal.SIGKILL)
        deadline = time.monotonic() + 10.0
        while process_group_rows(pgid) and time.monotonic() < deadline:
            time.sleep(0.1)
        process_group_after = process_group_rows(pgid)
        deadline = time.monotonic() + 10.0
        gpu_after = gpu_snapshot()
        while gpu_after["compute_apps"] and time.monotonic() < deadline:
            time.sleep(0.25)
            gpu_after = gpu_snapshot()
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
        stdout_handle.close()
        stderr_handle.close()
        monitor_handle.close()

    finished_ns = time.time_ns()
    wall_seconds = time.monotonic() - started_mono
    closure_after = closure_records()
    freeze_after = sha256(FREEZE)
    auth_after = sha256(AUTH)
    output_records = output_manifest()
    terminal = terminal_audit(returncode, output_records)
    clean = not process_group_after and not gpu_after["compute_apps"]
    immutable = (
        closure_after == closure_before
        and freeze_after == EXPECTED_MANIFEST_SHA256
        and auth_after == EXPECTED_AUTH_SHA256
    )
    valid = bool(
        terminal["valid_success"]
        and clean
        and immutable
        and not timed_out
        and not term_sent
        and not kill_sent
        and diagnostic_wall_seconds < TERM_AFTER_SECONDS
    )
    receipt = {
        "format": "graphene_r2r0d_external_launcher_receipt_v1",
        "bundle": str(BUNDLE),
        "output": str(OUTPUT),
        "control": str(CONTROL),
        "supervisor_sha256": supervisor_hash,
        "science_closure_manifest_sha256": closure_manifest_hash,
        "science_closure_file_count": len(closure_before),
        "science_closure_unchanged": closure_after == closure_before,
        "freeze_manifest_sha256_before": EXPECTED_MANIFEST_SHA256,
        "freeze_manifest_sha256_after": freeze_after,
        "authorization_marker_sha256_before": EXPECTED_AUTH_SHA256,
        "authorization_marker_sha256_after": auth_after,
        "command_sha256": command_hash,
        "stdout_sha256": sha256(CONTROL / "diagnostic.stdout"),
        "stderr_sha256": sha256(CONTROL / "diagnostic.stderr"),
        "monitor_sha256": sha256(monitor_path),
        "pid": proc.pid,
        "pgid": pgid,
        "started_time_ns": started_ns,
        "finished_time_ns": finished_ns,
        "wall_seconds": wall_seconds,
        "diagnostic_wall_seconds": diagnostic_wall_seconds,
        "returncode": returncode,
        "watchdog": {
            "term_after_seconds": TERM_AFTER_SECONDS,
            "kill_after_seconds": KILL_AFTER_SECONDS,
            "timed_out": timed_out,
            "term_sent": term_sent,
            "kill_sent": kill_sent,
            "cleanup_term_sent": cleanup_term_sent,
            "cleanup_kill_sent": cleanup_kill_sent,
        },
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "process_group_after": process_group_after,
        "process_and_gpu_clean": clean,
        "immutable_inputs_unchanged": immutable,
        "terminal_audit": terminal,
        "valid_success": valid,
    }
    receipt_path = CONTROL / "launcher_receipt.json"
    atomic_json(receipt_path, receipt)
    atomic_bytes(CONTROL / "LAUNCHER_EXIT_CODE", ("0\n" if valid else "2\n").encode())
    marker = CONTROL / ("LAUNCHER_DONE" if valid else "LAUNCHER_FAILED")
    atomic_bytes(marker, (sha256(receipt_path) + "\n").encode("ascii"))
    return 0 if valid else 2


if __name__ == "__main__":
    try:
        supervisor_returncode = main()
    except BaseException as exception:
        failure = {
            "format": "graphene_r2r0d_external_supervisor_failure_v1",
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "time_ns": time.time_ns(),
        }
        try:
            atomic_json(CONTROL / "supervisor_failure.json", failure)
        finally:
            raise
    raise SystemExit(supervisor_returncode)
