"""Opt-in physical check that a verified UUID is the device a process receives.

Ordinary CI never runs this. Without `--run` and NVIDIA GPUs it prints why it
skipped and exits 0.

The simulated smoke, `examples.ray_device_binding_smoke`, injects both NVML and
CUDA observations. This opt-in check uses the real APIs: NVML names the node's
devices, Ray assigns one to each rank, and the binding layer cross-checks the
CUDA-visible UUID before starting the worker. The worker additionally creates
a CUDA context and reads its device UUID. Both driver queries use `ctypes`, so
no CUDA toolkit, PyTorch, or other runtime has to be installed on the host.

The check creates and destroys one CUDA context and computes nothing. That is
enough to name the device the process holds, and far short of a benchmark.
"""
import argparse
import ctypes
import json
import os
import platform
import re
import socket
import sys
import uuid as uuidlib

from topology_scheduler import GPUDevice, Node, Plan
from topology_scheduler.device_binding import (
    DEVICE_ORDER_ENV_VAR, PCI_BUS_ID, VERIFY, DeviceBindingError,
    DevicePlacement, device_resources, run_with_devices,
)

# A device identity no host has, advertised on purpose: a plan can name a GPU
# that a node no longer holds, and resolution rather than preflight has to
# catch it. Only the mismatch check requests it.
ABSENT_UUID = "GPU-00000000-0000-0000-0000-000000000000"


class Counter:
    """Counts the worker bodies that actually executed."""

    def __init__(self):
        self.count = 0

    def record(self):
        self.count += 1

    def total(self) -> int:
        return self.count

    def reset(self):
        self.count = 0


def cuda_driver():
    """The installed CUDA driver library, or None when the host has none."""
    names = ("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def cuda_visible_device() -> dict:
    """Name the device the CUDA driver gives this process, through a context.

    Ray narrows `CUDA_VISIBLE_DEVICES` to the assigned accelerator, so ordinal
    0 is that device. A failure is reported rather than raised: the driver is
    evidence here, not the mechanism under test.
    """
    library = cuda_driver()
    if library is None:
        return {"available": False, "reason": "no CUDA driver library on this host"}
    device, raw, context = ctypes.c_int(), (ctypes.c_char * 16)(), ctypes.c_void_p()
    read_uuid = getattr(library, "cuDeviceGetUuid_v2", None) or library.cuDeviceGetUuid
    steps = (
        ("cuInit", lambda: library.cuInit(0)),
        ("cuDeviceGet", lambda: library.cuDeviceGet(ctypes.byref(device), 0)),
        ("cuDeviceGetUuid", lambda: read_uuid(ctypes.byref(raw), device)),
        ("cuCtxCreate", lambda: library.cuCtxCreate_v2(ctypes.byref(context), 0, device)),
    )
    for name, call in steps:
        code = call()
        if code != 0:
            return {"available": False, "reason": f"{name} returned {code}"}
    library.cuCtxDestroy_v2(context)
    return {"available": True, "context_created": True,
            "uuid": f"GPU-{uuidlib.UUID(bytes=bytes(raw))}"}


def host_devices() -> tuple[tuple[GPUDevice, ...], str | None]:
    """This host's NVML devices and driver version, empty when NVML is absent."""
    try:
        import ray._private.thirdparty.pynvml as pynvml

        from topology_scheduler.inventory import _read_nvml_snapshot

        devices = _read_nvml_snapshot(pynvml)[0]
        pynvml.nvmlInit()
        try:
            version = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
        return devices, version.decode() if isinstance(version, bytes) else version
    except Exception:  # NVML absent or unusable; the caller reports it as a skip.
        return (), None


def skip_reasons(*, opted_in: bool, devices, has_driver: bool) -> tuple[str, ...]:
    """Every reason this host cannot produce the physical evidence."""
    reasons = []
    if not opted_in:
        reasons.append("pass --run to let this start Ray and open a CUDA "
                       "context on this host's GPUs")
    if not devices:
        reasons.append("NVML reported no NVIDIA GPU on this host")
    if not has_driver:
        reasons.append("no CUDA driver library, so the device the process "
                       "receives cannot be named independently")
    return tuple(reasons)


def refusal(error: Exception) -> str:
    """The refusal itself, without Ray's task traceback wrapped around it."""
    plain = re.sub(r"\x1b\[[0-9;]*m", "", str(error))
    lines = [line.strip() for line in plain.splitlines() if line.strip()]
    return next((line for line in reversed(lines)
                 if "did not assign the planned devices" in line), lines[-1])


def run_checks(devices, **timeouts) -> dict:
    """Matched, mismatched, and untrustworthy-ordering runs on real devices."""
    import ray

    node = Node(socket.gethostname().lower(), devices[0].accelerator_type,
                len(devices), devices[0].memory_gb)
    plan = Plan(tuple(node for _ in devices), None)
    planned = DevicePlacement(tuple(item.uuid for item in devices))
    # Rank 0 asks for a device this host does not have; any others stay valid.
    absent = DevicePlacement((ABSENT_UUID,) + planned.uuids[1:])

    # Ray workers inherit this, so the ordering each rank reports is the real one.
    os.environ[DEVICE_ORDER_ENV_VAR] = PCI_BUS_ID
    ray.init(num_cpus=max(2, len(devices)), num_gpus=len(devices),
             include_dashboard=False,
             resources={node.resource_key: len(devices),
                        **device_resources(devices),
                        f"topology_gpu:{ABSENT_UUID}": 1})
    try:
        counter = ray.remote(num_cpus=0)(Counter).remote()

        def worker(rank):
            ray.get(counter.record.remote())
            return {"rank": rank, "cuda": cuda_visible_device()}

        results, assignments = run_with_devices(
            plan, planned, worker, mode=VERIFY, **timeouts)
        checks = {"matched": {
            "assignments": [item.as_dict() for item in assignments],
            "cuda_devices": [item["cuda"] for item in results],
            "cuda_device_is_the_resolved_one": all(
                item["cuda"].get("uuid") == assignment.assigned_uuid
                for item, assignment in zip(results, assignments)),
            "workers_that_ran": ray.get(counter.total.remote())}}

        for name, placement, environ in (
                ("mismatch_refused", absent, None),
                ("untrusted_order_refused", planned, {})):
            ray.get(counter.reset.remote())
            try:
                run_with_devices(plan, placement, worker, mode=VERIFY,
                                 environ=environ, **timeouts)
                checks[name] = {"refused": False}
            except DeviceBindingError as error:
                checks[name] = {"refused": True, "reason": refusal(error),
                                "workers_that_ran": ray.get(counter.total.remote())}
        return checks
    finally:
        ray.shutdown()


def passed(checks: dict, device_count: int) -> bool:
    """Every claim this example makes, checked against what the run produced."""
    matched = checks["matched"]
    return bool(
        matched["cuda_device_is_the_resolved_one"]
        and matched["workers_that_ran"] == device_count
        and all(item["matched"] for item in matched["assignments"])
        # A refused run must also prove the workload never started.
        and all(checks[name]["refused"] and checks[name]["workers_that_ran"] == 0
                for name in ("mismatch_refused", "untrusted_order_refused")))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true",
                        help="start Ray and open a CUDA context on this host")
    parser.add_argument("--reservation-timeout", type=float, default=120)
    parser.add_argument("--execution-timeout", type=float, default=300)
    args = parser.parse_args(argv)

    devices, driver_version = host_devices()
    report = {
        "host": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(), "nvidia_driver": driver_version,
        "devices": [{"uuid": item.uuid, "name": item.name,
                     "pci_bus_id": item.pci_bus_id} for item in devices],
        "performs_cuda_work": False,
    }
    reasons = skip_reasons(opted_in=args.run, devices=devices,
                           has_driver=cuda_driver() is not None)
    if reasons:
        print(json.dumps({**report, "status": "skipped",
                          "skipped_because": list(reasons)}, indent=2))
        return 0

    import ray

    report["ray"] = ray.__version__
    report["checks"] = run_checks(
        devices, reservation_timeout=args.reservation_timeout,
        execution_timeout=args.execution_timeout)
    report["status"] = "passed" if passed(report["checks"], len(devices)) else "failed"
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
