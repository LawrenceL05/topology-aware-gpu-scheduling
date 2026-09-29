"""Device binding on real Ray with simulated logical GPUs and device identities.

One local Ray node advertises two **simulated** GPUs and one
`topology_gpu:<uuid>` resource per device. The worker's device identity is
supplied by stand-ins instead of NVML and the CUDA driver, so this runs anywhere; no CUDA work
happens and no physical GPU is touched.

The two ranks consume the node's two logical GPUs; custom UUID resources
serialize their requests but do not control Ray's GPU pairing.
It does **not** make Ray hand a particular device to a particular rank, which
is why the default mode verifies the assignment rather than assuming it.
"""
import json
import re

import ray
from ray.cluster_utils import Cluster

from topology_scheduler import GPUDevice, Node, Plan
from topology_scheduler.device_binding import (
    DEVICE_ORDER_ENV_VAR, OBSERVE, PCI_BUS_ID, VERIFY, DeviceBindingError,
    DevicePlacement, device_resources, run_with_devices,
)

# Stand-in identities for the two simulated GPUs on the single node.
DEVICES = (
    GPUDevice(0, "GPU-sim-0", "SIMULATED", "SIMULATED", 80, "0000:00:00.0"),
    GPUDevice(1, "GPU-sim-1", "SIMULATED", "SIMULATED", 80, "0000:01:00.0"),
)
PLACEMENT = DevicePlacement(tuple(item.uuid for item in DEVICES))
SAFE_ENV = {DEVICE_ORDER_ENV_VAR: PCI_BUS_ID}


def worker(rank):
    return {"rank": rank}


def simulated_cuda():
    index = int(ray.get_gpu_ids()[0])
    return {"available": True, "device_count": 1, "uuid": DEVICES[index].uuid}


def swapped_cuda():
    index = int(ray.get_gpu_ids()[0])
    return {"available": True, "device_count": 1, "uuid": DEVICES[1 - index].uuid}


class Counter:
    def __init__(self):
        self.calls = 0

    def record(self):
        self.calls += 1

    def total(self):
        return self.calls


def main():
    cluster = Cluster()
    try:
        cluster.add_node(num_cpus=2, num_gpus=2, include_dashboard=False,
                         resources={"topology_node:a": 2, **device_resources(DEVICES)})
        ray.init(address=cluster.address)
        plan = Plan(tuple(Node("a", "SIMULATED", 2, 80) for _ in DEVICES), 1.0)

        results, assignments = run_with_devices(
            plan, PLACEMENT, worker, mode=OBSERVE,
            devices_provider=lambda: DEVICES, cuda_provider=simulated_cuda, environ=SAFE_ENV)

        assert [item["rank"] for item in results] == [0, 1], results
        assigned = sorted(item.assigned_uuid for item in assignments)
        # Ray assigned both logical GPUs in this simulated two-GPU reservation.
        assert assigned == sorted(PLACEMENT.uuids), assigned
        assert all(item.problem is None for item in assignments), assignments
        assert all(item.identity_source == "injected" for item in assignments)
        paired = all(item.matched for item in assignments)

        # Verification refuses a run whose device identity cannot be trusted:
        # without PCI_BUS_ID, CUDA and NVML may number devices differently.
        refused = None
        try:
            run_with_devices(plan, PLACEMENT, worker, mode=VERIFY,
                             devices_provider=lambda: DEVICES, cuda_provider=simulated_cuda, environ={})
        except DeviceBindingError as error:
            # Ray wraps a worker-side failure in its own task error, so keep
            # the line that explains the refusal rather than the traceback.
            plain = re.sub(r"\x1b\[[0-9;]*m", "", str(error))
            refused = next((line.strip() for line in reversed(plain.splitlines())
                            if DEVICE_ORDER_ENV_VAR in line), plain[-300:])
        assert refused and DEVICE_ORDER_ENV_VAR in refused, refused

        # Even PCI_BUS_ID is insufficient when CUDA and NVML disagree. Both
        # ranks must refuse before entering their workload bodies.
        counter = ray.remote(num_cpus=0)(Counter).remote()
        def counted_worker(rank):
            ray.get(counter.record.remote())
            return rank
        mismatch = None
        try:
            run_with_devices(plan, PLACEMENT, counted_worker, mode=VERIFY,
                devices_provider=lambda: DEVICES, cuda_provider=swapped_cuda, environ=SAFE_ENV)
        except DeviceBindingError as error:
            mismatch = str(error)
            assert error.assignments and error.assignments[0].assigned_uuid
            assert "numbering or visibility differs" in mismatch
        assert mismatch is not None
        calls = ray.get(counter.total.remote())
        assert calls == 0, calls

        print(json.dumps({
            "simulated_logical_gpus": True, "simulated_device_identities": True,
            "performs_cuda_work": False,
            "assignments": [item.as_dict() for item in assignments],
            "every_planned_device_used_once": assigned == sorted(PLACEMENT.uuids),
            "ray_happened_to_match_each_rank": paired,
            "unsafe_device_order_refused": refused,
            "cuda_nvml_disagreement_refused": mismatch is not None,
            "mismatched_worker_bodies_run": calls,
        }, indent=2))
    finally:
        ray.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    main()
