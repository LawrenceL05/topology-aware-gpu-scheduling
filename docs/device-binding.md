# Device identity binding

See [current implementation, validation status, and versions](current-status.md)
for the shared support summary and evidence boundaries.

[V1.2 discovery](v1.2-topology-discovery.md) records each GPU's UUID, PCI
ancestry and NVLink counts, but the task adapter reserves a *node* and Ray
chooses the device. Every device-level observation was therefore unusable for
placement: scoring on an NVLink edge would promise something the execution
layer does not deliver.

[device_binding.py](../topology_scheduler/device_binding.py) closes the part of
that gap which can be closed, and states precisely the part that cannot. It
provides **verification, not selection**: a rank that did not receive its
planned device fails before it runs, so a placement is honored or refused and
never silently moved.

## Why an index is not an identity

Ray's accelerator IDs are visibility tokens, not physical identity evidence.
Ray 2.55.0's [worker code](https://github.com/ray-project/ray/blob/ray-2.55.0/python/ray/_private/worker.py)
preserves original visibility tokens when configured; they may be numeric IDs
or GPU UUIDs. Its [NVIDIA accelerator manager](https://github.com/ray-project/ray/blob/ray-2.55.0/python/ray/_private/accelerators/nvidia_gpu.py)
sets `CUDA_VISIBLE_DEVICES` and does not set `CUDA_DEVICE_ORDER`.

[NVIDIA's NVML documentation](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)
states that NVML indices need not correlate with CUDA indices. Setting
`PCI_BUS_ID` alone therefore cannot prove agreement. The production verifier
uses [CUDA driver device queries](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html)
to require exactly one visible device and read its UUID through `cuDeviceGetUuid`.
It compares that observation with the requested UUID and the NVML candidate
for Ray's token. A disagreement or unavailable observation refuses execution.
Numeric tokens select an explicit NVML `index`, never a list position; full
UUID tokens require an exact match. Ambiguous identities are rejected.

The adapter retains its explicit ordering requirement as a configuration
precondition, in addition to the driver check. Start every worker with:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

The verifier does not set visibility, select a different GPU, or launch a CUDA
kernel. It loads the installed driver lazily through `ctypes`; no CUDA toolkit
or extra Python package is required. CUDA/NVML errors become refusal records.
Numeric mappings that cannot be cross-checked are refused conservatively;
this is not support for arbitrary container remapping, MIG, or fractional GPUs.

## What Ray can and cannot guarantee

| Option | What it gives | Why it is or is not used |
| --- | --- | --- |
| Per-device custom resource `topology_gpu:<uuid>` | Serializes requests for a UUID among cooperating callers | **Used.** A logical token only; GPU allocation is separately managed by Ray and need not select that UUID |
| Resolve and verify on the node | The rank learns its real UUID and refuses a mismatch | **Used.** This is the guarantee the adapter offers |
| `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` plus adapter-set visibility | True selection: the process sees exactly the chosen device | **Deferred.** It is explicitly experimental and moves device accounting out of Ray, so any workload on the cluster that does not follow the same discipline could claim the same GPU. That precondition needs a maintainer decision before it is safe |
| One GPU per node | The mapping is trivial | Supported as a degenerate case; it is not a general answer |

The reservation contains the requested UUID tokens and generic GPUs. These are
separate resources, so it does not guarantee even the requested physical set.
Each rank checks its own actual identity before running. This is a per-rank
check, not an all-rank startup barrier: a correctly assigned rank may begin
before another rank refuses. The [smoke example](../examples/ray_device_binding_smoke.py)
uses two simulated GPUs and separately injects CUDA/NVML disagreement to prove
that the mismatched workers do not enter their bodies.

## Advertise the devices

Each node advertises one resource per physical GPU alongside its node marker:

```python
from topology_scheduler import device_resources, discover_ray_gpu_inventory

for inventory in discover_ray_gpu_inventory():
    print(inventory.node_name, device_resources(inventory.devices))
```

Use the result in that node's `ray start --resources`, together with its
existing `topology_node:<name>` marker. Without it the reservation would wait
for a resource that never appears; `run_with_devices()` checks first and says
which device is missing from which node.

Each requested UUID must have exactly one unit on exactly one live node with
the intended topology marker. Duplicate advertisements or fractional/multiple
units are refused. Optional per-rank resources cannot override CPU, GPU,
memory, or topology-node reservations.

## Run with device identity

```python
import ray
from topology_scheduler import DevicePlacement, run_with_devices

ray.init(address="auto")
placement = DevicePlacement(("GPU-aaa", "GPU-bbb"))   # one device per rank
results, assignments = run_with_devices(plan, placement, worker)
for item in assignments:
    print(item.rank, item.requested_uuid, item.assigned_uuid, item.matched)
```

`DevicePlacement` sits beside `Plan` rather than inside `Node`, so the planner
keeps its one-model-per-node schema and a caller can still plan without naming
devices. One device cannot serve two ranks.

Modes:

- `verify` (default) — a rank that did not receive its planned device raises
  before the workload runs. The message names the rank, its node, the requested
  and received UUIDs, and the reason.
- `observe` — the assignment is recorded and the workload runs anyway. Use it
  to measure how often Ray's pairing matches before relying on verification.

Each `DeviceAssignment` records rank, node, requested UUID, actual CUDA-observed
`assigned_uuid`, NVML candidate `ray_assigned_uuid`, candidate index, actual
device PCI address when available, `CUDA_DEVICE_ORDER`, and any problem.
`identity_source` distinguishes `cuda_driver` from explicitly injected test
providers. A missing CUDA observation never becomes an assigned UUID just
because NVML returned a candidate. Records serialize to plain dictionaries;
older dictionaries without the new fields remain readable.

A verification failure happens inside the Ray task, so it reaches the driver
wrapped in Ray's task error; the underlying message is the one above.
`DeviceBindingError.assignments` retains the refusing rank's observation.
Workload exceptions are also wrapped with that rank's identity and preserve
the original cause. A failed run does not promise a complete record for every
other rank, particularly tasks canceled or lost before they report.

## Run it without a GPU

```bash
python -m unittest tests.test_device_binding tests.test_cuda_identity -v
python -m examples.ray_device_binding_smoke
```

The tests cover resolution, ordering, mismatches, preflight, both modes, and
the one-GPU-per-node case with constructed devices. The smoke runs the real
reservation and task path on one local Ray node with **simulated** logical GPUs
and stand-in NVML/CUDA identities. Driver-query tests cover missing libraries,
driver errors, wrong visible-device counts, UUID decoding, and index mismatch
even with `PCI_BUS_ID`; reservation tests cover duplicate UUIDs and resource
overrides. These fixtures do not establish physical device assignment.

## Physical evidence

[`device_binding_gpu_check.py`](../examples/device_binding_gpu_check.py) covers
the step the smoke cannot. It is opt-in and never runs in CI; without `--run`
or without a GPU it prints why it skipped and exits 0.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python -m examples.device_binding_gpu_check --run
```

Each worker asks the **CUDA driver** which device it would compute on, through
`ctypes`, so no CUDA toolkit or PyTorch has to be installed. The check then
compares that answer with the UUID this layer resolved. That comparison is the
point: not that two NVML reads agree with each other, but that the device the
process actually holds is the planned one.

Historical run on 2026-09-20 (before the production CUDA UUID verifier) — one NVIDIA GeForce RTX 5070 Laptop GPU, driver
610.74, Ray 2.55.0, Python 3.12.10, Windows 11:

| Check | Result |
| --- | --- |
| Rank asks for the device the plan names | Resolved to `GPU-aef71c8d-…` through index 0 at `00000000:02:00.0`, ordering `PCI_BUS_ID`, matched |
| CUDA context the worker opened | The driver named the same UUID this layer resolved |
| Rank asks for a device the host does not have | Refused, and **no worker body ran** |
| Same real device, ordering not `PCI_BUS_ID` | Refused, and **no worker body ran** |

The refusals are counted, not assumed: a counter actor records every worker
body that executes, and it stayed at zero in both.

What that run does **not** establish:

- **One GPU.** The degenerate case cannot show two ranks on one node each
  receiving their own planned device. Only the simulated smoke covers that, and
  only with stand-in identities.
- **No numerical work.** A CUDA context is created and destroyed. It names the
  device; it does not show a kernel running on it, and it is not a benchmark.
- **Windows on a consumer laptop GPU**, not the Linux amd64 hosts this project
  targets elsewhere.
- The updated production CUDA UUID verification path needs its own hardware
  run; this historical report is not reclassified as evidence for new code.

## Limits

- This is verification, not selection. Ray still decides which device a task
  receives; the adapter only refuses a result it cannot vouch for.
- The physical evidence above covers a single GPU on one Windows host. Multiple
  devices per node, Linux hosts, and any numerical workload running on the
  verified device are still unproven.
- Placement scoring still does not consume device-level relationships. Until
  selection exists, an NVLink pair can be observed and verified after the fact,
  not requested, so scoring on it would still overstate the guarantee.
- Multi-GPU ranks, MIG, and fractional devices are out of scope; each rank
  holds exactly one device.
