"""Tie a planned GPU identity to the device a worker actually receives.

V1.2 discovers each GPU's UUID and PCI identity, but the task adapter reserves
a node and Ray chooses the device. This module closes the gap it can and names
the gap it cannot:

- Ray's accelerator ids are visibility tokens, not proof of physical identity.
  Even ``PCI_BUS_ID`` does not guarantee that CUDA and NVML indices agree.
  The production path reads the actual CUDA-visible UUID and checks it against
  the NVML reservation candidate and the planned UUID before running work.
- A ``topology_gpu:<uuid>`` custom resource gives Ray's scheduler one unit per
  requested UUID among cooperating callers. These logical tokens do not bind
  physical devices: Ray still assigns the CUDA device separately.

What this provides, therefore, is **verification, not selection**. In
``VERIFY`` mode a worker whose device is not the planned one fails before it
runs, so a placement is honored or refused and never silently moved. Pinning a
process to a chosen device would need
``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES``; see the guide for why that is
deferred.

Importing this module never requires Ray or NVML.
"""

from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Sequence

from .policy import Plan

DEVICE_RESOURCE_PREFIX = "topology_gpu:"
PCI_BUS_ID = "PCI_BUS_ID"
DEVICE_ORDER_ENV_VAR = "CUDA_DEVICE_ORDER"
# Modes: record what happened, or refuse a placement that was not honored.
OBSERVE, VERIFY = "observe", "verify"
MODES = (OBSERVE, VERIFY)


class DeviceBindingError(RuntimeError):
    """Device verification or workload execution failed, with observed identity."""

    def __init__(self, message: str, assignments: Sequence["DeviceAssignment"] = ()):
        super().__init__(message)
        self.assignments = tuple(assignments)

    def as_dict(self) -> dict:
        return {"error": str(self),
                "assignments": [item.as_dict() for item in self.assignments]}


def device_resource_key(uuid: str) -> str:
    """The Ray custom resource that represents one physical GPU."""
    if not isinstance(uuid, str) or not uuid.strip():
        raise ValueError("GPU UUID must be a nonempty string")
    return f"{DEVICE_RESOURCE_PREFIX}{uuid.strip()}"


def device_resources(devices: Iterable) -> dict[str, float]:
    """One unit per device, for a node's ``ray start --resources``.

    These tokens serialize requests for a UUID among cooperating callers.
    They do not control which physical GPU Ray assigns.
    """
    resources: dict[str, float] = {}
    for device in devices:
        key = device_resource_key(getattr(device, "uuid", device))
        if key in resources:
            raise ValueError(f"Duplicate device {key!r} on one node")
        resources[key] = 1
    return resources


@dataclass(frozen=True)
class DevicePlacement:
    """The physical devices a plan asks for, one per rank.

    Kept beside ``Plan`` rather than inside ``Node`` so the planner keeps its
    one-model-per-node schema and a caller can plan without device identity.
    """

    uuids: tuple[str, ...]

    def __post_init__(self):
        if not self.uuids:
            raise ValueError("A device placement needs at least one device")
        for uuid in self.uuids:
            if not isinstance(uuid, str) or not uuid.strip():
                raise ValueError("Every device UUID must be a nonempty string")
        object.__setattr__(self, "uuids", tuple(uuid.strip() for uuid in self.uuids))
        if len(set(self.uuids)) != len(self.uuids):
            raise ValueError("One device cannot serve two ranks")

    def resource_keys(self) -> tuple[str, ...]:
        return tuple(device_resource_key(uuid) for uuid in self.uuids)

    def as_dict(self) -> dict:
        return {"uuids": list(self.uuids), "resources": list(self.resource_keys())}


@dataclass(frozen=True)
class DeviceAssignment:
    """What one rank asked for, what it received, and whether they agree."""

    rank: int
    node_name: str
    requested_uuid: str | None
    assigned_uuid: str | None = None
    assigned_index: int | None = None
    pci_bus_id: str | None = None
    device_order: str | None = None
    problem: str | None = None
    ray_assigned_uuid: str | None = None
    identity_source: str | None = None

    @property
    def matched(self) -> bool:
        return (self.problem is None and self.assigned_uuid is not None
                and self.assigned_uuid == self.requested_uuid)

    def as_dict(self) -> dict:
        return dict(asdict(self), matched=self.matched)

    @classmethod
    def from_dict(cls, value: Mapping) -> "DeviceAssignment":
        fields = {name: value.get(name) for name in cls.__dataclass_fields__}
        return cls(**fields)


def resolve_device(accelerator_ids: Sequence, devices: Sequence, *,
                   device_order: str | None = None, cuda_device: Mapping | None = None) -> dict:
    """Cross-check Ray's reservation candidate against actual CUDA identity.

    NVML indices are matched by their explicit index field, never list position.
    The CUDA observation is mandatory for success; an NVML guess alone cannot
    populate ``assigned_uuid`` or count as verified.
    """
    cuda_device = cuda_device or {}
    actual = cuda_device.get("uuid") if cuda_device.get("available") else None
    resolved = {"assigned_uuid": actual, "device_order": device_order}

    def problem(message):
        return {**resolved, "problem": message}

    if len(accelerator_ids) != 1:
        return problem(
            f"Ray assigned {len(accelerator_ids)} devices to this worker; "
            "device binding requires exactly one GPU per rank")
    token = accelerator_ids[0]
    if type(token) is int or isinstance(token, str) and token.isascii() and token.isdecimal():
        index = int(token)
        matches = [item for item in devices if item.index == index]
    elif isinstance(token, str) and token.startswith("GPU-"):
        matches = [item for item in devices if item.uuid == token]
    else:
        return problem(f"Ray accelerator id {token!r} is not an index or full GPU UUID")
    if len(matches) != 1:
        return problem(f"Ray accelerator id {token!r} has no unique NVML device; "
                       "the raylet and NVML disagree")
    device = matches[0]
    resolved.update(ray_assigned_uuid=device.uuid, assigned_index=device.index)
    actual_devices = [item for item in devices if item.uuid == actual]
    if len(actual_devices) == 1:
        resolved["pci_bus_id"] = actual_devices[0].pci_bus_id
    if not actual or cuda_device.get("device_count") != 1:
        return problem("CUDA identity unavailable: " + cuda_device.get("reason", "exactly one CUDA UUID is required"))
    if actual != device.uuid:
        return problem(f"CUDA reports {actual}, but Ray's NVML reservation candidate is {device.uuid}; "
                       "device numbering or visibility differs")
    if len(actual_devices) != 1:
        return problem("CUDA UUID has no unique NVML identity")
    if device_order != PCI_BUS_ID:
        return problem(
            f"{DEVICE_ORDER_ENV_VAR} is {device_order or 'unset'}; set it to "
            f"{PCI_BUS_ID} as required by this adapter. This setting alone does "
            "not prove agreement with NVML; the CUDA UUID is also checked.")
    return resolved


def assignment_for(rank: int, node_name: str, requested_uuid: str | None,
                   resolved: Mapping) -> DeviceAssignment:
    """Combine a request with what the node resolved into one record."""
    return DeviceAssignment(
        rank=rank, node_name=node_name, requested_uuid=requested_uuid,
        assigned_uuid=resolved.get("assigned_uuid"),
        assigned_index=resolved.get("assigned_index"),
        pci_bus_id=resolved.get("pci_bus_id"),
        device_order=resolved.get("device_order"),
        problem=resolved.get("problem"),
        ray_assigned_uuid=resolved.get("ray_assigned_uuid"),
        identity_source=resolved.get("identity_source"),
    )


def check_assignments(assignments: Sequence[DeviceAssignment], mode: str) -> None:
    """Refuse a placement that was not honored, when the mode asks for it."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    if mode == OBSERVE:
        return
    wrong = [item for item in assignments if not item.matched]
    if wrong:
        raise DeviceBindingError(
            "Ray did not assign the planned devices: " + "; ".join(
                f"rank {item.rank} on {item.node_name} asked for "
                f"{item.requested_uuid} and received "
                f"{item.assigned_uuid or 'nothing'}"
                + (f" ({item.problem})" if item.problem else "")
                for item in wrong),
            assignments)


def bundle_resources(plan: Plan, placement: DevicePlacement) -> list[dict]:
    """One device resource per rank, for the adapter's bundles and tasks."""
    if len(placement.uuids) != len(plan.workers):
        raise ValueError(
            f"Device placement names {len(placement.uuids)} devices for "
            f"{len(plan.workers)} planned workers")
    return [{key: 1} for key in placement.resource_keys()]


def preflight_devices(plan: Plan, placement: DevicePlacement,
                      nodes: Sequence[Mapping]) -> None:
    """Reject a placement whose devices no live node advertises.

    Without this the reservation would simply wait for a resource that will
    never appear and fail as a timeout, which hides the real cause.
    """
    bundle_resources(plan, placement)
    live = [node for node in nodes if node.get("Alive")]
    for rank, (node, key) in enumerate(zip(plan.workers, placement.resource_keys())):
        holders = [item for item in live if item["Resources"].get(key, 0) > 0]
        if (len(holders) != 1 or holders[0]["Resources"].get(key) != 1
                or holders[0]["Resources"].get(node.resource_key, 0) <= 0):
            raise ValueError(
                f"rank {rank}: {key} must be advertised by exactly one live node "
                f"with exactly one unit and marker {node.resource_key}. Start that node with "
                f"--resources including this device, as the guide describes.")


def _node_devices() -> tuple:
    """Read this node's NVML-ordered devices inside a worker."""
    import ray._private.thirdparty.pynvml as pynvml

    from .inventory import _read_nvml_snapshot

    return _read_nvml_snapshot(pynvml)[0]


def observe_worker_device(*, devices_provider: Callable[[], Sequence] | None = None,
                          accelerator_ids: Sequence | None = None,
                          environ: Mapping | None = None, cuda_provider=None) -> dict:
    """Resolve the device this worker received. Runs inside a Ray task."""
    import os

    if accelerator_ids is None:
        import ray

        accelerator_ids = ray.get_gpu_ids()
    environ = os.environ if environ is None else environ
    from .cuda_identity import read_cuda_identity

    source = "injected" if cuda_provider is not None or devices_provider is not None else "cuda_driver"
    cuda_device = {}
    try:
        cuda_device = (cuda_provider or read_cuda_identity)()
        devices = (devices_provider or _node_devices)()
        resolved = resolve_device(accelerator_ids, devices,
                                  device_order=environ.get(DEVICE_ORDER_ENV_VAR), cuda_device=cuda_device)
    except Exception as error:
        resolved = {"problem": f"Device identity query failed: {type(error).__name__}: {error}",
                    "device_order": environ.get(DEVICE_ORDER_ENV_VAR),
                    "assigned_uuid": cuda_device.get("uuid") if cuda_device.get("available") else None}
    return {**resolved, "identity_source": source}


def bind_worker(worker, placement: DevicePlacement, node_names: Sequence[str], *,
                mode: str = VERIFY, devices_provider=None, environ=None, cuda_provider=None):
    """Wrap a worker so it checks its device before doing any work.

    The wrapper returns ``{"device": ..., "result": ...}``; in ``VERIFY`` mode
    it raises before calling the worker, so a wrong device never runs the
    workload at all. ``devices_provider``, ``cuda_provider`` and ``environ``
    let tests and examples inject observations without hardware.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    uuids, names = tuple(placement.uuids), tuple(node_names)

    def bound(rank):
        resolved = observe_worker_device(
            devices_provider=devices_provider, environ=environ, cuda_provider=cuda_provider)
        assignment = assignment_for(rank, names[rank], uuids[rank], resolved)
        check_assignments([assignment], mode)
        try:
            result = worker(rank)
        except Exception as error:
            raise DeviceBindingError(
                f"Worker rank {rank} on {names[rank]} failed after device observation: "
                f"{type(error).__name__}: {error}", [assignment]) from error
        return {"device": assignment.as_dict(), "result": result}

    return bound


def run_with_devices(plan: Plan, placement: DevicePlacement, worker, *,
                     mode: str = VERIFY, devices_provider=None, environ=None, cuda_provider=None,
                     **run_options):
    """Run one worker per planned device and report what each one received.

    Returns ``(results, assignments)`` with results in rank order, exactly as
    the task adapter returns them. In ``VERIFY`` mode a rank that did not get
    its planned device fails the run instead of quietly using another GPU.
    """
    import ray

    from .ray_backend import run

    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    if not ray.is_initialized():
        raise RuntimeError("Call ray.init() before run_with_devices()")
    preflight_devices(plan, placement, ray.nodes())
    payloads = run(plan, bind_worker(
        worker, placement, [node.name for node in plan.workers],
        mode=mode, devices_provider=devices_provider, environ=environ, cuda_provider=cuda_provider),
        extra_resources=bundle_resources(plan, placement), **run_options)
    assignments = tuple(DeviceAssignment.from_dict(item["device"]) for item in payloads)
    check_assignments(assignments, mode)
    return [item["result"] for item in payloads], assignments
