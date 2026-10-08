"""Machine-readable planning, execution, and matched trace comparisons."""

from dataclasses import dataclass, replace
from math import isclose
from time import perf_counter_ns
from typing import Callable, Iterable, Mapping

from .device_binding import (
    MODES, VERIFY, DeviceAssignment, DeviceBindingError, DevicePlacement,
)
from .links import LinkCost, LinkCostSource
from .policy import Node, Plan, PolicyName, Workload, choose_placement


@dataclass(frozen=True)
class PlanningRecord:
    policy: str
    inputs: dict
    chosen_placement: tuple[str, ...]
    estimated_seconds: float | None
    planning_started_ns: int
    planning_finished_ns: int

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "inputs": self.inputs,
            "chosen_placement": list(self.chosen_placement),
            "estimated_seconds": self.estimated_seconds,
            "timing": {
                "planning_started_ns": self.planning_started_ns,
                "planning_finished_ns": self.planning_finished_ns,
            },
        }


@dataclass(frozen=True)
class ExecutionRecord:
    planning: PlanningRecord
    execution_started_ns: int
    execution_finished_ns: int
    status: str
    error: str | None = None
    backend: str = "custom"
    device_mode: str | None = None
    requested_device_uuids: tuple[str, ...] | None = None
    device_assignments: tuple[DeviceAssignment, ...] = ()

    def as_dict(self) -> dict:
        value = self.planning.as_dict()
        value["timing"].update({
            "execution_started_ns": self.execution_started_ns,
            "execution_finished_ns": self.execution_finished_ns,
        })
        value["status"] = self.status
        value["error"] = self.error
        value["backend"] = self.backend
        if self.device_mode is not None:
            value["device_verification"] = {
                "mode": self.device_mode,
                "requested_uuids": (list(self.requested_device_uuids)
                                    if self.requested_device_uuids is not None else None),
                "assignments": [item.as_dict() for item in self.device_assignments],
            }
        return value


class RecordedExecutionError(RuntimeError):
    """Expose a failure record while preserving the backend exception as cause."""

    def __init__(self, record: ExecutionRecord):
        super().__init__(record.error)
        self.record = record


def plan_with_record(
    nodes: list[Node], workload: Workload,
    bandwidth_gbps: Mapping[tuple[str, str], float], *,
    policy: PolicyName | str,
    accelerator_type: str | None = None,
    link_costs: Mapping[tuple[str, str], LinkCost] | None = None,
) -> tuple[Plan, PlanningRecord]:
    """Plan once and capture the inputs, decision, score, and timing boundary.

    ``link_costs``, as returned by ``resolve_link_costs``, records whether each
    bandwidth value was measured, advertised, or a fallback. Without it every
    value is recorded as supplied by the caller.
    """
    _validate_link_costs(bandwidth_gbps, link_costs)
    started = perf_counter_ns()
    plan = choose_placement(
        nodes, workload, bandwidth_gbps, policy=policy,
        accelerator_type=accelerator_type,
    )
    finished = perf_counter_ns()
    return plan, PlanningRecord(
        policy=plan.policy_name,
        inputs=_planning_inputs(nodes, workload, bandwidth_gbps, accelerator_type, link_costs),
        chosen_placement=tuple(node.name for node in plan.workers),
        estimated_seconds=plan.estimated_seconds,
        planning_started_ns=started,
        planning_finished_ns=finished,
    )


def _validate_link_costs(bandwidth_gbps, link_costs):
    if link_costs is not None and (
        set(link_costs) != set(bandwidth_gbps)
        or any(not isclose(link_costs[pair].gb_per_second, value)
               for pair, value in bandwidth_gbps.items())
    ):
        raise ValueError("link_costs must describe exactly the supplied bandwidth values")


def _planning_inputs(nodes, workload, bandwidth_gbps, accelerator_type, link_costs=None):
    return {
        "nodes": [
            {
                "name": node.name,
                "gpu_model": node.gpu_model,
                "available_gpus": node.available_gpus,
                "memory_gb_per_gpu": node.memory_gb_per_gpu,
            }
            for node in nodes
        ],
        "workload": {
            "workers": workload.workers,
            "memory_gb_per_worker": workload.memory_gb_per_worker,
            "compute_seconds_by_gpu": dict(workload.compute_seconds_by_gpu),
            "cross_node_gb_per_pair": workload.cross_node_gb_per_pair,
        },
        "bandwidth_gbps": {
            f"{left}|{right}": value
            for (left, right), value in sorted(bandwidth_gbps.items())
        },
        "bandwidth_sources": {
            f"{left}|{right}": (
                link_costs[(left, right)].provenance() if link_costs is not None
                else {"source": LinkCostSource.SUPPLIED.value, "measured_at": None}
            )
            for left, right in sorted(bandwidth_gbps)
        },
        "accelerator_type": accelerator_type,
    }


def run_with_record(plan: Plan, planning: PlanningRecord, worker, *, backend=None,
                    device_placement: DevicePlacement | Callable[[Plan], DevicePlacement] | None = None,
                    device_mode: str = VERIFY,
                    **backend_options):
    """Execute any policy through the same backend and record terminal status.

    Ray callers can opt into device verification with a fixed ``DevicePlacement``
    or a callable resolving one from each plan. Success and failure records keep
    the request and any available observations. The callable is invoked inside
    the execution boundary; invalid or missing mappings fail without running
    work. Device-level relationships still do not influence planning.
    """
    if plan.policy_name != planning.policy:
        raise ValueError("Plan and planning record policies must match")
    if device_placement is not None:
        if backend is not None and backend != "ray":
            raise ValueError("device_placement requires the Ray backend")
        if device_mode not in MODES:
            raise ValueError(f"device_mode must be one of: {', '.join(MODES)}")
        if not isinstance(device_placement, DevicePlacement) and not callable(device_placement):
            raise ValueError("device_placement must be a DevicePlacement or a callable")
    elif device_mode != VERIFY:
        raise ValueError("device_mode requires device_placement")
    if backend is None or backend == "ray":
        backend_name = "ray"
        from .ray_backend import run as backend
    elif backend == "kai":
        backend_name = "kai"
        from .kai_backend import run as backend
    elif isinstance(backend, str):
        raise ValueError("backend must be 'ray', 'kai', or a callable")
    else:
        backend_name = getattr(backend, "__name__", "custom")
    started = perf_counter_ns()
    placement = None
    assignments = ()

    def terminal(status, error=None):
        return ExecutionRecord(
            planning, started, perf_counter_ns(), status, error, backend_name,
            device_mode=device_mode if device_placement is not None else None,
            requested_device_uuids=placement.uuids if placement is not None else None,
            device_assignments=tuple(assignments),
        )

    try:
        if device_placement is None:
            results = backend(plan, worker, **backend_options)
        else:
            from .device_binding import run_with_devices

            selected = device_placement(plan) if callable(device_placement) else device_placement
            if not isinstance(selected, DevicePlacement):
                raise ValueError("device_placement callable must return a DevicePlacement")
            placement = selected
            results, assignments = run_with_devices(
                plan, placement, worker, mode=device_mode, **backend_options)
    except Exception as error:
        if device_placement is not None and isinstance(error, DeviceBindingError):
            assignments = error.assignments
        record = terminal("failed", f"{type(error).__name__}: {error}")
        raise RecordedExecutionError(record) from error
    return results, terminal("succeeded")


@dataclass(frozen=True)
class TraceJob:
    """A stable job identifier, workload profile, and shared rank callable."""

    job_id: str
    workload: Workload
    worker: Callable[[int], object]


@dataclass(frozen=True)
class TraceRecord:
    """One terminal job attempt; failed attempts remain in the comparison."""

    job_id: str
    planning: PlanningRecord
    execution: ExecutionRecord | None
    submitted_ns: int
    terminal_ns: int
    error: str | None = None
    baseline_jct_ns: int | None = None
    normalization_reason: str | None = None

    @property
    def status(self) -> str:
        return self.execution.status if self.execution else "failed"

    @property
    def jct_ns(self) -> int | None:
        return self.terminal_ns - self.submitted_ns if self.status == "succeeded" else None

    @property
    def normalized_jct(self) -> float | None:
        if self.jct_ns is None or not self.baseline_jct_ns or self.baseline_jct_ns < 0:
            return None
        return self.jct_ns / self.baseline_jct_ns

    def as_dict(self) -> dict:
        value = self.execution.as_dict() if self.execution else self.planning.as_dict()
        value.update({
            "job_id": self.job_id,
            "status": self.status,
            "error": self.execution.error if self.execution else self.error,
            "failure_stage": (None if self.status == "succeeded" else
                              "execution" if self.execution else "planning"),
            "jct_ns": self.jct_ns,
            "normalization": {
                "baseline_policy": PolicyName.GPU_COUNT.value,
                "baseline_job_id": self.job_id,
                "baseline_jct_ns": self.baseline_jct_ns,
                "normalized_jct": self.normalized_jct,
                "unavailable_reason": self.normalization_reason,
            },
        })
        value["timing"].update({
            "submitted_ns": self.submitted_ns,
            "terminal_ns": self.terminal_ns,
            "elapsed_ns": self.terminal_ns - self.submitted_ns,
        })
        return value


def run_matched_trace(
    nodes: list[Node], jobs: Iterable[TraceJob],
    bandwidth_gbps: Mapping[tuple[str, str], float], *,
    accelerator_type: str, backend=None,
    link_costs: Mapping[tuple[str, str], LinkCost] | None = None,
    **backend_options,
) -> list[TraceRecord]:
    """Replay one ordered, serial trace for each of the five reference policies.

    Each job uses the same worker and backend options under every policy. Ray is
    the default backend. Submission precedes planning; terminal follows backend
    return (including cleanup). Planning/execution failures are retained and the
    trace continues without retries. Interrupts still propagate. Callers own
    cluster isolation, warmup, workload seeds, and experimental metadata.

    Optional ``link_costs`` records the provenance of every bandwidth value.
    Both maps are copied and checked before any work runs, then held fixed for
    all policies, including planning/execution failure records. Resolve report
    freshness before starting the trace; this function does not run probes or
    refresh costs between jobs.

    ``device_placement`` and ``device_mode`` are forwarded to ``run_with_record``.
    Use the same deterministic placement callable for every policy when their
    plans name different nodes. Device selection/verification time is part of
    execution and JCT; requests and available observations survive trace export.
    """
    jobs = tuple(jobs)
    if not jobs:
        raise ValueError("Provide a nonempty job trace")
    if any(not isinstance(job.job_id, str) or not job.job_id.strip() for job in jobs):
        raise ValueError("Trace job IDs must be nonempty strings")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Trace job IDs must be unique")
    if any(not callable(job.worker) for job in jobs):
        raise ValueError("Trace workers must be callable")
    if not isinstance(accelerator_type, str) or not accelerator_type.strip():
        raise ValueError("Provide an accelerator_type for the accelerator baseline")
    if backend is not None and not callable(backend) and backend != "ray":
        raise ValueError("Trace backend must be 'ray' or a callable accepting rank workers")

    nodes, bandwidth_gbps = list(nodes), dict(bandwidth_gbps)
    link_costs = None if link_costs is None else dict(link_costs)
    _validate_link_costs(bandwidth_gbps, link_costs)
    jobs = tuple(replace(job, workload=replace(
        job.workload, compute_seconds_by_gpu=dict(job.workload.compute_seconds_by_gpu)
    )) for job in jobs)
    records = []
    for policy in PolicyName:
        constraint = accelerator_type if policy is PolicyName.ACCELERATOR_TYPE else None
        for job in jobs:
            submitted = perf_counter_ns()
            try:
                plan, planning = plan_with_record(
                    nodes, job.workload, bandwidth_gbps,
                    policy=policy, accelerator_type=constraint, link_costs=link_costs,
                )
            except Exception as error:
                terminal = perf_counter_ns()
                planning = PlanningRecord(
                    policy.value,
                    _planning_inputs(nodes, job.workload, bandwidth_gbps, constraint, link_costs),
                    (), None, submitted, terminal,
                )
                records.append(TraceRecord(
                    job.job_id, planning, None, submitted, terminal,
                    f"{type(error).__name__}: {error}",
                ))
                continue
            try:
                _, execution = run_with_record(
                    plan, planning, job.worker, backend=backend, **backend_options,
                )
            except RecordedExecutionError as error:
                execution = error.record
            records.append(TraceRecord(
                job.job_id, planning, execution, submitted, perf_counter_ns(),
            ))

    baselines = {record.job_id: record for record in records
                 if record.planning.policy == PolicyName.GPU_COUNT.value}
    normalized = []
    for record in records:
        baseline = baselines[record.job_id]
        reason = None
        if record.status != "succeeded":
            reason = "job_failed"
        elif baseline.status != "succeeded":
            reason = "baseline_failed"
        elif baseline.jct_ns <= 0:
            reason = "nonpositive_baseline_jct"
        normalized.append(replace(
            record, baseline_jct_ns=baseline.jct_ns, normalization_reason=reason,
        ))
    return normalized
