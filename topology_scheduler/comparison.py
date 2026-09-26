"""Machine-readable planning records for controlled policy comparisons."""

from dataclasses import dataclass
from math import isclose
from time import perf_counter_ns
from typing import Mapping

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

    def as_dict(self) -> dict:
        value = self.planning.as_dict()
        value["timing"].update({
            "execution_started_ns": self.execution_started_ns,
            "execution_finished_ns": self.execution_finished_ns,
        })
        value["status"] = self.status
        value["error"] = self.error
        value["backend"] = self.backend
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
    if link_costs is not None and (
        set(link_costs) != set(bandwidth_gbps)
        or any(not isclose(link_costs[pair].gb_per_second, value)
               for pair, value in bandwidth_gbps.items())
    ):
        raise ValueError("link_costs must describe exactly the supplied bandwidth values")
    started = perf_counter_ns()
    plan = choose_placement(
        nodes, workload, bandwidth_gbps, policy=policy,
        accelerator_type=accelerator_type,
    )
    finished = perf_counter_ns()
    inputs = {
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
    return plan, PlanningRecord(
        policy=plan.policy_name,
        inputs=inputs,
        chosen_placement=tuple(node.name for node in plan.workers),
        estimated_seconds=plan.estimated_seconds,
        planning_started_ns=started,
        planning_finished_ns=finished,
    )


def run_with_record(plan: Plan, planning: PlanningRecord, worker, *, backend=None,
                    **backend_options):
    """Execute any policy through the same backend and record terminal status."""
    if plan.policy_name != planning.policy:
        raise ValueError("Plan and planning record policies must match")
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
    try:
        results = backend(plan, worker, **backend_options)
    except Exception as error:
        record = ExecutionRecord(
            planning, started, perf_counter_ns(), "failed",
            f"{type(error).__name__}: {error}", backend_name,
        )
        raise RecordedExecutionError(record) from error
    return results, ExecutionRecord(
        planning, started, perf_counter_ns(), "succeeded", backend=backend_name,
    )
