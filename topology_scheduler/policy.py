"""Deterministic reference policies for small-cluster node placement."""

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, combinations_with_replacement
from math import comb, isfinite
from typing import Mapping


def positive(value: float, name: str, *, zero: bool = False) -> None:
    if not isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


class PolicyName(str, Enum):
    """Stable names written to comparison results."""

    GPU_COUNT = "gpu_count"
    ACCELERATOR_TYPE = "accelerator_type"
    WORKLOAD_COMPUTE = "workload_compute"
    TOPOLOGY_ONLY = "topology_only"
    COMBINED = "combined"


@dataclass(frozen=True)
class Node:
    name: str
    gpu_model: str
    available_gpus: int
    memory_gb_per_gpu: float

    def __post_init__(self):
        if not self.name or not self.gpu_model:
            raise ValueError("Node name and GPU model must be nonempty")
        if type(self.available_gpus) is not int or self.available_gpus < 0:
            raise ValueError("available_gpus must be a nonnegative integer")
        positive(self.memory_gb_per_gpu, "memory_gb_per_gpu")

    @property
    def resource_key(self):
        return f"topology_node:{self.name}"


@dataclass(frozen=True)
class Workload:
    workers: int
    memory_gb_per_worker: float
    compute_seconds_by_gpu: Mapping[str, float]
    cross_node_gb_per_pair: float = 0

    def __post_init__(self):
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        positive(self.memory_gb_per_worker, "memory_gb_per_worker", zero=True)
        positive(self.cross_node_gb_per_pair, "cross_node_gb_per_pair", zero=True)
        if not self.compute_seconds_by_gpu:
            raise ValueError("Provide workload-specific compute estimates")
        for value in self.compute_seconds_by_gpu.values():
            positive(value, "compute estimate")


@dataclass(frozen=True)
class Plan:
    workers: tuple[Node, ...]
    estimated_seconds: float | None
    policy_name: str = PolicyName.COMBINED.value

    def as_dict(self) -> dict:
        return {
            "policy": self.policy_name,
            "placement": [node.name for node in self.workers],
            "estimated_seconds": self.estimated_seconds,
        }


def _policy(value: PolicyName | str) -> PolicyName:
    try:
        return PolicyName(value)
    except ValueError as error:
        names = ", ".join(item.value for item in PolicyName)
        raise ValueError(f"Unknown policy {value!r}; choose one of: {names}") from error


def _communication_cost(
    workers: tuple[Node, ...], workload: Workload,
    bandwidth_gbps: Mapping[tuple[str, str], float],
) -> float:
    cost = 0.0
    for left, right in combinations(workers, 2):
        if left.name == right.name or not workload.cross_node_gb_per_pair:
            continue
        bandwidth = bandwidth_gbps.get(tuple(sorted((left.name, right.name))))
        if bandwidth is None:
            return float("inf")
        cost += workload.cross_node_gb_per_pair / bandwidth
    return cost


def choose_placement(
    nodes: list[Node], workload: Workload,
    bandwidth_gbps: Mapping[tuple[str, str], float], *,
    policy: PolicyName | str = PolicyName.COMBINED,
    accelerator_type: str | None = None,
    max_candidates: int = 100_000,
) -> Plan:
    """Choose a placement with one of five deterministic reference policies.

    Every policy applies GPU capacity and per-GPU memory feasibility. Candidate
    placements and ties are ordered lexicographically by node name. Bandwidth is
    GB/s and uses sorted node-name pairs. Missing links make a communicating
    candidate infeasible only for policies that use topology.
    """
    selected_policy = _policy(policy)
    if len({node.name for node in nodes}) != len(nodes):
        raise ValueError("Node names must be unique")
    for (left, right), value in bandwidth_gbps.items():
        if left >= right:
            raise ValueError("Bandwidth keys must be sorted distinct node-name pairs")
        positive(value, "bandwidth")
    if selected_policy is PolicyName.ACCELERATOR_TYPE and not accelerator_type:
        raise ValueError("accelerator_type policy requires accelerator_type")

    needs_compute = selected_policy in (PolicyName.WORKLOAD_COMPUTE, PolicyName.COMBINED)
    eligible = sorted(
        (
            node for node in nodes
            if node.available_gpus > 0
            and node.memory_gb_per_gpu >= workload.memory_gb_per_worker
            and (not needs_compute or node.gpu_model in workload.compute_seconds_by_gpu)
            and (selected_policy is not PolicyName.ACCELERATOR_TYPE
                 or node.gpu_model == accelerator_type)
        ),
        key=lambda node: node.name,
    )
    if sum(node.available_gpus for node in eligible) < workload.workers:
        raise ValueError("Insufficient compatible GPU capacity")
    count = comb(len(eligible) + workload.workers - 1, workload.workers)
    if count > max_candidates:
        raise ValueError(f"Search requires {count} candidates; limit is {max_candidates}")

    scored: list[tuple[float, tuple[str, ...], Plan]] = []
    for indices in combinations_with_replacement(range(len(eligible)), workload.workers):
        if any(indices.count(index) > eligible[index].available_gpus for index in set(indices)):
            continue
        workers = tuple(eligible[index] for index in indices)
        estimate: float | None = None
        if selected_policy is PolicyName.WORKLOAD_COMPUTE:
            estimate = max(workload.compute_seconds_by_gpu[node.gpu_model] for node in workers)
        elif selected_policy is PolicyName.TOPOLOGY_ONLY:
            estimate = _communication_cost(workers, workload, bandwidth_gbps)
        elif selected_policy is PolicyName.COMBINED:
            estimate = (
                max(workload.compute_seconds_by_gpu[node.gpu_model] for node in workers)
                + _communication_cost(workers, workload, bandwidth_gbps)
            )
        if estimate is not None and not isfinite(estimate):
            continue
        tie_break = tuple(node.name for node in workers)
        scored.append((estimate if estimate is not None else 0.0, tie_break,
                       Plan(workers, estimate, selected_policy.value)))
    if not scored:
        if selected_policy in (PolicyName.TOPOLOGY_ONLY, PolicyName.COMBINED):
            raise ValueError("No feasible placement with known communication links")
        raise ValueError("No feasible placement")
    return min(scored, key=lambda item: (item[0], item[1]))[2]
