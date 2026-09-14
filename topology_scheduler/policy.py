"""Small-cluster exhaustive node placement, with explicit estimated costs."""

from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from math import comb, isfinite
from typing import Mapping


def positive(value: float, name: str, *, zero: bool = False) -> None:
    if not isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


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
    estimated_seconds: float


def choose_placement(nodes: list[Node], workload: Workload,
                     bandwidth_gbps: Mapping[tuple[str, str], float],
                     *, max_candidates: int = 100_000) -> Plan:
    """Minimize slowest-rank compute + serialized cross-node pair traffic.

    Bandwidth is GB/s (not bits/s). Missing links make communicating placements
    infeasible. The symmetric link map uses lexically sorted node-name pairs.
    Local communication must already be included in the compute estimates.
    """
    if len({n.name for n in nodes}) != len(nodes):
        raise ValueError("Node names must be unique")
    for (a, b), value in bandwidth_gbps.items():
        if a >= b:
            raise ValueError("Bandwidth keys must be sorted distinct node-name pairs")
        positive(value, "bandwidth")
    eligible = sorted((n for n in nodes if n.available_gpus > 0
                       and n.memory_gb_per_gpu >= workload.memory_gb_per_worker
                       and n.gpu_model in workload.compute_seconds_by_gpu),
                      key=lambda n: n.name)
    if sum(n.available_gpus for n in eligible) < workload.workers:
        raise ValueError("Insufficient compatible GPU capacity")
    count = comb(len(eligible) + workload.workers - 1, workload.workers)
    if count > max_candidates:
        raise ValueError(f"Search requires {count} candidates; limit is {max_candidates}")
    best = None
    for indices in combinations_with_replacement(range(len(eligible)), workload.workers):
        if any(indices.count(i) > eligible[i].available_gpus for i in set(indices)):
            continue
        workers = tuple(eligible[i] for i in indices)
        cost = max(workload.compute_seconds_by_gpu[n.gpu_model] for n in workers)
        for a, b in combinations(workers, 2):
            if a.name != b.name and workload.cross_node_gb_per_pair:
                bandwidth = bandwidth_gbps.get(tuple(sorted((a.name, b.name))))
                if bandwidth is None:
                    cost = float("inf")
                    break
                cost += workload.cross_node_gb_per_pair / bandwidth
        if isfinite(cost) and (best is None or cost < best.estimated_seconds):
            best = Plan(workers, cost)
    if best is None:
        raise ValueError("No feasible placement with known communication links")
    return best
