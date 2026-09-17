"""Experimental placement policy; importing it does not require Ray."""

from .policy import Node, Plan, PolicyName, Workload, choose_placement
from .comparison import (
    ExecutionRecord, PlanningRecord, RecordedExecutionError, plan_with_record,
    run_with_record,
)
from .inventory import (
    GPUConnection, GPUDevice, RayNodeInventory, discover_planner_nodes,
    discover_ray_gpu_inventory,
)
from .kai_backend import (
    ClusterNode, KAIStatus, KAIWorkload, KubernetesKAIClient, PodState,
    build_kai_objects, cancel, preflight, run as run_kai, status, submit,
    validate_submission,
)
from .host_topology import (
    GPULocality, HostNIC, HostTopology, NICProximity, Proximity, SysfsReader,
    collect_host_topology, discover_host_topology, normalize_pci_address,
)

__all__ = [
    "ClusterNode", "ExecutionRecord", "GPUConnection", "GPUDevice",
    "GPULocality", "HostNIC", "HostTopology", "KAIStatus",
    "KAIWorkload", "KubernetesKAIClient", "NICProximity", "PodState",
    "Node", "Plan", "Proximity", "SysfsReader",
    "PlanningRecord", "PolicyName", "RayNodeInventory", "RecordedExecutionError",
    "Workload", "build_kai_objects", "cancel", "choose_placement",
    "collect_host_topology", "discover_host_topology", "normalize_pci_address",
    "discover_planner_nodes", "preflight", "run_kai", "status", "submit",
    "discover_ray_gpu_inventory", "plan_with_record", "run_with_record",
    "validate_submission",
]
