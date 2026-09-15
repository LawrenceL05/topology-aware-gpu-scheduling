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

__all__ = [
    "ExecutionRecord", "GPUConnection", "GPUDevice", "Node", "Plan",
    "PlanningRecord", "PolicyName", "RayNodeInventory", "RecordedExecutionError",
    "Workload", "choose_placement", "discover_planner_nodes",
    "discover_ray_gpu_inventory", "plan_with_record", "run_with_record",
]
