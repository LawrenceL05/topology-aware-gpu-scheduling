"""Experimental placement policy; importing it does not require Ray."""

from .policy import Node, Plan, Workload, choose_placement
from .inventory import (
    GPUConnection, GPUDevice, RayNodeInventory, discover_planner_nodes,
    discover_ray_gpu_inventory,
)

__all__ = [
    "GPUConnection", "GPUDevice", "Node", "Plan", "RayNodeInventory", "Workload",
    "choose_placement", "discover_planner_nodes", "discover_ray_gpu_inventory",
]
