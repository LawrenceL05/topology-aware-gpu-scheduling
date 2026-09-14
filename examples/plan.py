"""Run with python -m examples.plan; numbers are synthetic, not benchmarks."""
from topology_scheduler import Node, Workload, choose_placement

nodes = [Node("a", "H100", 2, 80), Node("b", "B200", 2, 180)]
workload = Workload(2, 40, {"H100": 10, "B200": 7}, 20)
plan = choose_placement(nodes, workload, {("a", "b"): 25})
print({"nodes": [n.name for n in plan.workers],
       "estimated_seconds": plan.estimated_seconds,
       "synthetic_inputs": True})
