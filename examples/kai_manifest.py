"""Print synthetic KAI objects as JSON; this does not submit a workload."""

import json

from topology_scheduler import (
    KAIWorkload, Node, PolicyName, Workload, build_kai_objects, choose_placement,
)


nodes = [Node("gpu-a", "H100", 1, 80), Node("gpu-b", "H100", 1, 80)]
workload = Workload(2, 40, {"H100": 10}, 20)
plan = choose_placement(
    nodes, workload, {("gpu-a", "gpu-b"): 100},
    policy=PolicyName.COMBINED,
)
kai = KAIWorkload(
    name="topology-demo",
    namespace="research",
    queue="gpu-team",
    image="example/worker:1",
    command=("python", "worker.py"),
    node_pool="gpu-nodes",
)
print(json.dumps({"synthetic_inputs": True, "objects": build_kai_objects(plan, kai)}, indent=2))
