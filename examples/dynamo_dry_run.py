"""Print the plan, replica configuration, and launch intent, running nothing.

This is a dry run: no Ray cluster, no Dynamo, no vLLM, no CUDA, and no network.
It shows exactly what a start would do so the configuration can be reviewed
before any GPU is involved. The inventory and workload below are synthetic.
"""
import json
import shlex

from topology_scheduler import Node, Workload, choose_placement
from topology_scheduler.dynamo_backend import (
    ADAPTER_OWNED, CALLER_OWNED, DynamoConfig,
)

# Synthetic inventory; a real run uses discover_planner_nodes(). One GPU each,
# so the two contract replicas land on separate nodes.
NODES = [Node("a", "H100", 1, 80), Node("b", "H100", 1, 80)]


def describe_launch(plan, config):
    """Describe what a start would do, without reserving or launching anything."""
    command = shlex.join(config.worker_command())
    return {
        "performs_inference": False,
        "policy": plan.policy_name,
        "placement": [node.name for node in plan.workers],
        "model": {"id": config.model, "revision": config.revision,
                  "max_model_length": config.max_model_length,
                  "tensor_parallel_size": config.tensor_parallel_size,
                  "gpu_memory_utilization": config.gpu_memory_utilization},
        "frontend_endpoint": config.frontend_url,
        "ownership": {"adapter": list(ADAPTER_OWNED), "caller": list(CALLER_OWNED)},
        "replicas": [
            {
                "rank": rank,
                "node_name": node.name,
                "resource_key": node.resource_key,
                "system_port": config.system_port_base + rank,
                "log_dir": config.log_dir,
                "command_line": command,
            }
            for rank, node in enumerate(plan.workers)
        ],
    }


def main():
    config = DynamoConfig.from_contract()
    # Independent replicas exchange nothing, so cross-node traffic stays zero.
    workload = Workload(len(NODES), 40, {"H100": 10}, 0)
    plan = choose_placement(NODES, workload, {})
    print(json.dumps({
        "synthetic_inputs": True,
        "estimated_seconds_is_a_placement_score": plan.estimated_seconds,
        **describe_launch(plan, config),
    }, indent=2))


if __name__ == "__main__":
    main()
