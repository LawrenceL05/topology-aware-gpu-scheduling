"""Two local Ray nodes with simulated GPUs; tests cross-node placement only."""
import ray
from ray.cluster_utils import Cluster
from topology_scheduler import Node, Workload, choose_placement
from topology_scheduler.ray_backend import run
from examples.ray_smoke import probe


def main():
    # Cluster is a Ray testing utility, not the production deployment API.
    cluster = Cluster()
    try:
        for name in ("a", "b"):
            cluster.add_node(num_cpus=1, num_gpus=1, include_dashboard=False,
                             resources={f"topology_node:{name}": 1})
        ray.init(address=cluster.address)
        node_ids = {name: next(n["NodeID"] for n in ray.nodes()
                              if n["Alive"] and f"topology_node:{name}" in n["Resources"])
                    for name in ("a", "b")}
        plan = choose_placement([Node(n, "SIMULATED", 1, 80) for n in ("a", "b")],
                                Workload(2, 1, {"SIMULATED": 1}, 1), {("a", "b"): 10})
        results = run(plan, probe)
        assert [r["node_id"] for r in results] == [node_ids["a"], node_ids["b"]]
        assert all(len(r["gpu_ids"]) == 1 for r in results)
        print(results)
    finally:
        ray.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    main()
