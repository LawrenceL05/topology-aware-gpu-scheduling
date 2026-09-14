"""Real Ray scheduling with simulated logical GPUs; performs no CUDA work."""
import os
import ray
from topology_scheduler import Node, Workload, choose_placement
from topology_scheduler.ray_backend import run


def probe(rank):
    import ray
    return {"rank": rank, "node_id": ray.get_runtime_context().get_node_id(),
            "gpu_ids": ray.get_gpu_ids(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def main():
    ray.init(num_cpus=2, num_gpus=2, include_dashboard=False,
             resources={"topology_node:local": 2})
    try:
        plan = choose_placement([Node("local", "SIMULATED", 2, 80)],
                                Workload(2, 1, {"SIMULATED": 1}), {})
        results = run(plan, probe)
        assert [r["rank"] for r in results] == [0, 1]
        assert len({r["node_id"] for r in results}) == 1
        assert all(len(r["gpu_ids"]) == 1 for r in results)
        assert len({str(r["gpu_ids"][0]) for r in results}) == 2
        print(results)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
