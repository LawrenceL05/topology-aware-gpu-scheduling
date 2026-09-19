"""Matched traces on two real Ray nodes with simulated GPUs; NOT a benchmark."""

import json

import ray
from ray.cluster_utils import Cluster

from topology_scheduler import Node, PolicyName, TraceJob, Workload, run_matched_trace


def probe(rank, *, node_ids):
    assert ray.get_runtime_context().get_node_id() == node_ids[rank]
    assert len(ray.get_gpu_ids()) == 1
    return rank


def fail(rank):
    raise RuntimeError("intentional matched-trace failure")


def main():
    cluster = Cluster()
    try:
        for name in ("a", "b"):
            cluster.add_node(num_cpus=1, num_gpus=1, include_dashboard=False,
                             resources={f"topology_node:{name}": 1})
        ray.init(address=cluster.address, log_to_driver=False)
        node_ids = tuple(next(n["NodeID"] for n in ray.nodes()
                              if n["Alive"] and f"topology_node:{name}" in n["Resources"])
                         for name in ("a", "b"))
        def worker(rank):
            return probe(rank, node_ids=node_ids)

        workload = Workload(2, 1, {"SIMULATED": 1}, 1)
        records = run_matched_trace(
            [Node(name, "SIMULATED", 1, 80) for name in ("a", "b")],
            [TraceJob("before-failure", workload, worker),
             TraceJob("intentional-failure", workload, fail),
             TraceJob("after-failure", workload, worker)],
            {("a", "b"): 10}, accelerator_type="SIMULATED",
            reservation_timeout=30, execution_timeout=30,
        )
        assert len(records) == len(PolicyName) * 3
        for before, failed, after in zip(records[::3], records[1::3], records[2::3]):
            assert before.status == after.status == "succeeded", (
                before.as_dict(), after.as_dict(),
            )
            assert failed.status == "failed"
            assert failed.execution is not None
            assert "intentional matched-trace failure" in failed.execution.error
            assert failed.normalized_jct is None
            assert before.normalized_jct is not None and after.normalized_jct is not None
        print(json.dumps({
            "synthetic_inputs": True,
            "simulated_gpus": True,
            "benchmark_evidence": False,
            "execution_mode": "serial_policy_then_job",
            "clock": "driver_perf_counter_ns",
            "backend_options": {"reservation_timeout": 30, "execution_timeout": 30},
            "records": [record.as_dict() for record in records],
        }, indent=2))
    finally:
        ray.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    main()
