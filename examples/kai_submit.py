"""Submit a GPU-count baseline to a live KAI Scheduler cluster."""

import argparse
import json

from topology_scheduler import (
    KAIWorkload, KubernetesKAIClient, Node, PolicyName, Workload,
    plan_with_record, run_with_record,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="topology-kai-smoke")
    parser.add_argument("--namespace", default="research")
    parser.add_argument("--queue", default="default-queue")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--image", default="nvcr.io/nvidia/cuda:12.8.1-base-ubuntu24.04"
    )
    parser.add_argument("--node-pool")
    parser.add_argument("--keep", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    client = KubernetesKAIClient.from_environment()
    inventory = client.cluster_nodes()
    nodes = [
        Node(
            item.labels.get("kubernetes.io/hostname", item.name),
            item.labels.get("nvidia.com/gpu.product", "kubernetes-gpu"),
            item.allocatable_gpus,
            float(item.labels.get("nvidia.com/gpu.memory", 1024)) / 1024,
        )
        for item in inventory if item.allocatable_gpus
    ]
    if not nodes:
        raise RuntimeError("Kubernetes reports no nodes with allocatable nvidia.com/gpu")
    workload = Workload(
        workers=args.workers, memory_gb_per_worker=0,
        compute_seconds_by_gpu={node.gpu_model: 1 for node in nodes},
    )
    plan, planning = plan_with_record(
        nodes, workload, {}, policy=PolicyName.GPU_COUNT,
    )
    kai = KAIWorkload(
        name=args.name, namespace=args.namespace, queue=args.queue,
        image=args.image, command=("nvidia-smi",), node_pool=args.node_pool,
    )
    result, execution = run_with_record(
        plan, planning, kai, backend="kai", client=client,
        cleanup_on_finish=not args.keep,
    )
    print(json.dumps({
        "planning": planning.as_dict(),
        "execution": execution.as_dict(),
        "kai": result.as_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
