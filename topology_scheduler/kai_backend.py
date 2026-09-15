"""Translate a backend-neutral placement plan into Kubernetes/KAI objects."""

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

from .policy import Plan


KAI_VERSION = "0.17.0"
KUBERNETES_VERSION = "1.34"
GPU_OPERATOR_VERSION = "25.10"
QUEUE_LABEL = "kai.scheduler/queue"
NODE_POOL_LABEL = "kai.scheduler/node-pool"
POD_GROUP_ANNOTATION = "pod-group-name"
SKIP_GROUPER_ANNOTATION = "kai.scheduler/skip-podgrouper"

_DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def _dns_label(value: str, field: str) -> None:
    if not value or len(value) > 63 or not _DNS_LABEL.fullmatch(value):
        raise ValueError(f"{field} must be a Kubernetes DNS label")


@dataclass(frozen=True)
class KAIWorkload:
    """KAI execution settings kept separate from placement-policy inputs."""

    name: str
    namespace: str
    queue: str
    image: str
    command: tuple[str, ...]
    node_label: str = "kubernetes.io/hostname"
    priority_class: str = "inference"
    node_pool: str | None = None

    def __post_init__(self):
        for value, field in (
            (self.name, "name"), (self.namespace, "namespace"), (self.queue, "queue")
        ):
            _dns_label(value, field)
        if not self.image:
            raise ValueError("image must be nonempty")
        if not self.command:
            raise ValueError("command must be nonempty")
        if not self.node_label:
            raise ValueError("node_label must be nonempty")
        if self.node_pool is not None:
            _dns_label(self.node_pool, "node_pool")


@dataclass(frozen=True)
class ClusterNode:
    """Kubernetes-visible resources used by the pre-submission check."""

    name: str
    labels: Mapping[str, str]
    allocatable_gpus: int


def validate_submission(
    plan: Plan, workload: KAIWorkload, cluster_nodes: Sequence[ClusterNode], *,
    queue_exists: bool, can_create_pods: bool, can_create_podgroups: bool,
) -> None:
    """Reject missing labels, GPUs, queue, or RBAC before creating objects."""
    if not plan.workers:
        raise ValueError("Plan must contain workers")
    if not queue_exists:
        raise ValueError(f"KAI queue {workload.queue!r} does not exist")
    missing_permissions = []
    if not can_create_pods:
        missing_permissions.append("pods")
    if not can_create_podgroups:
        missing_permissions.append("podgroups.scheduling.run.ai")
    if missing_permissions:
        raise PermissionError(
            "Missing create permission for " + ", ".join(missing_permissions)
        )

    by_label = {}
    for node in cluster_nodes:
        target = node.labels.get(workload.node_label)
        if target:
            if target in by_label:
                raise ValueError(
                    f"Node label {workload.node_label}={target!r} is not unique"
                )
            by_label[target] = node
    required = {}
    for worker in plan.workers:
        required[worker.name] = required.get(worker.name, 0) + 1
    for name, gpu_count in sorted(required.items()):
        node = by_label.get(name)
        if node is None:
            raise ValueError(
                f"No Kubernetes node has {workload.node_label}={name!r}"
            )
        if node.allocatable_gpus < gpu_count:
            raise ValueError(
                f"Node {name!r} exposes {node.allocatable_gpus} nvidia.com/gpu; "
                f"plan requires {gpu_count}"
            )
        if workload.node_pool and node.labels.get(NODE_POOL_LABEL) != workload.node_pool:
            raise ValueError(
                f"Node {name!r} is not in KAI node pool {workload.node_pool!r}"
            )


def build_kai_objects(plan: Plan, workload: KAIWorkload) -> list[dict]:
    """Return one external PodGroup followed by one node-pinned Pod per rank."""
    if not plan.workers:
        raise ValueError("Plan must contain workers")
    labels = {QUEUE_LABEL: workload.queue}
    if workload.node_pool:
        labels[NODE_POOL_LABEL] = workload.node_pool
    pod_group = {
        "apiVersion": "scheduling.run.ai/v2alpha2",
        "kind": "PodGroup",
        "metadata": {
            "name": workload.name,
            "namespace": workload.namespace,
            "labels": labels,
        },
        "spec": {
            "minMember": len(plan.workers),
            "queue": workload.queue,
            "priorityClassName": workload.priority_class,
        },
    }
    objects = [pod_group]
    for rank, node in enumerate(plan.workers):
        pod_labels = dict(labels)
        pod_labels.update({
            "app.kubernetes.io/name": workload.name,
            "topology-scheduler/rank": str(rank),
            "topology-scheduler/policy": plan.policy_name,
        })
        objects.append({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": f"{workload.name}-{rank}",
                "namespace": workload.namespace,
                "labels": pod_labels,
                "annotations": {
                    POD_GROUP_ANNOTATION: workload.name,
                    SKIP_GROUPER_ANNOTATION: "true",
                },
            },
            "spec": {
                "schedulerName": "kai-scheduler",
                "restartPolicy": "Never",
                "nodeSelector": {workload.node_label: node.name},
                "containers": [{
                    "name": "worker",
                    "image": workload.image,
                    "command": list(workload.command),
                    "env": [
                        {"name": "RANK", "value": str(rank)},
                        {"name": "WORLD_SIZE", "value": str(len(plan.workers))},
                    ],
                    "resources": {
                        "requests": {"nvidia.com/gpu": "1"},
                        "limits": {"nvidia.com/gpu": "1"},
                    },
                }],
            },
        })
    return objects
