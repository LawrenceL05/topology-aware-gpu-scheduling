"""Translate a backend-neutral placement plan into Kubernetes/KAI objects."""

from dataclasses import dataclass
import re
from time import monotonic, sleep
from typing import Mapping, Protocol, Sequence

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


@dataclass(frozen=True)
class PodState:
    """Small, backend-neutral view of a Kubernetes worker Pod."""

    name: str
    phase: str
    reason: str | None = None
    message: str | None = None
    node_name: str | None = None


@dataclass(frozen=True)
class KAIStatus:
    """Observed state of one externally managed KAI workload."""

    phase: str
    pods: tuple[PodState, ...]

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "pods": [
                {
                    "name": pod.name,
                    "phase": pod.phase,
                    "reason": pod.reason,
                    "message": pod.message,
                    "node_name": pod.node_name,
                }
                for pod in self.pods
            ],
        }


class KAIClient(Protocol):
    """Cluster operations needed by the KAI lifecycle."""

    def cluster_nodes(self) -> Sequence[ClusterNode]: ...
    def queue_exists(self, queue: str) -> bool: ...
    def can_create(
        self, resource: str, *, group: str = "", namespace: str | None = None
    ) -> bool: ...
    def create_pod_group(self, namespace: str, body: dict) -> None: ...
    def create_pod(self, namespace: str, body: dict) -> None: ...
    def list_pods(self, namespace: str, selector: str) -> Sequence[PodState]: ...
    def delete_pod(self, namespace: str, name: str) -> None: ...
    def delete_pod_group(self, namespace: str, name: str) -> None: ...


class KubernetesKAIClient:
    """Thin adapter over the official Kubernetes Python client APIs."""

    def __init__(self, core_api, custom_api, authorization_api):
        self.core_api = core_api
        self.custom_api = custom_api
        self.authorization_api = authorization_api

    @classmethod
    def from_environment(cls, *, in_cluster: bool | None = None):
        try:
            from kubernetes import client, config
        except ImportError as error:
            raise RuntimeError(
                "Install the Kubernetes client with `pip install -e .[kai]`"
            ) from error
        if in_cluster is True:
            config.load_incluster_config()
        elif in_cluster is False:
            config.load_kube_config()
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
        return cls(
            client.CoreV1Api(), client.CustomObjectsApi(),
            client.AuthorizationV1Api(),
        )

    def cluster_nodes(self) -> Sequence[ClusterNode]:
        nodes = []
        for item in self.core_api.list_node().items:
            raw_gpus = (item.status.allocatable or {}).get("nvidia.com/gpu", 0)
            nodes.append(ClusterNode(
                item.metadata.name,
                dict(item.metadata.labels or {}),
                int(raw_gpus),
            ))
        return nodes

    def queue_exists(self, queue: str) -> bool:
        try:
            self.custom_api.get_cluster_custom_object(
                group="scheduling.run.ai", version="v2", plural="queues",
                name=queue,
            )
        except Exception as error:
            if getattr(error, "status", None) == 404:
                return False
            raise
        return True

    def can_create(
        self, resource: str, *, group: str = "", namespace: str | None = None
    ) -> bool:
        attributes = {"verb": "create", "group": group, "resource": resource}
        if namespace:
            attributes["namespace"] = namespace
        review = self.authorization_api.create_self_subject_access_review({
            "apiVersion": "authorization.k8s.io/v1",
            "kind": "SelfSubjectAccessReview",
            "spec": {"resourceAttributes": attributes},
        })
        return bool(review.status.allowed)

    def create_pod_group(self, namespace: str, body: dict) -> None:
        self.custom_api.create_namespaced_custom_object(
            group="scheduling.run.ai", version="v2alpha2", namespace=namespace,
            plural="podgroups", body=body,
        )

    def create_pod(self, namespace: str, body: dict) -> None:
        self.core_api.create_namespaced_pod(namespace=namespace, body=body)

    def list_pods(self, namespace: str, selector: str) -> Sequence[PodState]:
        result = self.core_api.list_namespaced_pod(
            namespace=namespace, label_selector=selector,
        )
        return [
            PodState(
                item.metadata.name,
                item.status.phase or "Unknown",
                item.status.reason,
                item.status.message,
                item.spec.node_name,
            )
            for item in result.items
        ]

    def delete_pod(self, namespace: str, name: str) -> None:
        try:
            self.core_api.delete_namespaced_pod(name=name, namespace=namespace)
        except Exception as error:
            if getattr(error, "status", None) != 404:
                raise

    def delete_pod_group(self, namespace: str, name: str) -> None:
        try:
            self.custom_api.delete_namespaced_custom_object(
                group="scheduling.run.ai", version="v2alpha2",
                namespace=namespace, plural="podgroups", name=name,
            )
        except Exception as error:
            if getattr(error, "status", None) != 404:
                raise


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


def preflight(plan: Plan, workload: KAIWorkload, client: KAIClient) -> None:
    """Read live cluster state and validate the complete submission contract."""
    validate_submission(
        plan, workload, client.cluster_nodes(),
        queue_exists=client.queue_exists(workload.queue),
        can_create_pods=client.can_create("pods", namespace=workload.namespace),
        can_create_podgroups=client.can_create(
            "podgroups", group="scheduling.run.ai",
            namespace=workload.namespace,
        ),
    )


def submit(plan: Plan, workload: KAIWorkload, client: KAIClient) -> KAIStatus:
    """Validate and create the PodGroup before its worker Pods."""
    preflight(plan, workload, client)
    pod_group, *pods = build_kai_objects(plan, workload)
    created_pods = []
    client.create_pod_group(workload.namespace, pod_group)
    try:
        for pod in pods:
            client.create_pod(workload.namespace, pod)
            created_pods.append(pod["metadata"]["name"])
        return status(workload, client, expected_workers=len(plan.workers))
    except Exception:
        for name in reversed(created_pods):
            client.delete_pod(workload.namespace, name)
        client.delete_pod_group(workload.namespace, workload.name)
        raise


def status(
    workload: KAIWorkload, client: KAIClient, *, expected_workers: int | None = None
) -> KAIStatus:
    """Collapse worker Pod phases into one workload phase."""
    pods = tuple(client.list_pods(
        workload.namespace, f"app.kubernetes.io/name={workload.name}"
    ))
    phases = {pod.phase for pod in pods}
    if "Failed" in phases:
        phase = "failed"
    elif (
        pods and phases == {"Succeeded"}
        and (expected_workers is None or len(pods) == expected_workers)
    ):
        phase = "succeeded"
    elif "Running" in phases or "Succeeded" in phases:
        phase = "running"
    else:
        phase = "pending"
    return KAIStatus(phase, pods)


def cancel(
    workload: KAIWorkload, client: KAIClient, *, worker_count: int | None = None
) -> None:
    """Delete worker Pods first, then the externally managed PodGroup."""
    if worker_count is None:
        pod_names = [
            pod.name for pod in client.list_pods(
                workload.namespace, f"app.kubernetes.io/name={workload.name}"
            )
        ]
    else:
        pod_names = [f"{workload.name}-{rank}" for rank in range(worker_count)]
    for name in reversed(pod_names):
        client.delete_pod(workload.namespace, name)
    client.delete_pod_group(workload.namespace, workload.name)


cleanup = cancel


def run(
    plan: Plan, workload: KAIWorkload, *, client: KAIClient,
    timeout_s: float = 600, poll_interval_s: float = 1,
    cleanup_on_finish: bool = True, clock=monotonic, sleeper=sleep,
) -> KAIStatus:
    """Submit, wait for a terminal phase, and clean up external objects."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if poll_interval_s < 0:
        raise ValueError("poll_interval_s cannot be negative")
    submitted = False
    try:
        current = submit(plan, workload, client)
        submitted = True
        deadline = clock() + timeout_s
        while current.phase not in ("succeeded", "failed"):
            if clock() >= deadline:
                raise TimeoutError(
                    f"KAI workload {workload.name!r} did not finish in {timeout_s}s"
                )
            sleeper(poll_interval_s)
            current = status(
                workload, client, expected_workers=len(plan.workers)
            )
        if current.phase == "failed":
            failed = [
                f"{pod.name}: {pod.reason or pod.message or pod.phase}"
                for pod in current.pods if pod.phase == "Failed"
            ]
            raise RuntimeError("KAI workload failed: " + "; ".join(failed))
        return current
    finally:
        if submitted and cleanup_on_finish:
            cancel(workload, client, worker_count=len(plan.workers))
