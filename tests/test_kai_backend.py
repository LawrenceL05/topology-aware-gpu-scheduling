import unittest
from types import SimpleNamespace as Namespace

from topology_scheduler import Node, Plan
from topology_scheduler.kai_backend import (
    ClusterNode, KAIWorkload, KubernetesKAIClient, PodState, build_kai_objects,
    cancel, preflight, run, status, submit, validate_submission,
)


class FakeKAIClient:
    def __init__(self, nodes, snapshots=()):
        self.nodes = nodes
        self.snapshots = list(snapshots)
        self.created = []
        self.deleted = []
        self.permissions = []
        self.fail_pod = None

    def cluster_nodes(self):
        return self.nodes

    def queue_exists(self, queue):
        return queue == "gpu-team"

    def can_create(self, resource, *, group="", namespace=None):
        self.permissions.append((resource, group, namespace))
        return True

    def create_pod_group(self, namespace, body):
        self.created.append(("PodGroup", body["metadata"]["name"]))

    def create_pod(self, namespace, body):
        name = body["metadata"]["name"]
        if name == self.fail_pod:
            raise RuntimeError("pod create failed")
        self.created.append(("Pod", name))

    def list_pods(self, namespace, selector):
        if self.snapshots:
            return self.snapshots.pop(0)
        return []

    def delete_pod(self, namespace, name):
        self.deleted.append(("Pod", name))

    def delete_pod_group(self, namespace, name):
        self.deleted.append(("PodGroup", name))


class KAIBackendTests(unittest.TestCase):
    def setUp(self):
        self.plan = Plan(
            (Node("gpu-a", "H100", 2, 80), Node("gpu-b", "H100", 2, 80)),
            12,
            "topology_only",
        )
        self.workload = KAIWorkload(
            name="demo", namespace="research", queue="gpu-team",
            image="example/worker:1", command=("python", "worker.py"),
            node_pool="gpu-nodes",
        )
        self.nodes = [
            ClusterNode("node-a", {
                "kubernetes.io/hostname": "gpu-a",
                "kai.scheduler/node-pool": "gpu-nodes",
            }, 2),
            ClusterNode("node-b", {
                "kubernetes.io/hostname": "gpu-b",
                "kai.scheduler/node-pool": "gpu-nodes",
            }, 2),
        ]

    def test_builds_external_gang_and_one_gpu_pod_per_worker(self):
        group, first, second = build_kai_objects(self.plan, self.workload)
        self.assertEqual(group["kind"], "PodGroup")
        self.assertEqual(group["spec"]["minMember"], 2)
        self.assertEqual(group["spec"]["queue"], "gpu-team")
        self.assertEqual(first["spec"]["schedulerName"], "kai-scheduler")
        self.assertEqual(first["spec"]["nodeSelector"], {
            "kubernetes.io/hostname": "gpu-a"
        })
        self.assertEqual(second["spec"]["nodeSelector"], {
            "kubernetes.io/hostname": "gpu-b"
        })
        for pod in (first, second):
            self.assertEqual(
                pod["spec"]["containers"][0]["resources"]["limits"],
                {"nvidia.com/gpu": "1"},
            )
            self.assertEqual(pod["metadata"]["annotations"], {
                "pod-group-name": "demo",
                "kai.scheduler/skip-podgrouper": "true",
            })

    def test_validates_cluster_contract(self):
        validate_submission(
            self.plan, self.workload, self.nodes,
            queue_exists=True, can_create_pods=True,
            can_create_podgroups=True,
        )

    def test_missing_queue_and_permissions_are_actionable(self):
        with self.assertRaisesRegex(ValueError, "queue .* does not exist"):
            validate_submission(
                self.plan, self.workload, self.nodes,
                queue_exists=False, can_create_pods=True,
                can_create_podgroups=True,
            )
        with self.assertRaisesRegex(PermissionError, "pods.*podgroups"):
            validate_submission(
                self.plan, self.workload, self.nodes,
                queue_exists=True, can_create_pods=False,
                can_create_podgroups=False,
            )

    def test_missing_node_label_gpu_or_pool_is_actionable(self):
        cases = [
            ([], "No Kubernetes node"),
            ([ClusterNode("a", {"kubernetes.io/hostname": "gpu-a",
                                "kai.scheduler/node-pool": "gpu-nodes"}, 0)],
             "exposes 0"),
            ([ClusterNode("a", {"kubernetes.io/hostname": "gpu-a",
                                "kai.scheduler/node-pool": "other"}, 2)],
             "not in KAI node pool"),
        ]
        one_worker = Plan((self.plan.workers[0],), 1)
        for nodes, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_submission(
                    one_worker, self.workload, nodes,
                    queue_exists=True, can_create_pods=True,
                    can_create_podgroups=True,
                )

    def test_rejects_invalid_workload_names(self):
        with self.assertRaisesRegex(ValueError, "DNS label"):
            KAIWorkload(
                name="Not_Valid", namespace="research", queue="gpu-team",
                image="worker", command=("run",),
            )

    def test_preflight_reads_cluster_state_and_namespace_permissions(self):
        client = FakeKAIClient(self.nodes)
        preflight(self.plan, self.workload, client)
        self.assertEqual(client.permissions, [
            ("pods", "", "research"),
            ("podgroups", "scheduling.run.ai", "research"),
        ])

    def test_submit_orders_group_before_pods(self):
        client = FakeKAIClient(self.nodes, [[
            PodState("demo-0", "Pending"), PodState("demo-1", "Pending")
        ]])
        observed = submit(self.plan, self.workload, client)
        self.assertEqual(observed.phase, "pending")
        self.assertEqual(client.created, [
            ("PodGroup", "demo"), ("Pod", "demo-0"), ("Pod", "demo-1")
        ])

    def test_partial_submit_rolls_back_pods_then_group(self):
        client = FakeKAIClient(self.nodes)
        client.fail_pod = "demo-1"
        with self.assertRaisesRegex(RuntimeError, "pod create failed"):
            submit(self.plan, self.workload, client)
        self.assertEqual(client.deleted, [
            ("Pod", "demo-0"), ("PodGroup", "demo")
        ])

    def test_status_collapses_pod_phases(self):
        cases = [
            ([], "pending"),
            ([PodState("demo-0", "Running")], "running"),
            ([PodState("demo-0", "Succeeded")], "succeeded"),
            ([PodState("demo-0", "Failed", "OOMKilled")], "failed"),
        ]
        for pods, expected in cases:
            with self.subTest(expected=expected):
                client = FakeKAIClient(self.nodes, [pods])
                self.assertEqual(status(self.workload, client).phase, expected)

        client = FakeKAIClient(
            self.nodes, [[PodState("demo-0", "Succeeded")]]
        )
        self.assertEqual(
            status(self.workload, client, expected_workers=2).phase, "running"
        )

    def test_run_waits_and_cleans_up_on_success(self):
        pending = [PodState("demo-0", "Pending"), PodState("demo-1", "Pending")]
        running = [PodState("demo-0", "Running"), PodState("demo-1", "Running")]
        succeeded = [
            PodState("demo-0", "Succeeded", node_name="node-a"),
            PodState("demo-1", "Succeeded", node_name="node-b"),
        ]
        client = FakeKAIClient(self.nodes, [pending, running, succeeded, succeeded])
        ticks = iter((0, 0, 1, 2, 3))
        result = run(
            self.plan, self.workload, client=client, timeout_s=10,
            poll_interval_s=0, clock=lambda: next(ticks), sleeper=lambda _: None,
        )
        self.assertEqual(result.phase, "succeeded")
        self.assertEqual(client.deleted, [
            ("Pod", "demo-1"), ("Pod", "demo-0"), ("PodGroup", "demo")
        ])


class KubernetesKAIClientTests(unittest.TestCase):
    def setUp(self):
        self.plan = Plan(
            (Node("gpu-a", "H100", 2, 80), Node("gpu-b", "H100", 2, 80)),
            12,
            "topology_only",
        )
        self.workload = KAIWorkload(
            name="demo", namespace="research", queue="gpu-team",
            image="example/worker:1", command=("python", "worker.py"),
            node_pool="gpu-nodes",
        )
        self.nodes = [
            ClusterNode("node-a", {
                "kubernetes.io/hostname": "gpu-a",
                "kai.scheduler/node-pool": "gpu-nodes",
            }, 2),
            ClusterNode("node-b", {
                "kubernetes.io/hostname": "gpu-b",
                "kai.scheduler/node-pool": "gpu-nodes",
            }, 2),
        ]

    def test_maps_official_client_objects_and_api_calls(self):
        class Core:
            def __init__(self):
                self.created = []
                self.deleted = []

            def list_node(self):
                return Namespace(items=[Namespace(
                    metadata=Namespace(
                        name="node-a", labels={"kubernetes.io/hostname": "gpu-a"}
                    ),
                    status=Namespace(allocatable={"nvidia.com/gpu": "2"}),
                )])

            def create_namespaced_pod(self, *, namespace, body):
                self.created.append((namespace, body["metadata"]["name"]))

            def list_namespaced_pod(self, *, namespace, label_selector):
                return Namespace(items=[Namespace(
                    metadata=Namespace(name="demo-0"),
                    status=Namespace(phase="Running", reason=None, message=None),
                    spec=Namespace(node_name="node-a"),
                )])

            def delete_namespaced_pod(self, *, name, namespace):
                self.deleted.append((namespace, name))

        class Custom:
            def __init__(self):
                self.created = []
                self.deleted = []

            def get_cluster_custom_object(self, **kwargs):
                return {"metadata": {"name": kwargs["name"]}}

            def create_namespaced_custom_object(self, **kwargs):
                self.created.append(kwargs)

            def delete_namespaced_custom_object(self, **kwargs):
                self.deleted.append(kwargs)

        class Authorization:
            def __init__(self):
                self.reviews = []

            def create_self_subject_access_review(self, body):
                self.reviews.append(body)
                return Namespace(status=Namespace(allowed=True))

        core, custom, authorization = Core(), Custom(), Authorization()
        client = KubernetesKAIClient(core, custom, authorization)
        self.assertEqual(client.cluster_nodes()[0].allocatable_gpus, 2)
        self.assertTrue(client.queue_exists("gpu-team"))
        self.assertTrue(client.can_create("pods", namespace="research"))
        attributes = authorization.reviews[0]["spec"]["resourceAttributes"]
        self.assertEqual(attributes["namespace"], "research")

        client.create_pod_group("research", {
            "metadata": {"name": "demo"}
        })
        client.create_pod("research", {"metadata": {"name": "demo-0"}})
        pods = client.list_pods("research", "app.kubernetes.io/name=demo")
        self.assertEqual(pods, [PodState("demo-0", "Running", node_name="node-a")])
        client.delete_pod("research", "demo-0")
        client.delete_pod_group("research", "demo")
        self.assertEqual(core.deleted, [("research", "demo-0")])
        self.assertEqual(custom.deleted[0]["plural"], "podgroups")

    def test_run_failure_and_timeout_are_actionable_and_cleaned_up(self):
        failed = [
            PodState("demo-0", "Failed", "OOMKilled"),
            PodState("demo-1", "Pending"),
        ]
        client = FakeKAIClient(self.nodes, [failed, failed])
        with self.assertRaisesRegex(RuntimeError, "demo-0: OOMKilled"):
            run(self.plan, self.workload, client=client)
        self.assertEqual(client.deleted[-1], ("PodGroup", "demo"))

        pending = [PodState("demo-0", "Pending"), PodState("demo-1", "Pending")]
        client = FakeKAIClient(self.nodes, [pending, pending, pending])
        ticks = iter((0, 0, 2, 3))
        with self.assertRaisesRegex(TimeoutError, "did not finish"):
            run(
                self.plan, self.workload, client=client, timeout_s=1,
                poll_interval_s=0, clock=lambda: next(ticks), sleeper=lambda _: None,
            )
        self.assertEqual(client.deleted[-1], ("PodGroup", "demo"))

    def test_cancel_is_idempotent_at_the_client_boundary(self):
        pods = [PodState("demo-0", "Running"), PodState("demo-1", "Pending")]
        client = FakeKAIClient(self.nodes, [pods])
        cancel(self.workload, client)
        self.assertEqual(client.deleted, [
            ("Pod", "demo-1"), ("Pod", "demo-0"), ("PodGroup", "demo")
        ])


if __name__ == "__main__":
    unittest.main()
