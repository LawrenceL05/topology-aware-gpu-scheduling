import unittest

from topology_scheduler import Node, Plan
from topology_scheduler.kai_backend import (
    ClusterNode, KAIWorkload, build_kai_objects, validate_submission,
)


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


if __name__ == "__main__":
    unittest.main()
