import unittest
from topology_scheduler import Node, Workload, choose_placement


class PolicyTests(unittest.TestCase):
    def test_workload_changes_hardware_choice(self):
        nodes = [Node("a", "A", 2, 80), Node("b", "B", 2, 80)]
        for estimates, expected in [({"A": 4, "B": 8}, "a"), ({"A": 8, "B": 4}, "b")]:
            plan = choose_placement(nodes, Workload(2, 40, estimates), {})
            self.assertEqual([n.name for n in plan.workers], [expected, expected])

    def test_topology_changes_selected_pair(self):
        nodes = [Node(n, "A", 1, 80) for n in "abc"]
        plan = choose_placement(nodes, Workload(2, 40, {"A": 10}, 20),
                                {("a", "b"): 1, ("a", "c"): 20, ("b", "c"): 2})
        self.assertEqual([n.name for n in plan.workers], ["a", "c"])
        self.assertEqual(plan.estimated_seconds, 11)

    def test_memory_capacity_and_unknown_model(self):
        nodes = [Node("a", "A", 8, 20), Node("b", "B", 8, 80), Node("c", "A", 1, 80)]
        with self.assertRaisesRegex(ValueError, "capacity"):
            choose_placement(nodes, Workload(2, 40, {"A": 1}), {})

    def test_missing_link_rejected(self):
        with self.assertRaisesRegex(ValueError, "communication"):
            choose_placement([Node("a", "A", 1, 80), Node("b", "A", 1, 80)],
                             Workload(2, 40, {"A": 1}, 10), {})

    def test_bad_inputs(self):
        for value in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                Workload(1, 1, {"A": value})
        with self.assertRaises(ValueError):
            Node("a", "A", 1.5, 80)
        with self.assertRaises(ValueError):
            choose_placement([Node("a", "A", 1, 80)] * 2, Workload(1, 1, {"A": 1}), {})

    def test_search_limit(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            choose_placement([Node(str(i), "A", 10, 80) for i in range(20)],
                             Workload(10, 1, {"A": 1}), {})


if __name__ == "__main__":
    unittest.main()
