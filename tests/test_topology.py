from dataclasses import asdict, replace
import json
import random
import unittest

from topology_scheduler import (
    GPUConnection, GPUDevice, RayNodeInventory, RELATIONSHIP_TYPES,
    TopologyGraph, TopologyRelationship, TopologyVertex,
)


class TopologyTests(unittest.TestCase):
    def setUp(self):
        self.node = TopologyVertex("node", "host/one", "host/one", {"name": "a"})
        self.gpus = [TopologyVertex("gpu", self.node.node_id, f"GPU-{i}") for i in range(2)]
        self.nics = [TopologyVertex("nic", self.node.node_id, f"pci:0000:0{i}:00.0/port:1",
                                    {"name": f"eth{i}", "advertised_speed_gbps": None})
                     for i in range(3)]
        self.numa = [TopologyVertex("numa", self.node.node_id, str(i)) for i in range(2)]
        self.vertices = (self.node, *self.gpus, *self.nics, *self.numa)

    def affinity(self, gpu, nic, value, **kwargs):
        return TopologyRelationship("gpu_nic_affinity", gpu.id, nic.id, "fixture:sysfs",
                                    value=value, **kwargs)

    def graph(self):
        edges = [self.affinity(self.gpus[0], self.nics[i], value)
                 for i, value in enumerate(("same-numa", "same-numa", "cross-numa"))]
        edges += [self.affinity(self.gpus[1], self.nics[0], "same-numa"),
                  self.affinity(self.gpus[1], self.nics[2], "same-switch")]
        for i in range(2):
            edges.append(TopologyRelationship("numa_locality", self.gpus[i].id,
                                               self.numa[i].id, "fixture:sysfs", value="local"))
        return TopologyGraph(self.vertices, tuple(edges))

    def test_multiple_nics_numa_domains_and_equal_distance_ties(self):
        graph = self.graph()
        self.assertEqual(graph.nics_near_gpu(self.gpus[0].id), tuple(self.nics[:2]))
        self.assertEqual(graph.nics_near_gpu(self.gpus[1].id), (self.nics[2],))
        self.assertEqual(graph.gpus_near_nic(self.nics[0].id), tuple(self.gpus))
        locality = [edge for edge in graph.relationships if edge.kind == "numa_locality"]
        self.assertEqual({edge.target for edge in locality}, {v.id for v in self.numa})

    def test_unknown_and_unsupported_evidence_survives_but_is_not_ranked(self):
        edges = tuple(self.affinity(
            self.gpus[0], self.nics[i], None, state=state, confidence="unknown",
            reason="No permission" if i == 0 else "Driver does not expose ancestry",
            evidence={"errno": 13 if i == 0 else None, "raw": ["unavailable", {"path": "sysfs"}]},
        ) for i, state in enumerate(("unknown", "unsupported")))
        graph = TopologyGraph(self.vertices, edges)
        loaded = TopologyGraph.from_json(graph.to_json())
        self.assertEqual(loaded.as_dict(), graph.as_dict())
        self.assertEqual(loaded.nics_near_gpu(self.gpus[0].id), ())
        self.assertEqual(len(loaded.relationships_within_node(self.node.node_id)), 2)
        self.assertIsNone(loaded.vertex(self.nics[0].id).attributes["advertised_speed_gbps"])

    def test_canonical_serialization_ignores_discovery_order_and_edge_orientation(self):
        original = self.graph()
        randomizer = random.Random(10)
        for _ in range(8):
            vertices = list(original.vertices)
            edges = [replace(e, source=e.target, target=e.source)
                     if not RELATIONSHIP_TYPES[e.kind].directed else e
                     for e in original.relationships]
            randomizer.shuffle(vertices)
            randomizer.shuffle(edges)
            graph = TopologyGraph(vertices, edges)
            self.assertEqual(graph.to_json(), original.to_json())
            self.assertEqual(TopologyGraph.from_json(graph.to_json()).to_json(), original.to_json())

    def test_serialization_keeps_raw_evidence_array_order_and_nested_values(self):
        evidence = {"raw": [2, 1, None], "nested": {"speed": 12.5, "known": False}}
        edge = self.affinity(self.gpus[0], self.nics[0], "same-device", evidence=evidence)
        graph = TopologyGraph(self.vertices, (edge,))
        evidence["raw"].append(3)
        exported = graph.as_dict()
        exported["relationships"][0]["evidence"]["raw"].append(4)
        self.assertEqual(TopologyGraph.from_json(graph.to_json()).relationships[0].evidence["raw"],
                         [2, 1, None])

    def test_relationship_schema_is_machine_readable(self):
        data = self.graph().as_dict()
        for edge in data["relationships"]:
            spec = RELATIONSHIP_TYPES[edge["kind"]]
            self.assertEqual((edge["directed"], edge["unit"], edge["meaning"]),
                             (spec.directed, spec.unit, spec.meaning))
        data["relationships"][0]["unit"] = "GB/s"
        with self.assertRaisesRegex(ValueError, "Incorrect unit"):
            TopologyGraph.from_dict(data)

    def test_node_scopes_prevent_identity_collisions_and_cross_node_queries(self):
        other_node = TopologyVertex("node", "host%2Fone", "host%2Fone")
        other_gpu = TopologyVertex("gpu", other_node.node_id, self.gpus[0].key)
        other_nic = TopologyVertex("nic", other_node.node_id, self.nics[0].key)
        edge = self.affinity(other_gpu, other_nic, "same-device")
        graph = TopologyGraph((*self.vertices, other_node, other_gpu, other_nic), (edge,))
        self.assertNotEqual(other_gpu.id, self.gpus[0].id)
        self.assertIn("host%2Fone", self.gpus[0].id)
        self.assertEqual(graph.relationships_within_node(self.node.node_id), ())
        self.assertEqual(graph.relationships_within_node(other_node.node_id), (edge,))
        self.assertEqual(graph.nics_near_gpu(self.gpus[0].id), ())
        with self.assertRaisesRegex(ValueError, "intra-node"):
            TopologyGraph(graph.vertices, (self.affinity(self.gpus[0], other_nic, "same-numa"),))

    def legacy(self):
        return RayNodeInventory("ray-id", "a", "topology_node:a", 2,
                                tuple(GPUDevice(i, f"GPU-{i}", "NVIDIA H100", "H100", 80,
                                                f"0000:0{i}:00.0") for i in range(2)),
                                (GPUConnection("GPU-0", "GPU-1", "single-pci-bridge", 0),))

    def test_legacy_inventory_conversion_preserves_properties_and_planner_callers(self):
        legacy = self.legacy()
        before = legacy.as_planner_node()
        graph = TopologyGraph.from_inventory(legacy)
        self.assertEqual(legacy.as_planner_node(), before)
        self.assertEqual(graph.to_json(), TopologyGraph.from_json(json.dumps(asdict(legacy))).to_json())
        gpus = [v for v in graph.vertices if v.kind == "gpu"]
        self.assertEqual([v.attributes for v in gpus], [asdict(d) for d in legacy.devices])
        node = next(v for v in graph.vertices if v.kind == "node")
        self.assertEqual(node.attributes, {"node_name": "a", "resource_key": "topology_node:a",
                                           "configured_gpus": 2})
        nvlink = next(e for e in graph.relationships if e.kind == "nvlink")
        self.assertEqual((nvlink.value, nvlink.state, nvlink.as_dict()["unit"]), (0, "known", "links"))
        self.assertEqual(nvlink.evidence, asdict(legacy.connections[0]))
        self.assertEqual(nvlink.confidence, "unknown")

    def test_device_only_legacy_and_multiple_nodes_load_without_fabricated_edges(self):
        first = asdict(self.legacy())
        del first["connections"]
        second = dict(first, node_id="other", node_name="b", resource_key="topology_node:b")
        graph = TopologyGraph.from_dict([second, first])
        self.assertEqual(len(graph.vertices), 6)
        self.assertTrue(all(e.kind == "contains" for e in graph.relationships))
        self.assertEqual(graph.to_json(), TopologyGraph.from_dict([first, second]).to_json())

    def test_missing_and_unrecognized_legacy_connection_fields_remain_unknown(self):
        record = asdict(self.legacy())
        record["connections"] = [{"source_uuid": "GPU-0", "target_uuid": "GPU-1",
                                  "common_ancestor": "unknown-999", "diagnostic": ["raw"]}]
        graph = TopologyGraph.from_dict(record)
        edges = [e for e in graph.relationships if e.kind != "contains"]
        self.assertTrue(all(e.state == "unknown" and e.value is None for e in edges))
        self.assertEqual(edges[0].evidence, record["connections"][0])
        self.assertEqual(graph.to_json(), TopologyGraph.from_json(graph.to_json()).to_json())

    def test_duplicate_dangling_and_invalid_endpoint_edges_are_rejected(self):
        edge = self.affinity(self.gpus[0], self.nics[0], "same-numa")
        for vertices, edges, message in (
            ((*self.vertices, self.gpus[0]), (), "Duplicate topology"),
            (self.vertices, (edge, edge), "Duplicate relationship"),
            (self.vertices, (replace(edge, target="missing"),), "Dangling"),
            (self.vertices, (replace(edge, target=self.gpus[1].id),), "endpoint kinds"),
            (self.vertices[1:], (), "Missing owning node"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                TopologyGraph(vertices, edges)

    def test_invalid_values_states_and_non_json_evidence_are_rejected(self):
        for kwargs in ({"state": "missing"}, {"state": "unknown"},
                       {"confidence": "certain"}, {"value": "fast"},
                       {"evidence": {"speed": float("nan")}}, {"evidence": {1: "bad key"}}):
            arguments = dict(value="same-numa")
            arguments.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.affinity(self.gpus[0], self.nics[0], **arguments)
        for value in (True, -1, 1.5, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "link count"):
                TopologyRelationship("nvlink", self.gpus[0].id, self.gpus[1].id, "NVML", value=value)

    def test_invalid_schema_and_identity_are_rejected_without_dropping_data(self):
        data = self.graph().as_dict()
        for version in (0, 2, True):
            with self.subTest(version=version), self.assertRaises(ValueError):
                TopologyGraph.from_dict(dict(data, schema_version=version))
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            TopologyGraph.from_dict(dict(data, future_field="preserve me"))
        data["vertices"][0]["id"] = "corrupt"
        with self.assertRaisesRegex(ValueError, "Vertex ID"):
            TopologyGraph.from_dict(data)

    def test_queries_reject_wrong_or_missing_vertices(self):
        graph = self.graph()
        with self.assertRaises(KeyError):
            graph.nics_near_gpu("missing")
        with self.assertRaises(KeyError):
            graph.relationships_within_node("missing")
        with self.assertRaisesRegex(ValueError, "Expected a gpu"):
            graph.nics_near_gpu(self.nics[0].id)
        with self.assertRaisesRegex(ValueError, "Expected a nic"):
            graph.gpus_near_nic(self.gpus[0].id)

    def test_duplicate_json_keys_and_unrecognized_relationships_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            TopologyGraph.from_json('{"schema_version":1,"schema_version":2}')
        data = self.graph().as_dict()
        data["relationships"][0]["kind"] = "future-bandwidth"
        with self.assertRaisesRegex(ValueError, "Unsupported relationship"):
            TopologyGraph.from_dict(data)
        data = self.graph().as_dict()
        data["relationships"] = {}
        with self.assertRaisesRegex(ValueError, "must be arrays"):
            TopologyGraph.from_dict(data)

    def test_directed_relationships_and_numa_keys_are_validated(self):
        reversed_edge = TopologyRelationship(
            "numa_locality", self.numa[0].id, self.gpus[0].id, "fixture:sysfs", value="local")
        with self.assertRaisesRegex(ValueError, "endpoint kinds"):
            TopologyGraph(self.vertices, (reversed_edge,))
        for key in ("-1", "01", "unknown", "1.0"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "NUMA key"):
                TopologyVertex("numa", self.node.node_id, key)


if __name__ == "__main__":
    unittest.main()
