from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from topology_scheduler import GPUDevice, RayNodeInventory, TopologyGraph, TopologyVertex


def reading(value, confidence="reported"):
    return {"value": value, "source": "/sys/fixture", "confidence": confidence}


class ObservationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.inventory = RayNodeInventory("node-a", "a", "topology_node:a", 2, (
            GPUDevice(0, "GPU-a", "NVIDIA H100", "H100", 80, "00000000:01:00.0"),
            GPUDevice(1, "GPU-b", "NVIDIA H100", "H100", 80, "0000:81:00.0"),
        ))
        self.nic = {"node_id": "node-a", "node_name": "a", "problems": [],
                    "units": {"speed_mbps": "Mbit/s advertised"}, "interfaces": [
            {"name": name, "pci_address": reading("0000:02:00.0" if i < 2 else "0000:82:00.0"),
             "numa_node": reading(0 if i < 2 else 1), "speed_mbps": reading(100000),
             "rdma": [{"name": f"mlx5_{i}", "port_states": ["1:4: ACTIVE"]}]}
            for i, name in enumerate(("eth0", "eth1", "eth2"))
        ]}
        self.host = {"node_id": "node-a", "node_name": "a", "diagnostics": ["fixture"],
                     "sources": {"numa_node": "<pci device>/numa_node"}, "nics": [
            {"name": n["name"], "pci_address": n["pci_address"]["value"],
             "numa_node": n["numa_node"]["value"], "pci_path": ["pci0000:00"]}
            for n in self.nic["interfaces"]
        ], "gpus": [
            {"uuid": uuid, "pci_address": pci, "numa_node": index,
             "pci_path": ["pci0000:00", pci], "nearest_nic": "eth0" if index == 0 else "eth2",
             "nics": [
                {"nic_name": name, "proximity": "same-numa" if index == (i // 2) else "cross-numa",
                 "gpu_numa_node": index, "nic_numa_node": i // 2,
                 "shared_pci_ancestor": None, "reason": None}
                for i, name in enumerate(("eth0", "eth1", "eth2"))
             ]}
            for index, (uuid, pci) in enumerate((("GPU-a", "0000:01:00.0"), ("GPU-b", "0000:81:00.0")))
        ]}

    def graph(self):
        return TopologyGraph.from_observations(self.inventory,
                                              nic_inventory=self.nic, host_topology=self.host)

    def test_collector_snapshots_join_into_typed_graph_and_keep_ties(self):
        graph = self.graph()
        gpu = TopologyVertex("gpu", "node-a", "GPU-a").id
        peers = graph.nics_near_gpu(gpu)
        self.assertEqual([n.attributes["name"] for n in peers], ["eth0", "eth1"])
        self.assertNotEqual(peers[0].id, peers[1].id)  # Same PCI function, distinct interfaces.
        self.assertEqual([g.key for g in graph.gpus_near_nic(peers[0].id)], ["GPU-a"])
        self.assertEqual({v.kind for v in graph.vertices}, {"node", "gpu", "nic", "numa"})
        self.assertEqual({v.key for v in graph.vertices if v.kind == "numa"}, {"0", "1"})

    def test_raw_readings_rdma_and_derived_evidence_survive_round_trip(self):
        graph = self.graph()
        loaded = TopologyGraph.from_json(graph.to_json())
        self.assertEqual(loaded.to_json(), graph.to_json())
        nic = next(v for v in loaded.vertices if v.kind == "nic" and v.attributes["name"] == "eth0")
        self.assertEqual(nic.attributes["nic_inventory"], self.nic["interfaces"][0])
        edge = next(e for e in loaded.relationships if e.kind == "gpu_nic_affinity" and
                    e.evidence["observation"]["nic_name"] == "eth0")
        self.assertEqual(edge.evidence["sources"], self.host["sources"])
        self.assertEqual(edge.confidence, "unknown")  # Collector did not assign confidence.

    def test_unknown_unsupported_and_missing_pairs_are_not_ranked(self):
        self.host["gpus"][0]["nics"] = [
            {"nic_name": "eth0", "proximity": "unknown", "reason": "permission denied"},
            {"nic_name": "eth1", "proximity": "unsupported", "reason": "no driver support"},
        ]
        graph = self.graph()
        gpu = TopologyVertex("gpu", "node-a", "GPU-a").id
        self.assertEqual(graph.nics_near_gpu(gpu), ())
        edges = [e for e in graph.relationships if e.kind == "gpu_nic_affinity" and gpu in (e.source, e.target)]
        self.assertEqual(len(edges), 3)
        self.assertEqual(sorted(e.state for e in edges), ["unknown", "unknown", "unsupported"])
        self.assertTrue(all(e.reason for e in edges))

    def test_missing_numa_does_not_create_a_fictitious_domain(self):
        for interface in self.nic["interfaces"]:
            interface["numa_node"] = reading(None, "unreadable")
        for item in self.host["nics"] + self.host["gpus"]:
            item["numa_node"] = None
        graph = self.graph()
        self.assertFalse(any(v.kind == "numa" for v in graph.vertices))
        self.assertFalse(any(e.kind == "numa_locality" for e in graph.relationships))
        self.assertTrue(any(v.attributes.get("nic_inventory", {}).get("numa_node", {}).get("confidence")
                            == "unreadable" for v in graph.vertices))

    def test_input_permutations_do_not_change_canonical_json(self):
        expected = self.graph().to_json()
        self.nic["interfaces"].reverse()
        self.host["nics"].reverse()
        self.host["gpus"].reverse()
        for gpu in self.host["gpus"]:
            gpu["nics"].reverse()
        self.assertEqual(self.graph().to_json(), expected)

    def test_different_nodes_cannot_be_joined(self):
        for snapshot in (self.nic, self.host):
            snapshot["node_id"] = "another-node"
            with self.assertRaisesRegex(ValueError, "node_id"):
                self.graph()
            snapshot["node_id"] = "node-a"

    def test_conflicting_pci_or_numa_snapshots_fail_explicitly(self):
        for target, field, replacement in (
            (self.host["nics"][0], "pci_address", "0000:03:00.0"),
            (self.host["gpus"][0], "pci_address", "0000:03:00.0"),
            (self.host["nics"][0], "numa_node", 9),
        ):
            old = target[field]
            target[field] = replacement
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                self.graph()
            target[field] = old

    def test_duplicate_and_dangling_source_identities_are_rejected(self):
        for items in (self.nic["interfaces"], self.host["nics"], self.host["gpus"],
                      self.host["gpus"][0]["nics"]):
            items.append(deepcopy(items[0]))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.graph()
            items.pop()
        self.host["gpus"][0]["nics"][0]["nic_name"] = "not-present"
        with self.assertRaisesRegex(ValueError, "unknown NIC"):
            self.graph()
        self.host["gpus"][0]["uuid"] = "GPU-not-present"
        with self.assertRaisesRegex(ValueError, "UUIDs absent"):
            self.graph()

    def test_legacy_only_input_still_produces_the_same_graph(self):
        self.assertEqual(TopologyGraph.from_observations(self.inventory).to_json(),
                         TopologyGraph.from_inventory(self.inventory).to_json())
        # JSON snapshots, not tuples from dataclasses.asdict(), are the dictionary contract.
        gpu = asdict(self.inventory)
        gpu["devices"] = list(gpu["devices"])
        gpu["connections"] = list(gpu["connections"])
        self.assertEqual(TopologyGraph.from_observations(gpu).to_json(),
                         TopologyGraph.from_inventory(self.inventory).to_json())

    def test_nic_only_and_host_only_inputs_keep_their_information(self):
        nic_graph = TopologyGraph.from_observations(self.inventory, nic_inventory=self.nic)
        self.assertFalse(any(e.state == "known" for e in nic_graph.relationships if e.kind == "gpu_nic_affinity"))
        host_graph = TopologyGraph.from_observations(self.inventory, host_topology=self.host)
        self.assertEqual(len([v for v in host_graph.vertices if v.kind == "nic"]), 3)
        self.assertTrue(any(e.state == "known" for e in host_graph.relationships if e.kind == "gpu_nic_affinity"))

    def test_pci_structure_is_separate_from_affinity(self):
        self.host["gpus"][0]["nics"][0].update(
            proximity="same-switch", shared_pci_ancestor="0000:00:01.0")
        graph = self.graph()
        ancestry = [e for e in graph.relationships if e.kind == "pcie_ancestry" and e.state == "known"]
        self.assertEqual(len(ancestry), 1)
        self.assertEqual(ancestry[0].value, "shared-ancestor")
        self.assertEqual(ancestry[0].evidence["observation"]["shared_pci_ancestor"], "0000:00:01.0")

    def test_as_dict_objects_are_accepted_without_importing_collectors(self):
        class Snapshot:
            def __init__(self, value):
                self.value = value
            def as_dict(self):
                return self.value
        graph = TopologyGraph.from_observations(self.inventory,
                    nic_inventory=Snapshot(self.nic), host_topology=Snapshot(self.host))
        self.assertEqual(graph.to_json(), self.graph().to_json())
        self.assertEqual(self.nic["interfaces"][0]["name"], "eth0")

    def test_nic_collector_output_joins_locality_and_retains_full_evidence(self):
        from topology_scheduler.nic_inventory import collect_nic_inventory
        from tests.test_nic_inventory import ETHERNET, FakeSysfs, net

        tree = {}
        for interface in self.nic["interfaces"]:
            name = interface["name"]
            tree.update(net(name, ETHERNET,
                            uevent=f"PCI_SLOT_NAME={interface['pci_address']['value']}\nDRIVER=mlx5_core\n"))
            tree[f"sys/class/net/{name}/device"]["numa_node"] = str(interface["numa_node"]["value"])
        # Include both attached and unmatched RDMA evidence in the real export.
        for name, pci in (("mlx5_0", "0000:02:00.0"), ("mlx5_1", "0000:99:00.0")):
            tree[f"sys/class/infiniband/{name}"] = {"node_type": "1: CA"}
            tree[f"sys/class/infiniband/{name}/device"] = {"uevent": f"PCI_SLOT_NAME={pci}\n"}
            tree[f"sys/class/infiniband/{name}/ports/1"] = {
                "state": "4: ACTIVE", "link_layer": "InfiniBand"}
        snapshot = collect_nic_inventory(node_id="node-a", node_name="a", sysfs=FakeSysfs(tree))
        self.assertEqual(len(snapshot.unattached_rdma), 1)
        self.assertTrue(snapshot.interfaces[0].rdma)
        graph = TopologyGraph.from_observations(self.inventory,
                    nic_inventory=snapshot, host_topology=self.host)
        serialized = snapshot.as_dict()
        node = next(v for v in graph.vertices if v.kind == "node")
        self.assertEqual(node.attributes["nic_inventory"],
                         {k: v for k, v in serialized.items() if k != "interfaces"})
        for vertex in (v for v in graph.vertices if v.kind == "nic"):
            self.assertEqual(vertex.attributes["nic_inventory"], next(
                n for n in serialized["interfaces"] if n["name"] == vertex.attributes["name"]))
        gpu = TopologyVertex("gpu", "node-a", "GPU-a").id
        self.assertEqual([v.attributes["name"] for v in graph.nics_near_gpu(gpu)], ["eth0", "eth1"])
        from_dict = TopologyGraph.from_observations(self.inventory,
                    nic_inventory=serialized, host_topology=self.host)
        self.assertEqual(graph.to_json(), from_dict.to_json())
        self.assertEqual(TopologyGraph.from_json(graph.to_json()).to_json(), graph.to_json())

    def test_partial_nic_collector_output_does_not_invent_locality(self):
        from topology_scheduler.nic_inventory import collect_nic_inventory
        from tests.test_nic_inventory import DENIED, INVALID, ETHERNET, FakeSysfs, net

        tree = net("eth0", dict(ETHERNET, speed=INVALID))
        tree["sys/class/net/eth0/device"] = {"uevent": DENIED, "numa_node": DENIED}
        snapshot = collect_nic_inventory(node_id="node-a", sysfs=FakeSysfs(tree))
        graph = TopologyGraph.from_observations(self.inventory, nic_inventory=snapshot)
        nic = next(v for v in graph.vertices if v.kind == "nic")
        evidence = nic.attributes["nic_inventory"]
        self.assertEqual(evidence, snapshot.interfaces[0].as_dict())
        self.assertEqual(evidence["numa_node"]["confidence"], "unreadable")
        self.assertEqual(evidence["speed_mbps"]["confidence"], "unsupported")
        self.assertTrue(evidence["problems"])
        self.assertFalse(any(v.kind == "numa" for v in graph.vertices))
        self.assertEqual(graph.gpus_near_nic(nic.id), ())
        affinities = [e for e in graph.relationships if e.kind == "gpu_nic_affinity"]
        self.assertEqual(len(affinities), 2)
        self.assertTrue(all(e.state == "unknown" and e.reason for e in affinities))
        self.assertEqual(TopologyGraph.from_json(graph.to_json()).to_json(), graph.to_json())

    def test_file_example_loads_exported_snapshots_and_prints_queries(self):
        from examples.topology_graph import main

        with tempfile.TemporaryDirectory() as directory:
            args = ["topology_graph"]
            for flag, data in (("gpu-inventory", asdict(self.inventory)),
                               ("nic-inventory", self.nic), ("host-topology", self.host)):
                path = Path(directory) / f"{flag}.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                args += [f"--{flag}", str(path)]
            with patch("sys.argv", args), patch("sys.stdout", new_callable=io.StringIO) as output:
                main()
            value = json.loads(output.getvalue())
            self.assertEqual(value["input_source"], "snapshot_files")
            self.assertFalse(value["benchmark_evidence"])
            self.assertEqual(len(value["queries"]["nics_near_gpu"]), 2)
            self.assertEqual(TopologyGraph.from_dict(value["graph"]).to_json(), self.graph().to_json())

    def test_file_example_requires_gpu_identity_for_optional_snapshots(self):
        from examples.topology_graph import main

        with patch("sys.argv", ["topology_graph", "--nic-inventory", "unused.json"]), \
             patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as caught:
            main()
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
