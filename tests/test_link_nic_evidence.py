"""Join real NIC collector fixtures to probe records without network hardware."""

from dataclasses import replace
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from topology_scheduler import LinkMeasurement, LinkMeasurementReport, load_link_report, measure_loopback_link, resolve_link_costs
from topology_scheduler.links import _client_task, _endpoint, _interface_for_address, _ServerActor
from topology_scheduler.nic_inventory import collect_nic_inventory
from tests.test_links import NOW, PARAMETERS, measured
from tests.test_nic_inventory import DENIED, ETHERNET, INVALID, FakeSysfs, MELLANOX_UEVENT, net


class EndpointEvidenceTests(unittest.TestCase):
    def inventory(self, speed="100000", numa="1"):
        tree = net("eth0", dict(ETHERNET, speed=speed), uevent=MELLANOX_UEVENT)
        tree["sys/class/net/eth0/device"]["numa_node"] = numa
        tree.update({
            "sys/class/infiniband/mlx5_0": {"node_type": "1: CA"},
            "sys/class/infiniband/mlx5_0/device": {"uevent": MELLANOX_UEVENT},
            "sys/class/infiniband/mlx5_0/ports/1": {"state": "4: ACTIVE", "link_layer": "InfiniBand"},
        })
        # Add a faster, unrelated NIC: only the address-matched NIC may supply
        # the capacity estimate, regardless of ordering or link speed.
        tree.update(net("eth1", dict(ETHERNET, speed="400000"),
                        uevent="PCI_SLOT_NAME=0000:81:00.0\n"))
        return collect_nic_inventory(node_name="a", node_id="id-a", sysfs=FakeSysfs(tree))

    def endpoint(self, inventory):
        with patch("topology_scheduler.links._interface_for_address", return_value="eth0"), \
             patch("topology_scheduler.links.collect_nic_inventory", return_value=inventory) as collect:
            result = _endpoint("a", "id-a", "10.0.0.1")
        collect.assert_called_once_with(node_name="a", node_id="id-a")
        return result

    def test_collector_evidence_survives_measurement_report_round_trip(self):
        inventory = self.inventory()
        endpoint = self.endpoint(inventory)
        self.assertEqual(endpoint.nic, inventory.interfaces[0].as_dict())
        self.assertEqual(endpoint.nic["pci_address"]["value"], "0000:3b:00.0")
        self.assertEqual(endpoint.nic["numa_node"]["value"], 1)
        self.assertEqual(endpoint.nic["rdma"][0]["name"], "mlx5_0")
        self.assertEqual(endpoint.advertised_mbps, 100000)
        self.assertEqual(endpoint.interface_source, "Linux SIOCGIFADDR primary IPv4")
        self.assertGreater(endpoint.nic_collected_at, 0)
        record = replace(measured("a", "b"), source=endpoint)
        report = LinkMeasurementReport((record,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "links.json"
            report.save(path)
            self.assertEqual(load_link_report(path), report)

    def test_partial_fields_keep_confidence_and_cannot_supply_capacity(self):
        for speed, confidence in ((DENIED, "unreadable"), (INVALID, "unsupported"), ("0", "reported")):
            with self.subTest(speed=speed):
                endpoint = self.endpoint(self.inventory(speed=speed, numa=DENIED))
                self.assertIsNone(endpoint.advertised_mbps)
                self.assertEqual(endpoint.nic["speed_mbps"]["confidence"], confidence)
                self.assertEqual(endpoint.nic["numa_node"]["confidence"], "unreadable")
                self.assertTrue(endpoint.diagnostics)
                self.assertTrue(any("no capacity is inferred" in d for d in endpoint.diagnostics))
                evidence = LinkMeasurementReport((replace(
                    measured("a", "b", destination_mbps=25000), source=endpoint),))
                record = evidence.measurements[0]
                self.assertEqual(LinkMeasurement.from_dict(json.loads(json.dumps(record.as_dict()))), record)
                resolution = resolve_link_costs(evidence, ["a", "b"],
                    max_age_seconds=300, now=NOW, use_advertised=True)
                self.assertEqual(resolution.costs, {})

    def test_collected_advertised_capacity_remains_separate_from_measurement(self):
        endpoint = self.endpoint(self.inventory(speed="25000"))
        evidence = LinkMeasurementReport((replace(
            measured("a", "b", destination_mbps=100000), source=endpoint),))
        resolution = resolve_link_costs(evidence, ["a", "b"],
            max_age_seconds=300, now=NOW, use_advertised=True)
        self.assertEqual(resolution.costs[("a", "b")].source.value, "advertised")
        self.assertEqual(resolution.bandwidth_gbps[("a", "b")], 3.125)
        self.assertIn("b->a has no measurement", resolution.diagnostics[0])

    def test_missing_interface_does_not_borrow_another_nics_identity(self):
        inventory = replace(self.inventory(), interfaces=self.inventory().interfaces[1:])
        endpoint = self.endpoint(inventory)
        self.assertEqual(endpoint.interface, "eth0")
        self.assertIsNone(endpoint.nic)
        self.assertIsNone(endpoint.advertised_mbps)
        self.assertIn("no unique NIC inventory record", endpoint.diagnostics[0])

    def test_unresolved_interface_and_read_failure_are_explicit(self):
        with patch("topology_scheduler.links._interface_for_address", return_value=None), \
             patch("topology_scheduler.links.collect_nic_inventory") as collect:
            endpoint = _endpoint("a", "id-a", "2001:db8::1")
        collect.assert_not_called()
        self.assertIsNone(endpoint.nic)
        self.assertIn("IPv6", endpoint.diagnostics[0])
        with patch("topology_scheduler.links._interface_for_address", return_value="eth0"), \
             patch("topology_scheduler.links.collect_nic_inventory", side_effect=OSError("denied")):
            endpoint = _endpoint("a", "id-a", "10.0.0.1")
        self.assertIsNone(endpoint.nic)
        self.assertIn("NIC collection failed", endpoint.diagnostics[0])
        self.assertIn("denied", endpoint.diagnostics[0])

    def test_duplicate_address_match_is_not_silently_assigned_to_first_nic(self):
        reply = bytes(20) + socket.inet_aton("10.0.0.1")
        fcntl = SimpleNamespace(ioctl=Mock(return_value=reply))
        with patch("topology_scheduler.links.sys.platform", "linux"), \
             patch.dict("sys.modules", {"fcntl": fcntl}), \
             patch("topology_scheduler.links.socket.if_nameindex", return_value=[(1, "eth0"), (2, "eth1")]):
            self.assertIsNone(_interface_for_address("10.0.0.1"))

    def test_nic_read_failure_does_not_discard_successful_tcp_measurement(self):
        with patch("topology_scheduler.links._interface_for_address", return_value="lo"), \
             patch("topology_scheduler.links.collect_nic_inventory", side_effect=PermissionError("denied")):
            record = measure_loopback_link()
        self.assertEqual(record.status, "succeeded", record.error)
        self.assertGreater(record.throughput_bytes_per_second, 0)
        for endpoint in (record.source, record.destination):
            self.assertIsNone(endpoint.nic)
            self.assertIsNone(endpoint.advertised_mbps)
            self.assertIn("NIC collection failed", endpoint.diagnostics[0])

    def test_legacy_schema_one_endpoints_load_without_nic_metadata(self):
        value = measured("a", "b").as_dict()
        for side in ("source", "destination"):
            for field in ("nic", "nic_collected_at", "interface_source", "diagnostics"):
                del value[side][field]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps({"schema_version": 1, "measurements": [value]}))
            loaded = load_link_report(path).measurements[0]
        self.assertIsNone(loaded.source.nic)
        self.assertEqual(loaded.source.diagnostics, ())
        self.assertEqual(loaded.throughput_bytes_per_second, 1e9)

    def test_probe_tasks_collect_on_their_own_ray_node_and_keep_failure_evidence(self):
        ray = SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(get_node_id=lambda: "id-a"))
        with patch.dict("sys.modules", {"ray": ray}), \
             patch("topology_scheduler.links._interface_for_address", return_value="eth0"), \
             patch("topology_scheduler.links.collect_nic_inventory", return_value=self.inventory()) as collect, \
             patch("topology_scheduler.links._route_address", return_value="10.0.0.1"), \
             patch("topology_scheduler.links.run_link_client", side_effect=ConnectionRefusedError("refused")):
            task = _client_task("a", "10.0.0.2", 1234, PARAMETERS)
            server = _ServerActor.__new__(_ServerActor)
            server._server = SimpleNamespace(port=1234)
            ready = server.ready("a", "10.0.0.1")
        self.assertIsInstance(task["error"], ConnectionRefusedError)
        for endpoint in (task["endpoint"], ready["endpoint"]):
            self.assertEqual(endpoint.node_id, "id-a")
            self.assertEqual(endpoint.nic["name"], "eth0")
        self.assertEqual(collect.call_count, 2)
        for call in collect.call_args_list:
            self.assertEqual(call.kwargs, {"node_name": "a", "node_id": "id-a"})


if __name__ == "__main__":
    unittest.main()
