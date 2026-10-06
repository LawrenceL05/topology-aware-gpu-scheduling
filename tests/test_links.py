import errno
import importlib.util
import sys
import tempfile
import time
import unittest
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

from topology_scheduler import (
    LinkCost, LinkCostSource, LinkEndpoint, LinkMeasurement,
    LinkMeasurementReport, Node, ProbeParameters, Workload, choose_placement,
    load_link_report, measure_loopback_link, measure_ray_links,
    plan_with_record, resolve_link_costs,
)
from topology_scheduler.links import (
    FAILED, SUCCEEDED, TIMED_OUT, UNREACHABLE, ProbeServerStartError,
    _endpoint, _describe, _disjoint_batches, _gather,
    _interface_for_address, _live_topology_nodes, _measurement, _route_address,
)

PARAMETERS = ProbeParameters(duration_seconds=1, warmup_seconds=0, timeout_seconds=5)
NOW = 10_000.0


def endpoint(name, mbps=None):
    return LinkEndpoint(name, f"id-{name}", f"10.0.0.{len(name)}",
                        "eth0" if mbps else None, mbps)


def measured(source, destination, throughput=1e9, *, status=SUCCEEDED,
             at=NOW - 10, source_mbps=None, destination_mbps=None):
    ok = status == SUCCEEDED
    return LinkMeasurement(
        source=endpoint(source, source_mbps),
        destination=endpoint(destination, destination_mbps),
        status=status, started_at=at - 1, finished_at=at, parameters=PARAMETERS,
        throughput_bytes_per_second=throughput if ok else None,
        rtt_seconds=0.001 if ok else None,
        rtt_min_seconds=0.0009 if ok else None,
        error=None if ok else "ConnectionRefusedError: refused",
        hint=None if ok else "Allow ephemeral TCP ports.",
        software=(("python", "3.12"),),
    )


def report(*items):
    return LinkMeasurementReport(tuple(items))


class ParameterTests(unittest.TestCase):
    def test_defaults_are_bounded_and_valid(self):
        parameters = ProbeParameters()
        self.assertEqual(parameters.max_concurrent_pairs, 1)
        self.assertEqual(parameters.as_dict()["duration_seconds"], 5.0)

    def test_rejects_settings_that_become_a_load_test(self):
        for options in (
            {"duration_seconds": 61, "timeout_seconds": 90},
            {"streams": 9},
            {"streams": True},
            {"chunk_bytes": 0},
            {"latency_samples": 1001},
            {"max_concurrent_pairs": 5},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ProbeParameters(**options)

    def test_timeout_must_cover_warmup_and_duration(self):
        with self.assertRaisesRegex(ValueError, "timeout_seconds must exceed"):
            ProbeParameters(duration_seconds=5, warmup_seconds=1, timeout_seconds=7)
        ProbeParameters(duration_seconds=5, warmup_seconds=1, timeout_seconds=7.5)


class RecordTests(unittest.TestCase):
    def test_success_needs_positive_throughput_and_rtt(self):
        with self.assertRaises(ValueError):
            measured("a", "b", throughput=0)
        with self.assertRaisesRegex(ValueError, "needs throughput"):
            LinkMeasurement(endpoint("a"), endpoint("b"), SUCCEEDED, 0, 1, PARAMETERS)

    def test_failure_needs_an_error(self):
        with self.assertRaisesRegex(ValueError, "needs an error"):
            LinkMeasurement(endpoint("a"), endpoint("b"), FAILED, 0, 1, PARAMETERS)

    def test_record_identifies_endpoints_direction_units_time_and_parameters(self):
        value = measured("a", "b", source_mbps=100_000).as_dict()
        self.assertEqual(value["direction"], "a->b")
        self.assertEqual(value["source"]["interface"], "eth0")
        self.assertEqual(value["source"]["advertised_mbps"], 100_000)
        self.assertEqual(value["destination"]["node_id"], "id-b")
        self.assertEqual(value["status"], "succeeded")
        self.assertEqual(value["units"]["throughput_bytes_per_second"], "bytes/s")
        self.assertEqual(value["units"]["planner_bandwidth"], "GB/s (decimal)")
        self.assertEqual(value["parameters"]["timeout_seconds"], 5)
        self.assertTrue(value["started_at_utc"].endswith("+00:00"))
        self.assertEqual(value["software"], {"python": "3.12"})

    def test_report_round_trips_through_json(self):
        original = report(measured("a", "b", 2e9),
                          measured("b", "a", status=UNREACHABLE))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "links.json"
            original.save(path)
            self.assertEqual(load_link_report(path), original)

    def test_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "links.json"
            path.write_text('{"schema_version": 99, "measurements": []}')
            with self.assertRaisesRegex(ValueError, "schema"):
                load_link_report(path)


class ResolutionTests(unittest.TestCase):
    def resolve(self, evidence, names=("a", "b"), **options):
        options.setdefault("max_age_seconds", 3600)
        return resolve_link_costs(evidence, names, now=NOW, **options)

    def test_asymmetric_evidence_stays_raw_and_normalizes_to_slower_direction(self):
        evidence = report(measured("a", "b", 2e9, at=NOW - 5),
                          measured("b", "a", 1e9, at=NOW - 20))
        resolution = self.resolve(evidence)
        self.assertEqual(
            [item.throughput_bytes_per_second for item in evidence.measurements],
            [2e9, 1e9])
        self.assertEqual(resolution.costs[("a", "b")],
                         LinkCost(1.0, LinkCostSource.MEASURED, NOW - 20))
        self.assertEqual(resolution.bandwidth_gbps, {("a", "b"): 1.0})
        self.assertEqual(resolution.diagnostics, ())

    def test_partially_measured_pair_is_omitted_with_actionable_diagnostic(self):
        resolution = self.resolve(report(
            measured("a", "b"), measured("b", "a", status=UNREACHABLE)))
        self.assertEqual(resolution.costs, {})
        (message,) = resolution.diagnostics
        self.assertIn("a|b is not measured", message)
        self.assertIn("b->a unreachable: ConnectionRefusedError", message)
        self.assertIn("Allow ephemeral TCP ports.", message)
        self.assertIn("infeasible", message)
        self.assertIn("measure_ray_links()", message)

    def test_unmeasured_direction_is_named(self):
        (message,) = self.resolve(report(measured("a", "b"))).diagnostics
        self.assertIn("b->a has no measurement", message)

    def test_stale_evidence_is_never_reused(self):
        resolution = self.resolve(report(
            measured("a", "b", at=NOW - 7200), measured("b", "a", at=NOW - 7200)))
        self.assertEqual(resolution.costs, {})
        self.assertIn("a->b is stale (7200s old; max_age_seconds=3600)",
                      resolution.diagnostics[0])

    def test_newest_record_in_each_direction_wins(self):
        recovered = self.resolve(report(
            measured("a", "b", status=TIMED_OUT, at=NOW - 30),
            measured("a", "b", at=NOW - 10), measured("b", "a")))
        self.assertIn(("a", "b"), recovered.costs)
        regressed = self.resolve(report(
            measured("a", "b", at=NOW - 30),
            measured("a", "b", status=FAILED, at=NOW - 10), measured("b", "a")))
        self.assertEqual(regressed.costs, {})

    def test_advertised_capacity_is_opt_in_and_labelled(self):
        evidence = report(
            measured("a", "b", source_mbps=100_000, destination_mbps=25_000),
            measured("b", "a", status=TIMED_OUT))
        self.assertEqual(self.resolve(evidence).costs, {})
        resolution = self.resolve(evidence, use_advertised=True)
        self.assertEqual(resolution.costs[("a", "b")],
                         LinkCost(3.125, LinkCostSource.ADVERTISED))
        self.assertIn("using advertised capacity", resolution.diagnostics[0])

    def test_fallback_is_labelled_for_every_unmeasured_pair(self):
        resolution = self.resolve(None, ("c", "a", "b"), fallback_gb_per_second=1.5)
        self.assertEqual(sorted(resolution.costs), [("a", "b"), ("a", "c"), ("b", "c")])
        self.assertTrue(all(cost.source is LinkCostSource.FALLBACK
                            for cost in resolution.costs.values()))
        self.assertEqual(len(resolution.diagnostics), 3)

    def test_measured_map_replaces_manual_planner_input(self):
        nodes = [Node("a", "X", 1, 80), Node("b", "X", 1, 80)]
        workload = Workload(2, 1, {"X": 1}, 10)
        measured_map = self.resolve(report(
            measured("a", "b", 5e9), measured("b", "a", 5e9))).bandwidth_gbps
        plan = choose_placement(nodes, workload, measured_map)
        self.assertEqual(plan.estimated_seconds, 3)
        with self.assertRaisesRegex(ValueError, "known communication links"):
            choose_placement(nodes, workload, self.resolve(None).bandwidth_gbps)

    def test_rejects_invalid_age_and_fallback(self):
        with self.assertRaises(ValueError):
            self.resolve(None, max_age_seconds=0)
        with self.assertRaises(ValueError):
            self.resolve(None, fallback_gb_per_second=-1)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [Node("a", "X", 1, 80), Node("b", "X", 1, 80)]
        self.workload = Workload(2, 1, {"X": 1}, 1)

    def test_caller_values_are_recorded_as_supplied(self):
        _, record = plan_with_record(
            self.nodes, self.workload, {("a", "b"): 2}, policy="combined")
        self.assertEqual(record.as_dict()["inputs"]["bandwidth_sources"],
                         {"a|b": {"source": "supplied", "measured_at": None}})

    def test_resolved_costs_are_recorded_by_source(self):
        costs = {("a", "b"): LinkCost(2, LinkCostSource.MEASURED, NOW)}
        _, record = plan_with_record(
            self.nodes, self.workload, {("a", "b"): 2}, policy="combined",
            link_costs=costs)
        self.assertEqual(record.inputs["bandwidth_sources"]["a|b"],
                         {"source": "measured", "measured_at": NOW})

    def test_costs_must_match_the_bandwidth_used(self):
        for costs in ({}, {("a", "b"): LinkCost(3, LinkCostSource.FALLBACK)}):
            with self.subTest(costs=costs), self.assertRaisesRegex(
                    ValueError, "exactly the supplied"):
                plan_with_record(self.nodes, self.workload, {("a", "b"): 2},
                                 policy="combined", link_costs=costs)


class DiagnosticTests(unittest.TestCase):
    def outcome(self, client_error=None, server_errors=(), throughput=1e9):
        server = {"throughput_bytes_per_second": None if server_errors else throughput,
                  "errors": list(server_errors)}
        client = (None, client_error) if client_error else (
            {"rtt_seconds": 0.001, "rtt_min_seconds": 0.001}, None)
        return _measurement(endpoint("a"), endpoint("b"), PARAMETERS, 1, 2,
                            client, (server, None), ())

    def test_refused_connection_is_unreachable_and_names_the_target(self):
        record = self.outcome(ConnectionRefusedError("refused"))
        self.assertEqual(record.status, UNREACHABLE)
        self.assertIn("could not open TCP to b at 10.0.0.1", record.hint)

    def test_routing_errors_are_unreachable(self):
        if not hasattr(errno, "EHOSTUNREACH"):
            self.skipTest("platform has no EHOSTUNREACH")
        record = self.outcome(OSError(errno.EHOSTUNREACH, "No route to host"))
        self.assertEqual(record.status, UNREACHABLE)

    def test_timeout_is_classified(self):
        record = self.outcome(TimeoutError("timed out"))
        self.assertEqual(record.status, TIMED_OUT)
        self.assertIn("raise timeout_seconds", record.hint)

    def test_server_failure_overrides_a_successful_client(self):
        record = self.outcome(server_errors=["stream 0 closed early"])
        self.assertEqual(record.status, FAILED)
        self.assertIn("stream 0 closed early", record.error)
        self.assertIn("Rerun a->b alone", record.hint)

    def test_no_bytes_in_window_is_a_failure(self):
        record = self.outcome(throughput=0)
        self.assertEqual(record.status, FAILED)
        self.assertIn("no bytes arrived", record.error)

    def test_server_that_cannot_start_names_the_address_to_check(self):
        record = self.outcome(ProbeServerStartError("ActorDiedError: bind failed"))
        self.assertEqual(record.status, FAILED)
        self.assertIn("probe server on b could not listen on 10.0.0.1", record.hint)

    def test_error_text_drops_colors_and_keeps_the_root_cause(self):
        text = _describe(RuntimeError(
            "\x1b[36mray::Actor\x1b[39m " + "x" * 2000 + " OSError: address invalid"))
        self.assertNotIn("\x1b", text)
        self.assertTrue(text.startswith("RuntimeError: ray::Actor"))
        self.assertTrue(text.endswith("OSError: address invalid"))
        self.assertEqual(len(text), 1000)

    def test_successful_outcome_keeps_both_sides(self):
        record = self.outcome()
        self.assertEqual((record.status, record.throughput_bytes_per_second,
                          record.rtt_seconds), (SUCCEEDED, 1e9, 0.001))


class FakeRay:
    def __init__(self, ready, values):
        self.ready, self.values = ready, values

    def wait(self, refs, *, num_returns, timeout):
        return [ref for ref in refs if ref in self.ready], []

    def get(self, ref):
        value = self.values[ref]
        if isinstance(value, Exception):
            raise value
        return value


class OrchestrationTests(unittest.TestCase):
    def test_measurement_requires_explicit_opt_in(self):
        for value in (False, "yes", 1):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "allow_network_load=True"):
                measure_ray_links(allow_network_load=value)

    def test_batches_are_bounded_node_disjoint_and_complete(self):
        pairs = list(permutations("abcd", 2))
        for limit in (1, 2, 4):
            batches = _disjoint_batches(pairs, limit)
            with self.subTest(limit=limit):
                self.assertEqual(sorted(p for b in batches for p in b), sorted(pairs))
                for batch in batches:
                    nodes = [node for pair in batch for node in pair]
                    self.assertLessEqual(len(batch), limit)
                    self.assertEqual(len(nodes), len(set(nodes)))
        self.assertEqual(len(_disjoint_batches(pairs, 1)), len(pairs))

    def test_live_nodes_need_one_unique_marker(self):
        def node(node_id, marker, alive=True):
            resources = {"CPU": 1} | ({marker: 1} if marker else {})
            return {"NodeID": node_id, "NodeManagerAddress": f"10.0.0.{node_id}",
                    "Alive": alive, "Resources": resources}

        nodes = [node("1", "topology_node:a"), node("2", None),
                 node("3", "topology_node:b", alive=False)]
        self.assertEqual(_live_topology_nodes(nodes), {"a": ("1", "10.0.0.1")})
        with self.assertRaisesRegex(ValueError, "more than one live Ray node"):
            _live_topology_nodes([node("1", "topology_node:a"),
                                  node("2", "topology_node:a")])

    def test_gather_reports_each_timeout_and_error_separately(self):
        failure = RuntimeError("actor died")
        ray = FakeRay({"ok", "bad"}, {"ok": 1, "bad": failure})
        outcomes = _gather(ray, ["ok", "slow", "bad"], 3, "probe server")
        self.assertEqual(outcomes[0], (1, None))
        self.assertIsInstance(outcomes[1][1], TimeoutError)
        self.assertIn("probe server returned nothing within 3s", str(outcomes[1][1]))
        self.assertIs(outcomes[2][1], failure)

    @unittest.skipUnless(importlib.util.find_spec("ray"), "Install .[ray]")
    def test_rejects_unknown_or_single_node_selection(self):
        live = [{"NodeID": "1", "NodeManagerAddress": "10.0.0.1", "Alive": True,
                 "Resources": {"topology_node:a": 1}}]
        with patch("ray.is_initialized", return_value=True), \
             patch("ray.nodes", return_value=live):
            with self.assertRaisesRegex(ValueError, "topology_node:z"):
                measure_ray_links(["a", "z"], allow_network_load=True)
            with self.assertRaisesRegex(ValueError, "at least two"):
                measure_ray_links(allow_network_load=True)


class LoopbackTests(unittest.TestCase):
    """Real sockets on 127.0.0.1; these numbers are never cluster evidence."""

    def test_probe_runs_end_to_end_over_loopback(self):
        record = measure_loopback_link()
        self.assertEqual(record.status, SUCCEEDED, record.error)
        self.assertGreater(record.throughput_bytes_per_second, 0)
        self.assertGreaterEqual(record.rtt_seconds, record.rtt_min_seconds)
        self.assertEqual(record.source.address, "127.0.0.1")
        self.assertIn("python", dict(record.software))

    def test_parallel_streams_are_counted(self):
        record = measure_loopback_link(ProbeParameters(
            duration_seconds=0.3, warmup_seconds=0, streams=3, chunk_bytes=65536,
            latency_samples=2, timeout_seconds=10))
        self.assertEqual(record.status, SUCCEEDED, record.error)
        self.assertEqual(record.parameters.streams, 3)

    def test_route_to_loopback_uses_loopback(self):
        self.assertEqual(_route_address("127.0.0.1"), "127.0.0.1")

    @unittest.skipUnless(sys.platform.startswith("linux"), "interface lookup is Linux-only")
    def test_linux_resolves_loopback_interface_without_a_speed(self):
        self.assertEqual(_interface_for_address("127.0.0.1"), "lo")
        self.assertIsNone(_interface_for_address("192.0.2.1"))
        endpoint = _endpoint("local", "local-id", "127.0.0.1")
        self.assertIsNone(endpoint.advertised_mbps)
        self.assertEqual(endpoint.nic["name"], "lo")
        self.assertEqual(endpoint.nic["kind"], "loopback")
        self.assertTrue(endpoint.diagnostics)

    def test_client_failure_releases_the_server_promptly(self):
        started = time.monotonic()
        with patch("topology_scheduler.links.run_link_client",
                   side_effect=ConnectionRefusedError("refused")):
            record = measure_loopback_link()
        self.assertEqual(record.status, UNREACHABLE)
        self.assertLess(time.monotonic() - started, 5)


if __name__ == "__main__":
    unittest.main()
