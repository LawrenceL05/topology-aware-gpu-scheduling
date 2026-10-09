"""Validation reports must distinguish real evidence from unrelated failures."""
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from examples import dynamo_gpu_validation as validation
from topology_scheduler import Node, Plan
from topology_scheduler.dynamo_backend import DynamoConfig, DynamoLifecycleError

CONFIG = DynamoConfig.from_contract()
ARGS = SimpleNamespace(replicas=1, memory_gb=1, compute_seconds='{"X": 1}',
                       requests=2, request_timeout=1)


class FakeService:
    def __init__(self, *, start_error=None, close_error=None):
        self.config = CONFIG
        self.group = object()
        self.state = "new"
        self.start_error, self.close_error = start_error, close_error
        self.endpoint = "http://frontend:8000"
        self.records = [{"rank": 0, "node_id": "remote-node", "gpu_ids": [0],
                         "log_path": "/logs/worker", "receipt": "/logs/receipt", "token": "token"}]

    def start(self):
        self.state = "starting"
        if self.start_error:
            raise self.start_error
        self.state = "ready"

    def close(self):
        if self.close_error:
            self.state = "cleanup_failed"
            raise self.close_error
        self.state = "closed"
        self.group = None

    def status(self):
        return {"state": self.state, "deployment_id": "deployment", "namespace": CONFIG.namespace,
                "group_id": "group" if self.group is not None else None,
                "replicas": self.records, "error": None}


@unittest.skipUnless(importlib.util.find_spec("ray"), "Install .[ray] for lifecycle orchestration tests")
class ValidationLifecycleTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.init = stack.enter_context(patch("ray.init"))
        self.shutdown = stack.enter_context(patch("ray.shutdown"))
        self.table = stack.enter_context(patch("ray.util.placement_group.placement_group_table",
                                             return_value={"state": "REMOVED"}))
        self.nodes = stack.enter_context(patch("ray.nodes", return_value=[
            {"Alive": True, "NodeID": "remote-node", "Resources": {"topology_node:a": 1}}]))
        stack.enter_context(patch("topology_scheduler.discover_planner_nodes",
                                  return_value=[Node("a", "X", 1, 80)]))
        self.service = FakeService()
        self.factory = stack.enter_context(patch("topology_scheduler.DynamoService", return_value=self.service))
        self.completion = stack.enter_context(patch.object(validation, "completion", return_value=("ready", .01)))

    def validate(self):
        with patch.object(validation, "rollback_check", return_value={"status": validation.PASSED}):
            return validation.validate(ARGS, CONFIG)

    def test_remote_cleanup_uses_exact_reservation_and_adapter_proof(self):
        group = self.service.group
        with patch.object(validation, "reachable") as reachable:
            report = self.validate()
        self.assertEqual(report["status"], validation.PASSED)
        self.table.assert_called_once_with(group)
        reachable.assert_not_called()
        cleanup = report["steps"]["shutdown"]
        self.assertTrue(cleanup["process_cleanup_confirmed"])
        self.assertEqual(cleanup["reservation_state"], "REMOVED")
        self.assertEqual(cleanup["before_close"]["group_id"], "group")
        self.assertEqual(report["steps"]["requests"]["count"], 2)
        self.assertEqual(self.completion.call_count, 2)
        self.shutdown.assert_called_once()

    def test_partial_request_and_cleanup_errors_both_survive(self):
        self.completion.side_effect = [("first", .01), TimeoutError("request deadline")]
        self.service.close_error = RuntimeError("receipt missing")
        report = self.validate()
        self.assertEqual(report["status"], validation.FAILED)
        self.assertEqual(report["failure_stage"], "requests")
        self.assertIn("request deadline", report["error"])
        self.assertEqual(report["steps"]["requests"]["count"], 1)
        self.assertEqual(report["request_latency_seconds"], [.01])
        cleanup = report["steps"]["shutdown"]
        self.assertIn("receipt missing", cleanup["error"])
        self.assertEqual(cleanup["after_close"]["group_id"], "group")
        json.dumps(report)

    def test_startup_failure_retains_config_snapshot_and_cleanup(self):
        self.service.start_error = DynamoLifecycleError("engine failed")
        report = self.validate()
        self.assertEqual(report["status"], validation.FAILED)
        self.assertEqual(report["failure_stage"], "startup")
        self.assertEqual(report["steps"]["shutdown"]["before_close"]["deployment_id"], "deployment")
        self.assertEqual(report["config"]["revision"], CONFIG.revision)
        self.completion.assert_not_called()

    def test_connect_failure_returns_failed_report(self):
        self.init.side_effect = RuntimeError("no Ray cluster")
        report = self.validate()
        self.assertEqual(report["status"], validation.FAILED)
        self.assertEqual(report["failure_stage"], "connect")
        self.factory.assert_not_called()

    def test_disconnect_error_cannot_erase_request_failure(self):
        self.completion.side_effect = RuntimeError("bad reply")
        self.shutdown.side_effect = RuntimeError("disconnect failed")
        report = self.validate()
        self.assertEqual(report["failure_stage"], "requests")
        self.assertIn("bad reply", report["error"])
        self.assertEqual(report["steps"]["disconnect"]["status"], validation.FAILED)

    def test_close_without_confirmed_state_is_not_success(self):
        self.service.close = lambda: None
        report = validation.close_and_record(self.service)
        self.assertEqual(report["status"], validation.FAILED)
        self.table.assert_not_called()

    def test_placement_mismatch_stops_before_requests(self):
        self.service.records[0]["node_id"] = "wrong-node"
        report = self.validate()
        self.assertEqual(report["failure_stage"], "placement")
        self.completion.assert_not_called()

    def test_rollback_failure_prevents_real_deployment(self):
        with patch.object(validation, "rollback_check", return_value={"status": validation.FAILED}):
            report = validation.validate(ARGS, CONFIG)
        self.assertEqual(report["failure_stage"], "rollback")
        self.factory.assert_not_called()

    def test_reservation_removal_is_polled_and_timeout_is_not_a_pass(self):
        self.table.side_effect = [{"state": "CREATED"}, {"state": "REMOVED"}]
        with patch.object(validation.time, "sleep"):
            result = validation.close_and_record(self.service)
        self.assertEqual(result["status"], validation.PASSED)
        self.assertEqual(self.table.call_count, 2)
        self.service = FakeService()
        self.table.side_effect = None
        self.table.return_value = {"state": "CREATED"}
        with patch.object(validation.time, "monotonic", side_effect=[0, CONFIG.rpc_timeout + 1]):
            result = validation.close_and_record(self.service)
        self.assertEqual(result["status"], validation.FAILED)
        self.assertIn("reservation removal", result["error"])

    def test_only_observed_engine_failure_counts_as_controlled_rollback(self):
        plan = Plan((Node("a", "X", 1, 80),), 1)
        for error, expected in ((ValueError("wrong namespace"), validation.FAILED),
                                (DynamoLifecycleError("reservation timed out"), validation.FAILED),
                                (DynamoLifecycleError("Replica engine exited: failed"), validation.PASSED)):
            with self.subTest(error=error):
                self.factory.return_value = FakeService(start_error=error)
                report = validation.rollback_check(plan, CONFIG)
                self.assertEqual(report["status"], expected)
        self.factory.return_value = FakeService(
            start_error=DynamoLifecycleError("Replica engine exited: failed"),
            close_error=RuntimeError("unconfirmed cleanup"))
        self.assertEqual(validation.rollback_check(plan, CONFIG)["status"], validation.FAILED)

    def test_failed_run_is_written_by_cli_with_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            self.init.side_effect = RuntimeError("unreachable cluster")
            with patch.object(validation, "check_prerequisites", return_value=()), \
                 patch.object(validation, "nvidia_gpus", return_value=("GPU",)), \
                 patch.object(validation, "reachable", return_value=True), redirect_stdout(io.StringIO()):
                code = validation.main(["--run", "--report", str(path)])
            self.assertEqual(code, 1)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], validation.FAILED)
            self.assertEqual(report["failure_stage"], "connect")
            self.assertFalse(report["simulated_gpus_used"])


class ArgumentTests(unittest.TestCase):
    def test_optional_runtime_failure_still_writes_failed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with patch.object(validation, "check_prerequisites", return_value=()), \
                 patch.object(validation, "nvidia_gpus", return_value=("GPU",)), \
                 patch.object(validation, "reachable", return_value=True), \
                 patch.object(validation, "validate", side_effect=ImportError("Ray unavailable")), \
                 redirect_stdout(io.StringIO()):
                code = validation.main(["--run", "--report", str(path)])
            self.assertEqual(code, 1)
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], validation.FAILED)
            self.assertIn("Ray unavailable", report["error"])

    def test_invalid_arguments_fail_before_probes(self):
        for options in (["--requests", "0"], ["--requests", "1"], ["--replicas", "0"],
                        ["--request-timeout", "nan"], ["--memory-gb", "-1"],
                        ["--compute-seconds", "[]"], ["--compute-seconds", '{"X": 0}']):
            with self.subTest(options=options), patch.object(validation, "nvidia_gpus") as probe, \
                 patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                validation.main(options)
            probe.assert_not_called()

    def test_malformed_port_or_ipv6_endpoint_is_unreachable(self):
        for endpoint in ("http://host:wrong", "http://[broken", "http://host:99999"):
            self.assertFalse(validation.reachable(endpoint))
