import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from examples import dynamo_dry_run, dynamo_gpu_validation as validation
from topology_scheduler.dynamo_backend import (
    ADAPTER_OWNED, CALLER_OWNED, DynamoConfig,
)

CONFIG = DynamoConfig.from_contract()
READY_HOST = {"opted_in": True, "system": "Linux",
              "gpus": ("NVIDIA H100, 81559 MiB, 580.00.03",),
              "frontend": True, "etcd": True, "nats": True, "config": CONFIG}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


class PrerequisiteTests(unittest.TestCase):
    def test_a_complete_host_has_no_reasons_to_skip(self):
        self.assertEqual(validation.check_prerequisites(**READY_HOST), ())

    def test_every_missing_prerequisite_is_named(self):
        for missing, expected in (
            ({"opted_in": False}, "opt-in"),
            ({"system": "Windows"}, "linux/amd64"),
            ({"gpus": ()}, "no NVIDIA GPU"),
            ({"frontend": False}, "Dynamo frontend is not reachable"),
            ({"etcd": False}, "etcd is not reachable"),
            ({"nats": False}, "NATS is not reachable"),
        ):
            with self.subTest(missing=missing):
                reasons = validation.check_prerequisites(**{**READY_HOST, **missing})
                self.assertEqual(len(reasons), 1)
                self.assertIn(expected, reasons[0])

    def test_unreachable_services_name_their_endpoint(self):
        reasons = validation.check_prerequisites(**{**READY_HOST, "etcd": False})
        self.assertIn(CONFIG.etcd_endpoints, reasons[0])


class ProbeTests(unittest.TestCase):
    def test_reachable_rejects_closed_ports_and_malformed_endpoints(self):
        self.assertFalse(validation.reachable("http://127.0.0.1:9", timeout=0.2))
        self.assertFalse(validation.reachable("not-an-endpoint"))
        self.assertFalse(validation.reachable("http://127.0.0.1"))

    def test_gpu_query_parses_rows_and_survives_a_missing_driver(self):
        result = subprocess.CompletedProcess([], 0, stdout="H100, 81559 MiB, 580.00.03\n\n")
        with patch.object(validation.subprocess, "run", return_value=result):
            self.assertEqual(validation.nvidia_gpus(), ("H100, 81559 MiB, 580.00.03",))
        with patch.object(validation.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(validation.nvidia_gpus(), ())


class CompletionTests(unittest.TestCase):
    def response(self, content):
        return FakeResponse(json.dumps(
            {"choices": [{"message": {"content": content}}]}).encode())

    def test_returns_the_reply_and_its_own_latency(self):
        with patch.object(validation, "urlopen", return_value=self.response("ready")):
            content, latency = validation.completion("http://f", "model", timeout=5)
        self.assertEqual(content, "ready")
        self.assertGreaterEqual(latency, 0)

    def test_empty_completion_is_a_failure(self):
        with patch.object(validation, "urlopen", return_value=self.response("  ")):
            with self.assertRaisesRegex(AssertionError, "empty completion"):
                validation.completion("http://f", "model", timeout=5)


class SkipPathTests(unittest.TestCase):
    def run_main(self, **facts):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        report = Path(directory.name) / "report.json"
        stdout = io.StringIO()
        with patch.object(validation, "nvidia_gpus", return_value=facts.get("gpus", ())), \
             patch.object(validation, "reachable", return_value=facts.get("reachable", False)), \
             patch.object(validation.platform, "system",
                          return_value=facts.get("system", "Linux")), \
             redirect_stdout(stdout):
            code = validation.main(["--report", str(report)])
        return code, json.loads(report.read_text(encoding="utf-8")), stdout.getvalue()

    def test_a_host_without_gpus_skips_cleanly_and_says_why(self):
        code, report, output = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "skipped")
        self.assertFalse(report["simulated_gpus_used"])
        self.assertTrue(any("no NVIDIA GPU" in reason for reason in report["reasons"]))
        self.assertTrue(any("opt-in" in reason for reason in report["reasons"]))
        self.assertIn("skipped", output)

    def test_the_report_records_hardware_versions_and_endpoints(self):
        _, report, _ = self.run_main(gpus=("NVIDIA H100, 81559 MiB, 580.00.03",))
        environment = report["environment"]
        self.assertEqual(environment["gpus"], ["NVIDIA H100, 81559 MiB, 580.00.03"])
        self.assertEqual(environment["contract"]["dynamo"], "1.4.2")
        self.assertEqual(environment["contract"]["vllm"], "0.26.0")
        self.assertEqual(environment["endpoints"]["frontend"], CONFIG.frontend_url)
        self.assertIn("python", environment)

    def test_gpus_present_but_services_down_still_skips(self):
        _, report, _ = self.run_main(gpus=("NVIDIA H100",), reachable=False)
        self.assertEqual(report["status"], "skipped")
        self.assertTrue(any("not reachable" in reason for reason in report["reasons"]))

    def test_namespace_and_frontend_overrides_reach_the_config(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        report = Path(directory.name) / "report.json"
        with patch.object(validation, "nvidia_gpus", return_value=()), \
             patch.object(validation, "reachable", return_value=False), \
             redirect_stdout(io.StringIO()):
            validation.main(["--report", str(report), "--namespace", "experiment-001",
                             "--frontend-url", "http://head:8000"])
        endpoints = json.loads(report.read_text())["environment"]["endpoints"]
        self.assertEqual(endpoints["frontend"], "http://head:8000")


class DryRunTests(unittest.TestCase):
    def test_dry_run_prints_plan_replicas_and_intent_without_running(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            dynamo_dry_run.main()
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["synthetic_inputs"])
        self.assertFalse(report["performs_inference"])
        self.assertEqual(report["placement"], ["a", "b"])
        self.assertEqual(len(report["replicas"]), 2)
        self.assertEqual(report["model"]["id"], CONFIG.model)
        self.assertEqual(report["ownership"]["adapter"], list(ADAPTER_OWNED))
        self.assertEqual(report["ownership"]["caller"], list(CALLER_OWNED))
        self.assertEqual([item["system_port"] for item in report["replicas"]],
                         [CONFIG.system_port_base, CONFIG.system_port_base + 1])
        for replica in report["replicas"]:
            self.assertIn("dynamo.vllm", replica["command_line"])
            self.assertIn("--tensor-parallel-size 1", replica["command_line"])


if __name__ == "__main__":
    unittest.main()
