import json
import re
import unittest
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "deploy" / "dynamo-v1" / "contract.json"


class DynamoContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_runtime_is_reproducibly_pinned(self):
        runtime = self.contract["runtime"]
        self.assertEqual(runtime["dynamo"], "1.4.2")
        self.assertEqual(runtime["vllm"], "0.26.0")
        self.assertEqual(runtime["ray"], "2.55.0")
        self.assertRegex(runtime["dynamo_commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(runtime["image"], r"^nvcr\.io/.+@sha256:[0-9a-f]{64}$")
        self.assertRegex(runtime["platform_image_digest"], r"^sha256:[0-9a-f]{64}$")

    def test_single_gpu_replica_contract(self):
        service = self.contract["service"]
        self.assertGreater(service["replica_count"], 0)
        self.assertEqual(service["gpus_per_replica"], 1)
        self.assertEqual(service["tensor_parallel_size"], 1)
        self.assertGreater(service["startup_deadline_seconds"], 0)
        self.assertGreater(service["shutdown_deadline_seconds"], 0)
        utilization = self.contract["model"]["gpu_memory_utilization"]
        self.assertGreater(utilization, 0)
        self.assertLess(utilization, 1)

    def test_model_revision_and_ownership_are_explicit(self):
        self.assertRegex(self.contract["model"]["revision"], r"^[0-9a-f]{40}$")
        ownership = self.contract["ownership"]
        self.assertIn("dynamo_frontend", ownership["caller"])
        self.assertIn("dynamo_vllm_workers", ownership["adapter"])

    def test_ray_pin_matches_project_and_container_recipe(self):
        ray = self.contract["runtime"]["ray"]
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        requirements = (
            ROOT / "deploy" / "dynamo-v1" / "requirements.txt"
        ).read_text(encoding="utf-8")
        dockerfile = (
            ROOT / "deploy" / "dynamo-v1" / "Dockerfile"
        ).read_text(encoding="utf-8")
        self.assertRegex(pyproject, rf'ray = \["ray=={re.escape(ray)}"\]')
        self.assertEqual(requirements.strip(), f"ray=={ray}")
        self.assertIn(f"ARG DYNAMO_BASE={self.contract['runtime']['image']}", dockerfile)
        self.assertEqual(version("ray"), ray)


if __name__ == "__main__":
    unittest.main()
