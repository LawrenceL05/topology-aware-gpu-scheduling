"""Keep the adapter tied to deploy/dynamo-v1/contract.json.

The configuration defaults and the ownership split are both copies of that
document. A copy drifts silently, so these tests fail the moment the contract
and the code disagree.
"""
import json
import unittest
from dataclasses import replace

from topology_scheduler.dynamo_backend import (
    ADAPTER_OWNED, CALLER_OWNED, CONTRACT_PATH, DynamoConfig,
)

CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


class ContractBindingTests(unittest.TestCase):
    def test_every_derived_value_comes_from_the_contract(self):
        config = DynamoConfig.from_contract()
        model, service = CONTRACT["model"], CONTRACT["service"]
        self.assertEqual(config.namespace, service["namespace"])
        self.assertEqual(config.frontend_url,
                         f"http://127.0.0.1:{service['frontend_port']}")
        self.assertEqual(config.etcd_endpoints, service["etcd_endpoints"])
        self.assertEqual(config.nats_server, service["nats_server"])
        self.assertEqual(config.model, model["id"])
        self.assertEqual(config.revision, model["revision"])
        self.assertEqual(config.max_model_length, model["max_model_length"])
        self.assertEqual(config.gpu_memory_utilization, model["gpu_memory_utilization"])
        self.assertEqual(config.tensor_parallel_size, service["tensor_parallel_size"])
        self.assertEqual(config.system_port_base, service["worker_system_port_base"])
        self.assertEqual(config.startup_timeout, service["startup_deadline_seconds"])
        self.assertEqual(config.shutdown_timeout, service["shutdown_deadline_seconds"])
        config.validate(service["replica_count"])

    def test_hardcoded_defaults_do_not_drift_from_the_contract(self):
        derived = DynamoConfig.from_contract()
        defaults = DynamoConfig(namespace=derived.namespace,
                                frontend_url=derived.frontend_url)
        self.assertEqual(defaults, derived)

    def test_a_loopback_frontend_host_becomes_a_reachable_url(self):
        # The contract binds the frontend to 0.0.0.0, which is not a client URL.
        self.assertEqual(CONTRACT["service"]["frontend_host"], "0.0.0.0")
        self.assertNotIn("0.0.0.0", DynamoConfig.from_contract().frontend_url)

    def test_overrides_apply_and_are_still_validated(self):
        config = DynamoConfig.from_contract(frontend_url="http://head-node:8000")
        self.assertEqual(config.frontend_url, "http://head-node:8000")
        config.validate(2)
        with self.assertRaises(ValueError):
            DynamoConfig.from_contract(tensor_parallel_size=2).validate(2)

    def test_ownership_matches_the_contract(self):
        ownership = CONTRACT["ownership"]
        self.assertEqual(sorted(ADAPTER_OWNED), sorted(ownership["adapter"]))
        self.assertEqual(sorted(CALLER_OWNED), sorted(ownership["caller"]))
        # The two lists must stay disjoint, or close() would have to guess.
        self.assertEqual(set(ADAPTER_OWNED) & set(CALLER_OWNED), set())

    def test_a_changed_contract_is_reported_rather_than_ignored(self):
        moved = replace(DynamoConfig.from_contract(), max_model_length=8192)
        self.assertNotEqual(moved, DynamoConfig.from_contract())


if __name__ == "__main__":
    unittest.main()
