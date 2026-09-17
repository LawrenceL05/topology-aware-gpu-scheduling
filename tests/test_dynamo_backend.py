from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from topology_scheduler import Node, Plan
from topology_scheduler.dynamo_backend import (
    DynamoCleanupError, DynamoConfig, DynamoLifecycleError, DynamoService,
    RAY_NAMESPACE, RUNTIME_PINS, _Replica, _frontend_probe, _receipt,
)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = DynamoConfig("test-deployment", "http://localhost:8000")

    def test_defaults_match_pinned_contract_and_force_local_executor(self):
        contract = json.loads((Path(__file__).resolve().parents[1] /
                               "deploy/dynamo-v1/contract.json").read_text())
        self.config.validate(2)
        self.assertEqual(RUNTIME_PINS, {"ray": contract["runtime"]["ray"],
                                       "ai-dynamo": contract["runtime"]["dynamo"],
                                       "vllm": contract["runtime"]["vllm"]})
        self.assertEqual(self.config.revision, contract["model"]["revision"])
        command = self.config.worker_command()
        for option, value in (("--distributed-executor-backend", "mp"),
                              ("--tensor-parallel-size", "1"), ("--pipeline-parallel-size", "1"),
                              ("--data-parallel-size", "1"), ("--disaggregation-mode", "agg")):
            self.assertEqual(command[command.index(option) + 1], value)

    def test_unsupported_and_invalid_configurations_fail_before_ray(self):
        for kwargs in ({"tensor_parallel_size": 2}, {"pipeline_parallel_size": 2},
                       {"disaggregation_mode": "decode"}, {"revision": "main"},
                       {"gpu_memory_utilization": 1}, {"replica_cpus": 0},
                       {"startup_timeout": float("nan")}, {"system_port_base": 65535},
                       {"namespace": ""}, {"lease_timeout": 1}, {"log_dir": "relative"},
                       {"frontend_url": "http://user:secret@localhost:8000"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.config, **kwargs).validate(2)

    def test_frontend_requires_model_and_completion_not_only_http_success(self):
        with patch("topology_scheduler.dynamo_backend._http_json", return_value={"data": []}):
            self.assertTrue(_frontend_probe(self.config))
            self.assertFalse(_frontend_probe(self.config, completion=True))
        with patch("topology_scheduler.dynamo_backend._http_json", side_effect=[
            {"data": [{"id": self.config.model}]}, {"choices": [{"message": {"content": "ok"}}]},
        ]) as http:
            self.assertTrue(_frontend_probe(self.config, completion=True))
            self.assertEqual(http.call_args.args[2]["max_tokens"], 1)

    def test_receipt_requires_matching_deployment_and_positive_cleanup_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            self.assertFalse(_receipt(path, "a"))
            path.write_text('{"token":"a","stopped":true}')
            self.assertTrue(_receipt(path, "a"))
            self.assertFalse(_receipt(path, "b"))
            path.write_text('{"token":"a","stopped":false}')
            self.assertFalse(_receipt(path, "a"))


@unittest.skipUnless(importlib.util.find_spec("ray"), "Install .[ray]")
class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.plan = Plan((Node("a", "A", 2, 80), Node("a", "A", 2, 80)), 1)
        self.config = DynamoConfig("test-deployment", "http://localhost:8000")
        self.live = [{"Alive": True, "NodeID": "a" * 56,
                      "Resources": {"CPU": 4, "GPU": 2, "topology_node:a": 2}}]
        self.actors = [MagicMock(), MagicMock()]
        for rank, actor in enumerate(self.actors):
            actor.info.remote.return_value = {"rank": rank, "node_id": "a" * 56,
                "gpu_ids": [str(rank)], "cuda_visible_devices": str(rank),
                "log_path": f"log-{rank}", "receipt": f"receipt-{rank}", "token": f"token-{rank}"}
            actor.status.remote.return_value = {"alive": True, "ready": True}
            actor.heartbeat.remote.return_value = True
            actor.stop.remote.return_value = True
        self.actor_type = MagicMock()
        self.actor_type.options.return_value.remote.side_effect = self.actors
        self.probe_task = MagicMock()
        self.probe_task.options.return_value.remote.return_value = False

        def remote(*args, **kwargs):
            return (lambda function: self.probe_task) if kwargs else self.actor_type

        def get(ref, **kwargs):
            if isinstance(ref, Exception):
                raise ref
            return [get(item) for item in ref] if isinstance(ref, list) else ref

        self.get = self.enterContext(patch("ray.get", side_effect=get))
        self.enterContext(patch("ray.remote", side_effect=remote))
        self.enterContext(patch("ray.is_initialized", return_value=True))
        self.enterContext(patch("ray.get_runtime_context", return_value=MagicMock(namespace=RAY_NAMESPACE)))
        self.nodes = self.enterContext(patch("ray.nodes", return_value=self.live))
        self.create = self.enterContext(patch("ray.util.placement_group.placement_group"))
        self.remove = self.enterContext(patch("ray.util.placement_group.remove_placement_group"))
        self.kill = self.enterContext(patch("ray.kill"))
        self.enterContext(patch("topology_scheduler.dynamo_backend.threading.Thread"))
        self.frontend = MagicMock(return_value=True)
        self.service = DynamoService(self.plan, self.config, _probe=self.frontend)
        self.addCleanup(self.service._exit_cleanup)

    def test_reservation_bundle_binding_and_persistent_idempotent_handle(self):
        with self.service as service:
            self.assertEqual(service.endpoint, self.config.frontend_url)
            self.assertIs(service.start(), service)
            self.remove.assert_not_called()
            bundles = self.create.call_args.args[0]
            self.assertEqual(bundles, [{"CPU": 1, "GPU": 1, "topology_node:a": 1}] * 2)
            self.assertEqual(self.create.call_args.kwargs["lifetime"], "detached")
            for rank, call in enumerate(self.actor_type.options.call_args_list):
                options = call.kwargs
                self.assertEqual(options["num_gpus"], 1)
                self.assertEqual(options["max_restarts"], 0)
                strategy = options["scheduling_strategy"]
                self.assertEqual(strategy.placement_group_bundle_index, rank)
                self.assertFalse(strategy.placement_group_capture_child_tasks)
        self.service.close()
        self.remove.assert_called_once()
        with self.assertRaises(DynamoLifecycleError):
            _ = self.service.endpoint

    def test_duplicate_marker_or_insufficient_cpu_prevents_reservation(self):
        for nodes in (self.live * 2, [{**self.live[0], "Resources": {"CPU": 1, "GPU": 2, "topology_node:a": 2}}]):
            with self.subTest(nodes=nodes):
                self.nodes.return_value = nodes
                with self.assertRaises(ValueError):
                    self.service.start()
                self.create.assert_not_called()

    def test_frontend_must_be_running_before_reserving_replica_cpus(self):
        self.frontend.side_effect = OSError("frontend offline")
        with self.assertRaises(OSError):
            self.service.start()
        self.create.assert_not_called()

    def test_reservation_timeout_rolls_back_without_starting_actors(self):
        self.create.return_value.ready.return_value = TimeoutError("busy")
        with self.assertRaisesRegex(DynamoLifecycleError, "busy"):
            self.service.start()
        self.actor_type.options.assert_not_called()
        self.remove.assert_called_once()

    def test_wrong_node_or_partial_startup_rolls_back_all_actors(self):
        self.actors[1].launch.remote.return_value = RuntimeError("model failed")
        with self.assertRaisesRegex(DynamoLifecycleError, "model failed.*log-1"):
            self.service.start()
        for actor in self.actors:
            actor.stop.remote.assert_called_once()
        self.assertEqual(self.kill.call_count, 2)
        self.remove.assert_called_once()

    def test_wrong_node_is_rejected_before_any_engine_launch(self):
        self.actors[1].info.remote.return_value["node_id"] = "wrong"
        with self.assertRaisesRegex(DynamoLifecycleError, "wrong node"):
            self.service.start()
        self.actors[0].launch.remote.assert_not_called()
        self.remove.assert_called_once()

    def test_unready_frontend_times_out_and_cleans_up(self):
        self.service.config = replace(self.config, startup_timeout=0.02, poll_interval=0.005)
        self.frontend.side_effect = lambda *args, completion=False, **kw: not completion
        with self.assertRaisesRegex(DynamoLifecycleError, "deadline"):
            self.service.start()
        self.remove.assert_called_once()

    def test_engine_exit_is_not_readiness(self):
        self.actors[0].status.remote.return_value = {"alive": False, "ready": False}
        with self.assertRaisesRegex(DynamoLifecycleError, "engine exited"):
            self.service.start()
        self.remove.assert_called_once()

    def test_unconfirmed_cleanup_retains_resources_then_can_be_retried(self):
        self.service.start()
        self.service.config = replace(self.config, shutdown_timeout=0.01, kill_timeout=0.01)
        self.actors[0].stop.remote.return_value = RuntimeError("actor lost")
        with self.assertRaisesRegex(DynamoCleanupError, "reservation retained"):
            self.service.close()
        self.remove.assert_not_called()
        self.kill.assert_not_called()
        self.probe_task.options.return_value.remote.return_value = True
        self.service.close()
        self.remove.assert_called_once()

    def test_stops_every_replica_before_killing_actors_or_removing_group(self):
        self.service.start()
        events = []
        for rank, actor in enumerate(self.actors):
            actor.stop.remote.side_effect = lambda rank=rank: events.append(f"stop-{rank}") or True
        self.kill.side_effect = lambda *args, **kw: events.append("kill")
        self.remove.side_effect = lambda *args: events.append("remove")
        self.service.close()
        self.assertEqual(events, ["stop-0", "stop-1", "kill", "kill", "remove"])

    def test_running_actor_failure_triggers_supervised_cleanup(self):
        self.service.start()
        self.actors[0].heartbeat.remote.return_value = RuntimeError("actor disappeared")
        with patch.object(self.service._stop, "wait", return_value=False):
            self.service._watch()
        self.assertIn("actor disappeared", self.service.error)
        self.assertEqual(self.service.state, "closed")
        self.remove.assert_called_once()


class ReplicaTests(unittest.TestCase):
    def replica(self):
        replica = _Replica.__new__(_Replica)
        replica.config = DynamoConfig("test", "http://localhost:8000")
        replica.rank = 0
        replica.gpu_ids = ["1"]
        replica.visible = "0,1"
        replica.process = None
        replica.closed = False
        return replica

    def test_rejects_unreserved_or_mismatched_gpu_visibility(self):
        replica = self.replica()
        with patch.object(replica, "_validate_runtime"), patch("subprocess.Popen") as spawn:
            for visible in ("", "0,1", "0"):
                replica.visible = visible
                with self.assertRaisesRegex(RuntimeError, "exactly one matching"):
                    replica.launch()
            spawn.assert_not_called()

    def test_failed_validation_has_no_process_to_cleanup(self):
        replica = self.replica()
        self.assertTrue(replica.stop())
        self.assertTrue(replica.stop())


if __name__ == "__main__":
    unittest.main()
