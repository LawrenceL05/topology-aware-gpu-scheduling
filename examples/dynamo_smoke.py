"""Linux real-Ray lifecycle smoke with simulated GPUs and fake HTTP engines.

Performs no Dynamo/CUDA inference. Exercises the production guardian, process
ownership, resource reservation, persistent requests, and startup rollback.
"""

from dataclasses import replace
import json
import os
import sys
import tempfile
import time
from uuid import uuid4

from topology_scheduler import Node, Plan
from topology_scheduler.dynamo_backend import (
    DynamoConfig, DynamoLifecycleError, DynamoService, RAY_NAMESPACE, _Replica,
)


ENGINE = r'''
import json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    count = 0
    def do_GET(self):
        if self.path == '/complete':
            Handler.count += 1
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({'count': Handler.count,
            'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}).encode())
    def log_message(self, *args):
        pass
HTTPServer(('127.0.0.1', int(os.environ['DYN_SYSTEM_PORT'])), Handler).serve_forever()
'''


class FakeReplica(_Replica):
    def _validate_runtime(self):
        if sys.platform != "linux":
            raise RuntimeError("This simulated smoke still needs Linux process containment")

    def _command(self):
        return [sys.executable, "-c", ENGINE]

    def request(self):
        from topology_scheduler.dynamo_backend import _http_json
        return _http_json(f"http://127.0.0.1:{self.config.system_port_base + self.rank}/complete", 2)


class FailingReplica(FakeReplica):
    def _command(self):
        return [sys.executable, "-c", "raise SystemExit(7)"] if self.rank == 1 else super()._command()


def fake_frontend(*args, **kwargs):
    return True  # This smoke does not claim to validate Dynamo routing/model loading.


def wait_removed(ray, group):
    from ray.util.placement_group import placement_group_table
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if placement_group_table(group).get("state") == "REMOVED":
            return
        time.sleep(0.1)
    raise AssertionError("Placement group removal did not complete")


def main():
    if sys.platform != "linux":
        raise SystemExit("Run on Linux; no simulated pass is reported on other platforms")
    import ray
    ray.init(num_cpus=2, num_gpus=2, namespace=RAY_NAMESPACE, include_dashboard=False,
             resources={"topology_node:local": 2})
    try:
        with tempfile.TemporaryDirectory() as directory:
            config = DynamoConfig("smoke-" + uuid4().hex[:12], "http://127.0.0.1:8000",
                                  log_dir=directory, startup_timeout=60, shutdown_timeout=1,
                                  system_port_base=28181, heartbeat_interval=0.5)
            plan = Plan((Node("local", "SIMULATED", 2, 80),) * 2, None)
            service = DynamoService(plan, config, _replica_class=FakeReplica, _probe=fake_frontend)
            with service:
                group = service.group
                records = service.status()["replicas"]
                assert len({record["cuda_visible_devices"] for record in records}) == 2
                assert all(len(record["gpu_ids"]) == 1 for record in records)
                for expected in (1, 2):
                    results = ray.get([actor.request.remote() for actor in service.actors])
                    assert [result["count"] for result in results] == [expected, expected]
                    assert {result["cuda_visible_devices"] for result in results} == {"0", "1"}
                assert service.state == "ready" and service.group is group
            service.close()
            wait_removed(ray, group)
            failed = DynamoService(plan, replace(config, namespace="failure-" + uuid4().hex[:12]),
                                   _replica_class=FailingReplica, _probe=fake_frontend)
            try:
                failed.start()
            except DynamoLifecycleError:
                assert failed.state == "closed", failed.status()
            else:
                raise AssertionError("A dead engine was accepted as ready")
            print(json.dumps({"simulated_gpus": True, "dynamo_inference": False,
                              "persistent_requests": "passed", "partial_startup_rollback": "passed",
                              "replicas": records}))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
