"""Opt-in real-GPU validation of the V1 Dynamo deployment.

Ordinary CI never runs this. Without `--run`, real NVIDIA GPUs, and the
caller-owned frontend, etcd, and NATS, it prints why it skipped and exits 0. It
never substitutes simulated GPUs for the real inference path: the inventory
comes from NVML through `discover_planner_nodes()`.

It records hardware, versions, commands, timings, and a per-step status into a
run report, so a passing run is reproducible evidence and a missing run stays
explicitly unverified.
"""
import argparse
import json
import platform
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from topology_scheduler.dynamo_backend import CONTRACT_PATH, DynamoConfig

SKIPPED, PASSED, FAILED = "skipped", "passed", "failed"
RAY_NAMESPACE = "topology-scheduler-dynamo"


def reachable(endpoint: str, timeout: float = 2) -> bool:
    """True when a TCP connection to an endpoint's host and port succeeds."""
    parts = urlsplit(endpoint)
    if not parts.hostname or not parts.port:
        return False
    try:
        with socket.create_connection((parts.hostname, parts.port), timeout):
            return True
    except OSError:
        return False


def nvidia_gpus() -> tuple[str, ...]:
    """The NVIDIA GPUs this host reports, or an empty tuple when it has none."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def check_prerequisites(*, opted_in: bool, system: str, gpus: Sequence[str],
                        frontend: bool, etcd: bool, nats: bool,
                        config) -> tuple[str, ...]:
    """Every reason this host cannot run the real-GPU validation."""
    reasons = []
    if not opted_in:
        reasons.append(
            "--run was not passed; real-GPU validation is opt-in because it "
            "starts model workers on physical devices")
    if system != "Linux":
        reasons.append(f"the pinned runtime is linux/amd64 and this host is {system}")
    if not gpus:
        reasons.append("nvidia-smi reported no NVIDIA GPU on this host")
    for ok, name, endpoint in ((frontend, "Dynamo frontend", config.frontend_url),
                               (etcd, "etcd", config.etcd_endpoints),
                               (nats, "NATS", config.nats_server)):
        if not ok:
            reasons.append(
                f"the caller-owned {name} is not reachable at {endpoint}; start it "
                "from the contract guide before running this validation")
    return tuple(reasons)


def completion(endpoint: str, served_model: str, *, timeout: float) -> tuple[str, float]:
    """Send one OpenAI-compatible chat completion and time only that request."""
    payload = json.dumps({
        "model": served_model, "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with the word ready."}],
    }).encode()
    request = Request(f"{endpoint}/v1/chat/completions", data=payload,
                      headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    latency = time.perf_counter() - started
    content = body["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise AssertionError(f"the frontend returned an empty completion: {body}")
    return content, latency


def rollback_check(plan, config) -> dict:
    """Start a deployment that cannot come up, and confirm nothing is left."""
    from topology_scheduler import DynamoService
    from topology_scheduler.dynamo_backend import DynamoLifecycleError

    # A model that cannot be fetched: the engine exits, so startup must roll back.
    broken = replace(config, namespace=f"{config.namespace}-rollback",
                     model="topology-scheduler/does-not-exist", revision="0" * 40)
    service = DynamoService(plan, broken)
    try:
        service.start()
    except (DynamoLifecycleError, RuntimeError, ValueError) as error:
        return {"status": PASSED, "error": str(error)[:400],
                "state": service.status()["state"]}
    finally:
        try:
            service.close()
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            return {"status": FAILED, "error": f"cleanup failed: {error}"}
    return {"status": FAILED, "error": "a start that cannot come up still succeeded"}


def validate(arguments, config) -> dict:
    """Run the deployment end to end on real GPUs and report every step."""
    import ray

    from topology_scheduler import (
        DynamoService, Workload, choose_placement, discover_planner_nodes,
    )

    ray.init(address="auto", namespace=RAY_NAMESPACE)
    report: dict = {"steps": {}}
    service = None
    try:
        nodes = discover_planner_nodes()
        report["inventory"] = [
            {"name": node.name, "gpu_model": node.gpu_model,
             "available_gpus": node.available_gpus,
             "memory_gb_per_gpu": node.memory_gb_per_gpu} for node in nodes]
        # Independent replicas exchange nothing, so cross-node traffic is zero.
        workload = Workload(arguments.replicas, arguments.memory_gb,
                            json.loads(arguments.compute_seconds), 0)
        plan = choose_placement(nodes, workload, {})
        report["plan"] = plan.as_dict()
        report["commands"] = [config.worker_command()]

        report["steps"]["rollback"] = rollback_check(plan, config)

        started = time.perf_counter()
        service = DynamoService(plan, config)
        service.start()
        # Startup and model loading, kept separate from request latency.
        report["startup_seconds"] = time.perf_counter() - started
        status = service.status()
        report["replicas"] = status["replicas"]
        report["deployment_id"] = status["deployment_id"]

        planned = [node.name for node in plan.workers]
        node_ids = {name: next(
            node["NodeID"] for node in ray.nodes()
            if node["Alive"] and node["Resources"].get(f"topology_node:{name}", 0) > 0)
            for name in set(planned)}
        actual = [record["node_id"] for record in status["replicas"]]
        report["steps"]["placement"] = {
            "status": PASSED if actual == [node_ids[name] for name in planned] else FAILED,
            "planned": planned, "actual_node_ids": actual,
            "gpu_ids": [record["gpu_ids"] for record in status["replicas"]],
            "log_paths": [record["log_path"] for record in status["replicas"]],
        }

        replies, latencies = [], []
        for _ in range(arguments.requests):
            content, latency = completion(
                service.endpoint, config.model, timeout=arguments.request_timeout)
            replies.append(content)
            latencies.append(latency)
        # More than one request without releasing the reservation in between.
        report["request_latency_seconds"] = latencies
        report["steps"]["requests"] = {
            "status": PASSED if len(replies) == arguments.requests else FAILED,
            "count": len(replies), "replies": replies,
        }

        service.close()
        released = all(
            not reachable(f"http://127.0.0.1:{config.system_port_base + rank}")
            for rank in range(len(plan.workers)))
        report["steps"]["shutdown"] = {
            "status": PASSED if released else FAILED,
            "state": service.status()["state"],
            "available_gpus_after_close": ray.available_resources().get("GPU"),
        }
        service = None
    finally:
        if service is not None:
            try:
                service.close()
            except Exception:  # noqa: BLE001 - the report already carries the failure
                pass
        ray.shutdown()
    statuses = {step["status"] for step in report["steps"].values()}
    report["status"] = FAILED if FAILED in statuses else PASSED
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="opt in to starting real model workers on this host's GPUs")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--memory-gb", type=float, default=40)
    parser.add_argument("--compute-seconds", default='{"H100": 10}',
                        help="JSON of measured seconds per GPU model")
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--namespace", default=None,
                        help="deployment namespace; must match the caller's frontend")
    parser.add_argument("--frontend-url", default=None)
    parser.add_argument("--report", default="dynamo-gpu-validation.json")
    arguments = parser.parse_args(argv)

    overrides = {name: value for name, value in
                 (("namespace", arguments.namespace),
                  ("frontend_url", arguments.frontend_url)) if value}
    config = DynamoConfig.from_contract(**overrides)
    gpus = nvidia_gpus()
    environment = {
        "system": platform.system(), "platform": platform.platform(),
        "python": platform.python_version(), "gpus": list(gpus),
        "contract": json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["runtime"],
        "endpoints": {"frontend": config.frontend_url,
                      "etcd": config.etcd_endpoints, "nats": config.nats_server},
    }
    reasons = check_prerequisites(
        opted_in=arguments.run, system=platform.system(), gpus=gpus,
        frontend=reachable(config.frontend_url),
        etcd=reachable(config.etcd_endpoints), nats=reachable(config.nats_server),
        config=config)
    if reasons:
        report = {"status": SKIPPED, "reasons": list(reasons),
                  "environment": environment, "simulated_gpus_used": False}
    else:
        report = dict(validate(arguments, config), environment=environment,
                      simulated_gpus_used=False)
    Path(arguments.report).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in (SKIPPED, PASSED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
