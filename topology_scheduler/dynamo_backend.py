"""Ray-owned persistent, independent TP=1 Dynamo replicas. Imports are optional."""

import atexit
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from .policy import Plan, positive


RUNTIME_PINS = {"ray": "2.55.0", "ai-dynamo": "1.4.2", "vllm": "0.26.0"}
RAY_NAMESPACE = "topology-scheduler-dynamo"
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "deploy" / "dynamo-v1" / "contract.json"
# Mirrors contract.json: close() stops the first list and never the second.
ADAPTER_OWNED = ("ray_placement_group", "dynamo_vllm_workers", "worker_logs")
CALLER_OWNED = ("etcd", "nats", "dynamo_frontend", "model_credentials")


@dataclass(frozen=True)
class DynamoConfig:
    namespace: str
    frontend_url: str
    etcd_endpoints: str = "http://127.0.0.1:2379"
    nats_server: str = "nats://127.0.0.1:4222"
    model: str = "Qwen/Qwen3-0.6B"
    revision: str = "c1899de289a04d12100db370d81485cdf75e47ca"
    max_model_length: int = 4096
    gpu_memory_utilization: float = 0.80
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    disaggregation_mode: str = "agg"
    replica_cpus: float = 1
    system_port_base: int = 18081
    reservation_timeout: float = 60
    startup_timeout: float = 600
    shutdown_timeout: float = 30
    kill_timeout: float = 5
    rpc_timeout: float = 5
    poll_interval: float = 1
    heartbeat_interval: float = 2
    lease_timeout: float = 30
    log_dir: str = "/tmp/topology-scheduler-dynamo"

    def validate(self, replicas):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,62}", self.namespace):
            raise ValueError("namespace must be an explicit unique deployment identifier")
        for name, schemes, values in (
            ("frontend_url", {"http", "https"}, [self.frontend_url]),
            ("etcd_endpoints", {"http", "https"}, self.etcd_endpoints.split(",")),
            ("nats_server", {"nats", "tls"}, [self.nats_server]),
        ):
            for value in values:
                parsed = urlsplit(value)
                if parsed.scheme not in schemes or not parsed.hostname or parsed.username \
                        or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
                    raise ValueError(f"{name} must contain server URLs without credentials or paths")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.model) or \
                not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Supply a model repository and exact 40-character revision")
        if (type(self.tensor_parallel_size) is not int or self.tensor_parallel_size != 1
                or type(self.pipeline_parallel_size) is not int or self.pipeline_parallel_size != 1
                or self.disaggregation_mode != "agg"):
            raise ValueError("Only independent TP=1/PP=1 aggregated replicas are supported")
        if type(replicas) is not int or replicas < 1:
            raise ValueError("Plan must contain replicas")
        if type(self.max_model_length) is not int or self.max_model_length < 1:
            raise ValueError("max_model_length must be a positive integer")
        positive(self.gpu_memory_utilization, "gpu_memory_utilization")
        if self.gpu_memory_utilization >= 1:
            raise ValueError("gpu_memory_utilization must leave runtime headroom")
        for name in ("replica_cpus", "reservation_timeout", "startup_timeout", "shutdown_timeout",
                     "kill_timeout", "rpc_timeout", "poll_interval", "heartbeat_interval", "lease_timeout"):
            positive(getattr(self, name), name)
        if self.lease_timeout <= 2 * (self.heartbeat_interval + self.rpc_timeout):
            raise ValueError("lease_timeout must exceed two heartbeat plus RPC intervals")
        if type(self.system_port_base) is not int or not 1024 <= self.system_port_base <= 65536 - replicas:
            raise ValueError("Replica system ports must fit in 1024..65535")
        if not self.log_dir.startswith("/"):
            raise ValueError("log_dir must be an absolute Linux path")

    def worker_command(self):
        return [sys.executable, "-m", "dynamo.vllm", "--namespace", self.namespace,
                "--discovery-backend", "etcd", "--request-plane", "nats", "--event-plane", "nats",
                "--model", self.model, "--revision", self.revision, "--served-model-name", self.model,
                "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1",
                "--data-parallel-size", "1", "--distributed-executor-backend", "mp",
                "--disaggregation-mode", "agg", "--no-headless",
                "--max-model-len", str(self.max_model_length),
                "--gpu-memory-utilization", str(self.gpu_memory_utilization)]

    @classmethod
    def from_contract(cls, path=CONTRACT_PATH, **overrides):
        """Build a configuration from the V1 contract document.

        The field defaults above copy the contract, and a copy drifts the
        moment the contract changes. Deriving them keeps one source of truth;
        a test asserts the two agree. ``overrides`` still apply for whatever a
        deployment must change, such as a frontend on another host.

        The default path points into the repository, so an installed wheel
        that does not ship ``deploy/`` must pass its own path.
        """
        contract = json.loads(Path(path).read_text(encoding="utf-8"))
        model, service = contract["model"], contract["service"]
        host = service["frontend_host"]
        return cls(**{
            "namespace": service["namespace"],
            "frontend_url": "http://{}:{}".format(
                "127.0.0.1" if host == "0.0.0.0" else host, service["frontend_port"]),
            "etcd_endpoints": service["etcd_endpoints"],
            "nats_server": service["nats_server"],
            "model": model["id"],
            "revision": model["revision"],
            "max_model_length": model["max_model_length"],
            "gpu_memory_utilization": model["gpu_memory_utilization"],
            "tensor_parallel_size": service["tensor_parallel_size"],
            "system_port_base": service["worker_system_port_base"],
            "startup_timeout": service["startup_deadline_seconds"],
            "shutdown_timeout": service["shutdown_deadline_seconds"],
            **overrides,
        })


def _http_json(url, timeout, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _frontend_probe(config, *, completion=False, timeout=None):
    timeout = config.rpc_timeout if timeout is None else timeout
    deadline = time.monotonic() + timeout
    base = config.frontend_url.rstrip("/")
    models = _http_json(base + "/v1/models", timeout)
    if not isinstance(models, dict) or not isinstance(models.get("data"), list):
        raise RuntimeError("Frontend did not return an OpenAI model list")
    if not completion:
        return True  # Reachability is required BEFORE reserving replica CPUs.
    if config.model not in {item.get("id") for item in models["data"]}:
        return False
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    reply = _http_json(base + "/v1/chat/completions", remaining, {
        "model": config.model, "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1, "stream": False,
    })
    return bool(reply.get("choices")) and not reply.get("error")


def _receipt(path, token):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data.get("token") == token and data.get("stopped") is True
    except (OSError, ValueError):
        return False


class _Replica:
    """Each actor owns one guardian; no engine is spawned by the constructor."""

    def __init__(self, config, rank, deployment):
        import ray
        self.config, self.rank = config, rank
        self.node_id = ray.get_runtime_context().get_node_id()
        self.gpu_ids = ray.get_gpu_ids()
        self.visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.token = f"{deployment}-{rank}"
        directory = Path(config.log_dir) / deployment
        self.log_path = str(directory / f"replica-{rank}.log")
        self.receipt_path = str(directory / f"replica-{rank}.stopped.json")
        self.process = None
        self.closed = False

    def info(self):
        return {"rank": self.rank, "node_id": self.node_id, "gpu_ids": self.gpu_ids,
                "cuda_visible_devices": self.visible, "log_path": self.log_path,
                "receipt": self.receipt_path, "token": self.token}

    def _validate_runtime(self):
        if sys.platform != "linux" or platform.machine() not in {"x86_64", "AMD64"}:
            raise RuntimeError("Pinned Dynamo workers require Linux/amd64")
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("Pinned Dynamo image requires Python 3.12")
        for package, expected in RUNTIME_PINS.items():
            if version(package) != expected:
                raise RuntimeError(f"Expected {package}=={expected}")

    def _command(self):
        return self.config.worker_command()

    def launch(self):
        if self.closed or self.process is not None:
            raise RuntimeError("Replica cannot be launched twice")
        self._validate_runtime()
        assigned = [str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
                    for value in self.gpu_ids]
        if len(assigned) != 1 or self.visible != assigned[0]:
            raise RuntimeError("Ray must assign exactly one matching CUDA_VISIBLE_DEVICES entry")
        # Do not inherit overrides that can enable disaggregation or a nested Ray executor.
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("DYN_", "VLLM_", "RAY_"))}
        env.update(CUDA_VISIBLE_DEVICES=self.visible, DYN_NAMESPACE=self.config.namespace,
                   ETCD_ENDPOINTS=self.config.etcd_endpoints, NATS_SERVER=self.config.nats_server,
                   DYN_SYSTEM_PORT=str(self.config.system_port_base + self.rank), DYN_LOG="info")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.config.system_port_base + self.rank))
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab", buffering=0) as log:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "topology_scheduler._dynamo_guardian"],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                env=env, close_fds=True, start_new_session=True,
            )
        settings = {"command": self._command(), "receipt": self.receipt_path, "token": self.token,
                    "lease_timeout": self.config.lease_timeout,
                    "shutdown_timeout": self.config.shutdown_timeout, "kill_timeout": self.config.kill_timeout}
        self.process.stdin.write((json.dumps(settings) + "\n").encode())
        self.process.stdin.flush()
        return self.info()

    def heartbeat(self):
        if self.closed:
            return False
        if self.process is None:
            return True
        if self.process.poll() is not None:
            return False
        try:
            self.process.stdin.write(b"ping\n")
            self.process.stdin.flush()
            return True
        except OSError:
            return False

    def status(self):
        result = self.info()
        alive = self.process is not None and self.process.poll() is None
        ready = False
        if alive:
            try:
                with urlopen(f"http://127.0.0.1:{self.config.system_port_base + self.rank}/health",
                             timeout=self.config.rpc_timeout / 2) as response:
                    ready = response.status == 200
            except (OSError, ValueError):
                pass
        return {**result, "alive": alive, "ready": ready}

    def stop(self):
        self.closed = True
        if self.process is None:
            return True  # launch validation failed before spawning anything
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.write(b"stop\n")
                self.process.stdin.flush()
            except OSError:
                pass
            self.process.stdin.close()
        try:
            self.process.wait(timeout=self.config.shutdown_timeout + self.config.kill_timeout + 2)
        except subprocess.TimeoutExpired:
            return False  # Never kill the process responsible for cleaning up its tree.
        return _receipt(self.receipt_path, self.token)


class DynamoLifecycleError(RuntimeError):
    pass


class DynamoCleanupError(DynamoLifecycleError):
    """Reservations remain held when process cleanup cannot be confirmed."""


class DynamoService:
    """Explicit service owner. Call close before ray.shutdown; use as a context manager.

    Private injection arguments exist for the CPU-only lifecycle smoke/tests,
    never as an automatic production fallback.
    """

    def __init__(self, plan: Plan, config: DynamoConfig, *, _replica_class=_Replica,
                 _probe=_frontend_probe):
        config.validate(len(plan.workers))
        self.plan, self.config = plan, config
        self._replica_class, self._probe = _replica_class, _probe
        self.deployment_id = uuid4().hex
        self.group = None
        self.actors = []
        self.records = []
        self._launched = set()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._monitor = None
        self.state = "new"
        self.error = None
        self._atexit = None

    @property
    def endpoint(self):
        if self.state != "ready":
            raise DynamoLifecycleError(f"Deployment is {self.state}; no ready endpoint")
        return self.config.frontend_url.rstrip("/")

    def _context(self):
        return f"deployment={self.deployment_id}, namespace={self.config.namespace}, replicas={self.records}"

    def start(self):
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        if self.state == "ready":
            return self
        if self.state != "new":
            raise DynamoLifecycleError(f"Cannot start deployment in state {self.state}")
        if not ray.is_initialized() or ray.get_runtime_context().namespace != RAY_NAMESPACE:
            raise ValueError(f"Call ray.init(..., namespace={RAY_NAMESPACE!r}) first")
        self.config.validate(len(self.plan.workers))
        live = [node for node in ray.nodes() if node["Alive"]]
        node_ids = {}
        for key, count in Counter(node.resource_key for node in self.plan.workers).items():
            matches = [node for node in live if node["Resources"].get(key, 0) > 0]
            if len(matches) != 1:
                raise ValueError(f"{key} must identify exactly one live Ray node")
            node = matches[0]
            if any(node["Resources"].get(resource, 0) < amount for resource, amount in
                   ((key, count), ("GPU", count), ("CPU", count * self.config.replica_cpus))):
                raise ValueError(f"Insufficient total replica resources on {key}")
            node_ids[key] = node["NodeID"]
        # Caller starts and funds the frontend independently, before this reservation.
        self._probe(self.config, completion=False)
        self.state = "starting"
        try:
            self.group = placement_group(
                [{"CPU": self.config.replica_cpus, "GPU": 1, node.resource_key: 1}
                 for node in self.plan.workers], strategy="PACK", lifetime="detached",
                name=f"dynamo-{self.config.namespace}")
            ray.get(self.group.ready(), timeout=self.config.reservation_timeout)
            deadline = time.monotonic() + self.config.startup_timeout
            actor_type = ray.remote(self._replica_class)
            for rank, node in enumerate(self.plan.workers):
                actor = actor_type.options(
                    name=f"dynamo-{self.config.namespace}-{rank}", namespace=RAY_NAMESPACE,
                    lifetime="detached", num_cpus=self.config.replica_cpus, num_gpus=1,
                    resources={node.resource_key: 1}, max_restarts=0, max_task_retries=0,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.group, placement_group_bundle_index=rank,
                        placement_group_capture_child_tasks=False),
                ).remote(self.config, rank, self.deployment_id)
                self.actors.append(actor)
                record = ray.get(actor.info.remote(), timeout=self._remaining(deadline))
                self.records.append(record)
                if record["node_id"] != node_ids[node.resource_key]:
                    raise DynamoLifecycleError(f"Replica {rank} ran on the wrong node")
            self._atexit = lambda: self._exit_cleanup()
            atexit.register(self._atexit)
            self._monitor = threading.Thread(target=self._watch, daemon=True,
                                             name=f"dynamo-{self.config.namespace}")
            self._monitor.start()
            for rank, actor in enumerate(self.actors):
                self._launched.add(rank)
                ray.get(actor.launch.remote(), timeout=self._remaining(deadline))
            while True:
                if self.error:
                    raise DynamoLifecycleError(self.error)
                states = ray.get([actor.status.remote() for actor in self.actors],
                                 timeout=self._remaining(deadline))
                if any(not state["alive"] for state in states):
                    raise DynamoLifecycleError(f"Replica engine exited: {states}")
                if all(state["ready"] for state in states):
                    try:
                        ready = self._probe(self.config, completion=True,
                                            timeout=min(self.config.rpc_timeout, self._remaining(deadline)))
                    except (OSError, ValueError):
                        ready = False
                    if ready:
                        self._remaining(deadline)
                        with self._lock:
                            if self.error:
                                raise DynamoLifecycleError(self.error)
                            self.state = "ready"
                        return self
                self._stop.wait(min(self.config.poll_interval, self._remaining(deadline)))
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}; {self._context()}"
            try:
                self.close()
            except DynamoCleanupError as cleanup:
                raise DynamoCleanupError(f"{self.error}; {cleanup}") from exc
            if not isinstance(exc, Exception):
                raise
            raise DynamoLifecycleError(self.error) from exc

    @staticmethod
    def _remaining(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Dynamo startup/readiness deadline expired")
        return remaining

    def _watch(self):
        import ray
        while not self._stop.wait(self.config.heartbeat_interval):
            try:
                healthy = ray.get([actor.heartbeat.remote() for actor in self.actors],
                                  timeout=self.config.rpc_timeout)
                if not all(healthy):
                    raise DynamoLifecycleError("Replica guardian exited or lease expired")
            except Exception as exc:
                with self._lock:
                    if self._stop.is_set():
                        return
                    self.error = f"Replica/actor health failure: {exc}; {self._context()}"
                    # start() owns startup rollback; the monitor owns failures after readiness.
                    was_ready = self.state == "ready"
                if was_ready:
                    try:
                        self.close()
                    except DynamoCleanupError:
                        pass
                return

    def status(self):
        return {"deployment_id": self.deployment_id, "namespace": self.config.namespace,
                "group_id": self.group.id.hex() if self.group is not None else None,
                "state": self.state, "error": self.error, "replicas": list(self.records)}

    @classmethod
    def recover_for_cleanup(cls, plan, config, snapshot):
        """Reattach only to the exact recorded reservation; never restart a service."""
        import ray
        from ray.util.placement_group import get_placement_group
        if not ray.is_initialized() or ray.get_runtime_context().namespace != RAY_NAMESPACE:
            raise ValueError("Reconnect to the original Ray cluster and coordination namespace")
        if snapshot["namespace"] != config.namespace or len(snapshot["replicas"]) != len(plan.workers):
            raise ValueError("Recovery requires the original configuration and complete replica metadata")
        group = get_placement_group(f"dynamo-{config.namespace}")
        if group.id.hex() != snapshot["group_id"]:
            raise ValueError("The named reservation belongs to a different deployment")
        service = cls(plan, config)
        service.deployment_id = snapshot["deployment_id"]
        service.group = group
        service.records = snapshot["replicas"]
        for rank, record in enumerate(service.records):
            try:
                actor = ray.get_actor(f"dynamo-{config.namespace}-{rank}", namespace=RAY_NAMESPACE)
            except ValueError:
                actor = None
            if actor is not None:
                current = ray.get(actor.info.remote(), timeout=config.rpc_timeout)
                if current["token"] != record["token"]:
                    raise ValueError("Actor belongs to a different deployment")
            service.actors.append(actor)
        service._launched = set(range(len(service.actors)))
        service.state = "cleanup_failed"
        return service

    def close(self):
        import ray
        from ray.util.placement_group import remove_placement_group
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        with self._lock:
            if self.state == "closed":
                return
            if self.group is not None and not ray.is_initialized():
                self.state = "cleanup_failed"
                raise DynamoCleanupError("Reconnect to the original Ray cluster before cleanup; reservation retained")
            self.state = "closing"
            self._stop.set()
            failures = []
            refs = []
            for rank, actor in enumerate(self.actors):
                try:
                    refs.append((rank, actor.stop.remote()))
                except Exception:
                    refs.append((rank, None))
            deadline = time.monotonic() + self.config.shutdown_timeout + self.config.kill_timeout + 3
            for rank, ref in refs:
                stopped = False
                try:
                    stopped = ref is not None and ray.get(ref, timeout=max(0.01, deadline - time.monotonic())) is True
                except Exception:
                    pass
                if not stopped and rank not in self._launched:
                    stopped = True  # No launch RPC was issued; constructor never spawns.
                if not stopped and rank < len(self.records):
                    record = self.records[rank]
                    while not stopped and time.monotonic() < deadline:
                        proof = None
                        try:
                            proof = ray.remote(num_cpus=0)(_receipt).options(
                                scheduling_strategy=NodeAffinitySchedulingStrategy(record["node_id"], soft=False)
                            ).remote(record["receipt"], record["token"])
                            stopped = ray.get(proof, timeout=min(self.config.rpc_timeout,
                                                               max(0.01, deadline - time.monotonic()))) is True
                        except Exception:
                            if proof is not None:
                                try:
                                    ray.cancel(proof, force=True)
                                except Exception:
                                    pass
                        if not stopped:
                            time.sleep(min(0.1, max(0, deadline - time.monotonic())))
                if not stopped:
                    failures.append(rank)
            if failures:
                self.state = "cleanup_failed"
                raise DynamoCleanupError(f"Cleanup unconfirmed for replicas {failures}; detached reservation retained; {self._context()}")
            try:
                for actor in self.actors:
                    if actor is not None:
                        ray.kill(actor, no_restart=True)
                if self.group is not None:
                    remove_placement_group(self.group)
                    self.group = None
            except Exception as exc:
                self.state = "cleanup_failed"
                raise DynamoCleanupError(f"Processes stopped but Ray teardown failed: {exc}; {self._context()}") from exc
            self.state = "closed"
            if self._atexit:
                atexit.unregister(self._atexit)
                self._atexit = None

    def _exit_cleanup(self):
        try:
            self.close()
        except Exception as exc:
            print(f"Dynamo shutdown needs recovery: {exc}", file=sys.stderr)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
