# Ray-managed Dynamo replica lifecycle

The [service adapter](../topology_scheduler/dynamo_backend.py) implements the
[V1 contract](dynamo-v1-contract.md) using one persistent Ray actor and one
guarded `dynamo.vllm` process per planned GPU. Replicas are independent TP=1,
PP=1, DP=1 aggregated servers. The finite-task `ray_backend.run()` API is
unchanged. This implementation has CPU/fake-engine tests; actual CUDA inference
and the pinned container build remain unverified work in
[issue #3](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/3).

## Environment and caller-owned services

Use the [pinned container recipe](../deploy/dynamo-v1/README.md) on Linux/amd64
with Python 3.12. Before launching an engine, each actor checks Ray 2.55.0,
ai-dynamo 1.4.2, and vLLM 0.26.0. Normal package imports and unit tests do not
import or install Dynamo/vLLM. The process guardian needs Linux `/proc` and
`PR_SET_CHILD_SUBREAPER`; workers must be permitted to inspect and signal their
own descendants. Windows is not a production worker platform for this adapter.

Start etcd, NATS, and a frontend **before** calling `start()`. Use a separate
Dynamo namespace/frontend for each deployment and configure the frontend with
the same namespace, discovery, model, and messaging settings as its replicas.
The adapter checks `/v1/models` reachability before reserving any replica
resources, but cannot inspect the frontend's configuration for you. Loopback
endpoints are only appropriate when that service is actually on every relevant
host; use reachable shared endpoints across nodes.

Frontend CPU capacity is caller-owned and separate from `replica_cpus`. If the
frontend runs in Ray, start it with its own CPU allocation before the replica
reservation. If it runs outside Ray, account for its CPU needs when setting the
nodes' advertised Ray capacity. The adapter reserves only replica CPUs/GPUs and
never schedules a frontend behind a placement group consuming all those CPUs.
It never shuts down or restarts etcd, NATS, or the frontend.

All cooperating drivers must connect to the Ray namespace
`topology-scheduler-dynamo`. The placement group and actors use the explicit
`DynamoConfig.namespace` in their names; duplicate names fail rather than reuse
another service. Choose non-overlapping system-port ranges for concurrent
deployments on the same hosts. `system_port_base + replica_index` must be free;
the actor checks before launch. This is not a cluster-wide port allocator.

## Where the configuration comes from

`DynamoConfig` field defaults mirror
[`contract.json`](../deploy/dynamo-v1/contract.json): the model and its
revision, the maximum model length, GPU memory utilization, the namespace,
etcd and NATS endpoints, the first system port, and the startup and shutdown
deadlines. A copy drifts as soon as the contract changes, so
`DynamoConfig.from_contract()` derives them instead, and a test asserts the
derived configuration equals the defaults. Use it, and override only what a
deployment must change:

```python
from topology_scheduler.dynamo_backend import DynamoConfig

config = DynamoConfig.from_contract(
    namespace="experiment-001",
    frontend_url="http://FRONTEND_HOST:8000",
)
```

`from_contract()` also turns the contract's `0.0.0.0` bind address into a
reachable client URL. Its default path points into the repository, so an
installed wheel without `deploy/` must pass a path.

`ADAPTER_OWNED` and `CALLER_OWNED` name the same split the contract does, and
a test pins them to it. `close()` stops what the first names and never touches
the second.

## Driver API

On a prepared cluster, using an existing plan and caller-started frontend:

```python
import json
from pathlib import Path
import ray
from topology_scheduler import DynamoConfig, DynamoService

ray.init(address="auto", namespace="topology-scheduler-dynamo")
config = DynamoConfig.from_contract(
    namespace="experiment-001",  # also set on the caller's dedicated frontend
    frontend_url="http://FRONTEND_HOST:8000",
    etcd_endpoints="http://ETCD_HOST:2379",
    nats_server="nats://NATS_HOST:4222",
    replica_cpus=1,
)
service = DynamoService(plan, config)
try:
    with service:
        Path("deployment.json").write_text(json.dumps(service.status()))
        endpoint = service.endpoint
        # Send repeated OpenAI-compatible requests to endpoint; reservations persist.
finally:
    service.close()  # idempotent; close before disconnecting Ray
    ray.shutdown()
```

Construct `plan` with `cross_node_gb_per_pair=0`: these replicas do not exchange
activations or KV cache. `Plan.estimated_seconds` remains a placement score,
not serving latency or throughput. The number of plan workers is the replica
count. Model repository/revision and memory/length limits are configurable;
revision must be an immutable 40-character commit. Unsupported TP/PP,
disaggregation, malformed URLs, and invalid deadlines are rejected before
reservation. There is no arbitrary engine-argument or GPU-ID override.

Each actor consumes one GPU, `replica_cpus` CPUs, and its unique node marker
inside an explicit placement-group bundle. Marker uniqueness and total node
capacity are checked first; the placement group provides the actual atomic
reservation. Reservation and startup deadlines are separate. Ray-selected
`CUDA_VISIBLE_DEVICES` must contain exactly the single assigned GPU and is
passed unchanged to the guardian and engine. The command explicitly uses
vLLM's local `mp` executor. Inherited `DYN_*`, `VLLM_*`, and `RAY_*` overrides
are removed from the child environment before the supported configuration is
applied; model credentials and cache settings outside those prefixes remain
inherited. Do not place secrets in logs or configuration URLs.

## Readiness and ongoing failure detection

`start()` returns the service only after every engine is alive, every local
system-port `/health` succeeds, the frontend lists the model, and a one-token
chat completion succeeds. `endpoint` raises unless the service is ready.
Model loading alone or a successful process spawn is insufficient.

`status()` returns deployment/namespace/group identity, lifecycle state, latest
error, and per-replica node/GPU assignment and log/receipt paths. A driver thread
checks actor/guardian health every `heartbeat_interval`; status is an observed
state, not an atomic guarantee that the next request will succeed. Runtime
actor/engine failure closes the whole service instead of silently keeping a
smaller set of replicas. Startup errors include replica/node metadata and logs.

## Shutdown, driver death, and retained reservations

`close()` stops all replica trees before killing their actors and removing the
placement group. The [guardian](../topology_scheduler/_dynamo_guardian.py)
tracks its descendants, including children that create a new session, sends
SIGTERM, waits up to `shutdown_timeout`, then uses SIGKILL with a bounded
`kill_timeout`. It writes a deployment-specific receipt only after all owned
descendants are gone. Placement-group removal is asynchronous in Ray.

The guardian's input pipe belongs only to its Ray actor. Actor death closes
that pipe; driver death stops heartbeat renewals. The guardian therefore stops
the engine on EOF or `lease_timeout` expiry. Graceful interpreter exit also
attempts `close()` via `atexit`; context management is the preferred path.
SIGKILL, machine loss, Ray failure, or an unresponsive kernel cannot be treated
as successful cleanup. If actor/guardian cleanup cannot be verified, the
adapter raises `DynamoCleanupError` and **retains its detached reservation**.
It never calls Ray removal merely because a cleanup timeout elapsed. A dead
node requires operator confirmation/recovery; no remote software can prove
its processes stopped while it is unreachable.

Detached groups and actors intentionally outlive an abruptly disconnected
driver. Engine leases expire, but GPU reservations can remain until recovery.
Keep `deployment.json` and the original plan/config. Reconnect to the original
cluster/coordination namespace and, for a complete recorded deployment, run:

```python
snapshot = json.loads(Path("deployment.json").read_text())
recovery = DynamoService.recover_for_cleanup(plan, config, snapshot)
recovery.close()
```

Recovery verifies the original placement-group ID and actor tokens to avoid
stopping a newer deployment. It checks receipts on the recorded nodes when an
actor has died. Repeat `close()` if a temporarily unavailable node returns.
For incomplete startup metadata or missing receipts, inspect the named actors,
node-local logs, and process tree before an operator removes resources. Never
delete a reservation or receipt solely to silence a cleanup error. If the Ray
cluster itself is destroyed, the enclosing node/container supervisor must stop
remaining processes before those GPUs are returned to another scheduler.

## CPU-only validation

```bash
python -m unittest discover -s tests -v
# Linux, Ray installed; simulated logical GPUs and fake HTTP engines only:
python -m examples.dynamo_smoke
```

The [smoke](../examples/dynamo_smoke.py) exercises real Ray bundle assignment,
two requests per persistent fake replica, cleanup, and partial-startup rollback.
It consumes all two advertised CPUs for replicas without needing a third CPU
to schedule an adapter-owned frontend. Guardian tests kill actual Linux child
trees on actor-pipe EOF, driver-lease expiry, and engine exit. Unit tests cover
configuration, binding, timeouts, rollback, and retained reservations. Ordinary
CI needs no GPUs, Dynamo, model downloads, or infrastructure services. Missing
real-GPU validation remains unverified and is not replaced by these simulations.
