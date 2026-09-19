# Dynamo real-GPU validation

See [current implementation, validation status, and versions](current-status.md)
for the shared support summary and evidence boundaries.

The [replica lifecycle](dynamo-lifecycle.md) is implemented and covered by
tests with fake engines. None of that proves a model ever served a request.
This page is how that proof is produced, and until a run report exists the
integration stays **implemented but GPU-unverified**.

Ordinary CI never runs any of this. The validation is opt-in, and it refuses to
substitute simulated GPUs for the real inference path.

## Where the boundary is

| Layer | Responsibility |
| --- | --- |
| Ray | Reserves bundles, assigns physical GPUs, pins each replica to its planned node |
| This adapter | Launches and stops one `dynamo.vllm` worker per reserved GPU, and reports readiness |
| Dynamo | Frontend, discovery, routing, and the OpenAI-compatible HTTP surface |
| vLLM | Loads the model and generates tokens inside each worker |

A placement score from the planner is not a serving measurement, and a single
completion is not a benchmark. Startup and model-load time is recorded
separately from request latency, and neither is a normalized-JCT result: that
needs a stated job boundary, a matched baseline denominator, identical model
and hardware, and explicit failure handling.

## Dry run first, on any machine

```bash
python -m examples.dynamo_dry_run
```

This prints the plan, each replica's node, system port, log directory, and the
full worker command line, plus the ownership split, and performs no Dynamo or
CUDA work at all. It is the cheapest way to review configuration before
touching a GPU, and it runs in CI.

## What a real run needs

- **Hardware and OS.** A Linux amd64 host with at least one NVIDIA GPU and a
  driver at or above the contract's minimum. `nvidia-smi` must work. The
  adapter also refuses to launch a replica unless the Python and package
  versions match the contract exactly.
- **The pinned image.** Dynamo and vLLM are not Python dependencies of this
  project; they live in the image pinned by
  [the contract](dynamo-v1-contract.md). Run the driver and workers inside it.
- **Caller-owned services.** etcd, NATS, and one Dynamo frontend for this
  deployment's namespace, started as the lifecycle guide documents. The adapter
  never starts or stops them, and it refuses to reserve replica CPUs until the
  frontend answers.
- **Model access.** The contract pins a model id and revision. The weights must
  be downloadable on every node that will host a replica, with credentials in
  place if the repository is gated. Pre-downloading avoids a first-run timeout
  being mistaken for a failure.
- **Ray.** Started on each node with one unique `topology_node:<name>` resource
  and its real GPU count, and the driver must use the
  `topology-scheduler-dynamo` Ray namespace.

## Run it

```bash
python -m examples.dynamo_gpu_validation            # Reports why it would skip.
python -m examples.dynamo_gpu_validation --run \
  --namespace experiment-001 \
  --frontend-url http://FRONTEND_HOST:8000 \
  --replicas 2 --requests 3 \
  --compute-seconds '{"H100": 10}' \
  --report dynamo-gpu-validation.json
```

Without `--run`, or on a host missing any prerequisite, it prints each reason
and exits 0 with `"status": "skipped"`. A skipped run is not a pass.

With every prerequisite present it discovers the inventory through NVML, plans
a placement, and then, in order:

1. starts a deployment whose model cannot be fetched, and confirms the failure
   rolls back and cleans up;
2. starts the real replicas and records startup and model-load time;
3. checks each replica landed on its planned node, recording node ids, GPU ids,
   and log paths;
4. sends several OpenAI-compatible completions through the frontend, asserting a
   nonempty reply each time, without releasing the reservation in between;
5. closes the deployment and confirms no worker port still answers and the GPUs
   came back.

The report records the hardware, the pinned versions, the endpoints, the worker
command, per-replica log paths, the timings, and a status for every step.
Commit or attach it; that file is the completion evidence.

## Clean up

`close()` runs automatically, including after a failure, and the adapter keeps
a cleanup receipt so an unconfirmed cleanup is retained rather than forgotten.
If a run is killed before it can finish:

```bash
ray list actors --filter "state=ALIVE"    # Replica actors are dynamo-<namespace>-<rank>.
ray status                                # Placement groups still holding GPUs.
nvidia-smi                                # Any process still on a device.
```

Reservations and actors are detached, so they survive a dead driver on purpose;
recover them through the adapter rather than by deleting resources by hand.
Stop etcd, NATS, and the frontend yourself; they are caller-owned and other
deployments may share them.

## Multi-node replicas

Start Ray on two machines, each advertising its own marker and real GPUs, and
run the validation with `--replicas 2`. The planner places one replica per node
when each node has one GPU; step 3 checks planned against actual placement, so
a mismatch fails rather than passing quietly.

These are **independent replicas**: each is a complete vLLM server that loads
the whole model and serves requests on its own. That is not a tensor-parallel
model split across nodes, where ranks of one model exchange activations every
forward pass. V1 rejects tensor and pipeline parallelism outright, so a
multi-node run here scales replicas, never a single model.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `--run` was passed but the run still skips | A prerequisite is missing; the reasons name each one and its endpoint |
| Start fails complaining about the Ray namespace | The driver must call `ray.init(namespace="topology-scheduler-dynamo")` |
| A replica is rejected before launching | Ray did not assign exactly the planned device, or the runtime versions do not match the contract |
| Startup deadline passes | Model download or load is slower than the contract's deadline; pre-download the weights |
| Requests fail though replicas started | The frontend is not seeing the workers: check the shared namespace, etcd, and NATS |
| GPUs stay busy after a run | Cleanup was not confirmed; rerun the adapter's cleanup rather than killing processes by hand |

## What is still unverified

No run report exists yet. The worker command mirrors the contract but has never
been executed against the pinned image, no engine has ever reported itself
ready, and no completion has been served. Tensor-parallel and disaggregated
prefill/decode validation remain follow-on work and are rejected by the adapter
today.
