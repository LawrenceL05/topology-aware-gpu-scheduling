# Reference baseline policies

The planner exposes five stable policy names through the same
`choose_placement()` function. Every returned `Plan` can be passed unchanged to
`topology_scheduler.ray_backend.run`, so Ray reservation, execution, timeout,
failure, and cleanup behavior are shared.

| Policy name | Selection rule | Estimated score |
| --- | --- | --- |
| `gpu_count` | First feasible GPU placement | `null` |
| `accelerator_type` | First feasible placement matching the requested Ray-style accelerator type | `null` |
| `workload_compute` | Lowest slowest-worker compute estimate | Compute seconds |
| `topology_only` | Lowest serialized cross-node communication cost | Communication seconds |
| `combined` | Lowest slowest-worker compute plus communication cost | Total estimated seconds |

All policies reject insufficient GPU count and per-GPU memory. Policies that
use compute estimates also reject GPU models missing from the workload profile.
Policies that use topology reject a communicating candidate when its link is
unknown. Candidates and equal scores are ordered by the tuple of node names, so
the result does not depend on input order. `gpu_count` deliberately does not
prefer a GPU model.

## Synthetic comparison

Run:

```bash
python -m examples.compare_policies
```

The output records the stable policy name, full synthetic inputs, chosen node
placement, estimated score when the policy defines one, and monotonic planning
start and finish times. Use `run_with_record()` to send any returned plan through
the shared Ray backend and add execution start, finish, terminal status, and
failure details. A backend failure raises `RecordedExecutionError`; its `record`
remains available for serialization. The example demonstrates wiring and policy
differences; it is not benchmark evidence.

## Executing a matched trace

`run_matched_trace()` replays a nonempty ordered sequence of `TraceJob` objects
under all five policies in the table's order. Each job ID must be unique within
the trace. Every policy receives the same node inventory, workload profiles,
rank callables, bandwidth input, backend, and backend options. Only
`accelerator_type` uses the supplied accelerator constraint. The default
backend is the existing Ray finite-task adapter; a callable backend with the
same contract can also be supplied. This API does not accept KAI workloads,
which use a different worker representation.

On an initialized Ray cluster with the appropriate node markers:

```python
from topology_scheduler import TraceJob, run_matched_trace

records = run_matched_trace(
    nodes, [TraceJob("job-001", workload, worker)], bandwidth,
    accelerator_type="H100", reservation_timeout=60, execution_timeout=300,
)
serialized = [record.as_dict() for record in records]
```

The caller supplies `nodes`, `workload`, `worker(rank)`, and `bandwidth` as in
the [Ray guide](ray-integration.md). Ray workers must be serializable functions,
as required by the existing adapter; use a closure to bind additional arguments.
Worker return values are discarded by this
measurement harness; workload artifacts should be persisted by the worker.
Jobs run serially, and each policy completes the whole trace before the next
policy starts. This does not simulate an arrival schedule or concurrent queue.
There are no retries. Planning failures and backend exceptions produce terminal
records and subsequent jobs still run; process interrupts propagate.
Custom backends must perform cleanup on success and failure, as required by the
shared execution contract. Ray requests placement-group removal asynchronously;
the next job waits for its own reservation before launching workers.

Run the self-contained integration example after installing `.[ray]`:

```bash
python -m examples.compare_policy_traces
```

It starts two local Ray nodes with simulated GPUs and replays a successful job,
an intentional worker failure, and another successful job for every policy.
Successful workers verify their assigned Ray node and GPU allocation; the
post-failure job also checks that resources can be reserved again. Output is
JSON with full inputs, policy, placement, score, backend execution identity,
status/error, failure stage, timing boundaries, and normalization details.
Planning failures have an empty placement and no execution fields. The example
is simulated integration evidence, **not GPU or inference benchmark evidence**.
Linux CI runs it alongside the existing Ray smoke examples.

## Controlled JCT experiments

Real comparisons must run every policy against the same ordered workload trace,
cluster state, Ray backend, timeout behavior, failure handling, warmup, and
measurement window. Define a job from submission until every required worker
finishes. Record failed and timed-out jobs rather than dropping them.

For job `j`, normalized JCT is:

```text
normalized_jct(j, policy) = observed_jct(j, policy) / observed_jct(j, gpu_count)
```

The denominator is the matched `gpu_count` run for the same job and experimental
condition. Report the aggregation method and retain raw JCT values, failures,
policy inputs, placements, and timing boundaries with every result. Planning
timestamps alone are not job-completion boundaries; an experiment harness must
also record submission, execution start, and terminal success, failure, or
timeout times.

The matched trace harness uses driver-local `perf_counter_ns()` timestamps;
these are monotonic nanoseconds, not wall-clock dates or cross-host timestamps.
`submitted_ns` precedes planning. `terminal_ns` follows the shared backend's
return or exception, including its cleanup calls. Successful `jct_ns` therefore
includes planning, reservation wait, task startup, execution of all workers,
and adapter cleanup overhead. It is not inference request latency or the exact
instant the last worker finishes. Execution start/finish boundaries are also
retained so this convention is explicit.

Failed attempts retain elapsed time and error details, but have `jct_ns: null`
and no normalized JCT. A successful job is normalized only when the matching
`gpu_count` attempt succeeded with positive JCT. The result names its baseline
job and policy, retains the raw denominator, and explains unavailable ratios
with `job_failed`, `baseline_failed`, or `nonpositive_baseline_jct`. No aggregation
is performed and no failures are silently filtered.

For real experiments, callers must additionally record the hardware/runtime
versions, worker code and input revision, random seeds, timeout settings,
warmup and reset procedure, repetitions, and external cluster load. The serial
runner cannot reset cluster state, undo worker side effects, or enforce equal
cache/thermal conditions between policies. Fixed policy order and first-run
startup overhead can bias timings. Use controlled resets and repeated trials
before interpreting ratios; never interpret this smoke example's ratios as
performance comparisons. Device-level topology scoring and enforceable physical
GPU assignment remain outside this node-level baseline interface.
