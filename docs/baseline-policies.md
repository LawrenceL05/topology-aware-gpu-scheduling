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
