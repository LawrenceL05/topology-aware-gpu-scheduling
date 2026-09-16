# KAI Scheduler integration

See [current implementation, validation status, and versions](current-status.md)
for the shared support summary and evidence boundaries.

[Issue #6](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/6)
added a second execution backend for the same backend-neutral `Plan` used by Ray. The implementation targets KAI Scheduler 0.17.0, Kubernetes 1.34,
and NVIDIA GPU Operator 25.10. Validate that combination in a test cluster
before using it for experiments.

The lifecycle is implemented and covered by mocked Kubernetes tests and
synthetic manifests. A real KAI/GPU cluster run has not been recorded.

## Workflow

1. `plan_with_record()` runs any reference placement policy and produces a
   `Plan` plus a machine-readable planning record.
2. `preflight()` reads Kubernetes Nodes, the requested KAI Queue, and live
   SelfSubjectAccessReview results. It rejects missing labels, insufficient
   `nvidia.com/gpu`, a missing queue, a wrong node pool, or missing create
   permissions before changing the cluster.
3. `build_kai_objects()` creates one external KAI `PodGroup` and one Pod per
   planned rank. `submit()` creates the PodGroup first and rolls back a partial
   Pod submission.
4. `status()` reduces the worker Pod phases to `pending`, `running`,
   `succeeded`, or `failed`. `run()` polls this status until completion or a
   timeout.
5. `cancel()` deletes the Pods before deleting the externally managed
   PodGroup. `run()` performs this cleanup after success, failure, or timeout
   unless `cleanup_on_finish=False` is requested for debugging.

The official KAI documentation describes external PodGroups as a
controller-managed lifecycle: the caller creates the PodGroup, sets
`pod-group-name`, opts out of automatic grouping, and removes the PodGroup.
See [KAI Pod Grouper](https://github.com/kai-scheduler/KAI-Scheduler/blob/main/docs/developer/pod-grouper.md).

## Object mapping

The PodGroup uses `minMember` equal to the worker count, so all workers form one
gang. Every Pod:

- uses `schedulerName: kai-scheduler`;
- requests and limits one `nvidia.com/gpu`;
- carries the KAI queue and optional node-pool labels;
- joins the external PodGroup and disables automatic regrouping;
- selects the planned node through a configurable stable node label; and
- receives `RANK` and `WORLD_SIZE` environment variables.

Separate Pods preserve a plan that assigns different ranks to different nodes;
a single Kubernetes Job template cannot express that mapping.

Device UUID, PCI, NUMA, and NVLink facts discovered by V1.2 remain
observational inventory; individual GPU relationships do not feed the current
node-level placement score. Kubernetes node selection chooses a node but does
not bind a rank to a specific GPU UUID on that node.

## Selecting Ray or KAI

`run_with_record()` accepts `backend="ray"`, `backend="kai"`, or a custom
callable. The worker argument is a Python function for Ray and a `KAIWorkload`
for KAI:

```python
ray_result, ray_record = run_with_record(
    plan, planning, worker_function, backend="ray"
)

kai_result, kai_record = run_with_record(
    plan, planning, kai_workload, backend="kai", client=kubernetes_client
)
```

The execution record stores the selected backend. Policy inputs and placement
records therefore stay comparable while backend-specific execution remains
visible.

## Ray and KAI are different experimental conditions

| Concern | Ray | KAI Scheduler |
| --- | --- | --- |
| Atomic admission | Placement-group reservation | PodGroup gang admission |
| Resource governance | Ray logical resources | Kubernetes resources plus KAI queue quota |
| Priority/preemption | Ray scheduling semantics | KAI queue, priority, reclaim, and preemption rules |
| Placement enforcement | Ray node resource marker | Kubernetes node selector and KAI scheduler |
| Completion | Ray object references | Kubernetes Pod phases |
| Cleanup | Remove placement group | Delete Pods and external PodGroup |

Record queue configuration, quota, priority class, preemption settings, cluster
load, and backend version with benchmark results. These factors must not be
treated as equivalent merely because both backends consume the same `Plan`.

## Installation and RBAC

Install KAI in its own namespace and submit workloads to a separate namespace,
as required by the [KAI quickstart](https://github.com/kai-scheduler/KAI-Scheduler/blob/main/docs/quickstart/README.md).
Install the optional Python dependency and edit the example namespace in the
RBAC manifest before applying it:

```bash
python -m pip install -e '.[kai]'
kubectl apply -f deploy/kai-v1/rbac.yaml
```

The manifest grants namespaced lifecycle access to Pods and PodGroups plus
read-only discovery of Nodes and Queues. It also permits
SelfSubjectAccessReview creation for preflight checks. Use a leaf Queue; parent
Queues are organizational and cannot accept workloads. Queue behavior is
documented in [KAI scheduling queues](https://github.com/kai-scheduler/KAI-Scheduler/blob/main/docs/queues/README.md).

## Running and validating

Inspect generated objects without touching a cluster:

```bash
python -m examples.kai_manifest
python -m unittest tests.test_kai_backend -v
```

Run the GPU-count baseline against the current kubeconfig or in-cluster service
account. Node names, product labels, optional GPU-memory labels, and allocatable
GPU counts are read from the Kubernetes API; they are not typed into the
command. The smoke path uses the GPU-count policy and a zero-memory workload,
so a missing GPU-memory label does not affect placement:

```bash
python -m examples.kai_submit \
  --namespace research \
  --queue default-queue \
  --workers 2
```

The command prints planning, execution, selected backend, Pod phases, and bound
node names as JSON. It cleans up by default. Add `--keep` only when the objects
need to remain for inspection, then delete them explicitly.

For a real integration check, verify that all Pods are admitted together, each
Pod lands on its planned node, `nvidia-smi` succeeds, terminal phases are
reported, and no Pods or PodGroup remain after cleanup. This path requires a
KAI cluster and NVIDIA device plugin, so it is documented rather than run in
CPU-only CI.
