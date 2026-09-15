# KAI Scheduler integration

This first issue #6 slice translates the same `Plan` used by the Ray backend
into Kubernetes objects for KAI Scheduler. It targets KAI Scheduler 0.17.0,
Kubernetes 1.34, and NVIDIA GPU Operator 25.10. Validate that combination in a
test cluster before using it for experiments.

## Object mapping

`build_kai_objects()` creates one external KAI `PodGroup` followed by one Pod
per planned worker. The PodGroup uses `minMember` equal to the worker count, so
the workers form one gang. Every Pod:

- uses `schedulerName: kai-scheduler`;
- requests and limits one `nvidia.com/gpu`;
- carries the KAI queue and optional node-pool labels;
- joins the external PodGroup and disables automatic regrouping;
- selects the node recorded in the plan through a configurable stable node
  label; and
- receives `RANK` and `WORLD_SIZE` environment variables.

Separate Pods are intentional. One Kubernetes Job template cannot preserve a
plan that assigns different ranks to different nodes.

Run `python -m examples.kai_manifest` to print synthetic JSON objects. This
example performs no cluster submission and is not benchmark evidence.

## Pre-submission contract

`validate_submission()` checks a cluster snapshot before any object is created.
It requires an existing KAI queue, create permission for Pods and PodGroups,
one unique Kubernetes node label for each planned node, enough allocatable
`nvidia.com/gpu` on each node, and the requested KAI node-pool label.

The planner's node name must equal the value of the configured Kubernetes node
label. Device-level UUID, PCI, and NVLink relationships discovered by V1.2 are
still observational: Kubernetes node selection cannot bind a rank to a GPU UUID.

## Installation prerequisites

Install KAI in its own namespace and submit workloads to a separate namespace.
The target namespace needs a ServiceAccount with create/get/list/watch/delete
access to Pods and `podgroups.scheduling.run.ai`. A leaf KAI queue and NVIDIA
GPU Operator/device plugin must exist before submission.

## Remaining lifecycle work

The next collaborative slice will connect the pure object builder to Kubernetes
clients for discovery/RBAC checks, ordered PodGroup-and-Pod submission, status
watching, cancellation, cleanup, and integration tests. Queueing, quota,
preemption, and gang admission differ from Ray placement groups and must be
recorded as experimental conditions rather than normalized away.
