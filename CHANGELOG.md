# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and package versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Package `0.1.2` is in development; V1.2 names the topology milestone, not a
published release. See [current status and validation](docs/current-status.md).

### Added

- Matched serial job traces across all five reference policies, terminal failure
  records, observed GPU-count JCT normalization, shared execution conformance
  tests, and a simulated two-node Ray comparison example
  ([PR #34](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/34)).
- A concise V1.2 workflow covering discovery, planner inputs, node placement,
  Ray reservation, execution, and cleanup, with explicit device-enforcement
  and validation boundaries
  ([PR #8](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/8)).

- A reproducible single-GPU validation report and workflow covering physical
  NVML inventory, Ray GPU assignment, oversubscription rejection, baseline
  policies, backend contract suites, and a direct CUDA smoke test. Multi-GPU,
  live KAI, and Dynamo/vLLM validation remain outstanding.

- A persistent Ray-managed Dynamo replica adapter with atomic reservations,
  pinned TP=1 worker processes, readiness checks, Linux process-tree guardians,
  driver leases, rollback, and conservative cleanup/recovery. Includes CPU
  lifecycle tests and a real-Ray fake-engine smoke; real-GPU inference remains
  unverified ([PR #30](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/30)).

- A project roadmap defining topology, Dynamo, evaluation, and release-readiness
  milestones, issue triage, dependencies, exit criteria, and maintainer setup
  ([PR #26](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/26)).
- Documentation update rules, PR checklist, shared status/version summary, and
  GPU-free CI checks for local documentation links and approved examples
  ([PR #22](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/22)).

- Initial KAI Scheduler object adapter with gang scheduling, GPU requests,
  planned-node selection, queue/node-pool metadata, and preflight validation
  ([PR #12](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/12)).
- KAI Kubernetes client lifecycle for live discovery and RBAC checks, ordered
  submission, status polling, cancellation, rollback, timeout, and cleanup
  ([PR #13](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/13)).
- Named `ray` and `kai` backend selection with backend identity in execution
  records, a live KAI smoke example, and deployable RBAC manifests
  ([PR #13](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/13));
  a physical KAI/GPU cluster run is not yet recorded.
- Synthetic KAI manifest example and versioned integration guide
  ([PR #12](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/12)).
- Five deterministic reference policies for GPU-count, accelerator-type,
  workload-compute, topology-only, and combined placement comparisons
  ([PR #11](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/11)).
- Machine-readable planning records and a synthetic policy comparison example
  ([PR #11](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/11)).
- Controlled normalized-JCT comparison guidance
  ([PR #11](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/11)).
- Contribution guidance for issues, development, research evidence, testing,
  documentation, and pull requests
  ([commit 6468116](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/commit/6468116)).
- Automatic pairwise intra-node GPU topology discovery, including normalized
  PCI/NUMA ancestry and active direct NVLink counts
  ([PR #7](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/7)).
- A public `GPUConnection` data model and a V1.2 topology discovery guide
  ([PR #7](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/7)).
- A release-specific V1 Dynamo contract for aggregated vLLM serving with
  independent single-GPU replicas, plus a machine-readable example and
  container recipe
  ([PR #9](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/9));
  the lifecycle adapter now has CPU/fake-engine coverage; GPU inference
  validation remains outstanding.
- A documented release process covering semantic-versioning rules, a release
  checklist, annotated `vX.Y.Z` tags, release-note contents, and rollback and
  tag-correction rules
  ([PR #25](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/25));
  no tag or release is published by it.
- Network interface discovery on every live Ray node, recording MAC, PCI
  function, NUMA node, driver, state, MTU, advertised link speed, and RDMA
  devices matched by PCI address, with per-field source and confidence so an
  unavailable, unsupported, or unreadable value is never mistaken for zero, a
  runnable example, and stable serialization beside the GPU inventory
  ([PR #33](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/33));
  advertised speed is not measured throughput. PCI/driver read failures retain
  their confidence, virtual classification requires explicit sysfs evidence,
  and Ray discovery validates markers before dispatch, cancels failed probes,
  and has deterministic tests plus a same-host two-node smoke check.

- The Dynamo configuration is derived from the pinned contract through
  `DynamoConfig.from_contract()`, with declared adapter and caller ownership
  lists, so the contract stays the single source of truth instead of being
  copied into field defaults.

### Changed

- Per-node Ray probes now return a complete GPU relationship graph alongside
  the V1.1 device inventory
  ([PR #7](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/7)).
- Ray is pinned to 2.55.0 because Dynamo 1.4.2's vLLM dependency requires Ray
  2.55.0 or newer
  ([PR #9](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/9)).

### Known limitations

- The Ray adapter cannot yet bind a worker to a selected physical GPU UUID, so
  discovered device-level relationships are observational and are not used in
  placement scoring.
- NVML may not expose a topology property on every driver and GPU; unavailable
  relationship fields are reported as `None`.
- GPU-to-NIC affinity and inter-node bandwidth or latency are not discovered.
  The NIC inventory reports advertised link speed from Linux sysfs only; that
  is not measured throughput, and virtualized hosts often leave PCI, NUMA, or
  speed unavailable.

### Planned

- Dynamo image-build and real-GPU validation remain tracked in
  [issue #3](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/3).

## [0.1.1] - 2026-09-14

Source: [tagged commit 9dcb40f](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/commit/9dcb40f13d2d836a444ab0ddabfd53d03ba31266).

### Added

- Automatic NVIDIA GPU discovery on every live Ray GPU node.
- Per-GPU model, UUID, total memory, and PCI bus ID collection through Ray's
  bundled NVML bindings.
- Conversion from discovered cluster inventory to placement-planner `Node`
  inputs.
- A runnable inventory example and a dedicated V1.1 workflow guide.
- Inventory validation for node markers, logical GPU capacity, homogeneous GPU
  models, and uniform per-GPU memory.

### Changed

- GPU model, count, and memory no longer need to be entered manually when using
  `discover_planner_nodes()`.
- Simplified the README comparison between GPU-count scheduling and the V1.1
  placement policy.

### Known limitations

- Automatic discovery requires NVIDIA NVML and has not yet been validated on a
  physical multi-node GPU cluster.
- A planner node must contain one uniform GPU model and memory size.
- Network and NVLink topology are not discovered automatically.
- Workload compute estimates, memory requirements, communication volume, and
  link bandwidth remain experimental inputs.
- NVIDIA Dynamo integration is not implemented in this release.

## [0.1.0] - 2026-09-13

Source: [tagged commit 4db520f](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/commit/4db520f6343fcc4e7ecee086a811328ddee6bbc9).

### Added

- Exhaustive small-cluster placement planner using workload-specific GPU
  compute estimates, per-GPU memory, capacity, and inter-node link costs.
- Ray placement-group adapter that reserves all worker bundles together, binds
  each task to its planned node, and cleans up after success or failure.
- Synthetic planner example plus single-node and multi-node Ray smoke examples.
- Planner and mocked Ray-backend tests.
- Ray 2.49.0 source and integration guide.

### Known limitations

- GPU inventory and network bandwidth are supplied manually.
- Smoke examples use simulated logical GPUs and execute no CUDA workload.
- The cost model is not an NCCL simulator or measured JCT predictor.
- No queue model, fairness, preemption, automatic replanning, or Dynamo
  integration is included.

[Unreleased]: https://github.com/LawrenceL05/topology-aware-gpu-scheduling/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/LawrenceL05/topology-aware-gpu-scheduling/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/LawrenceL05/topology-aware-gpu-scheduling/releases/tag/v0.1.0
