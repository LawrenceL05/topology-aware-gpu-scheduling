# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and package versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Contribution guidance for issues, development, research evidence, testing,
  documentation, and pull requests.
- Automatic pairwise intra-node GPU topology discovery, including normalized
  PCI/NUMA ancestry and active direct NVLink counts.
- A public `GPUConnection` data model and a V1.2 topology discovery guide.

### Changed

- Per-node Ray probes now return a complete GPU relationship graph alongside
  the V1.1 device inventory.

### Known limitations

- The Ray adapter cannot yet bind a worker to a selected physical GPU UUID, so
  discovered device-level relationships are observational and are not used in
  placement scoring.
- NVML may not expose a topology property on every driver and GPU; unavailable
  relationship fields are reported as `None`.
- NIC affinity and inter-node bandwidth or latency are not discovered.

### Planned

- NVIDIA Dynamo integration is tracked in
  [issues 1–3](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues).

## [0.1.1] - 2026-09-14

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
