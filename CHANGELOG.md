# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and package versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/LawrenceL05/topology-aware-gpu-scheduling/compare/a3668b8...HEAD
[0.1.1]: https://github.com/LawrenceL05/topology-aware-gpu-scheduling/compare/4db520f...a3668b8
[0.1.0]: https://github.com/LawrenceL05/topology-aware-gpu-scheduling/commit/4db520f
