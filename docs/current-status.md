# Current implementation and validation status

This page summarizes the current source tree. V1, V1.1, and V1.2 name design
milestones, not package releases. The package metadata is `0.1.2` in development;
the latest published Git tag is `v0.1.1`. Changes since that tag stay under
[Unreleased](../CHANGELOG.md) until a release is made.

## Evidence and boundaries

**Implemented** means code exists with the tests linked below. **Simulated**
means a real runtime uses artificial resources or synthetic workloads;
**mocked** tests replace an external API. **Planned** means the implementation
or validation is still outstanding. **Real-cluster-validated** requires a
reproducible run artifact with hardware, versions, commands, and results. A
[single-GPU report](one-gpu-validation.md) records physical inventory, Ray
assignment, capacity rejection, and a direct CUDA smoke test. It is not an
inference benchmark or multi-GPU validation.

| Area | Current state | Evidence and remaining boundary |
| --- | --- | --- |
| Placement and five baseline policies | Implemented; synthetic examples | [Policy tests](../tests/test_policy.py), [baseline tests](../tests/test_baseline_policies.py); scores are not measured JCT. |
| Ray finite-task adapter | Implemented; real single-GPU assignment plus mocked and simulated coverage | The [single-GPU report](one-gpu-validation.md) verifies one physical GPU assignment and two-worker capacity rejection. [Tests](../tests/test_ray_backend.py), [single-node](../examples/ray_smoke.py), and [multi-node smoke](../examples/ray_multinode_smoke.py) cover other control paths. No CUDA workload ran inside a Ray task. |
| V1.1 inventory and V1.2 intra-node topology | Physical one-GPU inventory validated; pair topology remains unverified | The [single-GPU report](one-gpu-validation.md) records real NVML identity, memory, and PCI discovery. [Inventory tests](../tests/test_inventory.py) cover pair topology with mocked NVML. No physical GPU pair was available, and graph edges remain observational rather than scoring inputs or UUID enforcement. |
| KAI object and lifecycle adapter | Implemented; mocked Kubernetes tests and synthetic manifests | [KAI tests](../tests/test_kai_backend.py), [manifest example](../examples/kai_manifest.py); live-cluster admission, execution, and cleanup need validation. |
| Dynamo V1 | Contract, environment recipe, and lifecycle adapter implemented; GPU-unverified | [Contract tests](../tests/test_dynamo_contract.py), [contract](dynamo-v1-contract.md); [lifecycle guide](dynamo-lifecycle.md), [CPU tests](../tests/test_dynamo_backend.py), and [simulated Ray smoke](../examples/dynamo_smoke.py); [GPU validation #3](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/3) remains outstanding. |
| GPU-to-NIC affinity and inter-node discovery | Planned | [NIC inventory #14](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/14), [NUMA/NIC mapping #15](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/15), [affinity graph #16](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/16), [network measurements #17](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/17). |

## Versions

These are dependency pins or intended targets, not a claim that every
combination has passed a cluster test. The documentation checker compares this
table with [package metadata](../pyproject.toml), [KAI constants](../topology_scheduler/kai_backend.py),
and the [Dynamo contract](../deploy/dynamo-v1/contract.json). The existing
[contract tests](../tests/test_dynamo_contract.py) also compare Ray and image
pins with the container recipe and installed Ray version.

| Component | Version |
| --- | --- |
| Package (development) | `0.1.2` |
| Ray (exact dependency) | `2.55.0` |
| Kubernetes Python client (optional dependency) | `kubernetes>=34,<35` |
| KAI Scheduler (target) | `0.17.0` |
| Kubernetes (target) | `1.34` |
| GPU Operator (target) | `25.10` |
| Dynamo (contract only) | `1.4.2` |
| vLLM (contract only) | `0.26.0` |
| Python (Dynamo image / CI) | `3.12` |
| CUDA (Dynamo image) | `13.0.2` |
| Minimum NVIDIA driver (Dynamo) | `580.00.03` |

The Python package supports Python 3.10 or newer. The Dynamo image has the
narrower Python 3.12 / Linux amd64 contract; consult its guide for the model
revision, image digests, NIXL, and host requirements. Historical changelog
versions describe their own releases and must not be updated to today's pins.

## GPU-free documentation validation

From the repository root, run `python scripts/check_docs.py --run-examples`.
It uses only the Python standard library and checks local Markdown file links,
the version table above, and this explicit list of CPU-safe commands:

<!-- docs-check: cpu -->
```bash
python -m examples.plan
python -m examples.compare_policies
python -m examples.kai_manifest
python -m examples.kai_submit --help
```

The list must match the allowlist in [the checker](../scripts/check_docs.py).
Commands run with the current Python interpreter, no shell, and a 60-second
timeout each. KAI help does not load a kubeconfig or submit work. Add new safe
commands to both lists after review. The [CI workflow](../.github/workflows/tests.yml)
runs these checks in a separate job without Ray, Kubernetes, GPUs, or credentials;
the existing test job runs the full suite and simulated Ray smoke examples.

The local-link check covers inline links/images and full or collapsed reference
links, including reference definitions. It ignores code fences, inline code,
comments, external URLs, and heading fragments. It checks file/directory
existence, not heading anchors, HTML links, or shortcut reference resolution.
External URLs, issue/PR state, tag existence, and technical claims require
manual review. Cluster, Docker, NVML, and Dynamo commands are deliberately
excluded from automatic execution; their prerequisites remain in the guides.
