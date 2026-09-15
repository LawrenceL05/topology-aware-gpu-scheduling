# Contributing

Contributions to the scheduler, Ray integration, experiments, tests, and
documentation are welcome. Small fixes can go directly to a pull request. For
new scheduling policies, public APIs, or experimental methodology, open an
issue first so the design and evidence requirements can be agreed on before
implementation.

## Report an issue

Search the existing issues before opening a new one. A useful bug report
includes:

- the expected and actual behavior;
- minimal reproduction steps;
- Python, Ray, operating-system, CUDA, driver, and GPU versions when relevant;
- logs or stack traces with credentials and private data removed; and
- whether the run used simulated Ray resources or physical GPUs.

Feature requests should describe the scheduling problem, proposed behavior,
and how success could be measured. Use GitHub Discussions or an issue for design
questions rather than opening an incomplete implementation.

## Set up the project

Python 3.10 or newer is required. From a clone of the repository:

```bash
python -m venv .venv
python -m pip install -e '.[ray]'
python -m unittest discover -s tests -v
```

The CI workflow uses Python 3.12 and Ray 2.55.0. Run the planner and Ray smoke
examples when changing scheduling or execution behavior:

```bash
python -m examples.plan
python -m examples.ray_smoke
python -m examples.ray_multinode_smoke
```

The smoke examples use simulated logical GPUs. They do not replace validation
on physical hardware for changes involving NVML, CUDA, topology, or inference.

## Make a change

1. Fork the repository and create a descriptive branch, such as
   `feature/discover-nvlink` or `fix/placement-cleanup`.
2. Keep the change focused. Avoid mixing formatting or unrelated refactors with
   behavioral changes.
3. Add or update tests that demonstrate the behavior being changed.
4. Update the README or detailed documentation when commands, public APIs,
   assumptions, or limitations change.
5. Add a concise entry under `Unreleased` in `CHANGELOG.md` for user-visible
   features, fixes, and compatibility changes.
6. Run the relevant test and example commands before opening a pull request.

Do not commit models, datasets, benchmark outputs, credentials, cluster
addresses, or generated environments. Link to reproducible external artifacts
when results are too large for the repository.

## Research and benchmark changes

Performance claims need enough context to reproduce and interpret them. Record:

- the workload, model, revision, precision, batch/concurrency, and request mix;
- GPU models, counts, memory, topology, CUDA/driver, Ray, and backend versions;
- the baseline policy and identical conditions used for comparison;
- the JCT boundary and normalization denominator;
- warmup, repetition count, aggregation, failures, and uncertainty; and
- separate queue wait, startup/model-load time, and execution/request latency.

Label synthetic inputs, simulated GPUs, and mocked integrations clearly. A
planner score or successful smoke test is not evidence of a production
performance improvement.

## Pull requests

Use a clear, imperative title such as `Add NVLink topology discovery`. In the
description, explain the problem, the resulting behavior, validation performed,
and remaining limitations. Link the issue with `Fixes #123` only when the pull
request fully resolves it.

Before requesting review, confirm that:

- tests and relevant examples pass;
- new behavior has meaningful test coverage;
- public behavior and limitations are documented;
- user-visible changes appear in `CHANGELOG.md`; and
- the pull request contains no secrets or unrelated generated files.

Reviewers may request smaller scope, additional evidence, or a design issue for
changes that affect placement semantics or experimental conclusions.

## Upstream changes

This project integrates with Ray and KAI Scheduler but does not maintain forks
of them. Report or contribute general Ray defects through
[Ray's contribution process](https://github.com/ray-project/ray/blob/master/CONTRIBUTING.rst).
Report KAI-specific changes through
[KAI Scheduler's contribution process](https://github.com/kai-scheduler/KAI-Scheduler/blob/main/CONTRIBUTING.md).
Keep repository-specific adapters, policies, and experiments here.
