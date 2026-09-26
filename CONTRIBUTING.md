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

Use the [project roadmap](docs/roadmap.md) when proposing and triaging work.
Every implementation or evaluation issue needs a milestone, or an explicit
reason and next triage step recorded in the issue. Choose by the primary
deliverable, link dependencies, and keep task requirements and progress on the
issue. Completed research work enters a release through a linked readiness
issue in Next Release; it keeps its original research milestone. Maintainers
review completion evidence before closing milestones and set due dates only
when estimates, hardware availability, and dependencies justify them.

### Group and title issues

Use exactly one lowercase prefix in each issue title, chosen by the primary
deliverable: `[type] Clear description of the outcome`.

| Prefix | Use for | Example title |
| --- | --- | --- |
| `[bug]` | Existing behavior that is incorrect or broken | `[bug] Release GPU reservations after worker failure` |
| `[feature]` | A new capability | `[feature] Support topology-aware placement` |
| `[maintenance]` | Dependencies, CI, cleanup, or refactoring | `[maintenance] Update Ray dependency` |
| `[docs]` | Guides, examples, or explanations | `[docs] Clarify multi-node setup` |
| `[research]` | Investigating a question or comparing approaches | `[research] Compare placement scoring methods` |
| `[benchmark]` | Measuring performance or validating results | `[benchmark] Measure JCT on two GPU nodes` |

Choose the issue's main purpose rather than every activity it involves. A bug
fix that includes regression tests and documentation still uses `[bug]`.
Use `[research]` when the main output is a finding or recommendation, and
`[benchmark]` when it is a set of performance measurements. These titles are
examples, not claims about current capabilities or planned work.

Keep priority and status in labels or project fields, and target deliverables
or releases in milestones. The prefix describes the kind of work; it does not
replace the milestone and dependency requirements above. If the scope changes,
update the prefix to match the new primary deliverable.

Include these three items in every issue, alongside the relevant bug-report,
feature-request, or research evidence details:

- **Why it matters:** the problem, question, or motivation.
- **Scope:** the work included and any boundaries or dependencies.
- **Done when:** observable acceptance criteria, such as a reproducible fix,
  an updated guide, a documented finding, or a benchmark with its methodology.

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

## Documentation update rules

Review documentation impact in every PR, including code-only changes. Use the
[PR template](.github/pull_request_template.md) and explain each exclusion;
not every change requires editing every guide.

| Change | Documentation to review and update in the same PR |
| --- | --- |
| User-visible feature, fix, compatibility change, or corrected support claim | `CHANGELOG.md` under Unreleased; describe the behavior and its limits. |
| Setup, public API, example command, supported backend, or headline status | `README.md` and the affected example or integration guide. |
| Discovery, planning, reservation, execution, failure, or cleanup behavior | `docs/v1.1-workflow.md` plus the Ray or KAI integration guide. |
| NVML fields, topology scope, scoring inputs, or device enforcement | `docs/v1.2-topology-discovery.md` and the shared status guide. |
| Dependency pin, image/model revision, runtime ownership, readiness, or shutdown | Relevant Ray/KAI/Dynamo guide, deployment recipe, machine-readable contract, and contract tests. |
| New validation evidence or a change from planned to implemented | `docs/current-status.md`, README status, affected guide, and changelog. |

Use the [current status and versions](docs/current-status.md) as the shared
summary. Preserve the distinction between design milestones (V1/V1.1/V1.2),
package versions, and released tags. Keep historical release notes accurate to
their release; put subsequent changes under Unreleased. Label implemented,
mocked, simulated, planned, and real-cluster-validated behavior explicitly.
A synthetic score or logical-GPU smoke run is not GPU benchmark evidence.

For each changed guide, verify commands from the repository root with the
stated prerequisites, file links, issue/PR destinations and state, dependency
pins, and known limitations. Explain commands that could not be run and why.
Do not execute cluster-mutating examples just to satisfy documentation checks.
Run the GPU-free check locally before requesting review:

```bash
python scripts/check_docs.py --run-examples
```

CI checks local file links, the shared version table, and an explicit allowlist
of documented CPU commands. External URLs, heading anchors, release significance,
and technical correctness remain reviewer responsibilities. See the status
guide for exact coverage and how to add an approved command.

Every user-visible changelog entry should link its implementation PR or commit.
For a new PR, add its link once GitHub assigns the number; before release,
verify the PR is merged and included in the release, or use an immutable merged
commit link. An issue describes intent and does not prove implementation.
Keep planned work in a separate Planned section linked to its open issue.
Release, versioning, and tag policy lives in
[the release process](docs/releasing.md).

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

## Releases

Releases follow [the release process](docs/releasing.md). It defines the
semantic-versioning rules for this project, the checklist that must pass before
a tag, the annotated `vX.Y.Z` tag format and GitHub Release, the required
contents of release notes, and the rollback rules for a bad release.

Contributors do not create tags. A correct entry under `Unreleased` in
`CHANGELOG.md` is what makes a change releasable; a maintainer decides the
version and publishes.

## Upstream changes

This project integrates with Ray and KAI Scheduler but does not maintain forks
of them. Report or contribute general Ray defects through
[Ray's contribution process](https://github.com/ray-project/ray/blob/master/CONTRIBUTING.rst).
Report KAI-specific changes through
[KAI Scheduler's contribution process](https://github.com/kai-scheduler/KAI-Scheduler/blob/main/CONTRIBUTING.md).
Keep repository-specific adapters, policies, and experiments here.
