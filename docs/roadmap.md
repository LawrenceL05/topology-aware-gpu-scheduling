# Project milestones and roadmap

Use the upstream [GitHub milestones](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/milestones)
to group work and track completion. This guide defines scope, dependencies, and
exit criteria; issue bodies remain the source for task requirements, blockers,
implementation PRs, and validation artifacts. The [current status guide](current-status.md)
describes what the code supports and which claims have evidence.

The three research milestones below describe capabilities and experiments.
**Next Release** collects release readiness work for an explicitly selected
scope. None of these names is a package version, Git tag, or proof of research
results. V1/V1.1/V1.2 design milestones also remain separate from semantic
package versions. This roadmap creates no release or tag.

## Milestone definitions

Create the following four milestones with these exact titles and descriptions.
The descriptions are short enough to copy into GitHub; the exit criteria below
explain the evidence needed to close each milestone.

| Title | GitHub description |
| --- | --- |
| Topology Discovery | Discover NICs, GPU/NUMA/NIC affinity, and measured inter-node costs. Complete when scoped issues #14-#17 meet their acceptance criteria, with tested models, documented limits, and linked validation evidence. |
| Dynamo Integration | Implement and validate Ray-managed independent Dynamo/vLLM replicas. Complete when #2 and #3 satisfy the pinned contract, lifecycle/failure tests pass, and an actual Linux GPU run report is linked. |
| Evaluation | Establish deterministic baselines and reproducible, matched policy comparisons. Complete when #5 and scoped benchmark issues have reviewed conformance tests, measurement definitions, and reproducible results with failures and uncertainty. |
| Next Release | Prepare an explicitly selected set of merged changes for release. Complete when selected blockers, CI, documentation, changelog provenance, version review, and validation/limitation evidence pass the release readiness review. |

### Topology Discovery

Build on [V1.2 intra-node discovery](v1.2-topology-discovery.md). Sequence work as
[NIC inventory #14](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/14),
then [GPU/NUMA/NIC mapping #15](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/15),
then [typed affinity graph #16](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/16),
then [inter-node measurements #17](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/17).
Issue #16 consumes both #14 and #15; #17 uses all three to identify and interpret
measurement endpoints. Test design may proceed earlier using explicit fixtures.
[Host locality](host-topology.md) landed ahead of #14 on fixtures, reading
interfaces from sysfs directly; #14 can replace those reads without changing
its classification.

Exit criteria:

- Each scoped issue meets its own acceptance criteria and links merged code,
  deterministic tests, runnable examples, and documentation.
- Data retains stable identities, units, provenance, partial/unknown states,
  and the difference between advertised speed and measured throughput.
- Measurement evidence and the real multi-node validation procedure required by
  #17 are linked. Any hardware validation not performed stays explicitly
  unverified; observed affinity does not imply GPU/NIC binding by a backend.

### Dynamo Integration

The [pinned V1 contract](dynamo-v1-contract.md), established in
[issue #1](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/1),
precedes [worker lifecycle #2](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/2),
which precedes [examples and GPU validation #3](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/3).
This work can proceed alongside topology discovery: V1 independent TP=1
replicas do not require GPU-to-NIC graph placement or inter-replica traffic.

Exit criteria:

- The lifecycle satisfies the contract, preserves the finite-task adapter,
  and passes reservation, readiness, rollback, shutdown, and ownership tests.
- #3's CPU dry run is explicitly simulated, and its real-inference example
  checks actual assignment, repeated requests, cleanup, and controlled failure.
- A reproducible actual Linux GPU run report includes hardware, pinned software,
  commands, logs, and outcomes. Missing hardware evidence leaves this milestone
  open even if implementation is merged. A smoke response is not a benchmark.

### Evaluation

[Baseline work #5](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/5)
and future benchmark/harness issues belong here. The existing
[baseline guide](baseline-policies.md) defines policy names and normalized JCT.
Merged baseline code is evidence to review against #5, not a reason to silently
mark all evaluation complete.

Exit criteria:

- Shared conformance tests verify deterministic policies and matched backend
  behavior; all acceptance criteria in #5 are reviewed against merged evidence.
- Each scoped experiment defines job boundaries, normalization denominator,
  matched workload/model/hardware/backend conditions, repetitions, aggregation,
  failures, and uncertainty before performance conclusions are drawn.
- Benchmark issues link reproducible run artifacts and reviewed results.
  Synthetic examples validate wiring only. No performance improvement is
  required to close an experiment; negative or inconclusive results are valid
  when reported with the required evidence.

Baseline and harness development can start with current inventory and synthetic
inputs. Experiments claiming discovered network costs depend on the relevant
Topology Discovery outputs; Dynamo-serving comparisons depend on validated
Dynamo Integration. Physical device-placement comparisons additionally need an
enforceable device mapping tracked in a separate issue. These are per-experiment
dependencies, not a requirement to finish every research track first.

### Next Release

Group roadmap/triage work from
[issue #18](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/18),
[release process #20](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/20),
and future release preparation or release-blocking defects here.

Exit criteria:

- A maintainer records the selected release scope and explicitly names blocking
  issues, required validation, and deferred experimental work.
- Selected implementations are merged, required checks pass on the selected
  main-branch commit, and documentation/changelog claims match their evidence.
- Release notes trace to merged PRs or commits, package/version choices are
  reviewed under #20, and compatibility, known limitations, and missing GPU
  validation are explicit. The readiness review links its evidence.

Only the selected capabilities and their dependencies block this milestone;
the entire topology, Dynamo, and evaluation backlog need not be finished.
An experimental release may include merged but GPU-unverified code if it is
clearly labeled and the selected scope does not promise GPU validation. This
does not close the corresponding research milestone or waive its exit criteria.
Version selection, tags, and publication follow #20's release process separately.

## Issue assignment and ongoing triage

The following is the initial assignment map for #18, not a parallel status
tracker. Use each issue's live milestone field and linked evidence thereafter.

| Milestone | Initial issues |
| --- | --- |
| Topology Discovery | [#14](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/14), [#15](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/15), [#16](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/16), [#17](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/17) |
| Dynamo Integration | [#2](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/2), [#3](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/3) |
| Evaluation | [#5](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/5) |
| Next Release | [#18](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/18), [#20](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/20) |

At triage, assign every new implementation or evaluation issue to one milestone
based on its primary deliverable. If the scope is undecided or an issue is out
of scope, record the specific reason for no milestone and the next triage step
in that issue. Do not leave an unexplained empty milestone field. Split distinct
deliverables across issues when needed and link their dependencies; do not copy
acceptance checklists between them. Track blockers in GitHub issue dependencies
or linked dependency lists in the issues themselves.

GitHub allows one milestone per issue. Keep completed research issues in their
research milestone to preserve the record. For a release, create a release
readiness issue in **Next Release** that links the selected research issues,
merged PRs, and validation artifacts; do not move or duplicate those issues.
Record scope decisions and deferred work in that readiness issue. Milestone
progress, issue closure, and release readiness are different signals.

Update progress by assigning/closing issues and linking evidence on GitHub.
Edit this roadmap only when scope, dependencies, or exit criteria change. Before
closing a milestone, a maintainer reviews the linked evidence against its exit
criteria; a 100% issue counter alone is insufficient. Reopen a milestone only
with an explicit scope decision and linked follow-up issues. For the next release
cycle, preserve the completed readiness issue and start a new one rather than
rewriting its evidence.

## Dates and maintainer setup

Leave all four due dates unset initially: no owner estimates, reserved hardware
windows, or agreed release date currently justify deadlines. A maintainer may
set a date after recording the owner, estimate or hardware reservation, resolved
dependencies, and agreement in the relevant issue. Revisit the date when those
assumptions change, with a linked explanation; never infer it from an issue
number, design milestone, or package version.

Repository metadata is separate from files in a PR. Merging this guide does not
create GitHub milestones or assign issues. A maintainer with the necessary
upstream permissions completes this one-time setup:

1. Open [upstream milestones](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/milestones),
   including closed milestones, and reuse matching titles rather than creating
   duplicates. Create missing milestones using the definitions above, leave due
   dates unset, and link this roadmap from each description for detailed exits.
2. Apply the initial issue assignment map. Then review all currently open issues
   for newer work; assign a milestone or record an explicit exception as above.
3. Verify the live milestone descriptions and membership, and link that result
   from #18. Keep #18 open until this repository setup and its other acceptance
   criteria are complete. Milestones on a contributor's fork do not satisfy it.

For a read-only check using an authenticated GitHub CLI, run:

```bash
gh api 'repos/LawrenceL05/topology-aware-gpu-scheduling/milestones?state=all' --paginate
gh issue list --repo LawrenceL05/topology-aware-gpu-scheduling --state open --limit 1000 --json number,title,milestone
```

Review any unassigned issues against their explicit exceptions. These commands
require GitHub access and are not part of the offline documentation CI check.
