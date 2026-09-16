## Change

Describe the problem, resulting behavior, and remaining limitations.
Link the issue; use `Fixes #...` only when all acceptance criteria are met.

## Validation

List commands run and results. Label synthetic inputs, mocked APIs, simulated
logical GPUs, and real-cluster runs separately; link artifacts for the latter.

## Documentation impact

- [ ] Updated `CHANGELOG.md` under Unreleased for user-visible changes, or explained why no entry is needed.
- [ ] Updated `README.md` for changes to setup, commands, public behavior, or status, or explained why it is unaffected.
- [ ] Updated affected Ray, KAI, Dynamo, V1.1 workflow, and V1.2 topology guides, or named those reviewed and explained why no edits are needed.
- [ ] Checked `docs/current-status.md`, dependency pins, known limitations, and planned versus implemented status against code and tests.
- [ ] Verified changed commands, local file links, and issue/PR links; release notes link merged PRs or commits (the current PR may be linked after its number is assigned).
- [ ] Ran `python scripts/check_docs.py --run-examples` and relevant tests; recorded unavailable hardware validation above.

Documentation exclusions and reasons:
