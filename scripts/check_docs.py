"""Check repository Markdown file links, version summary, and CPU examples.

Standard-library only. This deliberately supports inline and full/collapsed
reference Markdown links, not HTML links or GitHub heading anchors. External
URLs are reviewed manually; no network or cluster is needed by this check.
"""

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
INLINE = re.compile(
    r"(?<!\\)!?\[[^\]\n]*\]\(\s*"
    r"(<[^>\n]+>|(?:\\.|[^()\s]|\([^()\n]*\))+?)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
DEFINITION = re.compile(r"^ {0,3}\[([^\]]+)\]:\s*(<[^>]+>|\S+)", re.MULTILINE)
REFERENCE = re.compile(r"\[([^\]\n]+)\]\[([^\]\n]*)\]")
# Explicitly approved commands: never execute arbitrary Markdown as shell code.
CPU_COMMANDS = (
    "python -m examples.plan",
    "python -m examples.compare_policies",
    "python -m examples.host_topology",
    "python -m examples.kai_manifest",
    "python -m examples.kai_submit --help",
    "python -m examples.nic_inventory",
)


def prose(text):
    """Remove fenced code, inline code and comments before inspecting links."""
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match[1]
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            lines.append("")
        else:
            lines.append(line if fence is None else "")
    text = re.sub(r"<!--.*?-->", "", "\n".join(lines), flags=re.DOTALL)
    return re.sub(r"(`+).*?\1", "", text)


def check_links(root, document):
    text = prose(document.read_text(encoding="utf-8"))
    definitions = {key.casefold(): value for key, value in DEFINITION.findall(text)}
    targets = INLINE.findall(text) + list(definitions.values())
    errors = []
    for label, reference in REFERENCE.findall(text):
        key = (reference or label).casefold()
        if key not in definitions:
            errors.append(f"{document.relative_to(root)}: undefined reference [{key}]")
    for raw in targets:
        target = raw.strip("<>")
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or not parts.path:
            continue
        base = root if parts.path.startswith("/") else document.parent
        resolved = (base / unquote(parts.path).lstrip("/")).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            errors.append(f"{document.relative_to(root)}: broken local link: {target}")
    return errors


def required_value(text, pattern, source, label, errors):
    """Extract required metadata while keeping failures actionable."""
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        errors.append(f"{source}: could not determine {label}")
        return None
    return match[1]


def check_versions(root):
    """Keep the shared summary tied to project metadata and runtime contracts."""
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    runtime = json.loads((root / "deploy/dynamo-v1/contract.json").read_text())["runtime"]
    kai = (root / "topology_scheduler/kai_backend.py").read_text(encoding="utf-8")
    errors = []
    expected = {
        "Package (development)": required_value(
            project, r'^version = "([^"]+)"', "pyproject.toml",
            "package version", errors),
        "Kubernetes Python client (optional dependency)": required_value(
            project, r'kai = \["([^"]+)"\]', "pyproject.toml",
            "Kubernetes client dependency", errors),
    }
    for label, key in (
        ("Ray (exact dependency)", "ray"),
        ("Dynamo (contract only)", "dynamo"),
        ("vLLM (contract only)", "vllm"),
        ("Python (Dynamo image / CI)", "python"),
        ("CUDA (Dynamo image)", "cuda"),
        ("Minimum NVIDIA driver (Dynamo)", "minimum_nvidia_driver"),
    ):
        expected[label] = runtime.get(key)
        if expected[label] is None:
            errors.append(f"deploy/dynamo-v1/contract.json: missing runtime.{key}")
    for label, constant in (
        ("KAI Scheduler (target)", "KAI_VERSION"),
        ("Kubernetes (target)", "KUBERNETES_VERSION"),
        ("GPU Operator (target)", "GPU_OPERATOR_VERSION"),
    ):
        expected[label] = required_value(
            kai, rf'^{constant} = "([^"]+)"',
            "topology_scheduler/kai_backend.py", constant, errors)
    summary = (root / "docs/current-status.md").read_text(encoding="utf-8")
    errors.extend(
        f"docs/current-status.md: expected '| {label} | `{value}` |'"
        for label, value in expected.items()
        if value is not None and f"| {label} | `{value}` |" not in summary
    )
    if expected["Ray (exact dependency)"] is not None and \
            f'ray=={expected["Ray (exact dependency)"]}' not in project:
        errors.append("pyproject.toml: Ray pin differs from the Dynamo contract")
    return errors


def documented_commands(root):
    """Only commands in the marked status-guide block are runnable by this tool."""
    text = (root / "docs/current-status.md").read_text(encoding="utf-8")
    blocks = re.findall(r"<!-- docs-check: cpu -->\s*```bash\n(.*?)\n```", text, re.S)
    commands = [line.strip() for block in blocks for line in block.splitlines() if line.strip()]
    if commands != list(CPU_COMMANDS):
        raise ValueError("docs/current-status.md: CPU commands differ from the approved list in scripts/check_docs.py")
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-examples", action="store_true")
    args = parser.parse_args()
    documents = [p for p in ROOT.rglob("*.md")
                 if not set(p.relative_to(ROOT).parts) & {".git", ".venv", "build", "dist"}]
    errors = [error for document in documents for error in check_links(ROOT, document)]
    errors.extend(check_versions(ROOT))
    try:
        commands = documented_commands(ROOT)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if args.run_examples:
        for command in commands:
            print(f"Checking: {command}", flush=True)
            try:
                subprocess.run([sys.executable, *shlex.split(command)[1:]],
                               cwd=ROOT, check=True, timeout=60)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                print(f"Failed documented command: {command}: {exc}", file=sys.stderr)
                return 1
    print(f"Documentation checks passed ({len(documents)} Markdown files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
