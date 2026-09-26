import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.check_docs import (
    CPU_COMMANDS, ROOT, check_links, check_versions, documented_commands, main,
)


class DocumentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        self.document = self.root / "docs/guide.md"

    def links(self, text):
        self.document.write_text(text, encoding="utf-8")
        return check_links(self.root, self.document)

    def test_relative_root_encoded_and_reference_links(self):
        (self.root / "a file.md").touch()
        (self.root / "image.png").touch()
        self.assertEqual(self.links(
            '[inline](../a%20file.md#heading)\n'
            '[root](/a%20file.md) ![image](../image.png)\n'
            '[full][target] [target][]\n[target]: <../a file.md>\n'
        ), [])

    def test_inline_links_support_parentheses_and_titles(self):
        (self.root / "guide(v1).md").touch()
        self.assertEqual(self.links(
            '[double](../guide(v1).md "Guide")\n'
            "[single](../guide(v1).md 'Guide')\n"
            '![image-like](../guide(v1).md)\n'
        ), [])

    def test_missing_files_and_undefined_references_fail(self):
        errors = self.links('[missing](missing.md) [bad][absent]\n[x]: missing.png')
        self.assertEqual(len(errors), 3)
        self.assertTrue(any('undefined reference [absent]' in error for error in errors))

    def test_link_cannot_escape_repository(self):
        self.assertIn('broken local link', self.links('[outside](../../)')[0])

    def test_external_urls_fragments_and_code_are_ignored(self):
        self.assertEqual(self.links(
            '[web](https://example.org/a) [email](mailto:a@example.org)\n'
            '[anchor](#heading)\n`[code](missing.md)`\n'
            '<!-- [comment](missing.md) -->\n'
            '```bash\n[code](missing.md)\n```\n'
            '~~~text\n[code](missing.md)\n~~~\n'
        ), [])

    def copy_contract_files(self):
        for name in ('pyproject.toml', 'deploy/dynamo-v1/contract.json',
                     'topology_scheduler/kai_backend.py', 'docs/current-status.md'):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def test_version_drift_is_reported(self):
        self.copy_contract_files()
        self.assertEqual(check_versions(self.root), [])
        guide = self.root / 'docs/current-status.md'
        guide.write_text(guide.read_text().replace('`2.55.0`', '`0.0.0`'))
        self.assertTrue(any('Ray (exact dependency)' in e for e in check_versions(self.root)))

    def test_missing_version_metadata_is_reported_without_crashing(self):
        self.copy_contract_files()
        project = self.root / 'pyproject.toml'
        project.write_text(project.read_text().replace('version = "0.1.2"', ''))
        kai = self.root / 'topology_scheduler/kai_backend.py'
        kai.write_text(kai.read_text().replace('KAI_VERSION = "0.17.0"', ''))
        errors = check_versions(self.root)
        self.assertTrue(any('package version' in error for error in errors))
        self.assertTrue(any('KAI_VERSION' in error for error in errors))

    def test_unapproved_command_is_rejected(self):
        self.copy_contract_files()
        guide = self.root / 'docs/current-status.md'
        guide.write_text(guide.read_text().replace(
            'python -m examples.kai_submit --help', 'python -m examples.kai_submit'))
        with self.assertRaisesRegex(ValueError, 'CPU commands differ'):
            documented_commands(self.root)

    def test_command_failure_makes_validation_fail(self):
        import subprocess
        with patch('sys.argv', ['check_docs.py', '--run-examples']), \
             patch('scripts.check_docs.subprocess.run', side_effect=
                   subprocess.CalledProcessError(1, 'example')) as run:
            self.assertEqual(main(), 1)
            self.assertEqual(run.call_count, 1)

    def test_commands_have_timeout_and_do_not_use_shell(self):
        with patch('sys.argv', ['check_docs.py', '--run-examples']), \
             patch('scripts.check_docs.subprocess.run') as run:
            self.assertEqual(main(), 0)
            self.assertEqual(run.call_count, len(CPU_COMMANDS))
            for call in run.call_args_list:
                self.assertEqual(call.kwargs['timeout'], 60)
                self.assertFalse(call.kwargs.get('shell', False))


if __name__ == '__main__':
    unittest.main()
