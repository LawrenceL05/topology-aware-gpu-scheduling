"""Real Linux process containment tests; no Ray, Dynamo, CUDA, or GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(sys.platform == "linux", "Guardian uses Linux subreaper and /proc")
class GuardianTests(unittest.TestCase):
    def run_guardian(self, *, eof=False, early_exit=False):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "stopped.json"
            pids_file = Path(directory) / "pids.json"
            grandchild = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
            engine = (
                "import os,sys,signal,subprocess,time,json; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                f"child=subprocess.Popen([sys.executable,'-c',{grandchild!r}],start_new_session=True); "
                f"open({str(pids_file)!r},'w').write(json.dumps([os.getpid(),child.pid])); "
                + ("sys.exit(7)" if early_exit else "time.sleep(60)")
            )
            settings = {"command": [sys.executable, "-c", engine], "token": "test",
                        "receipt": str(receipt), "lease_timeout": 1.5,
                        "shutdown_timeout": 0.2, "kill_timeout": 2}
            guardian = subprocess.Popen([sys.executable, "-m", "topology_scheduler._dynamo_guardian"],
                                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                guardian.stdin.write((json.dumps(settings) + "\n").encode())
                guardian.stdin.flush()
                deadline = time.monotonic() + 5
                while not pids_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pids_file.exists(), "Fake engine failed to start")
                if eof:
                    guardian.stdin.close()  # Simulates abrupt actor death.
                self.assertEqual(guardian.wait(timeout=8), 0)
                result = json.loads(receipt.read_text())
                self.assertTrue(result["stopped"])
                for pid in json.loads(pids_file.read_text()):
                    self.assertFalse(Path(f"/proc/{pid}").exists(), f"Owned process {pid} survived")
                return result["reason"]
            finally:
                if guardian.stdin and not guardian.stdin.closed:
                    guardian.stdin.close()
                if guardian.poll() is None:
                    guardian.terminate()
                    guardian.wait(timeout=8)
                guardian.stderr.close()

    def test_actor_pipe_eof_kills_session_escaping_descendants(self):
        self.assertIn("disconnected", self.run_guardian(eof=True))

    def test_driver_lease_expiry_kills_owned_process_tree(self):
        self.assertIn("heartbeat expired", self.run_guardian())

    def test_engine_exit_still_cleans_up_grandchildren(self):
        self.assertIn("engine exited", self.run_guardian(early_exit=True))


if __name__ == "__main__":
    unittest.main()
