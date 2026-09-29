import ctypes
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from examples.device_binding_gpu_check import (
    cuda_driver, cuda_visible_device, main, passed, refusal, skip_reasons,
)

MODULE = "examples.device_binding_gpu_check"
DEVICE = "GPU-01010101-0101-0101-0101-010101010101"


class FakeDriver:
    """A stand-in CUDA driver offering only the four calls the check makes."""

    def __init__(self, failing=None, uuid_bytes=b"\x01" * 16):
        self.failing, self.uuid_bytes = failing, uuid_bytes
        self.destroyed, self.ordinal = 0, None

    def code(self, name):
        return 1 if self.failing == name else 0

    def cuInit(self, flags):
        return self.code("cuInit")

    def cuDeviceGet(self, reference, ordinal):
        self.ordinal = ordinal
        return self.code("cuDeviceGet")

    def cuDeviceGetUuid(self, reference, device):
        if self.failing == "cuDeviceGetUuid":
            return 1
        ctypes.memmove(reference, self.uuid_bytes, len(self.uuid_bytes))
        return 0

    def cuCtxCreate_v2(self, reference, flags, device):
        return self.code("cuCtxCreate")

    def cuCtxDestroy_v2(self, context):
        self.destroyed += 1
        return 0


def checks(**overrides):
    """A run report in which every claim the example makes holds."""
    base = {
        "matched": {
            "assignments": [{"matched": True}],
            "cuda_device_is_the_resolved_one": True,
            "workers_that_ran": 1,
        },
        "mismatch_refused": {"refused": True, "workers_that_ran": 0},
        "untrusted_order_refused": {"refused": True, "workers_that_ran": 0},
    }
    base.update(overrides)
    return base


class DriverTests(unittest.TestCase):
    def test_uuid_comes_from_a_created_context_and_is_released(self):
        driver = FakeDriver()
        with patch(f"{MODULE}.cuda_driver", return_value=driver):
            self.assertEqual(cuda_visible_device(),
                             {"available": True, "context_created": True,
                              "uuid": DEVICE})
        # Ray narrows visibility to the assigned device, so ordinal 0 is it.
        self.assertEqual(driver.ordinal, 0)
        self.assertEqual(driver.destroyed, 1)

    def test_each_failed_call_is_named_and_no_context_is_leaked(self):
        for step in ("cuInit", "cuDeviceGet", "cuDeviceGetUuid", "cuCtxCreate"):
            driver = FakeDriver(failing=step)
            with self.subTest(step=step), \
                 patch(f"{MODULE}.cuda_driver", return_value=driver):
                self.assertEqual(cuda_visible_device(),
                                 {"available": False,
                                  "reason": f"{step} returned 1"})
            self.assertEqual(driver.destroyed, 0)

    def test_a_host_without_the_driver_reports_it_instead_of_raising(self):
        with patch(f"{MODULE}.cuda_driver", return_value=None):
            result = cuda_visible_device()
        self.assertFalse(result["available"])
        self.assertIn("no CUDA driver library", result["reason"])

    def test_every_candidate_library_name_is_tried_before_giving_up(self):
        with patch("ctypes.CDLL", side_effect=OSError("missing")) as load:
            self.assertIsNone(cuda_driver())
        self.assertGreaterEqual(load.call_count, 1)
        with patch("ctypes.CDLL", return_value="library") as load:
            self.assertEqual(cuda_driver(), "library")
        self.assertEqual(load.call_count, 1)


class SkipTests(unittest.TestCase):
    def test_an_opted_in_gpu_host_has_nothing_to_skip_for(self):
        self.assertEqual(
            skip_reasons(opted_in=True, devices=("gpu",), has_driver=True), ())

    def test_each_missing_prerequisite_is_reported_separately(self):
        reasons = skip_reasons(opted_in=False, devices=(), has_driver=False)
        self.assertEqual(len(reasons), 3)
        self.assertTrue(any("--run" in reason for reason in reasons))
        self.assertTrue(any("no NVIDIA GPU" in reason for reason in reasons))
        self.assertTrue(any("CUDA driver" in reason for reason in reasons))

    def test_a_host_without_gpus_skips_successfully_and_says_why(self):
        output = io.StringIO()
        with patch(f"{MODULE}.host_devices", return_value=((), None)), \
             patch(f"{MODULE}.cuda_driver", return_value=None), \
             redirect_stdout(output):
            self.assertEqual(main(["--run"]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(len(report["skipped_because"]), 2)
        self.assertFalse(report["performs_cuda_work"])


class VerdictTests(unittest.TestCase):
    def test_a_complete_run_passes(self):
        self.assertTrue(passed(checks(), 1))

    def test_a_cuda_device_other_than_the_resolved_one_fails(self):
        report = checks()
        report["matched"]["cuda_device_is_the_resolved_one"] = False
        self.assertFalse(passed(report, 1))

    def test_an_unmatched_assignment_fails(self):
        report = checks()
        report["matched"]["assignments"] = [{"matched": False}]
        self.assertFalse(passed(report, 1))

    def test_a_rank_whose_worker_never_ran_fails_the_matched_case(self):
        report = checks()
        report["matched"]["workers_that_ran"] = 0
        self.assertFalse(passed(report, 1))

    def test_a_wrong_device_that_was_not_refused_fails(self):
        self.assertFalse(passed(checks(mismatch_refused={"refused": False}), 1))

    def test_a_refusal_that_still_let_the_workload_run_fails(self):
        self.assertFalse(passed(checks(
            untrusted_order_refused={"refused": True, "workers_that_ran": 1}), 1))


class RefusalTests(unittest.TestCase):
    def test_the_refusal_is_extracted_from_rays_wrapped_task_error(self):
        error = RuntimeError(
            "\x1b[36mray::bound()\x1b[39m (pid=1)\n"
            '  File "device_binding.py", line 262, in bound\n'
            "    check_assignments([assignment], mode)\n"
            "DeviceBindingError: Ray did not assign the planned devices: "
            "rank 0 on a asked for GPU-a and received GPU-b")
        message = refusal(error)
        self.assertTrue(message.startswith("DeviceBindingError:"))
        self.assertNotIn("\x1b", message)

    def test_an_unrecognized_failure_keeps_its_last_line(self):
        self.assertEqual(refusal(RuntimeError("first\nlast\n")), "last")


if __name__ == "__main__":
    unittest.main()
