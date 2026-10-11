"""Offline lifecycle test for the generic Gate 4b scheduler."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import gate4b_scheduler as scheduler


class FakeProcess:
    next_pid = 91000
    active = 0
    max_active = 0

    def __init__(self, *_args, **_kwargs):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.poll_count = 0
        self.finished = False
        FakeProcess.active += 1
        FakeProcess.max_active = max(FakeProcess.max_active, FakeProcess.active)

    def poll(self):
        self.poll_count += 1
        if self.poll_count < 2:
            return None
        if not self.finished:
            self.finished = True
            FakeProcess.active -= 1
        return 0


class SchedulerConcurrencyTest(unittest.TestCase):
    def test_two_way_fill_and_completion(self):
        FakeProcess.active = 0
        FakeProcess.max_active = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime/locks").mkdir(parents=True)
            (root / "runs").mkdir()
            tasks = []
            for seed in (2021, 2022, 2023):
                tasks.append({
                    "task_id": f"etth1__seed{seed}__gaussian_head",
                    "dataset": "etth1", "variant": "gaussian_head",
                    "missing_type": "random", "missing_rate": 0.3,
                    "seed": seed, "experiment_seed": seed, "required_free_mib": 1,
                })
            queue = {
                "max_concurrent_total": 2,
                "experiment_stage_label": "offline_test",
                "state_file": "runtime/test_state.json",
                "pid_file": "runtime/test.pid",
                "lock_file": "runtime/locks/test.lock",
                "bounded_oom_retry": 3,
                "tasks": tasks,
            }
            queue_path = root / "queue.json"
            queue_path.write_text(json.dumps(queue))
            cards = [
                (20000, 0, "GPU-test-0", "test0"),
                (20000, 1, "GPU-test-1", "test1"),
            ]
            with mock.patch.object(scheduler, "ROOT", root), \
                    mock.patch.object(scheduler, "reusable", return_value=None), \
                    mock.patch.object(scheduler, "valid_run", return_value=True), \
                    mock.patch.object(scheduler, "gpu_free", return_value=cards), \
                    mock.patch.object(scheduler.subprocess, "Popen", FakeProcess), \
                    mock.patch.object(scheduler.time, "sleep", return_value=None):
                return_code = scheduler.main(str(queue_path))
            state = json.loads((root / "runtime/test_state.json").read_text())
            self.assertEqual(return_code, 0)
            self.assertEqual(FakeProcess.max_active, 2)
            self.assertEqual(state["status"], "COMPLETE")
            self.assertEqual(state["max_concurrent_total"], 2)
            self.assertTrue(all(task["status"] == "SUCCEEDED" for task in state["tasks"].values()))


if __name__ == "__main__":
    unittest.main()
