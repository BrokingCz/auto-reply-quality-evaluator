"""核心函数单元测试（标准库 unittest，无第三方依赖）。

运行：python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import bucket_of, compute_total, gate_triggered  # noqa: E402
from src.validate import _cohen_kappa, _spearman  # noqa: E402


class TestComputeTotal(unittest.TestCase):
    """加权总分：help 0.5 / slot 0.3 / acc 0.1 / tone 0.1"""

    def test_normal(self):
        scores = {
            "helpfulness": {"score": 2},
            "slot_completeness": {"score": 2},
            "accuracy": {"score": 5},
            "tone": {"score": 4},
        }
        self.assertAlmostEqual(compute_total(scores), 2.5)  # 1.0+0.6+0.5+0.4

    def test_full_marks(self):
        scores = {m: {"score": 5} for m in ("helpfulness", "slot_completeness", "accuracy", "tone")}
        self.assertAlmostEqual(compute_total(scores), 5.0)

    def test_ignores_missing(self):
        scores = {"helpfulness": {"score": 3}}  # 只有一维
        self.assertAlmostEqual(compute_total(scores), 3.0)  # 归一化后仍 3.0


class TestGate(unittest.TestCase):
    def test_faithfulness_2_triggers(self):
        self.assertTrue(gate_triggered({"faithfulness": {"score": 2}}))

    def test_faithfulness_5_no(self):
        self.assertFalse(gate_triggered({"faithfulness": {"score": 5}}))

    def test_missing_no(self):
        self.assertFalse(gate_triggered({}))


class TestBucket(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(bucket_of(4.0), "好")
        self.assertEqual(bucket_of(3.0), "中")
        self.assertEqual(bucket_of(2.99), "差")


class TestCohenKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        pairs = [("好", "好")] * 10 + [("中", "中")] * 5
        self.assertAlmostEqual(_cohen_kappa(pairs), 1.0)

    def test_no_agreement(self):
        pairs = [("好", "差")] * 5 + [("差", "好")] * 5
        self.assertAlmostEqual(_cohen_kappa(pairs), -1.0)


class TestSpearman(unittest.TestCase):
    def test_monotonic(self):
        self.assertAlmostEqual(_spearman([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)

    def test_reverse(self):
        self.assertAlmostEqual(_spearman([1, 2, 3, 4], [8, 6, 4, 2]), -1.0)


if __name__ == "__main__":
    unittest.main()
