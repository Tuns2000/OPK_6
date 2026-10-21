import math
import unittest

from variant_1 import make_workload, run_sequential
from variant_1_parallel import (
    MODELS,
    amdahl_speedup,
)


class ParallelModelsTests(unittest.TestCase):
    def test_models_match_sequential_reference(self) -> None:
        cpu_seeds, io_delays = make_workload(1)
        expected_sum, expected_count = run_sequential(cpu_seeds, io_delays)

        for name, model in MODELS.items():
            with self.subTest(model=name):
                actual_sum, actual_count = model(cpu_seeds, io_delays)
                self.assertTrue(math.isclose(actual_sum, expected_sum, rel_tol=1e-12))
                self.assertEqual(actual_count, expected_count)

    def test_amdahl_formula(self) -> None:
        self.assertAlmostEqual(amdahl_speedup(0.5, 2), 4 / 3)
        self.assertEqual(amdahl_speedup(0.0, 8), 8)


if __name__ == "__main__":
    unittest.main()
