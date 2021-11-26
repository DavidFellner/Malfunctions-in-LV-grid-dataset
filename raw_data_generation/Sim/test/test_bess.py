import unittest
import random

import simulated_components.sim_bess as bess


N_TESTS = 1000


class TestBess(unittest.TestCase):

    @staticmethod
    def create_random_test_bess():
        return bess.BESS(
            capacity=random.uniform(2.0, 10.0),
            max_p=random.uniform(2.0, 7.0),
            min_p=random.uniform(-2.0, -7.0),
            init_power=random.random(),
            init_soc=random.random()
        )

    def test_charging_max(self):
        for i in range(N_TESTS):
            test_bess = self.create_random_test_bess()
            test_power = random.uniform(test_bess.max_p,
                                        test_bess.max_p + 5.0 * random.random())
            test_bess.go(test_power)

            if test_bess.soc < test_bess.max_soc:
                self.assertEqual(test_bess.power, test_bess.max_p)
            else:
                self.assertLessEqual(test_bess.power, test_bess.max_p)

            self.assertGreaterEqual(test_bess.max_soc, test_bess.soc)

    def test_charging_min(self):
        for i in range(N_TESTS):
            test_bess = self.create_random_test_bess()
            test_power = random.uniform(test_bess.min_p,
                                        (test_bess.min_p - 5.0 * random.random()))
            test_bess.go(test_power)

            if test_bess.soc > test_bess.min_soc:
                self.assertEqual(test_bess.power, test_bess.min_p)
            else:
                self.assertGreaterEqual(test_bess.power, test_bess.min_p)

            self.assertGreaterEqual(test_bess.soc, test_bess.min_soc)

    def test_calc_soc(self):
        for i in range(N_TESTS):
            test_bess = self.create_random_test_bess()
            test_power = random.uniform(test_bess.min_p,
                                        test_bess.max_p)

            test_soc = test_bess.calc_soc(test_power)
            extreme_value = test_bess.go(test_power)

            if extreme_value:
                self.assertTrue(test_bess.soc == test_bess.max_soc or test_bess.soc == test_bess.min_soc)
            else:
                self.assertAlmostEqual(test_bess.soc, test_soc, places=4)


if __name__ == '__main__':
    unittest.main()

