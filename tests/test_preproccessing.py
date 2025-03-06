import unittest
from preprocessing.dimensions import nearest_power_of_two

class TestNearestPowerOfTwo(unittest.TestCase):
    def test_nearest_power_of_two_positive(self):
        self.assertEqual(nearest_power_of_two(1), 1)
        self.assertEqual(nearest_power_of_two(2), 2)
        self.assertEqual(nearest_power_of_two(3), 4)
        self.assertEqual(nearest_power_of_two(5), 4)
        self.assertEqual(nearest_power_of_two(8), 8)
        self.assertEqual(nearest_power_of_two(9), 8)
        self.assertEqual(nearest_power_of_two(15), 16)
        self.assertEqual(nearest_power_of_two(16), 16)
        self.assertEqual(nearest_power_of_two(17), 16)

    def test_nearest_power_of_two_zero_or_negative(self):
        with self.assertRaises(AssertionError):
            nearest_power_of_two(0)
        with self.assertRaises(AssertionError):
            nearest_power_of_two(-1)
        with self.assertRaises(AssertionError):
            nearest_power_of_two(-10)

if __name__ == "__main__":
    unittest.main()