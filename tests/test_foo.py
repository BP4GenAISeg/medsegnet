import unittest

class TestFoobar(unittest.TestCase):
    def setUp(self):
      self.foo = 'bar'

    def test_foo(self):
        self.assertEqual(self.foo, 'bar')

if __name__ == "__main__":
    unittest.main()