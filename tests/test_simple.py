import unittest


class TestSimple(unittest.TestCase):
    def setUp(self) -> None:
        print("Setting up a testcase...")

    def test_simple(self) -> None:
        self.assertTrue(True, "This should be True.")

    def test_another(self) -> None:
        self.assertTrue(True, "This should also be True.")

    def tearDown(self) -> None:
        print("Tearing down a testcase...")
