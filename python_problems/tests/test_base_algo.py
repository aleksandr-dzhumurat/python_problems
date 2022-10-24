import unittest

from python_problems.base_algo import timeConversion


class TestStringMethods(unittest.TestCase):
    def test_timeConversion(self):
        self.assertEqual(timeConversion('07:05:45PM'), '19:05:45')


if __name__ == '__main__':
    unittest.main()
    # print(timeConversion('07:05:45PM'))
    # print(timeConversion('17:05:45PM'))
    # print(timeConversion('07:05:45AM'))
    # print(timeConversion('17:05:45AM'))
