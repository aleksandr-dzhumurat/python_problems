import numpy as np

from app.base_algo import timeConversion

if __name__ == '__main__':
    print(np.__version__)
    print(timeConversion('07:05:45PM'))
    print(timeConversion('17:05:45PM'))
    print(timeConversion('07:05:45AM'))
    print(timeConversion('17:05:45AM'))