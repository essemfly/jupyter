# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

import math


def solution(N, M):
    # write your code in Python 3.6
    lcm = N * M // math.gcd(M, N)
    return lcm // M
