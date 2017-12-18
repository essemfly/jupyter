# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
import math

def solution(N):
    for i in range(math.floor(math.sqrt(N)),0,-1):
        if N % i == 0:
            return (N // i + i)*2
    # write your code in Python 3.6
