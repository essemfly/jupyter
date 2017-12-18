# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")


def solution(A):
    # write your code in Python 3.6
    A.sort()

    ans = A[-1] * A[-2] * A[-3]

    if A[-1] < 0:
        return ans
    if A[0] * A[1] > A[-2] * A[-3]:
        ans = A[0] * A[1] * A[-1]
    return ans