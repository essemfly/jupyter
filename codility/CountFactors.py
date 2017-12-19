import math


def solution(N):
    ans = 0
    median = math.floor(math.sqrt(N))
    for i in range(median):
        if N % (i + 1) == 0:
            ans += 1

    if median ** 2 == N:
        return ans * 2 - 1
    else:
        return ans * 2
