# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")


def solution(A, B):
    # write your code in Python 3.6
    fishes = len(A)
    ans = fishes
    downstream = []

    for i in range(fishes):
        if B[i] == 0:
            while len(downstream) > 0:
                if downstream[-1] < A[i]:
                    ans -= 1
                    downstream.pop()
                else:
                    ans -= 1
                    break

        else:
            downstream.append(A[i])

    return ans