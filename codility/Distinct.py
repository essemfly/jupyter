# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")


def solution(A):
    prev = -1000001
    ans = []
    A.sort()
    for elem in A:
        if prev != elem:
            prev = elem
            ans.append(elem)
        else:
            pass

    return len(ans)
