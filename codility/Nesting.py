# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")


def solution(S):
    stacks = []
    for letter in S:
        if letter == '(':
            stacks.append(letter)
        else:
            if len(stacks) < 1:
                return 0
            stacks.pop()
    return 0 if len(stacks) != 0 else 1