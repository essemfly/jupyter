# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")


def solution(S):
    # write your code in Python 3.6
    stacks = []

    for letter in S:
        if letter == '[' or letter == '{' or letter == '(':
            stacks.append(letter)
        else:
            if len(stacks) < 1:
                return 0
            else:
                if letter == ']':
                    if stacks[-1] != '[':
                        return 0
                elif letter == '}':
                    if stacks[-1] != '{':
                        return 0
                else:
                    if stacks[-1] != '(':
                        return 0
                stacks.pop()

    return 1 if len(stacks) == 0 else 0
