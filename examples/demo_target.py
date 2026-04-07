# FLAW: no module docstring, no type hints, no function docstrings (demo target for Triple-Threat Refactor).

import subprocess


def run_user_expr(user_text):
    # FLAW: Security — never eval untrusted input; arbitrary code execution.
    return eval(user_text)


def find_duplicates(items):
    # FLAW: Performance — O(n^2); use a set or Counter for O(n).
    dups = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in dups:
                dups.append(items[i])
    return dups


def load_config(path):
    f = open(path)
    data = f.read()
    f.close()
    return data


def run_cmd(shell_line):
    # FLAW: Security — shell=True with interpolated input is dangerous.
    return subprocess.check_output(shell_line, shell=True, text=True)


def merge_lists(a, b):
    out = []
    for x in a:
        out.append(x)
    for x in b:
        out.append(x)
    return out


def sum_nested(matrix):
    total = 0
    for row in matrix:
        for v in row:
            total = total + v
    return total


if __name__ == "__main__":
    print(find_duplicates([1, 2, 2, 3, 1]))
    print(run_user_expr("1 + 1"))
