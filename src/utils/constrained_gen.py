import re

def makeDynamicOutputGridRegex(n_rows, n_cols):
    base_string = "\["
    core = "[0-9]"
    for i in range(n_rows):
        base_string += "\["
        for k in range(n_cols):
            base_string += core + (", " if k < n_cols - 1 else "")
        base_string += "\]" + (", " if i < n_rows - 1 else "")
    base_string += "\]"

    return base_string


if __name__ == "__main__":
    test_grid = "[[2, 1, 3], [3, 1, 9]]"
    test_regex = makeDynamicOutputGridRegex(2, 3)
    re_obj = re.compile(test_regex)
    print(test_regex)
    print(re_obj.findall(test_grid))
