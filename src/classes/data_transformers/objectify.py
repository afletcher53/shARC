import numpy as np
import src.arc_dsl.dsl as dsl
from src.classes.helper import *
from pprint import pprint

# https://github.com/mxbi/arckit/blob/main/arckit/vis.py
# TODO: double-check if this colour mapping is correct,
#  also we might want to rename 'light blue' if we're using LLMs (to avoid 'blue' being more likely?)
NUM2COLOR = {0: "black",
             1: "blue",
             2: "red",
             3: "green",
             4: "yellow",
             5: "grey",
             6: "pink",
             7: "orange",
             8: "light blue",
             9: "brown"}


class StringCompatibleObjects:
    def __init__(self, array_2d):
        self.array_2d = tuple([tuple(x) for x in array_2d])
        self.dsl_objects_returned_item = dsl.objects(self.array_2d, univalued=True, diagonal=False, without_bg=True)

    def toString(self,
                 separator="\n",
                 mode="xu_2024"):

        if mode == "basic":
            # a basic format, to be modified
            objects_str_list = []
            for grouping in self.dsl_objects_returned_item:
                cells_str_list = []

                for cell in grouping:
                    colour, (row_i, col_i) = cell
                    cells_str_list.append(f"({colour}, ({row_i}, {col_i}))")

                objects_str_list.append("Object(cells=[" + ", ".join(cells_str_list) + "])")

            return separator.join(objects_str_list)

        elif mode == "xu_2024":
            # see p.9 - https://openreview.net/pdf?id=E8m8oySvPJ
            # this only works where there is only one colour per grouping
            objects_list = []
            for i, grouping in enumerate(self.dsl_objects_returned_item):
                object_color, object_size = "", ""

                cells_list = []
                for cell_i, cell in enumerate(grouping):
                    colour, (row_i, col_i) = cell
                    cells_list.append(f"({row_i}, {col_i})")

                    if cell_i == len(grouping) - 1:
                        object_color = NUM2COLOR[colour]
                        object_size = str(cell_i + 1)

                object_str = (f"Object {i + 1}: coordinates=[" +
                              ", ".join(cells_list) +
                              f"], color=\"{object_color}\", size={object_size}")

                objects_list.append(object_str)

            objects_str = "\n".join(objects_list)
            return f"Image Size:({len(self.array_2d)}, {len(self.array_2d[0])})\nObjects\n{objects_str}"


def test_grid_to_object():
    input_grids, output_grids = get_sample_input_output(fpath_overrides=get_my_fpaths("../my_fpaths.txt"))
    output_grid = output_grids[1]
    print("---RAW OUTPUT GRID---")
    pprint(output_grid)

    print("---PROCESSED OBJECTS IN STRING FORMAT---")
    dsl_object = StringCompatibleObjects(output_grid)
    print(dsl_object.toString(mode="xu_2024"))


if __name__ == '__main__':
    test_grid_to_object()
