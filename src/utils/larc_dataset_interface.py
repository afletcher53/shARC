import pandas as pd
from collections import defaultdict

with open("../classes/my_fpaths.txt") as f_obj:
    fpaths = [line.strip() for line in f_obj.readlines() if line]
    description_csv_fpath = fpaths[2]
    task_csv_fpath = fpaths[3]
    join_csv_fpath = fpaths[4]

description_csv = pd.read_csv(description_csv_fpath)
task_csv = pd.read_csv(task_csv_fpath)
join_csv = pd.read_csv(join_csv_fpath)

task_name_to_description_dict = defaultdict(list)

for task_id, task_name in zip(task_csv["task_id"], task_csv["task_name"]):
    task_subset = join_csv[join_csv["task_id"] == task_id]["description_id"]
    filtered = description_csv[description_csv["description_id"].isin(task_subset)].copy().reset_index(drop=True)
    for i in range(len(filtered)):
        task_name_to_description_dict[task_name].append({
            "description_input": filtered["description_input"][i],
            "description_output_grid_size": filtered["description_output_grid_size"][i],
            "description_output": filtered["description_output"][i]
        })

# for k, v in task_name_to_description_dict.items():
#     print(k)
#     print(v)
#     break

# with open("larc_task_names.txt", "w") as f_obj:
#     f_obj.write("\n".join([k.replace(".json", "") for k in task_name_to_description_dict.keys()]))
#
