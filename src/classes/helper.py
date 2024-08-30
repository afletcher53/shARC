from src.classes.data_loader import DataLoader
import numpy as np
from pprint import pprint

def get_sample_input_output(challenge_id="00d62c1b",
                            train_or_test="train",
                            fpath_overrides=()):
    dl = DataLoader()
    dataset = dl.load_dataset("training",
                              dataset_locations_override=fpath_overrides)

    challenge = dl.get_specific_sample(challenge_id)

    n_train_examples = len(challenge["train_examples"])

    if train_or_test == "test":
        return challenge["test_input"], challenge["solution"]

    else:
        return ([challenge["train_examples"][n]["input"] for n in range(n_train_examples)],
                [challenge["train_examples"][n]["output"] for n in range(n_train_examples)])

def get_my_fpaths(txt_file="my_fpaths.txt"):
    with open(txt_file) as f_obj:
        fpaths = f_obj.read().splitlines()

    return tuple(fpaths)


if __name__ == "__main__":
    pass