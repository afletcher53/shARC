# pip install pydantic torch transformers datasets accelerate outlines==0.1.3 matplotlib scikit-learn
import os

hf_cache_dir = "/mnt/parscratch/users/acp23jlc/phd_projects/hf_cache/"  # REPLACE WITH YOUR OWN HF CACHE DIR HERE
# hf_cache_dir = ""
if hf_cache_dir:
    os.environ['HF_HOME'] = hf_cache_dir

from typing import List
import numpy as np
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from pydantic import BaseModel
import outlines

from classes.data_loader import DataLoader
from utils.generate_aug_training import get_augmented_training_examples



# def load_tokenizer_and_model(target_model="meta-llama/Llama-3.2-3B-Instruct"):
#     torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
#
#     with open("my_hf_token.txt", "r") as f:  # REPLACE WITH TXT FPATH CONTAINING YOUR HUGGINGFACE TOKEN
#         my_hf_token = f.read().strip()
#     login(my_hf_token)
#
#     tokenizer = AutoTokenizer.from_pretrained(target_model,
#                                               torch_dtype=torch_dtype,
#                                               padding_side="left")
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     model = AutoModelForCausalLM.from_pretrained(target_model,
#                                                  torch_dtype=torch_dtype,
#                                                  device_map="auto")
#
#     return tokenizer, model

class OutputGrid(BaseModel):
    outputGrid: List[List[int]]

def load_outlines_model(target_model="meta-llama/Llama-3.2-3B-Instruct"):
    with open("my_hf_token.txt", "r") as f:  # REPLACE WITH TXT FPATH CONTAINING YOUR HUGGINGFACE TOKEN
        my_hf_token = f.read().strip()
    login(my_hf_token)
    _model = outlines.models.transformers(target_model, model_kwargs={"device_map": "auto"},
                                          tokenizer_kwargs={"padding_side": "left"})

    generator = outlines.generate.json(_model, OutputGrid)
    tokenizer = AutoTokenizer.from_pretrained(target_model,
                                              padding_side="left")

    return tokenizer, generator

def grid_to_str(grid: list[list[int]]):
    grid_strs = []
    for row in grid:
        row_str = ", ".join([str(x) for x in row])
        grid_strs.append("[" + row_str + "]")
    return "[" + ", ".join(grid_strs) + "]"


def data_instance_to_chat_input(challenge_data_instance, systemPrompt):
    """

    :param challenge_data_instance: dict with keys "train_examples" (list of dicts, each with two keys ("input", "output"), "test_input", "solution"
    :param systemPrompt: system prompt to be used in chat template
    :return: list(dict) messages in chat template [...{"role": "user", "content": "..."},]
    """
    instance_string_list = []
    for io_pair in challenge_data_instance["train_examples"]:
        io_pair_as_string = grid_to_str(io_pair["input"]) + " -> " + grid_to_str(io_pair["output"])
        instance_string_list.append(io_pair_as_string)
    test_input_as_string = grid_to_str(challenge_data_instance["test_input"]) + " -> ?"
    instance_string_list.append(test_input_as_string)

    instance_string = "\n".join(instance_string_list)

    messages = [{"role": "user",
                 "content": f"You are given example pairs of input and output grids "
                            f"which follow the same transformation rule. "
                            f"Infer this rule, transform the final input grid and predict its corresponding output grid."
                            f"Return the output grid as a list of list of integers:\n\n{instance_string}"}]

    if systemPrompt:
        messages.insert(0, {"role": "system", "content": systemPrompt})

    return messages


def load_data(first_n=2):
    dl = DataLoader()
    training_data = dl.load_dataset("training",
                                    dataset_locations_override=("../data/arc-agi_training_challenges.json",
                                                                "../data/arc-agi_training_solutions.json"))
    print(f"Training data loaded. Number of instances: {len(set(training_data.keys()))}")
    ids = list(training_data.keys())
    for i in range(first_n):
        yield dl.get_specific_sample(ids[i])


def pred_v_gt(predGrid, gtGrid, print_result=True):
    predGrid = np.array(predGrid)
    gtGrid = np.array(gtGrid)
    if predGrid.shape != gtGrid.shape:
        print(f"Grid shape mismatch - gt: {gtGrid.shape}, pred: {predGrid.shape}")
        return gtGrid.shape[0]*gtGrid.shape[1]  # counts as getting all cells wrong

    diffGrid = predGrid != gtGrid
    error = np.sum(diffGrid)
    if print_result:
        print("Printing difference grid (TRUE = mismatch between pred and gt found:")
        print(diffGrid)

    return error

def main():
    print("Loading model...")
    tokenizer, outlines_model = load_outlines_model(target_model="meta-llama/Llama-3.2-3B-Instruct")
    # tokenizer, outlines_model = load_outlines_model(target_model="microsoft/Phi-3.5-mini-instruct")
    print("Model loaded")
    print("Running inference...")
    for challenge_data_instance in load_data(first_n=2):
        msg = data_instance_to_chat_input(challenge_data_instance,
                                          systemPrompt="You are a helpful assistant that obeys instructions.")
        ground_truth_grid = challenge_data_instance["solution"]

        chat_templated_prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        print("Example input:")
        print(chat_templated_prompt)

        out = outlines_model(chat_templated_prompt)
        print("Example output:")
        print(out)
        output_grid = out.outputGrid

        print("Example output grid:")
        print(output_grid)

        error = pred_v_gt(output_grid["output_grid"], ground_truth_grid)
        print(f"Example error: {error}")
        print(f"Example ground truth grid: {ground_truth_grid}")

    print("Inference complete")


if __name__ == "__main__":
    main()
