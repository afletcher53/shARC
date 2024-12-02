# pip install pydantic torch transformers datasets accelerate outlines==0.1.3 matplotlib scikit-learn
import os

with open("my_absolute_fpaths.txt") as f:
    fpaths = [line.strip() for line in f.readlines()]

hf_cache_dir = fpaths[0]  # REPLACE WITH YOUR OWN HF CACHE DIR HERE
# hf_cache_dir = ""
if hf_cache_dir:
    os.environ['HF_HOME'] = hf_cache_dir

from pprint import pprint
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


class OutputGrid(BaseModel):
    outputGrid: List[List[int]]
    # TODO: we might consider dynamically constrain the dimensions of the outputs based on an upstream LLM prediction of grid size
    # TODO: this predicted info can also appear as part of the prompt

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
                            f"Infer this rule, transform the final input grid and predict its corresponding output grid. "
                            f"Return the output grid as a list of list of integers.\n\n{instance_string}"}]

    if systemPrompt:
        messages.insert(0, {"role": "system", "content": systemPrompt})

    return messages


def load_data(first_n=2):
    dl = DataLoader()
    training_data = dl.load_dataset("training",
                                    dataset_locations_override=(fpaths[1], fpaths[2]))
    print(f"Training data loaded. Number of instances: {len(set(training_data.keys()))}")
    ids = list(training_data.keys())
    for i in range(first_n):
        yield dl.get_specific_sample(ids[i])

def get_dataloader_and_ids(dataset_type="training"):
    dl = DataLoader()
    training_data = dl.load_dataset(dataset_type=dataset_type,
                                    dataset_locations_override=(fpaths[1], fpaths[2]))
    print(f"Training data loaded. Number of instances: {len(set(training_data.keys()))}")
    ids = list(training_data.keys())
    return dl, ids

def pred_v_gt(predGrid, gtGrid, print_result=False):
    gtGrid = np.array(gtGrid)
    try:
        predGrid = np.array(predGrid)
    except ValueError:
        print("Predicted arrays are jagged and do not form a grid.")
        return gtGrid.shape[0]*gtGrid.shape[1], True
    
    
    if predGrid.shape != gtGrid.shape:
        print(f"Grid shape mismatch - gt: {gtGrid.shape}, pred: {predGrid.shape}")
        return gtGrid.shape[0]*gtGrid.shape[1], True  # counts as getting all cells wrong, flag to indicate isShapeMismatch

    diffGrid = predGrid != gtGrid
    error = np.sum(diffGrid)
    if print_result:
        print("Printing difference grid (TRUE = mismatch between pred and gt found:")
        print(diffGrid)

    return error, False  # flag to indicate error is not shape-mismatch

def batched_inference(batch_size=3):
    target_model = "meta-llama/Llama-3.2-1B-Instruct"
    run_on_n_data_samples = 30
    systemPrompt = "You are a helpful assistant that obeys instructions."

    print("Loading model...", flush=True)
    tokenizer, outlines_model = load_outlines_model(target_model=target_model)
    print("Model loaded", flush=True)
    print("Running inference...", flush=True)

    dl, ids = get_dataloader_and_ids(dataset_type="training")

    errors = []
    areShapeMismatches = []
    gtGridSizes = []

    start_time = time.time()
    for i in range(0, run_on_n_data_samples, batch_size):
        print(f"Processing examples starting from {i}... time lapsed: {(time.time()-start_time)/60:.2f} minutes", flush=True)
        if (i + batch_size) > run_on_n_data_samples:
            batch = [dl.get_specific_sample(example_id) for example_id in ids[i:run_on_n_data_samples]]
        else:
            batch = [dl.get_specific_sample(example_id) for example_id in ids[i:i+batch_size]]

        batch_msgs = [data_instance_to_chat_input(instance, systemPrompt=systemPrompt) for instance in batch]
        ground_truth_grids = [instance["solution"][0] for instance in batch]

        chat_templated_prompts = tokenizer.apply_chat_template(batch_msgs, tokenize=False, add_generation_prompt=True)

        outputs = outlines_model(chat_templated_prompts)
        output_grids = [x.outputGrid for x in outputs]

        compares = [pred_v_gt(output_grid, ground_truth_grid)
                    for output_grid, ground_truth_grid in zip(output_grids, ground_truth_grids)]

        errors.extend([x[0] for x in compares])
        areShapeMismatches.extend([x[1] for x in compares])
        gtGridSizes.extend([len(ground_truth_grid)*len(ground_truth_grid[0]) for ground_truth_grid in ground_truth_grids])

    relative_errors = [error/gtGridSize for error, gtGridSize in zip(errors, gtGridSizes)]

    print("Inference complete")

    print("SUMMARY")
    print(f"Model: {target_model}")
    print(f"No. of samples: {run_on_n_data_samples}")
    print(f"No. of Exact grid matches: {errors.count(0)}")
    print(f"No. of instances where predicted grid shape != ground truth grid shape: {areShapeMismatches.count(True)}")
    print(f"Mean 'relative' error (i.e. wrong cells divided by all cells in ground truth grid): {np.mean(relative_errors)}")



def serial_inference():
    target_model = "meta-llama/Llama-3.2-1B-Instruct"
    run_on_n_data_samples = 200
    print("Loading model...")
    tokenizer, outlines_model = load_outlines_model(target_model=target_model)
    print("Model loaded")
    print("Running inference...")

    errors = []
    areShapeMismatches = []
    gtGridSizes = []

    start_time = time.time()
    for i, challenge_data_instance in enumerate(load_data(first_n=run_on_n_data_samples)):
        if i % 5 == 0: 
            print(f"Running sample {i+1} out of {run_on_n_data_samples}. Time lapsed: {(time.time()-start_time)/60:.2f} minutes.", flush=True)
        msg = data_instance_to_chat_input(challenge_data_instance,
                                          systemPrompt="You are a helpful assistant that obeys instructions.")
        ground_truth_grid = challenge_data_instance["solution"][0]

        chat_templated_prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

        output_grid = outlines_model(chat_templated_prompt).outputGrid
        # TODO: this is currently run in serial, to be parallelised using the outlines API
        # TODO: train on original unpermuted grids (200?), hold out test set (50) i.e. offline fine-tuning 
        # TODO: implement test-time fine-tuning

        error, isShapeMismatch = pred_v_gt(output_grid, ground_truth_grid)
        errors.append(error)
        areShapeMismatches.append(isShapeMismatch)
        gtGridSizes.append(len(ground_truth_grid)*len(ground_truth_grid[0]))
    
    relative_errors = [error/gtGridSize for error, gtGridSize in zip(errors, gtGridSizes)]

    print("Inference complete", flush=True)

    print("SUMMARY")
    print(f"Model: {target_model}")
    print(f"No. of samples: {run_on_n_data_samples}")
    print(f"No. of Exact grid matches: {errors.count(0)}")
    print(f"No. of instances where predicted grid shape != ground truth grid shape: {areShapeMismatches.count(True)}")
    print(f"Mean 'relative' error (i.e. wrong cells divided by all cells in ground truth grid): {np.mean(relative_errors)}", flush=True)


if __name__ == "__main__":
    batched_inference(batch_size=3)
