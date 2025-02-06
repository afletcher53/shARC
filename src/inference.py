import json
import time

import numpy as np
import outlines
from huggingface_hub import login
from pydantic import BaseModel, field_validator, conlist, Field
from transformers import AutoTokenizer
from classes.data_loader import DataLoader
from classes.project_strings import ProjectStrings

ps = ProjectStrings()
login(ps.HF_TOKEN)

cfg = json.load(open("config.json"))

class PredictedShape(BaseModel):
    """Schema for capturing the predicted shape of the grid."""

    rows: int
    cols: int

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the shape as a tuple (rows, cols)."""
        return (self.rows, self.cols)

class OutputGrid(BaseModel):
    """Schema for capturing the output grid."""

    outputGrid: list[list[int]]

    @field_validator("outputGrid")
    def check_consistency(cls, v):
        """Validates that all rows in the grid have the same length."""
        if len(set(map(len, v))) > 1:
            raise ValueError("Inconsistent number of columns in grid rows")
        return v

def load_outlines_model(target_model=cfg["model_name"]):
    """Loads the outlines model for the given target model."""
    _model = outlines.models.transformers(
        target_model,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    tokenizer = AutoTokenizer.from_pretrained(target_model, padding_side="left")

    return tokenizer, _model

def grid_to_str(grid: list[list[int]]):
    """Converts a grid to a string representation."""
    grid_strs = []
    for row in grid:
        row_str = ", ".join([str(x) for x in row])
        grid_strs.append("[" + row_str + "]")
    return "[" + ", ".join(grid_strs) + "]"

def data_instance_to_chat_input(challenge_data_instance, systemPrompt):
    """
    Formats a data instance into a chat input for the model.
    No longer prompts the LLM to guess rows and cols,
    just shows input-output examples and ends with the test input -> ?
    """
    instance_string_list = []
    for io_pair in challenge_data_instance["train_examples"]:
        io_pair_as_string = (
            grid_to_str(io_pair["input"]) + " -> " + grid_to_str(io_pair["output"])
        )
        instance_string_list.append(io_pair_as_string)

    test_input_as_string = grid_to_str(challenge_data_instance["test_input"]) + " -> ?"
    instance_string_list.append(test_input_as_string)
    instance_string = "\n test input:".join(instance_string_list)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Based on the provided example input-output grids, "
                "generate the final output grid for the test input."
            ),
        },
        {
            "role": "user",
            "content": f"{instance_string}",
        },
    ]

    # Optionally prepend a system prompt if provided:
    if systemPrompt:
        messages.insert(0, {"role": "system", "content": systemPrompt})

    return messages

def get_dataloader_and_ids(dataset_type="training"):
    """Loads the dataset and returns the dataloader and corresponding IDs."""
    dl = DataLoader()
    training_data = dl.load_dataset(dataset_type=dataset_type)
    print(
        f"Training data loaded. Number of instances: {len(set(training_data.keys()))}"
    )
    ids = list(training_data.keys())
    return dl, ids

def pred_v_gt(predGrid, gtGrid, print_result=False):
    """Compares the predicted grid with the ground truth grid."""
    gtGrid = np.array(gtGrid)
    try:
        predGrid = np.array(predGrid)
        if predGrid.shape != gtGrid.shape:
            print(
                f"Grid shape mismatch - gt: {gtGrid.shape}, pred: {predGrid.shape}"
            )
            return (
                gtGrid.shape[0] * gtGrid.shape[1],
                True,
            )
        diffGrid = predGrid != gtGrid
        error = np.sum(diffGrid)
        if print_result:
            print(
                "Printing difference grid (TRUE = mismatch between pred and gt found:"
            )
            print(diffGrid)
        return error, False
    except ValueError:
        print("Predicted arrays are jagged and do not form a grid.")
        return gtGrid.shape[0] * gtGrid.shape[1], True

def analyze_dimensions(instance):
    """
    Analyzes a single instance's input and output dimensions and infers the transformation rule.
    Enhanced to provide more specific transformation rules.
    """

    width_ratios = []
    height_ratios = []

    for io_pair in instance["train_examples"]:
        output_width = len(io_pair["output"][0])
        output_height = len(io_pair["output"])
        input_width = len(io_pair["input"][0])
        input_height = len(io_pair["input"])

        width_ratio = output_width / input_width if input_width != 0 else 0
        height_ratio = output_height / input_height if input_height != 0 else 0

        width_ratios.append(width_ratio)
        height_ratios.append(height_ratio)

    if all(
        w == width_ratios[0] and w.is_integer() for w in width_ratios
    ) and all(h == height_ratios[0] and h.is_integer() for h in height_ratios):
        if width_ratios[0] == height_ratios[0] == 1:
            transformation_type = "no_change"
        elif width_ratios[0] == height_ratios[0]:
            transformation_type = "replication"
        elif width_ratios[0] == 1:
            transformation_type = "vertical_scaling"
        elif height_ratios[0] == 1:
            transformation_type = "horizontal_scaling"
        else:
            transformation_type = "complex"
    else:
        transformation_type = "complex"

    return {
        "width_ratio": width_ratios[0],
        "height_ratio": height_ratios[0],
        "transformation_type": transformation_type,
    }

def apply_transformation(input_shape, transformation_rule):
    """Applies the inferred transformation rule to the input shape."""
    input_width, input_height = input_shape

    if transformation_rule["transformation_type"] == "no_change":
        return PredictedShape(rows=input_height, cols=input_width)
    elif transformation_rule["transformation_type"] == "replication":
        return PredictedShape(
            rows=int(input_height * transformation_rule["height_ratio"]),
            cols=int(input_width * transformation_rule["width_ratio"]),
        )
    elif transformation_rule["transformation_type"] == "vertical_scaling":
        return PredictedShape(
            rows=int(input_height * transformation_rule["height_ratio"]),
            cols=input_width,
        )
    elif transformation_rule["transformation_type"] == "horizontal_scaling":
        return PredictedShape(
            rows=input_height,
            cols=int(input_width * transformation_rule["width_ratio"]),
        )
    elif transformation_rule["transformation_type"] == "complex":
        avg_width_ratio = transformation_rule["width_ratio"]
        avg_height_ratio = transformation_rule["height_ratio"]
        return PredictedShape(
            rows=int(input_height * avg_height_ratio),
            cols=int(input_width * avg_width_ratio),
        )
    else:
        raise ValueError(
            f"Unknown transformation type: {transformation_rule['transformation_type']}"
        )

def batched_inference(batch_size=1):
    """Performs batched inference on the dataset."""
    target_model = cfg["model_name"]
    run_on_n_data_samples = 400
    systemPrompt = "You are a helpful assistant that obeys instructions."

    dl, ids = get_dataloader_and_ids(dataset_type="training")

    errors = []
    areShapeMismatches = []
    gtGridSizes = []
    incorrect_size_predictions = []
    incorrect_size_predictions_details = {}

    start_time = time.time()
    for i in range(0, run_on_n_data_samples, batch_size):
        print(
            f"Processing examples starting from {i}... time lapsed: {(time.time()-start_time)/60:.2f} minutes",
            flush=True,
        )

        # # Load the model and tokenizer at the beginning of each batch
        # print("Loading model...", flush=True)
        # tokenizer, outlines_model = load_outlines_model(target_model=target_model)
        # print("Model loaded", flush=True)

        if (i + batch_size) > run_on_n_data_samples:
            batch = [
                dl.get_specific_sample(example_id)
                for example_id in ids[i:run_on_n_data_samples]
            ]
        else:
            batch = [
                dl.get_specific_sample(example_id)
                for example_id in ids[i : i + batch_size]
            ]

        batch_msgs = [
            data_instance_to_chat_input(instance, systemPrompt=systemPrompt)
            for instance in batch
        ]

        ground_truth_grids = [instance["solution"][0] for instance in batch]

        transformation_rules = []
        for instance in batch:
            transformation_rule = analyze_dimensions(instance)
            transformation_rules.append(transformation_rule)

        predicted_shapes = []
        for instance, transformation_rule in zip(batch, transformation_rules):
            input_shape = (
                len(instance["test_input"][0]),
                len(instance["test_input"]),
            )
            predicted_shape = apply_transformation(input_shape, transformation_rule)
            predicted_shapes.append(predicted_shape)
           

        # Check if this is actually correct (from batch[i]['solution'])
        for k, (instance, predicted_shape) in enumerate(
            zip(batch, predicted_shapes)
        ):
            rows = len(instance["solution"][0])
            cols = len(instance["solution"][0][0])
            if (rows, cols) != predicted_shape.shape:
                incorrect_size_predictions.append(i + k)  # Adjusted index
                incorrect_size_predictions_details[i + k] = {
                "input_shape": input_shape,
                "transformation_rule": transformation_rule,
                "predicted_shape": predicted_shape,
            }

        # chat_templated_prompts = tokenizer.apply_chat_template(
        #     batch_msgs, tokenize=False, add_generation_prompt=True
        # )

        # output_grids = []
        # for j, predicted_shape in enumerate(predicted_shapes):
        #     (rows, cols) = predicted_shape.shape
        #     current_index = i + j

            # if current_index in incorrect_size_predictions:
            #     print(f"Skipping instance {current_index} due to incorrect size prediction")
            #     continue

            # @outlines.prompt
            # def generate_grid_prompt(chat_templated_prompt, rows, cols):
            #     """
            #     {{ chat_templated_prompt }}
            #     The final output grid has {{ rows }} rows and {{ cols }} columns.
            #     Generate the grid in JSON format with key 'outputGrid'.
            #     """

            # grid_prompt = generate_grid_prompt(chat_templated_prompts[j], rows, cols)

            # print(grid_prompt)

            # class DynamicOutputGrid(BaseModel):
            #     outputGrid: list[list[int]]

            #     @field_validator("outputGrid")
            #     def check_dimensions(cls, v, values, **kwargs):
            #         if len(v) != rows:
            #             print(
            #                 f"Warning: Expected {rows} rows, but got {len(v)}. Adjusting the number of rows."
            #             )

            #         adjusted_grid = []
            #         for row_idx in range(rows):
            #             if row_idx < len(v):
            #                 row = v[row_idx]
            #                 if len(row) != cols:
            #                     print(
            #                         f"Warning: Expected {cols} columns in row {row_idx}, but got {len(row)}. Adjusting row."
            #                     )

            #                 adjusted_row = row[:cols] + [0] * max(
            #                     0, cols - len(row)
            #                 )
            #             else:
            #                 adjusted_row = [0] * cols

            #             adjusted_grid.append(adjusted_row)

            #         return adjusted_grid

            # grid_generator = outlines.generate.json(
            #     outlines_model, DynamicOutputGrid
            # )

            # try:
            #     output_grid = grid_generator(grid_prompt).outputGrid
            # except ValueError as e:
            #     print(f"Error during grid generation: {e}")
            #     output_grid = [[0] * cols for _ in range(rows)]

            # output_grids.append(output_grid)

            # print(output_grid)

            # current_instance_index = i + j
            # if current_instance_index < len(batch):
            #     current_instance = batch[current_instance_index]
            #     output_directory = "inference_plots"
            #     dl.plot_inference_results(
            #         current_instance, output_grid, output_directory
            #     )


        # compares = [
        #     pred_v_gt(output_grid, ground_truth_grid)
        #     for output_grid, ground_truth_grid in zip(
        #         output_grids, ground_truth_grids
        #     )
        # ]

        # errors.extend([x[0] for x in compares])
        # areShapeMismatches.extend([x[1] for x in compares])
        # gtGridSizes.extend(
        #     [
        #         len(ground_truth_grid) * len(ground_truth_grid[0])
        #         for ground_truth_grid in ground_truth_grids
        #     ]
        # )

        # Unload the model and clear the CUDA cache at the end of each batch
        # del tokenizer
        # del outlines_model
        # torch.cuda.empty_cache()

        print("-" * 100)
        print(
            f"Batch {i} complete in {(time.time()-start_time)/60:.2f} minutes"
            f" - Incorrect size predictions: {len(incorrect_size_predictions)}"
        )
        print("-" * 100)

    relative_errors = [
        error / gtGridSize for error, gtGridSize in zip(errors, gtGridSizes)
    ]

    print("Inference complete")

    print("SUMMARY")
    print(f"Model: {target_model}")
    print(f"No. of samples: {run_on_n_data_samples}")
    print(f"No. of Exact grid matches: {errors.count(0)}")
    print(
        f"No. of instances where predicted grid shape != ground truth grid shape: {areShapeMismatches.count(True)}"
    )
    print(
        f"Mean 'relative' error (i.e. wrong cells divided by all cells in ground truth grid): {np.mean(relative_errors)}"
    )

if __name__ == "__main__":
    batched_inference(batch_size=1)