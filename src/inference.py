import json
import time

import numpy as np
import outlines
from huggingface_hub import login
from pydantic import BaseModel, field_validator
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

class OutputGrid(BaseModel):
    """Schema for capturing the output grid."""
    outputGrid: list[list[int]]

    @field_validator('outputGrid')
    def check_consistency(cls, v):
        """Validates that all rows in the grid have the same length."""
        if len(set(map(len, v))) > 1:
            raise ValueError('Inconsistent number of columns in grid rows')
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
    """Formats a data instance into a chat input for the model."""
    instance_string_list = []
    for io_pair in challenge_data_instance["train_examples"]:
        io_pair_as_string = (
            grid_to_str(io_pair["input"])
            + " -> "
            + grid_to_str(io_pair["output"])
        )
        instance_string_list.append(io_pair_as_string)
    test_input_as_string = (
        grid_to_str(challenge_data_instance["test_input"]) + " -> ?"
    )
    instance_string_list.append(test_input_as_string)

    instance_string = "\n".join(instance_string_list)

    messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can predict the shape of output grids based on input-output examples. The output grid's shape is determined by a transformation rule applied to the input grid's shape."
    },
    {
        "role": "user",
        "content": f"You are given example pairs of input and output grids "
        f"which follow the same transformation rule. "
        f"Infer this rule, and apply it to the final input grid. "
        f"First, predict the number of rows and columns the output grid will have. "
        f"Express this as a JSON object with the keys 'rows' and 'cols', where 'rows' is the number of rows and 'cols' is the number of columns. "
        f"Then, generate the output grid itself as a list of lists of integers.\n\n{instance_string}",
    }
]
    if systemPrompt:
        messages.insert(0, {"role": "system", "content": systemPrompt})

    return messages

def get_dataloader_and_ids(dataset_type="training"):
    """Loads the dataset and returns the dataloader and corresponding IDs."""
    dl = DataLoader()
    training_data = dl.load_dataset(
        dataset_type=dataset_type
    )
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
    """Analyzes a single instance's input and output dimensions and infers the transformation rule."""
    print(f"Instance:")
    
    # Initialize variables to store inferred rules
    width_ratio = None
    height_ratio = None
    transformation_type = None

    for io_pair in instance["train_examples"]:
        output_width = len(io_pair["output"][0])
        output_height = len(io_pair["output"])
        input_width = len(io_pair["input"][0])
        input_height = len(io_pair["input"])

        # Calculate ratios only for the first example to infer the rule
        if width_ratio is None:
            width_ratio = output_width / input_width
        if height_ratio is None:
            height_ratio = output_height / input_height

        print(f"  Shape of output: {output_width} x {output_height}")
        print(f"  Shape of input: {input_width} x {input_height}")
        print(f"  Width Ratio: {width_ratio}")
        print(f"  Height Ratio: {height_ratio}")

        # Infer basic transformation type based on ratios
        if width_ratio == height_ratio == 1:
            transformation_type = "no_change"
            print("  Possible Transformation: No spatial change or complex rearrangement")
        elif width_ratio == height_ratio == int(width_ratio):
            transformation_type = "replication"
            print(f"  Possible Transformation: Replication/Scaling by a factor of {int(width_ratio)}")
        elif width_ratio == 1:
            transformation_type = "vertical_scaling"
            print(f"  Possible Transformation: Vertical scaling/stretching by a factor of {height_ratio}")
        elif height_ratio == 1:
            transformation_type = "horizontal_scaling"
            print(f"  Possible Transformation: Horizontal scaling/stretching by a factor of {width_ratio}")
        else:
            transformation_type = "complex"
            print("  Possible Transformation: Complex transformation (e.g., scaling with interpolation, cropping, etc.)")

    # Return the inferred transformation rule
    return {
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "transformation_type": transformation_type
    }

def apply_transformation(input_shape, transformation_rule):
    """Applies the inferred transformation rule to the input shape."""
    input_width, input_height = input_shape

    if transformation_rule["transformation_type"] == "no_change":
        return PredictedShape(rows=input_height, cols=input_width)
    elif transformation_rule["transformation_type"] == "replication":
        return PredictedShape(
            rows=int(input_height * transformation_rule["height_ratio"]),
            cols=int(input_width * transformation_rule["width_ratio"])
        )
    elif transformation_rule["transformation_type"] == "vertical_scaling":
        return PredictedShape(
            rows=int(input_height * transformation_rule["height_ratio"]),
            cols=input_width
        )
    elif transformation_rule["transformation_type"] == "horizontal_scaling":
        return PredictedShape(
            rows=input_height,
            cols=int(input_width * transformation_rule["width_ratio"])
        )
    elif transformation_rule["transformation_type"] == "complex":
        # TODO: Add more complex transformations
        return PredictedShape(rows=input_height, cols=input_width)
    else:
        raise ValueError(f"Unknown transformation type: {transformation_rule['transformation_type']}")

def batched_inference(batch_size=3):
    """Performs batched inference on the dataset."""
    target_model = cfg["model_name"]
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
        print(
            f"Processing examples starting from {i}... time lapsed: {(time.time()-start_time)/60:.2f} minutes",
            flush=True,
        )
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
        
        # Analyse dimensions and infer transformation rules for each instance in the batch
        transformation_rules = []
        for instance in batch:
            transformation_rule = analyze_dimensions(instance)
            transformation_rules.append(transformation_rule)

        # Apply transformation rules to predict shapes
        predicted_shapes = []
        for instance, transformation_rule in zip(batch, transformation_rules):
            input_shape = (len(instance["test_input"][0]), len(instance["test_input"])) 
            predicted_shape = apply_transformation(input_shape, transformation_rule)
            predicted_shapes.append(predicted_shape)

        chat_templated_prompts = tokenizer.apply_chat_template(
            batch_msgs, tokenize=False, add_generation_prompt=True
        )

        output_grids = []
        for j, predicted_shape in enumerate(predicted_shapes):
            # 2. Generate the output grid based on the predicted shape
            rows, cols = predicted_shape.rows, predicted_shape.cols
            
            @outlines.prompt
            def generate_grid_prompt(chat_templated_prompt, rows, cols):
                """
                {{ chat_templated_prompt }}
                The predicted output grid shape is {{ rows }} rows and {{ cols }} columns.
                Now generate the output grid itself:
                """
            # Generate the prompt for grid generation
            grid_prompt = generate_grid_prompt(
                chat_templated_prompts[j],
                rows,
                cols
            )
            # Dynamically define the output schema based on predicted shape
            class DynamicOutputGrid(BaseModel):
                outputGrid: list[list[int]]

                @field_validator('outputGrid')
                def check_dimensions(cls, v):
                    if len(v) != rows:
                        print(f"Warning: Expected {rows} rows, but got {len(v)}. Padding or truncating as needed.")
                    
                    padded_or_truncated_grid = []
                    for row_idx in range(rows):
                        if row_idx < len(v):
                            row = v[row_idx]
                            if len(row) != cols:
                                print(f"Warning: Expected {cols} columns in row {row_idx}, but got {len(row)}. Padding or truncating row.")
                            
                            padded_or_truncated_row = row[:cols] + [0] * max(0, cols - len(row))
                        
                        else:
                            padded_or_truncated_row = [0] * cols
                        
                        padded_or_truncated_grid.append(padded_or_truncated_row)

                    return padded_or_truncated_grid
            grid_generator = outlines.generate.json(outlines_model, DynamicOutputGrid)
            
            try:
                output_grid = grid_generator(grid_prompt).outputGrid
            except ValueError as e:
                print(f"Error during grid generation: {e}")
                output_grid = [[0] * cols for _ in range(rows)]

            output_grids.append(output_grid)

        compares = [
            pred_v_gt(output_grid, ground_truth_grid)
            for output_grid, ground_truth_grid in zip(
                output_grids, ground_truth_grids
            )
        ]

        errors.extend([x[0] for x in compares])
        areShapeMismatches.extend([x[1] for x in compares])
        gtGridSizes.extend(
            [
                len(ground_truth_grid) * len(ground_truth_grid[0])
                for ground_truth_grid in ground_truth_grids
            ]
        )

        print("-" * 100)
        print(f"Batch {i} complete in {(time.time()-start_time)/60:.2f} minutes")
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
    batched_inference(batch_size=3)