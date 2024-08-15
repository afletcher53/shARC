# shARC


# classes/data_transfomers 

This should contain a class containing how your data is transformed from the JSON file to the input for the model used.

# DataLoader Class

The `DataLoader` class is a utility for loading, processing, and handling datasets for challenges. It ensures that the datasets adhere to a specific structure. The class also provides functionalities to randomly sample data points and retrieve specific challenges by their ID.

## Usage

### Loading a dataset
The DataLoader class can load different types of datasets such as training, evaluation, and test. Here's how to load a dataset:
```
from classes.data_loader import DataLoader

# Initialize the DataLoader
dl = DataLoader()

# Load a dataset (e.g., 'training')
training_data = dl.load_dataset("training")

# Check the number of unique challenges
unique_keys = set(training_data.keys())
print(f"Number of training challenges: {len(unique_keys)}")
```

### Randomly Sampling Datapoints
You can randomly sample a specific number of data points from the loaded datasets:

```python
# Randomly sample 5 challenges from the loaded dataset
sampled_data = dl.randomly_sample_datapoints(5)

# Print out the sampled challenge IDs
for challenge_id in sampled_data.keys():
    print(f"Sampled Challenge ID: {challenge_id}")

```

### Retrieving a Specific Challenge
To retrieve a specific challenge by its ID:

```python
# Retrieve a specific challenge by ID
challenge_id = "007bbfb7"
challenge_data = dl.get_specific_sample(challenge_id)

# Access different parts of the challenge
print("Test Input:", challenge_data['test_input'])
print("Training Examples:", challenge_data['train_examples'])
print("Solution:", challenge_data['solution'])
```

### Plotting Data
The DataLoader class provides methods to visualize the data:

#### Plotting Train and Test Examples
```python
# Plot train and test examples for a specific challenge
dl.plot_train_and_test_examples({challenge_id: challenge_data})
```
![Train_test_plot](./output/d22278a0_train_and_test_example.png)

```python
# Plot a specific solution grid
dl.plot_solution(challenge_data['test_input'], f"{challenge_id}_test_input")
```
![specific_solution](./output/007bbfb7_train_3_input.png)

### Finding similar solutions

Given an input grid, we can find similar solutions via:

```python

grid = [[...][...][...]]
similar_grids, original_idx = find_similar_solutions(grid, 5)

```

This function

1. Finds the N most similar grids from all solutions, using cosine similarity on the colour distribution, and grid size.
2. Augments the original solution via rotation, jittering, masking, scaling and cropping
3. Returns unique values within that pool, and add ensures solution is within that pool. 


