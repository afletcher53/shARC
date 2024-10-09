from classes import ProjectStrings
from classes.data_loader import DataLoader
from utils.find_similar_grid import find_similar_solutions, generate_augmentations
#  from utils.generate_train_output_for_llm import generate_nld_of_example  # Assuming 'classes' is a package with an __init__.py file
import json

def main():
    project_strings = ProjectStrings()


    # Initialize the DataLoader
    dl = DataLoader()

    # Load a dataset (e.g., 'training')
    training_data = dl.load_dataset("training")

    # for each training data, count the number of training inputs
    total_training_inputs = 0
    for key, value in training_data.items():
        total_training_inputs += len(value['train_examples'])
    print(f"Total training inputs: {total_training_inputs}")

    # Check the number of unique challenges
    unique_keys = set(training_data.keys())
    print(f"Number of training challenges: {len(unique_keys)}")

    # Retrieve a specific challenge by ID
    challenge_id = "00d62c1b"
    challenge_data = dl.get_specific_sample(challenge_id)

    # Access different parts of the challenge
    print("Test Input:", challenge_data['test_input'])
    print("Training Examples:", challenge_data['train_examples'])
    print("Solution:", challenge_data['solution'])

    # Plot train and test examples for a specific challenge
    # dl.plot_train_and_test_examples({challenge_id: challenge_data})
    # Plot a specific solution grid

    # dl.plot_solution(challenge_data['test_input'], f"{challenge_id}_test_input")

    # get a specific solution grid
    solution = challenge_data['solution']
    

    # generate augmented solutions
    aug_solutions = generate_augmentations(solution)



    similar_grids, original_idx = find_similar_solutions(solution, dl, 5)  
        
    # plot the similar grids
    for idx, grid in enumerate(similar_grids):
        dl.plot_solution(grid, f"similar_grid_{idx}")

    # plot the original solution
    dl.plot_solution(solution, "original_solution")
if __name__ == "__main__":
    # dl = DataLoader()
    # train = dl.load_dataset("training")
    # dl.plot_single_train_examples(train, 'output/singles')

    # get all the image files for the training data
    # responses = generate_nld_of_example(sample=5)

    # save the responses to a file
    # with open('output/singles/responses.json', 'w') as f:
    #     f.write(json.dumps(responses))
        

    main()