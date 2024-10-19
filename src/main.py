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

    def get_augmented_training_examples(id: str):
        sample = dl.get_specific_sample(id)
        test_input = sample['test_input']
        train_examples = sample['train_examples']
        solution = sample['solution']

        augmented_train_examples_input = []
        augmented_train_examples_output = []
        train_example_count = 0
        for train_example in train_examples:
            
            train_example_input = train_example['input']
            train_example_output = train_example['output']
            
            # Ensure inputs are in the correct format for generate_augmentations
            if not isinstance(train_example_input[0], list):
                train_example_input = [train_example_input]
            if not isinstance(train_example_output[0], list):
                train_example_output = [train_example_output]
            
            aug_input_dict = generate_augmentations([train_example_input], challenge_id, "input", train_example_count)
            aug_output_dict = generate_augmentations([train_example_output], challenge_id, "output", train_example_count)

            # work out which one has the fewest unique augmentations
            unique_augmentations_input = len(aug_input_dict)
            unique_augmentations_output = len(aug_output_dict)
            
            final_pairs = []
            if unique_augmentations_input < unique_augmentations_output:
                # we want to keep only output augmentations that have a corresponding input augmentation matched on description hash and colour map hash
                for output_augmentation in aug_output_dict:
                    for input_augmentation in aug_input_dict:
                        aug_output = aug_output_dict[output_augmentation]
                        aug_input = aug_input_dict[input_augmentation]

                        if aug_output['description_hash'] ==  aug_input['description_hash'] and aug_output['colour_map_hash'] == aug_input['colour_map_hash']:
                            final_pairs.append((output_augmentation, input_augmentation))
            else: 
                for output_augmentation in aug_output_dict:
                    for input_augmentation in aug_input_dict:
                        if aug_output['description_hash'] ==  aug_input['description_hash'] and aug_output[output_augmentation]['colour_map_hash'] == aug_input[input_augmentation]['colour_map_hash']:
                            final_pairs.append((output_augmentation, input_augmentation))

            train_example_count += 1

            print("")

        print(augmented_train_examples_input)
        # # Ensure test_input and solution are in the correct format
        # if not isinstance(test_input[0], list):
        #     test_input = [test_input]
        # if not isinstance(solution[0], list):
        #     solution = [solution]

        # augmented_test_input = generate_augmentations([test_input])
        # augmented_solution = generate_augmentations(solution)

        # return augmented_train_examples_input, augmented_train_examples_output, augmented_test_input, augmented_solution



    get_augmented_training_examples(challenge_id)
        

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