"""This is the main file that demonstrates how to use the DataLoader class."""

from classes.data_loader import DataLoader
from utils.find_similar_grid import find_similar_solutions

def main():
    """This is the main function that demonstrates how to use the DataLoader class."""
    dl = DataLoader()
    print(dl.show_available_datasets())

    evaluation = dl.load_dataset("evaluation")

    unique_keys = set(evaluation.keys())
    print(f"len(evaluation): {len(unique_keys)}")

    test = dl.load_dataset("test")
    unique_keys = set(test.keys())
    print(f"len(test): {len(unique_keys)}")

    training = dl.load_dataset("training")
    unique_keys = set(training.keys())
    print(f"len(training): {len(unique_keys)}")

    random_samples = dl.randomly_sample_datapoints(5)
    for challenge_id, challenge_data in random_samples.items():
        dl.plot_train_and_test_examples({challenge_id: challenge_data})

    challenge_id = "007bbfb7"
    challenge = dl.get_specific_sample(challenge_id)

    dl.plot_solution(challenge["test_input"], f"{challenge_id}_test")
    dl.plot_solution(
        challenge["train_examples"][3]["input"],
        f"{challenge_id}_train_{3}_input",
    )
    dl.plot_solution(
        challenge["train_examples"][3]["output"],
        f"{challenge_id}_train_{3}_output",
    )

    dl.plot_train_and_test_examples({challenge_id: challenge})
    grids, original_solution_index = find_similar_solutions(
        challenge["solution"], dl
    )
    dl.plot_multiple_solutions(grids, f"{challenge_id}_similar_solutions")
    dl.plot_solution(
        challenge["solution"][0],
        f"./{challenge_id}_similar_solutions/original_solution",
    )
    print(challenge["solution"][0])
    print(f"original_solution_index: {original_solution_index}")


if __name__ == "__main__":
    main()
