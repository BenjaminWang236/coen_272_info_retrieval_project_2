import numpy as np
import pprintpp as ppp
import glob, os
import numpy as np
import math

# Global variables
emptyRating = 0
minRating = 1
maxRating = 5
unknownNumPred = -1
rec_indices = {
    "user_id": 0,
    "num_known": 1,
    "known_ratings": 2,
    "average_rating": 3,
    "num_predict": 4,
    "predict_ratings": 5,
    "prediction_done": 6,
    "num_neighbors": 7,
    "neighbor_ids": 8,
    "neighbor_weights": 9,
}

"""
Define a Custom Class for Recommender System / User-based Collaborative Filtering:
Recommender_type (np.dtype):
    - UserID  (1-based when explicitely stored, ie. 1-200, 201-300)
    - List of Known Ratings (1-5)
    - Number of Known Ratings
    - List of Predictions. Initialize to Empty Ratings (Empty Ratings == 0)
    - Number of Predictions Needed
    - PredictionDone (Boolean). Initialize to False
    - Number of Neighbors
    - List of Neighbors (UserIDs)
    - List of Neighboring Weights (Cosine Similarity, Pearson Correlation) to other users (200 training users only)
Accessible as:
    - user_id: "user_id"
    - num_known: "num_known"
    - known_ratings: "known_ratings"
    etc.
"""
recommender_type = np.dtype(
    [
        ("user_id", np.uint16),  # 1-based
        ("num_known", np.uint16),
        (
            "known_ratings",
            np.float32,
            (1000, 1),
        ),  # Pearson Ratings are Floats [-1, 1] # Index (0-based) of ratings + 1 = Item-ID as given (1-based)
        ("average_rating", np.uint8),  # Average of Known Ratings
        ("num_predict", np.int16),  # -1 for unknown
        (
            "predict_ratings",
            np.float32,
            (1000, 1),
        ),  # Index (0-based) of ratings + 1 = Item-ID as given (1-based)
        ("prediction_done", np.bool_),
        ("num_neighbors", np.uint16),
        ("neighbor_ids", np.uint16, (200, 1)),
        ("neighbor_weights", np.float32, (200, 1)),
    ]
)


def recommender_init(
    dataset_size: tuple = (200, 1),
    user_id_offset: int = 1,
    num_known: int = 1000,
    num_predict: int = 0,
    prediction_done: np.bool_ = np.True_,
):
    """
    Initialize a Recommender System / User-based Collaborative Filtering dataset.
    Initialize to 0/Empty Ratings for all fields, then set fields to desired values
    as specified by the parameter arguments.
    By default, initialize to a dataset of 200 users, each with 1000 known ratings, and 0 predictions. (Training Dataset)

    Args:
        dataset_size (tuple, optional): Shape of dataset. Defaults to (200, 1).
        user_id_offset (int, optional): Given dataset is 1-based indexing. Defaults to 1.
        num_known (int, optional): Number of Known Ratings. Defaults to 1000.
        num_predict (int, optional): Number of Unknown Ratings to Predict. -1 if this is not known yet. Defaults to 0.
        prediction_done (np.bool_, optional): True if done with Predictions, else False. Defaults to np.True_.

    Returns:
        dataset (ndarray of dtype=recommender_type): Initialized recommender dataset with Zeros 
        and parameters inserted.
    """
    dataset = np.zeros(dataset_size, dtype=recommender_type)
    for user_id, user in enumerate(dataset):
        user["user_id"] = user_id + user_id_offset
        user["num_known"] = num_known
        user["num_predict"] = num_predict
        user["prediction_done"] = prediction_done

    return dataset


def recommender_load(
    output_suffix: str = "",
    filetype: str = ".txt",
    exclusion: str = "result",
    filepath: os.path = "default",
):
    """
    Initialize datasets and Load the datasets of dtype=recommender_type from corresponding files.
    Files are expected to be in the same directory as this file.
    Subdirectories are ignored. Ignores Example result files. All test datasets has 100 users as given.

    Args:
        output_suffix (str, optional): Suffix to append to output file name as '_[output_suffix]'. Defaults to "".
        filetype (str, optional): Specify type of file to import data from. Defaults to ".txt".
        exclusion (str, optional): Files starting with this string will be ignored. Defaults to "result".
        filepath (os.path, optional): [description]. Defaults to "default".

    Returns:
        (datasets, output_filenames): (list(), list())
            datasets (list()): List of initialized and loaded datasets of dtype=recommender_type.
            output_filenames (list()): List of output filenames starting with 'result'. Only generated for datasets/files starting with "test".

    """
    # Change current directory to filepath (default to script file directory) so glob can find files
    if filepath != "default":
        os.chdir(
            os.path.dirname(filepath)
        )  # Change directory to the directory specified
    else:
        os.chdir(
            os.path.dirname(os.path.abspath(__file__))
        )  # Change directory to the directory of this file

    datasets = []
    output_filenames = []
    for file in glob.glob(f"*{filetype}", recursive=False):
        if file.startswith(exclusion) or file.startswith(
            "example"
        ):  # Ignore files starting with exclusion or are example result files
            continue
        if not file.startswith("test") and not file.startswith("train"):
            print(
                f"Error: File '{file}' does not start with 'train' or 'test'. No output. Quitting."
            )
            quit()
        fileName = file[: -len(filetype)]  # remove .txt from file name

        # Initializations:
        if fileName.startswith("train"):
            dataset = recommender_init()  # Initialize training dataset
        else:  # elif fileName.startswith("test"):
            # Create output filename
            if fileName[len("test") :].isdigit():
                out_filename = f"result{int(fileName[len('test') :])}"
                if output_suffix:
                    out_filename += f"_{output_suffix}"
                out_filename += ".txt"
                output_filenames.append(out_filename)
            else:
                print(
                    f"Error: Test file '{file}' does not start with 'test' followed by a number. No output. Quitting."
                )
                quit()

            # Extract the User_ID offset from the first line of file:
            with open(file, "r") as f:
                first_line = f.readline().strip()
                offset = [int(word) for word in first_line.split() if word.isdigit()][0]
                # print(f"first line of file '{file}':\t{numbers}")  # DEBUG   # Works
                # print(f"file '{file}' User_ID offset:\t{offset}")  # DEBUG   # Works

            # Initialize test dataset
            dataset = recommender_init(
                dataset_size=(100, 1),  # 100 users for all 3 test datasets
                user_id_offset=offset,
                num_known=int(fileName[len("test") :]),
                num_predict=unknownNumPred,
                prediction_done=np.False_,
            )

        # Load dataset from file:
        with open(file, "r") as f:
            fileContents = f.read().strip().splitlines()
            # print(fileContents)  # DEBUG   # Works
            user_indices, item_indices, ratings = [], [], []

        datasets.append(dataset)

    return datasets, output_filenames


# M_j (Count of number of users that have rated item j) for all 1000 items


def main():
    """
    Driver function for Recommender System / User-based Collaborative Filtering Project.
    """
    (datasets, out_filenames,) = recommender_load(output_suffix="v2")
    [test10, test20, test5, train,] = datasets
    ppp.pprint(out_filenames)
    ppp.pprint(test10[:]["user_id"])
    ppp.pprint(test20[:]["user_id"])
    ppp.pprint(test5[:]["user_id"])
    ppp.pprint(train[:]["user_id"])


if __name__ == "__main__":
    """
    If this file is run as a script, execute main()
    """
    main()

