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
    "known_indices": 2,
    "known_ratings": 3,
    "average_rating": 4,
    "num_predict": 5,
    "predict_indices": 6,
    "predict_ratings": 7,
    "prediction_done": 8,
    "num_neighbors": 9,
    "neighbor_ids": 10,
    "neighbor_weights": 11,
}

"""
Define a Custom Class for Recommender System / User-based Collaborative Filtering:
Recommender_type (np.dtype):
    - UserID  (1-based when explicitely stored, ie. 1-200, 201-300)
    - Number of Known Ratings
    - List of Known Rating Indices (Compressed Notation, only store indices of known ratings)
    - List of Known Ratings (1-5) (Compressed Notation, only store ratings of known ratings)
    - List of Predictions. Initialize to Empty Ratings (Empty Ratings == 0)
    - Number of Predictions Needed
    - PredictionDone (Boolean). Initialize to False
    - Number of Neighbors
    - List of Neighbors (UserIDs)   (Compressed Notation, only store indices of neighbors)
    - List of Neighboring Weights (Cosine Similarity, Pearson Correlation) to other users (200 training users only) (Spare Notation, access weights by index)
Accessible as:
    - user_id: "user_id"
    - num_known: "num_known"
    - known_ratings: "known_ratings"
    etc.
"""
recommender_type = np.dtype(
    [
        ("user_id", np.uint16),  # 1-based
        ("num_known", np.uint32),
        ("known_indices", np.uint32, (1000, 1)),  # 1-based
        (
            "known_ratings",
            np.float32,
            (1000, 1),
        ),  # Pearson Ratings are Floats [-1, 1] # Index (0-based) of ratings + 1 = Item-ID as given (1-based)
        ("average_rating", np.float64),  # Average of Known Ratings
        ("num_predict", np.int32),  # -1 for unknown
        ("predict_indices", np.uint32, (1000, 1)),  # 1-based
        (
            "predict_ratings",
            np.float32,
            (1000, 1),
        ),  # Index (0-based) of ratings + 1 = Item-ID as given (1-based)
        ("prediction_done", np.bool_),
        ("num_neighbors", np.uint32),
        ("neighbor_ids", np.uint32, (200, 1)),
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


def train_load(dataset, file, fileName):
    """[summary]

    Args:
        dataset ([type]): [description]
        file ([type]): [description]
        fileName ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Training dataset is 200 users, each with 1000 known/given ratings and 0 predictions
    with open(file, "r") as f:
        fileContents = f.read().strip().splitlines()  # 200 lines of user ratings
        print(f"{fileName} has {len(fileContents)} users")
        lines = [
            line.split("\t") for line in fileContents
        ]  # Split each line into a list of ints with delimiter "\t" tab
        # for line in lines:    # Somehow this didn't work, numbers remain as strings
        for i in range(len(lines)):
            lines[i] = [
                int(word) for word in lines[i] if word.isdigit()
            ]  # Convert to ints
            # for word in line:
            #     if word.isdigit():
            #         word = int(word)
            #     else:
            #         print(f"{word} is not an int")  # DEBUG # Works
        # print(f"{fileName} has {len(lines[0])} ratings for each row")
        # print(f"line 0:\n{lines[0]}")
        # print(type(lines[0][0]))
        print(f"lines size: {len(lines)} x {len(lines[0])}")

        # known_ratings is still Compressed Notation, only store indices of known ratings, but predict_ratings is IGNORED
        # Go through each user, insert known_ratings
        # Compare current dataset_index with previous dataset_index to see if user_id has changed
        prev_index = -1
        # If user_id has changed, restart the internal known counters
        known_counter, predict_counter = 0, 0
        for line_id, line in enumerate(lines):
            for column_index, rating in enumerate(line):
                if line_id != prev_index and line_id != 0:
                    # Insert num_predict
                    print(
                        f"row: {prev_index} col: {column_index}'s num_known is: {known_counter} num_predict is: {predict_counter} for total of {known_counter + predict_counter} values"
                    )
                    dataset[prev_index]["num_known"] = known_counter
                    dataset[prev_index]["num_predict"] = predict_counter
                    # Insert the average of known ratings into the dataset's average_rating field:
                    dataset[prev_index]["average_rating"] = np.mean(
                        dataset[prev_index][0]["known_ratings"][:known_counter]
                    ).astype(np.float64)
                    if int(np.round(dataset[prev_index]["average_rating"])) == 0:
                        print(f"ERROR: Average of 0 means no known_rating")
                    # Reset counters
                    known_counter = 0
                    predict_counter = 0
                prev_index = line_id

                # Insert: known_ratings, num_predict, predict_ratings
                # known ratings and predict_ratings need an internal counter to keep track of the number of known/predict ratings
                # At end insert internal counter of num_predit_ratings into num_predict
                # known_indices, predict_indices = [], []
                if rating != 0 and rating in range(1, 6, 1):
                    # Out of Bound error was because the indices/ratings are 2D arrays of size (1000, 1), so need to access row 0 first with [0]
                    dataset[line_id][0]["known_indices"][known_counter] = (
                        column_index + 1
                    )
                    dataset[line_id][0]["known_ratings"][known_counter] = rating
                    known_counter += 1
                elif rating == 0:
                    # IGNORED
                    # Out of Bound error was because the indices/ratings are 2D arrays of size (1000, 1), so need to access row 0 first with [0]
                    dataset[line_id][0]["predict_indices"][predict_counter] = (
                        column_index + 1
                    )
                    dataset[line_id][0]["predict_ratings"][predict_counter] = rating
                    predict_counter += 1
                else:
                    print(
                        f"Error: file '{file}' line_index: {line_id} column_index: {column_index} rating: {rating} is neither 0 nor within [1-5]. Invalid Rating. Quitting."
                    )
                    quit()

                # Last iteration only:
                if (line_id + 1) == len(lines) and (column_index + 1) == len(line):
                    # Insert num_predict
                    print(
                        f"row: {prev_index} col: {column_index}'s num_known is: {known_counter} num_predict is: {predict_counter} for total of {known_counter + predict_counter} values"
                    )
                    dataset[prev_index]["num_known"] = known_counter
                    dataset[prev_index]["num_predict"] = predict_counter
                    # Insert the average of known ratings into the dataset's average_rating field:
                    dataset[prev_index]["average_rating"] = np.mean(
                        dataset[prev_index][0]["known_ratings"][:known_counter]
                    ).astype(np.float64)
                    if int(np.round(dataset[prev_index]["average_rating"])) == 0:
                        print(f"ERROR: Average of 0 means no known_rating")

        # Check if num_known + num_predict adds up to the number of line entries in file:
        if sum(dataset[:]["num_known"]) + sum(dataset[:]["num_predict"]) != (
            len(lines) * len(lines[0])
        ):
            print(
                f"Error: num_known: {sum(dataset[:]['num_known'])} + num_predict: {sum(dataset[:]['num_predict'])} in file '{file}' does not equal number of values in file {len(lines) * len(lines[0])}. Quitting."
            )
            quit()
        else:
            print(
                f"file '{file}' total num_known: {sum(dataset[:]['num_known'])} + total num_predict: {sum(dataset[:]['num_predict'])} = {sum(dataset[:]['num_known']) + sum(dataset[:]['num_predict'])} == number of values in file: {len(lines) * len(lines[0])}"
            )

    return dataset


def train_init_load(file, fileName):
    """[summary]

    Args:
        file ([type]): [description]
        fileName ([type]): [description]

    Returns:
        [type]: [description]
    """
    dataset = (
        recommender_init()
    )  # Initialize training dataset, simple so no need to further sub-function
    dataset = train_load(dataset, file, fileName)
    return dataset


def test_init(file, fileName, output_suffix):
    """[summary]

    Args:
        file ([type]): [description]
        fileName ([type]): [description]
        output_suffix ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Create output filename
    if fileName[len("test") :].isdigit():
        out_filename = f"result{int(fileName[len('test') :])}"
        if output_suffix:
            out_filename += f"_{output_suffix}"
        out_filename += ".txt"
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
    return dataset, out_filename


def test_load(dataset, file):
    """[summary]

    Args:
        dataset ([type]): [description]
        file ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    with open(file, "r") as f:
        fileContents = f.read().strip().splitlines()
        # print(fileContents)  # DEBUG   # Works
        # print(f"file '{file}' num lines: {len(fileContents)}")  # DEBUG   # Works
        user_indices, item_indices, ratings = [], [], []
        for line in fileContents:
            [a, b, c] = [int(word) for word in line.split() if word.isdigit()]
            user_indices.append(a)
            item_indices.append(b)
            ratings.append(c)
        # print(f"file '{file}' user_indices: {user_indices}")    # DEBUG   # Works
        # num_known = dataset[0]["num_known"]

        # Go through each user, insert known_ratings, count num_predict, and insert predict_ratings
        # Compare current dataset_index with previous dataset_index to see if user_id has changed
        prev_index = -1
        # If user_id has changed, restart the internal known/prediction counters
        known_counter, predict_counter = 0, 0
        for line_id, line in enumerate(fileContents):
            # Find index of user in dataset:
            dataset_index = np.where(dataset[:]["user_id"] == user_indices[line_id])[0][
                0
            ]
            # print(f"line: {line_id} dataset_index: {dataset_index}")    # DEBUG   # Works

            if dataset_index != prev_index and line_id != 0:
                # Insert num_predict and check if num_known is correct
                dataset[prev_index]["num_predict"] = predict_counter
                if known_counter != dataset[prev_index]["num_known"]:
                    print(
                        f"Error: known_counter {known_counter} in file '{file}' line {line_id} does not match num_known in dataset {dataset[prev_index]['num_known']}. Quitting."
                    )
                    quit()
                # Insert the average of known ratings into the dataset's average_rating field:
                dataset[prev_index]["average_rating"] = np.mean(
                    dataset[prev_index][0]["known_ratings"][:known_counter]
                ).astype(np.float64)
                if int(np.round(dataset[prev_index]["average_rating"])) == 0:
                    print(f"ERROR: Average of 0 means no known_rating")
                # Reset counters
                known_counter = 0
                predict_counter = 0
            prev_index = dataset_index

            # Insert: known_ratings, num_predict, predict_ratings
            # known ratings and predict_ratings need an internal counter to keep track of the number of known/predict ratings
            # At end insert internal counter of num_predit_ratings into num_predict
            # known_indices, predict_indices = [], []
            if ratings[line_id] != 0 and ratings[line_id] in range(1, 6, 1):
                # print(
                #     f"file '{file}' line: {line_id} dataset_index: {dataset_index} known_counter: {known_counter}"
                # )  # DEBUG  # Works
                # print(
                #     f"Checking known_indices array:\n{dataset[dataset_index][0]['known_indices'][known_counter]}"
                # )  # DEBUG  # Works
                # Out of Bound error was because the indices/ratings are 2D arrays of size (1000, 1), so need to access row 0 first with [0]
                dataset[dataset_index][0]["known_indices"][
                    known_counter
                ] = item_indices[line_id]
                dataset[dataset_index][0]["known_ratings"][known_counter] = ratings[
                    line_id
                ]
                known_counter += 1
            elif ratings[line_id] == 0:
                # print(
                #     f"file '{file}' line: {line_id} dataset_index: {dataset_index} predict_counter: {predict_counter}"
                # )  # DEBUG  # Works
                # print(
                #     f"Checking predict_indices array:\n{dataset[dataset_index][0]['predict_indices'][predict_counter]}"
                # )  # DEBUG  # Works
                # Out of Bound error was because the indices/ratings are 2D arrays of size (1000, 1), so need to access row 0 first with [0]
                dataset[dataset_index][0]["predict_indices"][
                    predict_counter
                ] = item_indices[line_id]
                dataset[dataset_index][0]["predict_ratings"][predict_counter] = ratings[
                    line_id
                ]
                predict_counter += 1
            else:
                print(
                    f"Error: file '{file}' line_index: {line_id} rating: {ratings[line_id]} is neither 0 nor within [1-5]. Invalid Rating. Quitting."
                )
                quit()

            # Last iteration only:
            if (line_id + 1) == len(fileContents):
                # Insert num_predict and check if num_known is correct
                dataset[prev_index]["num_predict"] = predict_counter
                if known_counter != dataset[prev_index]["num_known"]:
                    print(
                        f"Error: known_counter {known_counter} in file '{file}' line {line_id} does not match num_known in dataset {dataset[prev_index]['num_known']}. Quitting."
                    )
                    quit()
                # Insert the average of known ratings into the dataset's average_rating field:
                dataset[prev_index]["average_rating"] = np.mean(
                    dataset[prev_index][0]["known_ratings"][:known_counter]
                ).astype(np.float64)
                if int(np.round(dataset[prev_index]["average_rating"])) == 0:
                    print(f"ERROR: Average of 0 means no known_rating")

        # Check if num_known + num_predict adds up to the number of line entries in file:
        if sum(dataset[:]["num_known"]) + sum(dataset[:]["num_predict"]) != len(
            fileContents
        ):
            print(
                f"Error: num_known + num_predict in file '{file}' does not equal number of lines in file. Quitting."
            )
            quit()
        else:
            print(
                f"file '{file}' total num_known: {sum(dataset[:]['num_known'])} + total num_predict: {sum(dataset[:]['num_predict'])} = {sum(dataset[:]['num_known']) + sum(dataset[:]['num_predict'])} == number of lines in file: {len(fileContents)}"
            )
    return dataset


def test_init_load(file, fileName, output_suffix):
    """[summary]

    Args:
        file ([type]): [description]
        fileName ([type]): [description]
        output_suffix ([type]): [description]

    Returns:
        [type]: [description]
    """
    dataset, out_filename = test_init(file, fileName, output_suffix)
    dataset = test_load(dataset, file)
    return dataset, out_filename


def recommender_import(
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

        # Initializations and loading of datasets:
        if fileName.startswith("train"):
            dataset = train_init_load(file, fileName)
        else:  # elif fileName.startswith("test"):
            dataset, out_filename = test_init_load(file, fileName, output_suffix)
            output_filenames.append(out_filename)

        datasets.append(dataset)

    return datasets, output_filenames


# M_j (Count of number of users that have rated item j) for all 1000 items


def main():
    """
    Driver function for Recommender System / User-based Collaborative Filtering Project.
    """
    (datasets, out_filenames,) = recommender_import(output_suffix="v2")
    [test10, test20, test5, train,] = datasets
    ppp.pprint(out_filenames)
    ppp.pprint(test10[:]["average_rating"])
    # ppp.pprint(test10[:]["user_id"])
    # print(f"total num_known for test10:\t{sum(test10[:]['num_known'])}")
    # ppp.pprint(test20[:]["user_id"])
    # print(f"total num_known for test20:\t{sum(test20[:]['num_known'])}")
    # ppp.pprint(test5[:]["user_id"])
    # print(f"total num_known for test5:\t{sum(test5[:]['num_known'])}")
    # ppp.pprint(train[:]["user_id"])
    print(f"total num_known for train:\t{sum(train[:]['num_known'])}")
    print(
        f"traing Dataset has a fill factor of {float(sum(train[:]['num_known'])) / float(len(train) * 1000)}"
    )


if __name__ == "__main__":
    """
    If this file is run as a script, execute main()
    """
    main()

