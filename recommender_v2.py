import numpy as np
import pprintpp as ppp
import glob, os
import numpy as np
import math
import copy

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
    "centered": 5,
    "num_predict": 6,
    "predict_indices": 7,
    "predict_ratings": 8,
    "prediction_done": 9,
    "num_neighbor": 10,
    "neighbor_ids": 11,
    "neighbor_weights": 12,
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
        ("user_id", np.uint32),  # 1-based
        ("num_known", np.uint32),
        ("known_indices", np.uint32, (1000,)),  # 1-based
        (
            "known_ratings",
            np.float32,
            (1000,),
        ),  # Pearson Ratings are Floats [-1, 1] # Index (0-based) of ratings + 1 = Item-ID as given (1-based)
        ("average_rating", np.float64),  # Average of Known Ratings
        ("centered", np.bool_),  # Centered or not, Boolean
        ("num_predict", np.int32),  # -1 for unknown
        ("predict_indices", np.uint32, (1000,)),  # 1-based
        (
            "predict_ratings",
            np.float32,
            (1000,),
        ),  # Index (0-based) of ratings + 1 = Item-ID as given (1-based)
        ("prediction_done", np.bool_),
        ("num_neighbor", np.uint32),
        ("neighbor_ids", np.uint32, (200,)),  # 1-based
        ("neighbor_weights", np.float32, (200,)),
    ]
)


def recommender_init(
    dataset_size: tuple = (200,),
    user_id_offset: int = 1,
    num_known: int = 1000,
    num_predict: int = 0,
    prediction_done: np.bool_ = np.True_,
    centered: np.bool_ = np.False_,
):
    """
    Initialize a Recommender System / User-based Collaborative Filtering dataset.
    Initialize to 0/Empty Ratings for all fields, then set fields to desired values
    as specified by the parameter arguments.
    By default, initialize to a dataset of 200 users, each with 1000 known ratings, and 0 predictions. (Training Dataset)

    Args:
        dataset_size (tuple, optional): Shape of dataset. Defaults to (200,). For vectors DO NOT use (#, 1), use (#,) instead.
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
        user["centered"] = centered
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
        # print(f"{fileName} has {len(fileContents)} users")
        lines = [
            line.split("\t") for line in fileContents
        ]  # Split each line into a list of ints with delimiter "\t" tab
        # for line in lines:    # Somehow this didn't work, numbers remain as strings
        for i in range(len(lines)):
            lines[i] = [
                int(word) for word in lines[i] if word.isdigit()
            ]  # Convert to ints
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
                    dataset[prev_index]["num_known"] = known_counter
                    dataset[prev_index]["num_predict"] = predict_counter
                    # Insert the average of known ratings into the dataset's average_rating field:
                    dataset[prev_index]["average_rating"] = np.mean(
                        dataset[prev_index]["known_ratings"][:known_counter]
                    ).astype(np.float64)
                    if int(np.round(dataset[prev_index]["average_rating"])) == 0:
                        print(f"ERROR: Average of 0 means no known_rating. Quitting")
                        quit()
                    # Reset counters
                    known_counter = 0
                    predict_counter = 0
                prev_index = line_id

                # Insert: known_ratings, num_predict, predict_ratings
                # known ratings and predict_ratings need an internal counter to keep track of the number of known/predict ratings
                # At end insert internal counter of num_predit_ratings into num_predict
                # known_indices, predict_indices = [], []
                if rating != 0 and rating in range(1, 6, 1):
                    dataset[line_id]["known_indices"][known_counter] = column_index + 1
                    dataset[line_id]["known_ratings"][known_counter] = rating
                    known_counter += 1
                elif rating == 0:
                    dataset[line_id]["predict_indices"][predict_counter] = (
                        column_index + 1
                    )
                    dataset[line_id]["predict_ratings"][predict_counter] = rating
                    predict_counter += 1
                else:
                    print(
                        f"Error: file '{file}' line_index: {line_id} column_index: {column_index} rating: {rating} is neither 0 nor within [1-5]. Invalid Rating. Quitting."
                    )
                    quit()

                # Last iteration only:
                if (line_id + 1) == len(lines) and (column_index + 1) == len(line):
                    dataset[line_id]["num_known"] = known_counter
                    dataset[line_id]["num_predict"] = predict_counter
                    # Insert the average of known ratings into the dataset's average_rating field:
                    dataset[line_id]["average_rating"] = np.mean(
                        dataset[line_id]["known_ratings"][:known_counter]
                    ).astype(np.float64)
                    if int(np.round(dataset[line_id]["average_rating"])) == 0:
                        print(f"ERROR: Average of 0 means no known_rating. Quitting")
                        quit()

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

    # Initialize test dataset
    dataset = recommender_init(
        dataset_size=(100,),  # 100 users for all 3 test datasets
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
        user_indices, item_indices, ratings = [], [], []
        for line in fileContents:
            [a, b, c] = [int(word) for word in line.split() if word.isdigit()]
            user_indices.append(a)
            item_indices.append(b)
            ratings.append(c)

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
                    dataset[prev_index]["known_ratings"][:known_counter]
                ).astype(np.float64)
                if int(np.round(dataset[prev_index]["average_rating"])) == 0:
                    print(f"ERROR: Average of 0 means no known_rating. Quitting")
                    quit()
                # Reset counters
                known_counter = 0
                predict_counter = 0
            prev_index = dataset_index

            # Insert: known_ratings, num_predict, predict_ratings
            # known ratings and predict_ratings need an internal counter to keep track of the number of known/predict ratings
            # At end insert internal counter of num_predit_ratings into num_predict
            # known_indices, predict_indices = [], []
            if ratings[line_id] != 0 and ratings[line_id] in range(1, 6, 1):
                dataset[dataset_index]["known_indices"][known_counter] = item_indices[
                    line_id
                ]
                dataset[dataset_index]["known_ratings"][known_counter] = ratings[
                    line_id
                ]
                known_counter += 1
            elif ratings[line_id] == 0:
                dataset[dataset_index]["predict_indices"][
                    predict_counter
                ] = item_indices[line_id]
                dataset[dataset_index]["predict_ratings"][predict_counter] = ratings[
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
                    dataset[prev_index]["known_ratings"][:known_counter]
                ).astype(np.float64)
                if int(np.round(dataset[prev_index]["average_rating"])) == 0:
                    print(f"ERROR: Average of 0 means no known_rating. Quitting.")
                    quit()

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


def calcNeighborWeights(
    testSet: np.ndarray,
    trainSet: np.ndarray,
    transposed_testSet: np.ndarray,
    transposed_trainSet: np.ndarray,
    pearson: bool = False,
    IUF: bool = False,
    adjustedCosSim: bool = False,
):
    """[summary]

    Args:
        testSet (np.ndarray): [description]
        trainSet (np.ndarray): [description]
    """
    # If adjustedCosSim, then no Pearson & no IUF, and it means this is Item-Based!!!
    if adjustedCosSim:
        for item in testSet:
            for known_id, known_user in enumerate(
                item["known_ratings"][: item["num_known"]]
            ):
                # print(
                #     f"test: {item['known_indices'][known_id] - transposed_testSet[0]['user_id']}"
                # )
                known_user -= transposed_testSet[
                    item["known_indices"][known_id]
                    - transposed_testSet[0]["user_id"]
                    - 1
                ]["average_rating"]
        for item in trainSet:
            for known_id, known_user in enumerate(
                item["known_ratings"][: item["num_known"]]
            ):
                # print(f"train: {item['known_indices'][known_id]- transposed_trainSet[0]['user_id']}")
                known_user -= transposed_trainSet[
                    item["known_indices"][known_id]
                    - transposed_trainSet[0]["user_id"]
                    - 1
                ]["average_rating"]

    # Before Pearson, do IUF is specified to be true
    # Re-calculate the test user's average rating after applying IUF
    if IUF:
        # Using this case's total known users in training set of 200 possible at most for each item-rating pair
        max_users = 200
        for test_user in testSet:
            for known_id_id, known_index in enumerate(
                test_user["known_indices"][: test_user["num_known"]]
            ):
                if transposed_trainSet[known_index - 1]["num_known"] == 0:
                    IUF_value = 1
                else:
                    # known_index is 1-based by the numpy array of transposed train dataset is 0-based, so subtract 1
                    IUF_value = math.log(
                        max_users / (transposed_trainSet[known_index - 1]["num_known"]),
                        max_users,
                    )
                # Multiply the test user's original known rating by the IUF value
                test_user["known_ratings"][known_id_id] *= IUF_value
            # Re-Calculate this user's average rating:
            test_user["average_rating"] = np.mean(
                test_user["known_ratings"][: test_user["num_known"]]
            ).astype(np.float64)
        for train_user in trainSet:
            for known_id_id, known_index in enumerate(
                train_user["known_indices"][: train_user["num_known"]]
            ):
                if transposed_trainSet[known_index - 1]["num_known"] == 0:
                    IUF_value = 1
                else:
                    # known_index is 1-based by the numpy array of transposed train dataset is 0-based, so subtract 1
                    IUF_value = math.log(
                        max_users / (transposed_trainSet[known_index - 1]["num_known"]),
                        max_users,
                    )
                # Multiply the test user's original known rating by the IUF value
                train_user["known_ratings"][known_id_id] *= IUF_value
            # Re-Calculate this user's average rating:
            train_user["average_rating"] = np.mean(
                train_user["known_ratings"][: train_user["num_known"]]
            ).astype(np.float64)
    if pearson:
        test_centered_knowns, train_centered_knowns = [], []
        for i in range(len(testSet)):
            if not testSet[i]["centered"]:
                # print(f"Average Rating: {testSet[i]['average_rating']}")
                testSet[i]["known_ratings"][: testSet[i]["num_known"]] -= testSet[i][
                    "average_rating"
                ]
                testSet[i]["centered"] = np.True_
                test_centered_knowns.extend(
                    testSet[i]["known_ratings"][: testSet[i]["num_known"]]
                )
        for j in range(len(trainSet)):
            if not trainSet[j]["centered"]:
                trainSet[j]["known_ratings"][: trainSet[j]["num_known"]] -= trainSet[j][
                    "average_rating"
                ]
                trainSet[j]["centered"] = np.True_
                train_centered_knowns.extend(
                    trainSet[j]["known_ratings"][: trainSet[j]["num_known"]]
                )
        # CHECK HERE:
        if test_centered_knowns:
            print(
                f"Test's Centered Known Ratings: min: {np.min(test_centered_knowns)}, max: {np.max(test_centered_knowns)}"
            )
        if train_centered_knowns:
            print(
                f"Train's Centered Known Ratings: min: {np.min(train_centered_knowns)}, max: {np.max(train_centered_knowns)}"
            )

    # Go through each test user in testSet and calculate the cosine similarity of each test user with each neighboring train user
    for i in range(len(testSet)):
        # for i in range(1):  # DEBUG
        for j in range(len(trainSet)):
            # Neighbor is defined as a train user who shares at least 1 known item-rating with the test user's known item-ratings
            # AND the train user must have the item-rating that the test user does not have / need to predict
            # For pearsons correlation, subtract each rating from the average rating of the user to get the user's rating deviation
            sharedItemRatings = np.intersect1d(
                testSet[i]["known_indices"][: testSet[i]["num_known"]],
                trainSet[j]["known_indices"][: trainSet[j]["num_known"]],
            )
            # print(
            #     f"Test user_id {testSet[i]['user_id']} & Train user_id {trainSet[j]['user_id']} shares {len(sharedItemRatings)} ratings at indices:",
            #     end="\t",
            # )
            # ppp.pprint(sharedItemRatings)
            # Check if the training user has a non-empty rating for the item that the test user needs to predict
            TrainKnowsTestPredictItemRatings = np.intersect1d(
                testSet[i]["predict_indices"][: testSet[i]["num_predict"]],
                trainSet[j]["known_indices"][: trainSet[j]["num_known"]],
            )
            if (
                len(TrainKnowsTestPredictItemRatings) == 0
                or len(sharedItemRatings) == 0
            ):
                # if len(TrainKnowsTestPredictItemRatings) == 0:
                #     print(f"{len(TrainKnowsTestPredictItemRatings)}")
                # else:
                #     print(f"{len(sharedItemRatings)}")
                continue
            else:
                # Calculate cosine similarity of test user with train user, add train user_id to
                # test user's neighbor_ids, increment test user's num_neighbor, and add cosine
                # similarity to test user's neighbor_weights

                # Calculate the indices of the shared known item-ratings' indices for test and train users:
                test_shared_indices_indices = [
                    np.where(
                        testSet[i]["known_indices"][: testSet[i]["num_known"]]
                        == itemRating
                    )[0][0]
                    for itemRating in sharedItemRatings
                ]
                train_shared_indices_indices = [
                    np.where(
                        trainSet[j]["known_indices"][: trainSet[j]["num_known"]]
                        == itemRating
                    )[0][0]
                    for itemRating in sharedItemRatings
                ]
                # print(
                #     f"test {testSet[i]['user_id']} shared indices indices: {test_shared_indices_indices}"
                # )
                if len(test_shared_indices_indices) != len(sharedItemRatings):
                    print(
                        f"Error: test_shared_indices_indices: {test_shared_indices_indices} ({len(test_shared_indices_indices)}) doesn't match sharedItemRatings: {sharedItemRatings} ({len(sharedItemRatings)})"
                    )
                    quit()
                if len(train_shared_indices_indices) != len(sharedItemRatings):
                    print(
                        f"Error: train_shared_indices_indices: {train_shared_indices_indices} ({len(train_shared_indices_indices)}) doesn't match sharedItemRatings: {sharedItemRatings} ({len(sharedItemRatings)})"
                    )
                sumOfProduct, sumOfSquaresU1, sumOfSquaresU2 = 0, 0, 0
                for k in range(len(sharedItemRatings)):
                    # print(
                    #     f"test {testSet[i]['user_id']} train {trainSet[j]['user_id']} @ {k}-th itemRating: test rating: {testSet[i]['known_ratings'][test_shared_indices_indices[k]]} train value: {trainSet[j]['known_ratings'][train_shared_indices_indices[k]]}"
                    # )   # DEBUG
                    sumOfProduct += (
                        testSet[i]["known_ratings"][test_shared_indices_indices[k]]
                        * trainSet[j]["known_ratings"][train_shared_indices_indices[k]]
                    )
                    sumOfSquaresU1 += (
                        testSet[i]["known_ratings"][test_shared_indices_indices[k]] ** 2
                    )
                    sumOfSquaresU2 += (
                        trainSet[j]["known_ratings"][train_shared_indices_indices[k]]
                        ** 2
                    )
                    # print(
                    #     f"sumOfProduct: {sumOfProduct}, sumOfSquaresU1: {sumOfSquaresU1}, sumOfSquaresU2: {sumOfSquaresU2}"
                    # )   # DEBUG
                if sumOfProduct == 0:
                    cosineSimilarity = 0
                elif sumOfSquaresU1 == 0 or sumOfSquaresU2 == 0:
                    cosineSimilarity = 0
                else:
                    cosineSimilarity = sumOfProduct / (
                        math.sqrt(sumOfSquaresU1) * math.sqrt(sumOfSquaresU2)
                    )
                # if cosineSimilarity == 0:
                #     # print(
                #     #     f"test {testSet[i]['user_id']} train {trainSet[j]['user_id']} has cosSim Weight of {cosineSimilarity} (Pearson = {pearson})"
                #     # )
                #     ...
                # else:
                #     print(
                #         f"test {testSet[i]['user_id']} train {trainSet[j]['user_id']} has cosSim Weight of {cosineSimilarity} (Pearson = {pearson})"
                #     )

                testSet[i]["neighbor_weights"][
                    testSet[i]["num_neighbor"]
                ] = cosineSimilarity
                testSet[i]["neighbor_ids"][testSet[i]["num_neighbor"]] = trainSet[j][
                    "user_id"
                ]
                trainSet[j]["num_neighbor"] += 1
                testSet[i]["num_neighbor"] += 1
    all_neighboring_weights = []
    [
        all_neighboring_weights.extend(user["neighbor_weights"][: user["num_neighbor"]])
        for user in testSet
    ]
    if all_neighboring_weights:
        print(
            f"Overall Weights statistics: avg {np.mean(all_neighboring_weights)} min {np.min(all_neighboring_weights)} max {np.max(all_neighboring_weights)}"
        )
    else:
        print("all_neighboring_weights is empty.")


def calcPredictions(
    testSet: np.ndarray,
    trainSet: np.ndarray,
    testSet_2: np.ndarray,
    trainSet_2: np.ndarray,
    pearson: bool = False,
    IUF: bool = False,
    caseMod: bool = False,
):
    """[summary]

    Args:
        testSet (np.ndarray): [description]
        trainSet (np.ndarray): [description]
    """
    numNoNeighborsTestSet = []
    # For each user in the testing set
    for i in range(len(testSet)):
        numNoNeighbors = []
        # For each empty item-rating that the test user needs to predict
        for j in range(testSet[i]["num_predict"]):
            # Item that needs predicting's index:
            item_index = testSet[i]["predict_indices"][j]
            sumRelevantWeights, sumWeightRatingProduct = 0, 0
            numRelevantNeighbors = 0
            # For each neighbor who has a known item-rating that the test user needs to predict: # "For similar k users"
            for k in range(testSet[i]["num_neighbor"]):
                neighbor_id = testSet[i]["neighbor_ids"][k]  # 1-based id of neighbor
                # Check this particular neighbor has the desired item-rating, since previous superset is for all desired item-ratings of this test user:
                check_neighbor_has_item_rating = np.where(
                    trainSet[neighbor_id - 1]["known_indices"][
                        : trainSet[neighbor_id - 1]["num_known"]
                    ]
                    == item_index
                )[0]
                if len(check_neighbor_has_item_rating) == 0:
                    # This mean this neighbor have 1 or more known item-ratings the test user needs to predict but not this particular item-rating
                    # print(
                    #     f"Train user {trainSet[neighbor_id-1]['user_id']} is a neighbor but doesn't have a rating for item {item_index}"
                    # )
                    continue
                else:
                    # Reaching here means the train_user is a neighbor, and has the desired item-rating that test user needs to predict
                    numRelevantNeighbors += 1
                    # print(
                    #     trainSet[neighbor_id - 1]["known_ratings"][
                    #         check_neighbor_has_item_rating[0]
                    #     ],
                    #     end=" ",
                    # )
                    caseMod_factor = 1
                    if caseMod:
                        caseMod_factor = abs(testSet[i]["neighbor_weights"][k]) ** (
                            2.5 - 1
                        )  # p of 2.5
                    if IUF:
                        sumWeightRatingProduct += (
                            caseMod_factor
                            * testSet[i]["neighbor_weights"][k]
                            * trainSet_2[neighbor_id - 1]["known_ratings"][
                                check_neighbor_has_item_rating[0]
                            ]  # Original TrainSet no multiplied by IUF
                        )
                    else:
                        sumWeightRatingProduct += (
                            caseMod_factor
                            * testSet[i]["neighbor_weights"][k]
                            * trainSet[neighbor_id - 1]["known_ratings"][
                                check_neighbor_has_item_rating[0]
                            ]
                        )
                    if pearson:
                        sumRelevantWeights += abs(
                            caseMod_factor * testSet[i]["neighbor_weights"][k]
                        )
                    else:
                        sumRelevantWeights += (
                            caseMod_factor * testSet[i]["neighbor_weights"][k]
                        )
                    # sumRelevantWeights += (
                    #     abs(testSet[i]["neighbor_weights"][k])
                    #     if pearson
                    #     else testSet[i]["neighbor_weights"][k]
                    # )
            if numRelevantNeighbors == 0:
                # print(
                #     f"WARNING: Test user {testSet[i]['user_id']} has NO NEIGHBOR with rquired item-rating for item index {item_index}!!!"
                # )
                numNoNeighbors.append(item_index)
                rating_prediction = (
                    testSet[i]["average_rating"]
                    if not IUF
                    else testSet_2[i]["average_rating"]
                )

            else:
                if pearson:
                    if IUF:
                        rating_prediction = testSet_2[i][
                            "average_rating"
                        ]  # Original Rating
                    else:
                        rating_prediction = testSet[i]["average_rating"]
                    rating_prediction += (
                        (sumWeightRatingProduct / sumRelevantWeights)
                        if sumRelevantWeights != 0
                        and not math.isnan(sumRelevantWeights)
                        else 0
                    )

                else:
                    rating_prediction = (
                        (sumWeightRatingProduct / sumRelevantWeights)
                        if sumRelevantWeights != 0
                        and not math.isnan(sumRelevantWeights)
                        else 0
                    )
            # rating_prediction = (testSet[i]["average_rating"] if pearson else 0) + (
            #     sumWeightRatingProduct / sumRelevantWeights
            #     if sumRelevantWeights != 0
            #     else 0
            # )
            rating_prediction = np.around(rating_prediction, decimals=0)
            rating_prediction = rating_prediction.astype(int)
            if not rating_prediction in range(1, 6):
                # print(
                #     f"WARNING: Rating prediction error of {rating_prediction} at test user {testSet[i]['user_id']} item {item_index}: avg: {testSet[i]['average_rating']} nominator: {sumWeightRatingProduct}, denominator: {sumRelevantWeights}"
                # )
                if rating_prediction == 0:
                    rating_prediction = 3  # Default Rating for this above scenario
                if rating_prediction > 5:
                    rating_prediction = 5
                if rating_prediction < 1:
                    rating_prediction = 1
            testSet[i]["predict_ratings"][j] = rating_prediction
        if len(numNoNeighbors) > 0:
            # print(
            #     f">>> Test user_id {testSet[i]['user_id']} ran into no neighbors {len(numNoNeighbors)} times @ {numNoNeighbors}"
            # )
            numNoNeighborsTestSet.append(testSet[i]["user_id"])
        testSet[i]["prediction_done"] = np.True_
    if len(numNoNeighborsTestSet) > 0:
        print(
            f">>> Total number of test users with no neighbors {len(numNoNeighborsTestSet)}: {numNoNeighborsTestSet}"
        )
    else:
        print(f">>> All test users have neighbors for all predict item-ratings")


def writeOutputToFile(testSet: np.ndarray, outputFile: str):
    """[summary]

    Args:
        testSet (np.ndarray): [description]
        outputFile (str): [description]
    """
    with open(outputFile, "w") as f:
        for i in range(len(testSet)):
            for j in range(testSet[i]["num_predict"]):
                f.write(
                    f"{testSet[i]['user_id']} {testSet[i]['predict_indices'][j]} {testSet[i]['predict_ratings'][j].astype(int)}\n"
                )


# M_j (Count of number of users that have rated item j) for all 1000 items for Case Modification
# By transposing train dataset
def transpose_t(originalSet: np.ndarray, max_size: int, offset: int = 1):
    # Initialize test dataset
    transposed_dataset = recommender_init(
        dataset_size=(max_size,),  # 1000 items max possible
        user_id_offset=offset,
        num_known=0,
        num_predict=0,
        prediction_done=np.True_,
    )
    # Transpose from user-major to item-major
    # Go through all users and add their item-ratings to item-major dataset
    for user_id, user in enumerate(originalSet):
        # Go through the user's known ratings
        for item_id, item in enumerate(user["known_indices"][: user["num_known"]]):
            transposed_dataset[item - 1]["known_indices"][
                transposed_dataset[item - 1]["num_known"]
            ] = user["user_id"]
            transposed_dataset[item - 1]["known_ratings"][
                transposed_dataset[item - 1]["num_known"]
            ] = user["known_ratings"][item_id]
            transposed_dataset[item - 1]["num_known"] += 1
    for item in transposed_dataset:
        item["average_rating"] = np.mean(
            item["known_ratings"][: item["num_known"]]
        ).astype(np.float64)
    return transposed_dataset


def updateOutputFilenames(outputFiles: list, suffix: str = "", filetype: str = ".txt"):
    """[summary]

    Args:
        outputFiles (list): [description]
        prevSuffix (str, optional): [description]. Defaults to "".
        suffix (str, optional): [description]. Defaults to "".
        filetype (str, optional): [description]. Defaults to ".txt".

    Returns:
        [type]: [description]
    """
    return [
        outFile[: -len(filetype)] + "_" + suffix + filetype for outFile in outputFiles
    ]


def customAggregate(outTests: np.ndarray, successful_tests: np.ndarray):
    for i in range(len(outTests)):
        for j in range(len(outTests[i])):
            for k in range(outTests[i][j]["num_predict"]):
                # predict_ratings
                outTests[i][j]["predict_ratings"][k] = (
                    0.5 * successful_tests[0][i][j]["predict_ratings"][k]
                    + 0.25
                    * successful_tests[1][i][j]["predict_ratings"][k]
                    * 0.25
                    * successful_tests[2][i][j]["predict_ratings"][k]
                )
                outTests[i][j]["predict_ratings"][k] = np.around(
                    outTests[i][j]["predict_ratings"][k], decimals=0
                )
                outTests[i][j]["predict_ratings"][k] = outTests[i][j][
                    "predict_ratings"
                ][k].astype(int)
                if outTests[i][j]["predict_ratings"][k] > 5:
                    outTests[i][j]["predict_ratings"][k] = 5
                if outTests[i][j]["predict_ratings"][k] < 1:
                    outTests[i][j]["predict_ratings"][k] = 1
    return outTests


def main():
    """
    Driver function for Recommender System / User-based Collaborative Filtering Project.
    """
    """
    0.  Import data from the .txt files in the same directory as this script.
    """
    suffices = ["v2", "Pearson", "IUF", "CaseMod", "ItemBased", "Custom"]
    pearsons = [False, True, True, True, False, False]
    IUFs = [False, False, True, False, False, False]
    caseMods = [False, False, False, True, False, False]
    itemBased = [False, False, False, False, True, False]
    adjustedCosSim = [False, False, False, False, True, False]
    successful_runs = [0, 1, 3]
    failed_runs = [2, 4]
    (datasets, original_out_filenames,) = recommender_import()
    trains = datasets[3]
    tests = datasets[:3]
    # For both transposes, the item-major datasets have max possible of 1000 items
    transposed_trains = transpose_t(
        trains, max_size=1000, offset=1
    )  # To get the data for IUF, transpose the train dataset
    transposed_tests = [transpose_t(t, max_size=1000, offset=1) for t in tests]
    # print(
    #     f"Transposed num_known total {sum(transposed_trains[:]['num_known'])}, detail:"
    # )  # DEBUG # Worked; Same total number of known ratings as un-transposed train dataset
    # print(transposed_trains[:]["num_known"])
    successful_run_tests = []
    for i in range(len(suffices)):
        # Item Based failed, SKIP for computing Custom (5)
        if i in failed_runs:
            continue
        # for i in range(4, 5, 1):
        #   1.1.1   Cosine Similarity method (Naive Algorithm)
        #   1.1.2   Pearson Correlation method (Standard Algorithm)
        #   1.2.1   Pearson Correlation + Inverse user frequency
        #   1.2.2   Pearson Correlation + Case Modification
        #   2.1     Item-based Collaborative Filtering based on Adjusted Cosine Similarity
        #   3.1     Custom Collaborative Filtering Algorithm
        testSets = [copy.deepcopy(t) for t in tests]
        testSets_2 = [copy.deepcopy(t) for t in tests]
        trainSet = copy.deepcopy(trains)
        trainSet_2 = copy.deepcopy(trains)
        transposed_train = copy.deepcopy(transposed_trains)
        transposed_testSets = [copy.deepcopy(t) for t in transposed_tests]
        out_filenames = updateOutputFilenames(
            original_out_filenames, suffix=suffices[i],
        )
        print(f"out_filenames: {out_filenames}")
        if i == 5:  # Custom
            # Get the custom dataset
            testSets = [copy.deepcopy(t) for t in tests]
            # Aggregate rating is 50% Cos-Sim (Best MAE) + 25% Pearson Correlation (2nd Worst MAE) + 25% Case Modification (Worst MAE)
            customAggregate(testSets, successful_run_tests)
            [
                writeOutputToFile(test, out_filenames[idx])
                for idx, test in enumerate(testSets)
            ]
            continue
        if not itemBased[i]:
            [
                calcNeighborWeights(
                    test,
                    trainSet,
                    transposed_testSets[test_id],
                    transposed_train,
                    pearsons[i],
                    IUFs[i],
                )
                for test_id, test in enumerate(testSets)
            ]
            [
                calcPredictions(
                    test,
                    trainSet,
                    testSets_2[test_id],
                    trainSet_2,
                    pearsons[i],
                    IUFs[i],
                    caseMods[i],
                )
                for test_id, test in enumerate(testSets)
            ]
            [
                writeOutputToFile(test, out_filenames[idx])
                for idx, test in enumerate(testSets)
            ]
            if i in successful_runs:
                successful_run_tests.append(testSets)
        else:  # if item-based:
            [
                calcNeighborWeights(
                    test,
                    transposed_train,
                    testSets[test_id],
                    trainSet,
                    pearsons[i],
                    IUFs[i],
                    adjustedCosSim[i],
                )
                for test_id, test in enumerate(transposed_testSets)
            ]
            [
                calcPredictions(
                    test,
                    transposed_train,
                    testSets[test_id],
                    trainSet,
                    pearsons[i],
                    IUFs[i],
                    caseMods[i],
                )
                for test_id, test in enumerate(transposed_testSets)
            ]
            [print(t) for t in transposed_testSets]  # DEBUG # FAILED, OUT OF TIME
            # Transpose back to user-major order    #TBD
            [
                writeOutputToFile(test, out_filenames[idx])
                for idx, test in enumerate(transposed_testSets)
            ]
            if i in successful_runs:
                successful_run_tests.append(transposed_testSets)


if __name__ == "__main__":
    """
    If this file is run as a script, execute main()
    """
    main()

