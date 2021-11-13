import dataclasses
import glob, os
import numpy as np
import math
import pprint as pp
from dataclasses import dataclass
import attr

# Global variables:
userID = 0
trainID = 1
itemID = 1
rating = 2
noRating = 0
minRating = 1
maxRating = 5


def userWeightCosineSimilarity(
    numTestUsers: int,
    numKnownPerTestUser: int,
    numTrainUsers: int,
    numItems: int,
    testKnownArr: np.ndarray,
    trainArr: np.ndarray,
):
    """
    Calculate the user weight as user similarity using Cosine Similarity (range: 0 to 1)
    
    Output:
        userWeights: numpy array of user weights, format = [testUserIndex, trainUserIndex, weight]
    """
    userWeights = np.zeros(
        shape=(numTestUsers * numTrainUsers, 3)
    )  # Format = [testUserIndex, trainUserIndex, weight]
    # training Data is 200 user rows x 1000 item columns
    testUserOffset = testKnownArr[0, userID]
    # print(f"TestKnownArr: {testKnownArr}")
    lowestSimilarity = 100
    for testUserIndex in range(numTestUsers):
        userWeights[
            testUserIndex * numTrainUsers : (testUserIndex + 1) * numTrainUsers, userID
        ] = int(testUserIndex + testUserOffset)
        for trainUserIndex in range(numTrainUsers):
            userWeights[testUserIndex * numTrainUsers + trainUserIndex, trainID] = int(
                trainUserIndex
            )
            # Calculate Cosine Similarity of Test User and Train User on all items here:
            cosSim = 0
            nominator, sum_squared_user_1, sum_squared_user_2 = (
                0,
                sum(
                    testKnownArr[
                        testUserIndex
                        * numKnownPerTestUser : (testUserIndex + 1)
                        * numKnownPerTestUser,
                        rating,
                    ]
                    ** 2
                ),
                0,
            )

            # Only calculate at the indices of the known ratings given by the test user:
            for i in range(numKnownPerTestUser):
                known_item_index = int(
                    testKnownArr[testUserIndex * numKnownPerTestUser + i, itemID]
                )
                # if trainArr[trainUserIndex, known_item_index] != noRating:
                # Since noRating == 0, could just add:
                nominator += (
                    testKnownArr[testUserIndex * numKnownPerTestUser + i, rating]
                    * trainArr[trainUserIndex, known_item_index]
                )
                # sum_squared_user_1 += (
                #     testKnownArr[testUserIndex * numKnownPerTestUser + i, rating] ** 2
                # )
                sum_squared_user_2 += trainArr[trainUserIndex, known_item_index] ** 2
            if sum_squared_user_2 != 0:
                cosSim = nominator / (
                    math.sqrt(sum_squared_user_1) * math.sqrt(sum_squared_user_2)
                )
            # print(
            #     f"Cosine Similarity of test user {testUserIndex} and training user {trainUserIndex} is {cosSim} for nominator: {nominator}, sum_squared_user_1: {sum_squared_user_1}, sum_squared_user_2: {sum_squared_user_2}"
            # )

            userWeights[testUserIndex * numTrainUsers + trainUserIndex, rating] = cosSim
            if (lowestSimilarity > cosSim) and (cosSim != 0):
                lowestSimilarity = cosSim
    print(
        f"Lowest Weight: {lowestSimilarity}, Highest Weight: {max(userWeights[:, rating])}"
    )

    return userWeights


def userWeightPearsonCorrelation(
    numTestUsers: int,
    numKnownPerTestUser: int,
    numTrainUsers: int,
    numItems: int,
    testKnownArr: np.ndarray,
    trainArr: np.ndarray,
):
    """
    Calculate the user weight as user similarity using Pearson Correlation (range: -1 to 1)
    """
    userWeights = np.zeros(
        shape=(numTestUsers * numTrainUsers, 3)
    )  # Format = [testUserIndex, trainUserIndex, rating]

    return userWeights


def readData(numUsersPerTest: int):
    """
    Return a dictionary of datasets, a list of test names, a dictionary of test sizes,
    a dictionary of empty array of test known ratings, and a dictionary of empty array of test predictions.
    """
    datasets = dict()
    testNames = []
    testSizes = dict()
    testKnownRatings = dict()
    testPredictions = dict()

    os.chdir(
        os.path.dirname(os.path.abspath(__file__))
    )  # Change directory to the directory of this file
    for file in glob.glob("*.txt"):
        if file.startswith("result"):
            continue
        fileName = file[: -len(".txt")]  # remove .txt from file name
        datasets[fileName] = np.loadtxt(file, dtype=int)
        print(f"{fileName} shape:\t{datasets[fileName].shape}")
        # test files names have to be in format "test#", where # is a integer number
        if fileName[: len("test")] == "test" and fileName[len("test") :].isdigit():
            testNames.append(fileName)
            testSizes[fileName] = int(fileName[len("test") :])
            # For each test files, split test file per user (100 users per test file, starting at 201-300 for test5)
            # For each user, split into the testSize number of given ratings and empty ratings to predict
            testKnownRatings[fileName] = np.zeros(
                shape=(
                    testSizes[fileName] * numUsersPerTest,
                    datasets[fileName].shape[1],
                )
            )
            testPredictions[fileName] = np.zeros(
                shape=(
                    datasets[fileName].shape[0]
                    - (testSizes[fileName] * numUsersPerTest),
                    datasets[fileName].shape[1],
                )
            )
            print(testKnownRatings[fileName].shape)
            print(testPredictions[fileName].shape)

    print(testNames)
    print(f"Test Sizes:\t{testSizes}")
    # pp.pprint(datasets)

    return datasets, testNames, testSizes, testKnownRatings, testPredictions


def PredictionsSizeOK(
    testNames: list, testSizes: dict, testPredictions: dict, datasets: dict
):
    """Check the test Prediction size against corresponding test file size:"""
    for test in testNames:
        testSize = testSizes[test]
        correspondingExampleTestFile = datasets[f"example_result{testSize}"]
        if testPredictions[test].shape != correspondingExampleTestFile.shape:
            print(
                f"Test Prediction {test} shape does not match corresponding example test file example_result{testSize} shape"
            )
            return False
        else:
            print(
                f"Test Prediction {test} shape matches corresponding example test file example_result{testSize} shape:\tYES"
            )
    return True


def splitTests(
    testNames: list, testKnownRatings: dict, testPredictions: dict, datasets: dict,
):
    """Split the dataset (Fill in) into testSize number of given ratings and empty ratings to predict"""
    for test in testNames:
        row_id_known, row_id_predict = 0, 0
        for row in datasets[test]:
            if row[rating] != noRating:
                testKnownRatings[test][row_id_known] = row
                row_id_known += 1
            else:
                testPredictions[test][row_id_predict] = row
                row_id_predict += 1


def splitTestsOK(testNames: list, testKnownRatings: dict, testPredictions: dict):
    """Checking if the test files are correctly split"""
    for test in testNames:
        for row_id, row in enumerate(testKnownRatings[test]):
            if row[rating] == noRating:
                print(
                    f"Error: Knowns Test file {test} row {row_id} not correctly split"
                )
                return False
        print(f"Test file {test} Knowns correctly split:\tYES")
        for row_id, row in enumerate(testPredictions[test]):
            if row[itemID] > 1000:
                print(f"ERROR: item_id only goes up to 999")
                return False
            if row[rating] != noRating:
                print(
                    f"Error: Predictions Test file {test} row {row_id} not correctly split"
                )
                return False
        print(f"Test file {test} Predictions correctly split:\tYES")
    return True


def findNeighborhood(
    test: str,
    numUsersPerTest: int,
    testPredictions: dict,
    trainingData: np.ndarray,
    numTrainUsers: int,
    userWeights: np.ndarray,
):
    """
    Find the neighborhood as neighbors of each test user.
    A neighbor must have rating for the desired item (item_id) AND 
    share at least one item-rating in common with the test user.
    
    We know if a training user has a weight of 0, then it shares no rating in common with the test user.
    
    Output:
        neighborhood (list): Format: [userID, item_id, [matching training users]]

    Args:
        test (str): Name of test file
        numUsersPerTest (int): 100 users per test as given
        testPredictions (dict): test array with only the empty ratings awaiting predictions, access by index "test"
        trainingData (numpy ndarray): 200 x 1000 array of training data
    """
    predictionTotal = 0
    # For each test user:
    neighborhood = []  # Format: [userID, item_id, [matching training users]]
    for testUser in range(numUsersPerTest):
        neighbors = []  # Format: [item_id, [matching training users]]
        testUserOffset = int(testPredictions[test][0, userID])
        # Get all the predictions it needs to make by going through the Prediction-only array
        predictionCount = 0
        for prediction in testPredictions[test][predictionTotal:]:
            if prediction[userID] != (testUser + testUserOffset):
                break
            # print(f"testUser {testUser} prediction number {predictionCount}")
            # print(prediction)
            item_index = int(prediction[itemID])
            neighbors.append([testUser + testUserOffset, item_index, []])
            # Go through each training user (Each row = one training user's rating for all items)
            for row_id, row in enumerate(trainingData):
                if (
                    row[item_index - 1] > 0
                    and userWeights[testUser * numTrainUsers + row_id][rating] > 0
                ):
                    neighbors[predictionCount][rating].append(row_id)
            predictionCount += 1
        if predictionCount == 0:
            print(
                f"No matching neighbors for tester {testUser + testUserOffset} item {item_index} (1-based)"
            )
        # concatenated = [testUser + testUserOffset] + [neighbors]
        # print(f"concatenated: {concatenated}")
        # neighborhood.append(concatenated)
        neighborhood.extend(neighbors)
        predictionTotal += predictionCount
        # print(predictionTotal)
    # print(neighborhood)
    # print(f"First neighborhood: {neighborhood[0][1]}")

    return neighborhood


def makePredictions(
    testPrediction: np.ndarray,
    userWeights: np.ndarray,
    neighborhood: list,
    trainingData: np.ndarray,
    numTrainUsers: int,
):
    """
    Make predictions for each test user's empty ratings.

    Args:
        testPrediction (np.ndarray): test array with only the empty ratings awaiting predictions
        userWeights (np.ndarray): format = [testUserIndex, trainUserIndex, weight]
        neighborhood (list): format = [userID, item_id, [matching training users]]
        trainingData (np.ndarray): Given 200 rows of users x 1000 columns of item-ratings
        numTrainUsers (int): 200 (rows of users) in trainingData

    Returns:
        testPrediction (np.ndarray): test prediction-only array with filled in ratings
    """
    # predictions = np.zeros(
    #     shape=(len(neighborhood), 1)
    # )
    # print(userWeights)
    # print(neighborhood[0][0])
    # Row Format: [userID, itemID, predicted-Rating]
    testUserIndexOffset = userWeights[0, userID]
    print(f"offset: {testUserIndexOffset}")
    for row in testPrediction:
        user_id = int(row[userID])  # a
        item_id = int(
            row[itemID]
        )  # 1-based, so decrement by 1 when fetching from trainingData by index
        actual_user_id = int(user_id - testUserIndexOffset)

        # Compute Prediction here:
        prediction = 0
        sum_neighbor_weights = 0  # Get the sum of all neighboring weights
        nominator = 0
        neighborFound = False
        for neighbor in neighborhood:
            if neighborFound:
                break
            # print(neighbor)
            if neighbor[userID] == user_id and neighbor[itemID] == item_id:
                neighborFound = True
                # print(neighbor)
                for trainUser in neighbor[rating]:
                    sum_neighbor_weights += userWeights[
                        int(actual_user_id * numTrainUsers + trainUser), rating
                    ]
                    nominator += (
                        userWeights[
                            int(actual_user_id * numTrainUsers + trainUser), rating
                        ]
                        * trainingData[trainUser, item_id - 1]
                    )
        # print(sum_neighbor_weights)
        # print(nominator)
        if sum_neighbor_weights == 0:
            prediction = 3
        else:
            prediction = nominator / sum_neighbor_weights
            if prediction > maxRating:
                prediction = maxRating
            if prediction < minRating:
                prediction = minRating
            if int(prediction) == noRating:
                prediction = 3

        row[rating] = prediction

    # Predicted Ratings are in floats, so convert to ints:
    testPrediction = np.around(testPrediction)
    testPrediction = testPrediction.astype(int)

    return testPrediction


def main():
    """
    Driver function for the recommender system.
    Rating range: [1, 5]
    Rating of 0 means no rating NOT lowest rating
    """

    """
    0.  Import data from the .txt files in the same directory as this script.
    """
    # Metadata given in the assignment:
    numUsersPerTest = 100
    numTrainUsers = 200
    numItems = 1000

    datasets, testNames, testKnownSizes, testKnownRatings, testPredictions = readData(
        numUsersPerTest
    )
    # Create Output files' names:
    outputFileNames = [f"result{testKnownSizes[test]}.txt" for test in testNames]
    print(f"outputFileNames: {outputFileNames}")

    if not PredictionsSizeOK(testNames, testKnownSizes, testPredictions, datasets):
        exit()
    splitTests(testNames, testKnownRatings, testPredictions, datasets)
    if not splitTestsOK(testNames, testKnownRatings, testPredictions):
        exit()

    # pp.pprint(testKnownRatings)
    # pp.pprint(testPredictions)

    """
    1.  User-Based Collaborative Filtering Algorithms
    """
    #   1.1.1   Cosine Similarity method (Naive Algorithm)
    for test_id, test in enumerate(testNames):
        # test = "test5"
        print(f"{test}:")

        # Weight as Cosine Similarity between current test user and all training users
        userWeights = userWeightCosineSimilarity(
            numUsersPerTest,
            testKnownSizes[test],
            numTrainUsers,
            numItems,
            testKnownRatings[test],
            datasets["train"],
        )
        pp.pprint(userWeights)
        print(f"userWeights shape:\t{userWeights.shape}")

        # Find neighbors for all users in this test:
        neighborhood = findNeighborhood(
            test,
            numUsersPerTest,
            testPredictions,
            datasets["train"],
            numTrainUsers,
            userWeights,
        )
        # print(neighborhood[0])

        # Make predictions for all users in this test:
        testPredictions[test] = makePredictions(
            testPredictions[test],
            userWeights,
            neighborhood,
            datasets["train"],
            numTrainUsers,
        )
        # print(testPredictions[test])
        # for row in testPredictions[test]:
        #     print(row)
        print(
            f"max predicted: {max(testPredictions[test][:,rating])} min predicted: {min(testPredictions[test][:,rating])}"
        )
        print(
            f"max item_index: {max(testPredictions[test][:,itemID])} min item_index: {min(testPredictions[test][:,itemID])}"
        )

        # Write out the predictions to the output file
        np.savetxt(
            outputFileNames[test_id], testPredictions[test], fmt="%i", delimiter=" "
        )

    #   1.1.2   Pearson Correlation method (Standard Algorithm)
    #   1.2.1   Pearson Correlation + Inverse user frequency
    #   1.2.2   Pearson Correlation + Case Modification

    """
    2.  Item-Based Collaborative Filtering Algorithm based on Adjusted Cosine Similarity
    """

    """
    3.  Implement MY OWN algorithm
    """


if __name__ == "__main__":
    """
    Execute main() if this file is run directly.
    """
    main()
