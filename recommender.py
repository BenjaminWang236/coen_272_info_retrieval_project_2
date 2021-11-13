import glob, os
import numpy as np
import math
import pprint as pp
from dataclasses import dataclass
import attr

# Global variables:
userID = 0
itemID = 1
rating = 2
noRating = 0


def userWeightCosineSimilarity(
    numTestUsers: int,
    numKnownPerTestUser: int,
    numTrainUsers: int,
    numItems: int,
    testArr: np.ndarray,
    trainArr: np.ndarray,
):
    """
    Calculate the user weight as user similarity using Cosine Similarity (range: 0 to 1)
    """
    userWeights = np.zeros(
        shape=(numTestUsers * numTrainUsers, 3)
    )  # Format = [testUserIndex, trainUserIndex, rating]
    # training Data is 200 user rows x 1000 item columns

    return userWeights


def userWeightPearsonCorrelation(
    numTestUsers: int,
    numKnownPerTestUser: int,
    numTrainUsers: int,
    numItems: int,
    testArr: np.ndarray,
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
            if row[rating] != noRating:
                print(
                    f"Error: Predictions Test file {test} row {row_id} not correctly split"
                )
                return False
        print(f"Test file {test} Predictions correctly split:\tYES")
    return True


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

    datasets, testNames, testSizes, testKnownRatings, testPredictions = readData(
        numUsersPerTest
    )
    # Create Output files' names:
    outputFileNames = [f"result{testSizes[test]}.txt" for test in testNames]
    print(f"outputFileNames: {outputFileNames}")

    if not PredictionsSizeOK(testNames, testSizes, testPredictions, datasets):
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
    # # Weight as Cosine Similarity between current test user and all training users
    for test in testNames:
        userWeights = userWeightCosineSimilarity(
            numUsersPerTest,
            testSizes[test],
            numTrainUsers,
            numItems,
            testKnownRatings[test],
            datasets["train"],
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
