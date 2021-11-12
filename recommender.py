import glob, os
import numpy as np
import math
import pprint as pp
from dataclasses import dataclass
import attr


def userWeight(user):
    """
    Calculate the user weight based on the number of ratings
    """
    return 1 / (1 + math.log(user.size))


def main():
    """
    Driver function for the recommender system.
    """

    """
    0.  Import data from the .txt files in the same directory as this script.
    """
    datasets = dict()
    testSizes = list()
    os.chdir(
        os.path.dirname(os.path.abspath(__file__))
    )  # Change directory to the directory of this file
    for file in glob.glob("*.txt"):
        fileName = file[: -len(".txt")]  # remove .txt from file name
        datasets[fileName] = np.loadtxt(file, dtype=int)
        print(f"{fileName} shape:\t{datasets[fileName].shape}")
        if fileName[: len("test")] == "test" and fileName[len("test") :].isdigit():
            testSizes.append(int(fileName[len("test") :]))
    print(f"Test Sizes:\t{testSizes}")
    pp.pprint(datasets)

    """
    1.  User-Based Collaborative Filtering Algorithms
    """
    #   1.1.1   Cosine Similarity method (Naive Algorithm)
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
