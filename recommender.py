import glob, os
import numpy as np
import math
import pprint as pp
from dataclasses import dataclass
import attr


def main():
    """
    Driver function for the recommender system.
    """
    datasets = dict()
    os.chdir(
        os.path.dirname(os.path.abspath(__file__))
    )  # Change directory to the directory of this file
    for file in glob.glob("*.txt"):
        fileName = file[: -len(".txt")]  # remove .txt from file name
        datasets[fileName] = np.loadtxt(file, dtype=int)
        print(f"{fileName} shape:\t{datasets[fileName].shape}")
    pp.pprint(datasets)


if __name__ == "__main__":
    """
    Execute main() if this file is run directly.
    """
    main()
