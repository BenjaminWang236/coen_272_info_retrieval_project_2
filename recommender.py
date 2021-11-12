import os
import numpy as np
import math
import pprint as pp
from dataclasses import dataclass
import attr


def main():
    """
    Driver function for the recommender system.
    """
    datasets = {"train": None, "test5": None, "test10": None, "test20": None}
    for dataset in datasets:
        datasets[dataset] = np.loadtxt(f"{dataset}.txt", dtype=int)
        print(f"{dataset} shape: {datasets[dataset].shape}")
    pp.pprint(datasets)


if __name__ == "__main__":
    """
    Execute main() if this file is run directly.
    """
    main()
