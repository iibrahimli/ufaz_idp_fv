import numpy as np

from fv import config


if __name__ == "__main__":

    print(config.PROJECT_ROOT)
    print(config.DATA_ROOT)
    print()

    for dname, dpath in config.DATASET_PATHS.items():
        print(f"{dname:12}: {dpath}")

    print()

    for dname, dpath in config.DATASET_RAW_PATHS.items():
        print(f"{dname:12}: {dpath}")

    print()

    for dname, dpath in config.DATASET_PROC_PATHS.items():
        print(f"{dname:12}: {dpath}")