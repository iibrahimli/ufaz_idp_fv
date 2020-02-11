from os.path import abspath, join
from functools import partial


PROJECT_ROOT = abspath("..")
DATA_ROOT    = join(PROJECT_ROOT, "data")

# paths to datasets
DATASET_PATHS = {
    'celeba'     : join(DATA_ROOT, "celeba"),
    'colorferet' : join(DATA_ROOT, "colorferet"),
    'lfwa'       : join(DATA_ROOT, "lfwa"),
    'sof'        : join(DATA_ROOT, "sof"),
}

DATASET_RAW_PATHS = {
    dname: join(dpath, "raw") for dname, dpath in DATASET_PATHS.items()
}

# paths to processed datasets
DATASET_PROC_PATHS = {
    dname: join(dpath, "proc") for dname, dpath in DATASET_PATHS.items()
}

