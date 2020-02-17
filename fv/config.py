from os.path import abspath, join
from functools import partial


PROJECT_ROOT = abspath("..")
DATA_ROOT    = join(PROJECT_ROOT, "data")

# paths to the dataset
DATASET_PATH = join(DATA_ROOT, "vggface2")