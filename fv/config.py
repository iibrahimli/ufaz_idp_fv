from os import makedirs
from os.path import abspath, join
from multiprocessing import cpu_count


N_WORKERS = cpu_count() - 1

PROJECT_ROOT = abspath("..")
DATA_ROOT    = join(PROJECT_ROOT, "data")

# path to the dataset
DATASET_PATH       = join(DATA_ROOT, "vggface2")
DATASET_TRAIN_PATH = join(DATASET_PATH, "train")
DATASET_TEST_PATH  = join(DATASET_PATH, "test")

# paths to csv files
TRAIN_CSV_PATH = join(DATASET_PATH, "train.csv")
TEST_CSV_PATH  = join(DATASET_PATH, "test.csv")

# triplets
N_TRAIN_TRIPLETS    = 25000
TRAIN_TRIPLETS_DIR  = join(DATASET_PATH, "triplets")
TRAIN_TRIPLETS_PATH = join(TRAIN_TRIPLETS_DIR, "train_triplets.npy")
makedirs(TRAIN_TRIPLETS_DIR, exist_ok=True)

N_TEST_TRIPLETS     = 2500
TEST_TRIPLETS_DIR   = join(DATASET_PATH, "triplets")
TEST_TRIPLETS_PATH  = join(TEST_TRIPLETS_DIR, "test_triplets.npy")
makedirs(TEST_TRIPLETS_DIR, exist_ok=True)

# model checkpoint
MODEL_CHECKPOINT_DIR = join(PROJECT_ROOT, "fv/model/checkpoints")
makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
MODEL_CHECKPOINT_PATH = join(MODEL_CHECKPOINT_DIR, "ckpt_epoch_{}.pt")

# logs
TRAINING_LOG_PATH = join(PROJECT_ROOT, "fv/model/logs")
makedirs(TRAINING_LOG_PATH, exist_ok=True)

# training params (ik it's a bad idea)
IMG_SIZE       = (160, 160)
BATCH_SIZE     = 12
N_EPOCHS       = 50
INITIAL_LR     = 0.0001
TRIPLET_MARGIN = 0.5
EMBEDDING_DIM  = 128
USE_PRETRAINED = True
RESUME_PATH    = None