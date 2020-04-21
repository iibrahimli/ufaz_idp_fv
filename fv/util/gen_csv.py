import os
import glob
import pandas as pd
import time
# import argparse
from tqdm import tqdm


# parser = argparse.ArgumentParser()
# parser.add_argument('--dataroot', '-d', type=str, required=True,
#                     help="Absolute path to the dataset folder")
# parser.add_argument('--csv_name', type=str, required=True,
#                     help="Name of the output csv file.")
# args = parser.parse_args()
# dataroot = args.dataroot
# csv_name = args.csv_name


def generate_csv(dataroot, csv_name):
    """
    Generates a csv file containing the image paths of the VGGFace2 dataset
    for use in triplet selection in triplet loss training.
    
    Args:
        dataroot (str): absolute path to the training dataset.
        csv_name (str): name of the resulting csv file.
   
    """
    
    print("Loading image paths ...")
    files = glob.glob(dataroot + "/*/*")

    start_time = time.time()
    list_rows = []

    print("Number of files: {}".format(len(files)))
    print("Generating csv file ...")

    progress_bar = enumerate(tqdm(files))

    for file_index, file in progress_bar:

        face_id = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))

        # better alternative than dataframe.append()
        row = {'id': face_id, 'name': face_label}
        list_rows.append(row)

    dataframe = pd.DataFrame(list_rows)
    dataframe = dataframe.sort_values(by=['name', 'id']).reset_index(drop=True)

    # encode names as categorical classes
    dataframe['class'] = pd.factorize(dataframe['name'])[0]
    dataframe.to_csv(path_or_buf=csv_name, index=False)

    elapsed_time = time.time()-start_time
    print("Done! Elapsed time: {:.2f} minutes.".format(elapsed_time/60))


# if __name__ == '__main__':
#     generate_csv(dataroot=dataroot, csv_name=csv_name)