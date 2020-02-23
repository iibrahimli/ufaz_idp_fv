import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

from fv import config


class triplet_dataset(Dataset):

    def __init__(self, root_dir, csv_name, num_triplets,
                 training_triplets_path=None, save_dir=config.TRAIN_TRIPLETS_PATH,
                 transform=None):

        self.df = pd.read_csv(csv_name, dtype={'id':    object,
                                               'name':  object,
                                               'class': int})
        self.root_dir = root_dir
        self.num_triplets = num_triplets
        self.transform = transform

        if training_triplets_path is None:
            self.training_triplets = self.generate_triplets(
                self.df,
                self.num_triplets,
                save_dir)
        else:
            self.training_triplets = np.load(training_triplets_path)


    @staticmethod
    def generate_triplets(df, num_triplets, save_path):

        def make_dictionary_for_face_class(df):
            """
            face_classes = {'class0': [class0_id0, ...],
                            'class1': [class1_id0, ...],
                            ...}
            """
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                face_classes[label].append(df.iloc[idx, 0])

            return face_classes

        triplets = []
        classes = df['class'].unique()
        face_classes = make_dictionary_for_face_class(df)

        print("\nGenerating {} triplets...".format(num_triplets))

        progress_bar = tqdm(range(num_triplets))
        for _ in progress_bar:

            """
            * randomly choose anchor, positive and negative images for
              triplet loss
            * anchor and positive images in pos_class
            * negative image in neg_class
            * at least, two images needed for anchor and positive images
              in pos_class
            * negative image should have different class as anchor and
              positive images by definition
            """

            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)

            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)

            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)

            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))

                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))

            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [
                    face_classes[pos_class][ianc],
                    face_classes[pos_class][ipos],
                    face_classes[neg_class][ineg],
                    pos_class,
                    neg_class,
                    pos_name,
                    neg_name
                ]
            )

        # save training triplets as a numpy file
        print("Saving training triplets list in {} ...".format(save_path))
        np.save(save_path, triplets)
        print("Training triplets list saved")

        return triplets


    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = self.add_extension(os.path.join(self.root_dir, str(pos_name), str(anc_id)))
        pos_img = self.add_extension(os.path.join(self.root_dir, str(pos_name), str(pos_id)))
        neg_img = self.add_extension(os.path.join(self.root_dir, str(neg_name), str(neg_id)))

        anc_img = Image.open(anc_img)
        pos_img = Image.open(pos_img)
        neg_img = Image.open(neg_img)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample


    def __len__(self):
        return len(self.training_triplets)


    def add_extension(self, path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError("No file '{}' with extension png or jpg".format(path))