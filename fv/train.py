import time
from datetime import datetime
from os.path import join, isfile
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from tqdm import tqdm

from fv import config as cfg
from fv.model import face_resnet, triplet_loss
from fv.util import generate_csv, triplet_dataset


if isfile(cfg.TRAIN_CSV_PATH) == False:
    generate_csv(cfg.DATASET_TRAIN_PATH, cfg.TRAIN_CSV_PATH)

if isfile(cfg.TEST_CSV_PATH) == False:
    generate_csv(cfg.DATASET_TEST_PATH, cfg.TEST_CSV_PATH)

data_transforms = transforms.Compose([
    transforms.Resize(size=cfg.IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_dataloader = torch.utils.data.DataLoader(
    dataset=triplet_dataset(
        root_dir=cfg.DATASET_TRAIN_PATH,
        csv_name=cfg.TRAIN_CSV_PATH,
        num_triplets=cfg.N_TRAIN_TRIPLETS,
        training_triplets_path=cfg.TRAIN_TRIPLETS_PATH
            if isfile(cfg.TRAIN_TRIPLETS_PATH) else None,
        transform=data_transforms
    ),
    batch_size=cfg.BATCH_SIZE,
    num_workers=cfg.N_WORKERS,
    shuffle=True
)

model = face_resnet(
    embedding_dimension=cfg.EMBEDDING_DIM,
    pretrained=cfg.USE_PRETRAINED
).cuda()

opt = optim.Adam(model.parameters(), lr=cfg.INITIAL_LR)

start_epoch = 0

# optionally resume from a checkpoint
if cfg.RESUME_PATH is not None:

    if isfile(cfg.RESUME_PATH):
        print("Loading checkpoint {} ...".format(cfg.RESUME_PATH))

        checkpoint = torch.load(cfg.RESUME_PATH)
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['model_state_dict'])

        opt.load_state_dict(checkpoint['opt_state_dict'])

        print("Checkpoint loaded: start epoch from checkpoint = {}".format(start_epoch))
        print("Running for {} epochs".format(cfg.N_EPOCHS - start_epoch))
    else:
        print("WARNING: No checkpoint found at {}, training from scratch".format(cfg.RESUME_PATH))


dt = datetime.now()
dt_str = dt.strftime("%d-%m-%Y_%H-%M-%S")
total_time_start = time.time()
start_epoch = start_epoch
end_epoch = start_epoch + cfg.N_EPOCHS
l2_distance = PairwiseDistance(2).cuda()
margin = cfg.TRIPLET_MARGIN
loss = triplet_loss(margin=margin)


print("Training on {} triplets for {} epochs".format(
    cfg.N_TRAIN_TRIPLETS,
    cfg.N_EPOCHS - start_epoch))

for epoch in range(start_epoch, end_epoch):
    epoch_time_start = time.time()

    epoch_loss_sum = 0
    num_valid_training_triplets = 0

    # Training pass
    model.train()
    pbar = tqdm(train_dataloader)

    for batch_idx, (batch_sample) in enumerate(pbar):
        
        batch_loss_sum = 0

        anc_img = batch_sample['anc_img'].cuda()
        pos_img = batch_sample['pos_img'].cuda()
        neg_img = batch_sample['neg_img'].cuda()

        # Forward pass - compute embeddings
        anc_embedding, pos_embedding, neg_embedding = model(anc_img), model(pos_img), model(neg_img)

        # Forward pass - choose hard negatives only for training
        pos_dist = l2_distance.forward(anc_embedding, pos_embedding)
        neg_dist = l2_distance.forward(anc_embedding, neg_embedding)

        batch_dist = (neg_dist - pos_dist < margin).cpu().numpy().flatten()

        hard_triplets = np.where(batch_dist == 1)
        if len(hard_triplets[0]) == 0:
            continue

        anc_hard_embedding = anc_embedding[hard_triplets].cuda()
        pos_hard_embedding = pos_embedding[hard_triplets].cuda()
        neg_hard_embedding = neg_embedding[hard_triplets].cuda()

        # Calculate triplet loss
        batch_loss = loss.forward(
            anchor=anc_hard_embedding,
            positive=pos_hard_embedding,
            negative=neg_hard_embedding
        ).cuda()

        # Calculating loss
        batch_loss_sum += batch_loss.item()
        epoch_loss_sum += batch_loss.item()
        num_valid_training_triplets += len(anc_hard_embedding)

        # Backward pass
        opt.zero_grad()
        batch_loss.backward()
        opt.step()


        if batch_idx % 10 == 0:
            pbar.set_description("epoch {} - batch {} - loss {:.5f}".format(
                epoch+1,
                batch_idx,
                batch_loss_sum / cfg.BATCH_SIZE
                )
            )
            # print('Epoch {} batch {}:\tavg batch loss: {:.4f}'.format(
            #         epoch+1,
            #         batch_idx,
            #         batch_loss_sum / cfg.BATCH_SIZE,
            #     )
            # )

    # Model only trains on hard negative triplets
    avg_triplet_loss = 0 if (num_valid_training_triplets == 0) else epoch_loss_sum / num_valid_training_triplets
    epoch_time_end = time.time()

    # Print training statistics and add to log
    print('Epoch {}:\tavg loss: {:.4f}\tepoch time: {:.3f} minutes\t# valid triplets: {}'.format(
            epoch+1,
            avg_triplet_loss,
            (epoch_time_end - epoch_time_start)/60,
            num_valid_training_triplets
        )
    )
    with open(join(cfg.TRAINING_LOG_PATH, f"training_log.txt"), 'a') as f:
        val_list = [
            epoch+1,
            avg_triplet_loss,
            num_valid_training_triplets
        ]
        log = ';'.join(str(value) for value in val_list)
        f.writelines(log + '\n')

    # Save model checkpoint
    state = {
        'epoch': epoch+1,
        'embedding_dimension': cfg.EMBEDDING_DIM,
        'batch_size_training': cfg.BATCH_SIZE,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict()
    }
    torch.save(state, cfg.MODEL_CHECKPOINT_PATH.format(epoch+1))


# training done
total_time_end = time.time()
total_time_elapsed = total_time_end - total_time_start
print("Training finished: total time elapsed: {:.2f} minutes".format(total_time_elapsed / 60))
