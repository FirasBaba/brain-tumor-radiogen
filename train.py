import argparse
import os

import monai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
from tqdm import tqdm

import config
from dataset import BrainRSNADataset

parser = argparse.ArgumentParser()
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--type", default="FLAIR", type=str)
parser.add_argument("--model_name", default="b0", type=str)
args = parser.parse_args()

data = pd.read_csv("../input/train.csv")
train_df = data[data.fold != args.fold].reset_index(drop=False)
val_df = data[data.fold == args.fold].reset_index(drop=False)


device = torch.device("cuda")

print(f"train_{args.type}_{args.fold}")
train_dataset = BrainRSNADataset(data=train_df, mri_type=args.type, ds_type=f"train_{args.type}_{args.fold}")

valid_dataset = BrainRSNADataset(data=val_df, mri_type=args.type, ds_type=f"val_{args.type}_{args.fold}")


train_dl = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.TRAINING_BATCH_SIZE,
    shuffle=True,
    num_workers=config.n_workers,
    drop_last=True,
    pin_memory=True,
)

validation_dl = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=config.TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=config.n_workers,
    pin_memory=True,
)


model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5, last_epoch=-1, verbose=True)

model.zero_grad()
model.to(device)
best_loss = 9999
best_auc = 0
criterion = nn.BCEWithLogitsLoss()
for counter in range(config.N_EPOCHS):

    epoch_iterator_train = tqdm(train_dl)
    tr_loss = 0.0
    for step, batch in enumerate(epoch_iterator_train):
        model.train()
        images, targets = batch["image"].to(device), batch["target"].to(device)

        outputs = model(images)
        targets = targets  # .view(-1, 1)
        loss = criterion(outputs.squeeze(1), targets.float())

        loss.backward()
        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()

        tr_loss += loss.item()
        epoch_iterator_train.set_postfix(
            batch_loss=(loss.item()), loss=(tr_loss / (step + 1))
        )
    scheduler.step()  # Update learning rate schedule

    if config.do_valid:
        with torch.no_grad():
            val_loss = 0.0
            preds = []
            true_labels = []
            case_ids = []
            epoch_iterator_val = tqdm(validation_dl)
            for step, batch in enumerate(epoch_iterator_val):
                model.eval()
                images, targets = batch["image"].to(device), batch["target"].to(device)

                outputs = model(images)
                targets = targets  # .view(-1, 1)
                loss = criterion(outputs.squeeze(1), targets.float())
                val_loss += loss.item()
                epoch_iterator_val.set_postfix(
                    batch_loss=(loss.item()), loss=(val_loss / (step + 1))
                )
                preds.append(outputs.sigmoid().detach().cpu().numpy())
                true_labels.append(targets.cpu().numpy())
                case_ids.append(batch["case_id"])
        preds = np.vstack(preds).T[0].tolist()
        true_labels = np.hstack(true_labels).tolist()
        case_ids = np.hstack(case_ids).tolist()
        auc_score = roc_auc_score(true_labels, preds)
        auc_score_adj_best = 0
        for thresh in np.linspace(0, 1, 50):
            auc_score_adj = roc_auc_score(true_labels, list(np.array(preds) > thresh))
            if auc_score_adj > auc_score_adj_best:
                best_thresh = thresh
                auc_score_adj_best = auc_score_adj

        print(
            f"EPOCH {counter}/{config.N_EPOCHS}: Validation average loss: {val_loss/(step+1)} + AUC SCORE = {auc_score} + AUC SCORE THRESH {best_thresh} = {auc_score_adj_best}"
        )
        if auc_score > best_auc:
            print("Saving the model...")

            all_files = os.listdir("../weights/")

            for f in all_files:
                if f"{args.model_name}_{args.type}_fold{args.fold}" in f:
                    os.remove(f"../weights/{f}")

            best_auc = auc_score
            torch.save(
                model.state_dict(),
                f"../weights/3d-{args.model_name}_{args.type}_fold{args.fold}_{round(best_auc,3)}.pth",
            )

print(best_auc)
