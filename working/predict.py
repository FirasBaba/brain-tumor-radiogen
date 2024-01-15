import argparse
import os

import monai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm

from dataset import BrainRSNADataset

parser = argparse.ArgumentParser()
parser.add_argument("--type", default="FLAIR", type=str)
parser.add_argument("--model_name", default="b0", type=str)
args = parser.parse_args()

data = pd.read_csv("../input/sample_submission.csv")


# model
model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
device = torch.device("cpu")
model.to(device)
all_weights = os.listdir("../weights/")
fold_files = [f for f in all_weights if args.type in f]
criterion = nn.BCEWithLogitsLoss()


test_dataset = BrainRSNADataset(data=data, mri_type=args.type, is_train=False)
test_dl = torch.utils.data.DataLoader(
    test_dataset, batch_size=8, shuffle=False, num_workers=4
)

preds_f = np.zeros(len(data))
for fold in range(5):
    image_ids = []
    model.load_state_dict(torch.load(f"../weights/{fold_files[fold]}"))
    preds = []
    epoch_iterator_test = tqdm(test_dl)
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_test):
            model.eval()
            images = batch["image"].to(device)

            outputs = model(images)
            preds.append(outputs.sigmoid().detach().cpu().numpy())
            image_ids.append(batch["case_id"].detach().cpu().numpy())

    preds_f += np.vstack(preds).T[0] / 5

    ids_f = np.hstack(image_ids)

data["BraTS21ID"] = ids_f
data["MGMT_value"] = preds_f

data = data.sort_values(by="BraTS21ID").reset_index(drop=True)
data.to_csv("submission.csv", index=False)
