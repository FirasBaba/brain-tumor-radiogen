import os

import monai
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from dataset import BrainRSNADataset

data = pd.read_csv("../input/train.csv")

targets = data.MGMT_value.values

device = torch.device("cuda")
model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, n_classes=1)
model.to(device)

tta_true_labels = []
tta_preds = []
preds_f = np.zeros(len(data))

for type_ in ["T1wCE"]:
    preds_type = np.zeros(len(data))
    all_weights = os.listdir("../weights")
    fold_files = [f for f in all_weights if type_+"_" in f]
    for fold in range(5):
        val_df = data[data.fold == fold]
        val_index = val_df.index
        val_df = val_df.reset_index(drop=True)

        test_dataset = BrainRSNADataset(data=val_df, mri_type=type_, is_train=True, do_load=True, ds_type=f"val_{type_}_{fold}")
        test_dl = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False, num_workers=4
            )
        image_ids = []
        model.load_state_dict(torch.load(f"../weights/{fold_files[fold]}"))
        preds = []
        case_ids = []
        with torch.no_grad():
            for  step, batch in enumerate(test_dl):
                model.eval()
                images = batch["image"].to(device)

                outputs = model(images)
                preds.append(outputs.sigmoid().detach().cpu().numpy())
                case_ids.append(batch["case_id"])

        case_ids = np.hstack(case_ids).tolist()

        preds_f[val_index] += np.vstack(preds).T[0]/5
        preds_type[val_index] += np.vstack(preds).T[0]
        score_fold = roc_auc_score(targets[val_index], np.vstack(preds).T[0])
        print(f"the score of the fold number {fold} and the type {type_}: {score_fold}")

    print(f"the final socre of the type {type_}")
    print(roc_auc_score(targets, preds_type))
    print("\n"*2)
print("the final score is")
print(roc_auc_score(targets, preds_f))
