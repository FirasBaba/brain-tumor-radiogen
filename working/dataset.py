import glob
import os
import re

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import config
import utils


class BrainRSNADataset(Dataset):
    def __init__(
        self, data, transform=None, target="MGMT_value", mri_type="FLAIR", is_train=True, ds_type="forgot", do_load=True
    ):
        self.target = target
        self.data = data
        self.type = mri_type

        self.transform = transform
        self.is_train = is_train
        self.folder = "train" if self.is_train else "test"
        self.do_load = do_load
        self.ds_type = ds_type
        self.img_indexes = self._prepare_biggest_images()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.loc[index]
        case_id = int(row.BraTS21ID)
        target = int(row[self.target])
        _3d_images = self.load_dicom_images_3d(case_id)
        _3d_images = torch.tensor(_3d_images).float()
        if self.is_train:
            return {"image": _3d_images, "target": target, "case_id": case_id}
        else:
            return {"image": _3d_images, "case_id": case_id}

    def _prepare_biggest_images(self):
        big_image_indexes = {}
        if (f"big_image_indexes_{self.ds_type}.pkl" in os.listdir("../input/"))\
            and (self.do_load) :
            print("Loading the best images indexes for all the cases...")
            big_image_indexes = joblib.load(f"../input/big_image_indexes_{self.ds_type}.pkl")
            return big_image_indexes
        else:
            
            print("Caulculating the best scans for every case...")
            for row in tqdm(self.data.iterrows(), total=len(self.data)):
                case_id = str(int(row[1].BraTS21ID)).zfill(5)
                path = f"../input/{self.folder}/{case_id}/{self.type}/*.dcm"
                files = sorted(
                    glob.glob(path),
                    key=lambda var: [
                        int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
                    ],
                )
                resolutions = [utils.extract_cropped_image_size(f) for f in files]
                middle = np.array(resolutions).argmax()
                big_image_indexes[case_id] = middle

            joblib.dump(big_image_indexes, f"../input/big_image_indexes_{self.ds_type}.pkl")
            return big_image_indexes



    def load_dicom_images_3d(
        self,
        case_id,
        num_imgs=config.NUM_IMAGES_3D,
        img_size=config.IMAGE_SIZE,
        rotate=0,
    ):
        case_id = str(case_id).zfill(5)

        path = f"../input/{self.folder}/{case_id}/{self.type}/*.dcm"
        files = sorted(
            glob.glob(path),
            key=lambda var: [
                int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
            ],
        )

    
        middle = self.img_indexes[case_id]

        # # middle = len(files) // 2
        num_imgs2 = num_imgs // 2
        p1 = max(0, middle - num_imgs2)
        p2 = min(len(files), middle + num_imgs2)
        image_stack = [utils.load_dicom_image(f, rotate=rotate, voi_lut=True) for f in files[p1:p2]]
        
        img3d = np.stack(image_stack).T
        if img3d.shape[-1] < num_imgs:
            n_zero = np.zeros((img_size, img_size, num_imgs - img3d.shape[-1]))
            img3d = np.concatenate((img3d, n_zero), axis=-1)

        return np.expand_dims(img3d, 0)
