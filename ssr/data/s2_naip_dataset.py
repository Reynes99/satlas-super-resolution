from basicsr.utils.registry import DATASET_REGISTRY
import os
import glob
import torch
from torch.utils import data

@DATASET_REGISTRY.register()
class S2NAIPDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.lq_path = opt['dataroot_lq']
        self.gt_path = opt['dataroot_gt']

        self.lq_images = sorted(glob.glob(os.path.join(self.lq_path, '*.jp2')))
        self.gt_images = sorted(glob.glob(os.path.join(self.gt_path, '*.jpg')))

    def __getitem__(self, index):
        import rasterio
        import cv2
        import numpy as np
        import torch

        # ---- LOAD JP2 (ECHT BELANGRIJK) ----
        with rasterio.open(self.lq_images[index]) as src:
            img_lq = src.read([1,2,3])  # RGB bands
            img_lq = np.transpose(img_lq, (1,2,0))

        # ---- NORMALISATIE (zoals Satlas) ----
        img_lq = img_lq.astype(np.float32) / 255.0

        # ---- resize naar 32x32 ----
        img_lq = cv2.resize(img_lq, (32,32), interpolation=cv2.INTER_AREA)

        # ---- GT ----
        img_gt = cv2.imread(self.gt_images[index])
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        # 🔥 GEEN aggressive downscale meer!
        img_gt = cv2.resize(img_gt, (128,128), interpolation=cv2.INTER_AREA)
        img_gt = img_gt.astype(np.float32) / 255.0

        # ---- naar tensor ----
        img_lq = torch.from_numpy(img_lq.transpose(2,0,1))
        img_gt = torch.from_numpy(img_gt.transpose(2,0,1))

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': self.lq_images[index],
            'gt_path': self.gt_images[index]
        }

    def __len__(self):
        return len(self.lq_images)