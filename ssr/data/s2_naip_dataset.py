from basicsr.utils.registry import DATASET_REGISTRY
import os
import glob
import torch
from torch.utils import data


@DATASET_REGISTRY.register()
class S2NAIPDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.gt_path = opt['dataroot_gt']
        self.phase = opt.get('phase', 'train')

        self.gt_images = sorted(glob.glob(os.path.join(self.gt_path, '*.jpg')))

        print(f"📦 Dataset loaded: {len(self.gt_images)} samples")

    def __getitem__(self, index):
        import cv2
        import numpy as np

        # ---- LOAD GT ----
        img_gt = cv2.imread(self.gt_images[index])
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        # ---- TRAIN: random crop ----
        if self.phase == 'train':
            h, w, _ = img_gt.shape

            if h < 128 or w < 128:
                raise ValueError("GT image te klein")

            top = np.random.randint(0, h - 128)
            left = np.random.randint(0, w - 128)

            img_gt = img_gt[top:top+128, left:left+128]

        else:
            # ---- TEST ----
            img_gt = cv2.resize(img_gt, (128, 128), interpolation=cv2.INTER_AREA)

        # ---- MAKE LR (synthetic) ----
        img_lq = cv2.resize(img_gt, (32, 32), interpolation=cv2.INTER_AREA)

        # ---- BLUR ----
        img_lq = cv2.GaussianBlur(img_lq, (3, 3), 0)

        # ---- NOISE ----
        noise = np.random.normal(0, 2, img_lq.shape)  # klein beetje noise
        img_lq = img_lq.astype(np.float32) + noise
        img_lq = np.clip(img_lq, 0, 255)

        # ---- NORMALIZE ----
        img_gt = img_gt.astype(np.float32) / 255.0
        img_lq = img_lq.astype(np.float32) / 255.0

        # ---- TO TENSOR ----
        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1))
        img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1))

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': self.gt_images[index],  # fake maar nodig voor BasicSR
            'gt_path': self.gt_images[index]
        }

    def __len__(self):
        return len(self.gt_images)