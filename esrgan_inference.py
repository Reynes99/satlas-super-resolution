import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

# ========================
# RRDBNet (oude ESRGAN)
# ========================
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        return self.RDB3(self.RDB2(self.RDB1(x))) * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)

        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(torch.nn.functional.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

# ========================
# SETTINGS
# ========================
model_path = 'RRDB_ESRGAN_x4_fine_tuned.pth'
input_folder = 'inputs'
output_folder = 'results'

os.makedirs(output_folder, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# LOAD MODEL
# ========================
model = RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

print(f"✅ Model loaded: {model_path}")

# ========================
# INFERENCE
# ========================
image_paths = []
for ext in ['*.png', '*.jpg', '*.jpeg']:
    image_paths.extend(glob.glob(osp.join(input_folder, ext)))

for path in tqdm(image_paths):
    base = osp.splitext(osp.basename(path))[0]

    # ---- Load ----
    img = cv2.imread(path)
    img = img.astype(np.float32) / 255.0

    # BGR → RGB + HWC → CHW
    img = torch.from_numpy(
        np.transpose(img[:, :, ::-1].copy(), (2, 0, 1))
    ).unsqueeze(0).to(device)

    # ---- Inference ----
    with torch.no_grad():
        output = model(img)

    output = output.squeeze().cpu().clamp_(0, 1).numpy()

    # CHW → HWC
    output = np.transpose(output, (1, 2, 0))

    # RGB → BGR
    output = (output[:, :, ::-1] * 255.0).round().astype(np.uint8)

    # ---- Save ----
    cv2.imwrite(osp.join(output_folder, f"{base}_SR.png"), output)

print("\n🚀 Done! Check your 'results' folder.")