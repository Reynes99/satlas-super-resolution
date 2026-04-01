import os
import sys

from skimage import data_dir

# fix import pad
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import torch
import argparse
import numpy as np
import cv2
import rasterio
import skimage.io

from ssr.utils.infer_utils import format_s2naip_data
from ssr.utils.options import yaml_load
from ssr.utils.model_utils import build_network




# =========================
# 🔥 JP2 → RGB loader (CORRECT)
# =========================
def load_sentinel_rgb(path):
    with rasterio.open(path) as src:
        # Sentinel bands:
        # 1 = B02 (blue)
        # 2 = B03 (green)
        # 3 = B04 (red)
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)

    # stack naar RGB
    im = np.stack([red, green, blue], axis=-1)

    # normalisatie (Sentinel reflectance)
    im = im.astype(np.float32)
    im = im / 10000.0
    im = np.clip(im, 0, 1)

    # naar uint8
    im = (im * 255).astype(np.uint8)

    return im


# =========================
# 🚀 MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path naar YAML config')
    args = parser.parse_args()

    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("⚠️ GPU niet beschikbaar, fallback naar CPU")

    torch.backends.cudnn.benchmark = True

    # load config
    opt = yaml_load(args.opt)

    n_lr_images = opt['n_lr_images']
    data_dir = opt['data_dir']
    save_path = opt['save_path']

    print("🔥 YAML pad:", args.opt)
    print("📂 data_dir uit YAML:", data_dir)
    print("📂 bestaat:", os.path.exists(data_dir))

    if os.path.exists(data_dir):
        print("📂 inhoud:", os.listdir(data_dir)[:5])  # eerste 5 files
    else:
        print("❌ map bestaat niet")

    os.makedirs(save_path, exist_ok=True)

    # build model
    print("🔧 Model bouwen...")
    model = build_network(opt)

    # load weights
    weights_path = opt['path']['pretrain_network_g']

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"❌ Weights niet gevonden: {weights_path}")

    print(f"📦 Laden van weights: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)

    model.load_state_dict(
        state_dict[opt['path']['param_key_g']],
        strict=opt['path']['strict_load_g']
    )

    model = model.to(device).eval()

    print("✅ Model klaar!")

    # =========================
    # 📸 LOAD IMAGES (.jp2)
    # =========================
    print("DATA DIR:", data_dir)
    print("FILES:", os.listdir(data_dir))
    print("📂 data_dir:", data_dir)
    print("📂 bestaat:", os.path.exists(data_dir))

    print("📂 alle files:", os.listdir(data_dir))

    # 🔥 BETERE GLOB
    images = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".jp2")
    ]

    print("📂 glob resultaat:", images[:5])

    print(f"📸 {len(images)} JP2 images gevonden")

    if len(images) == 0:
        raise ValueError("❌ Geen JP2 images gevonden")

    # =========================
    # 🔥 INFERENCE
    # =========================
    for i, path in enumerate(images):

        print(f"\n🟢 Processing ({i+1}/{len(images)}): {path}")

        name = os.path.splitext(os.path.basename(path))[0]
        save_dir = os.path.join(save_path, name)
        os.makedirs(save_dir, exist_ok=True)

        # 🔥 JP2 → correcte RGB
        im = load_sentinel_rgb(path)

        # resize naar model input (32x32)
        im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)

        # preprocess (NIET zelf normalizen verder!)
        input_tensor, s2_image = format_s2naip_data(im, n_lr_images, device)

        # inference
        with torch.no_grad():
            output = model(input_tensor)

        # save LR
        skimage.io.imsave(os.path.join(save_dir, 'lr.png'), s2_image)

        # postprocess
        output = torch.clamp(output, 0, 1)
        output = output.squeeze().cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255).astype(np.uint8)

        # save SR
        skimage.io.imsave(
            os.path.join(save_dir, 'sr.png'),
            output,
            check_contrast=False
        )

    print("\n🎉 Done! Inference klaar.")


if __name__ == "__main__":
    main()