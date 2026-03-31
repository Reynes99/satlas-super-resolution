import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import torch
import argparse
import skimage.io
import numpy as np

from ssr.utils.infer_utils import format_s2naip_data
from ssr.utils.options import yaml_load
from ssr.utils.model_utils import build_network


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path naar YAML config')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("⚠️ GPU niet beschikbaar, fallback naar CPU")

    torch.backends.cudnn.benchmark = True

    # ====== LOAD CONFIG ======
    opt = yaml_load(args.opt)

    n_lr_images = opt['n_lr_images']
    data_dir = opt['data_dir']
    save_path = opt['save_path']

    os.makedirs(save_path, exist_ok=True)

    # ====== BUILD MODEL ======
    print("🔧 Model bouwen...")
    model = build_network(opt)

    # ====== LOAD WEIGHTS ======
    if 'pretrain_network_g' not in opt['path']:
        raise ValueError("❌ Geen weights gevonden in YAML")

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
    print(model)

    # ====== LOAD IMAGES ======
    images = glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True) + \
             glob.glob(os.path.join(data_dir, "**/*.jpg"), recursive=True) + \
             glob.glob(os.path.join(data_dir, "**/*.jpeg"), recursive=True)

    print(f"📸 {len(images)} images gevonden")

    if len(images) == 0:
        raise ValueError("❌ Geen images gevonden in data_dir")

    # ====== INFERENCE ======
    for i, path in enumerate(images):

        print(f"\n🟢 Processing ({i+1}/{len(images)}): {path}")

        name = os.path.splitext(os.path.basename(path))[0]
        save_dir = os.path.join(save_path, name)
        os.makedirs(save_dir, exist_ok=True)

        # ====== READ IMAGE ======
        import cv2
        im = skimage.io.imread(path)

        # 🔥 CRUCIAAL: model verwacht 32x32
        if im.shape[:2] != (32, 32):
            im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)

        # 🔥 GEEN resize doen!
        if im is None:
            print(f"⚠️ Kon image niet laden: {path}")
            continue

        if im.ndim != 3 or im.shape[2] != 3:
            print(f"⚠️ Skip (geen RGB): {path}")
            continue

        # ====== PREPROCESS ======
        input_tensor, s2_image = format_s2naip_data(im, n_lr_images, device)

        # ====== INFERENCE ======
        with torch.no_grad():
            output = model(input_tensor)

        # ====== SAVE LR ======
        skimage.io.imsave(os.path.join(save_dir, 'lr.png'), s2_image)

        # ====== POSTPROCESS ======
        output = torch.clamp(output, 0, 1)
        output = output.squeeze().cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255).astype(np.uint8)

        # ====== SAVE SR ======
        skimage.io.imsave(
            os.path.join(save_dir, 'sr.png'),
            output,
            check_contrast=False
        )

    print("\n🎉 Done! Inference klaar.")


if __name__ == "__main__":
    main()