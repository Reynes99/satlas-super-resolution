# flake8: noqa
import sys
import os

# ✅ FIX: correcte root toevoegen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import logging
import yaml

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str

# ✅ lokale imports (geen ssr. meer)
import archs
import data
import models


def test_pipeline():
    # -------------------------
    # ARGPARSE
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path naar YAML config')
    args = parser.parse_args()

    # -------------------------
    # LOAD YAML (zonder BasicSR parser)
    # -------------------------
    print("📄 YAML PATH:", args.opt)

    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)

    print("📦 YAML LOADED:", opt is not None)

    if opt is None:
        raise ValueError("❌ YAML kon niet geladen worden")

    # -------------------------
    # BASIC SETTINGS
    # -------------------------
    opt['dist'] = False
    opt['num_gpu'] = 0
    opt['is_train'] = False

    torch.backends.cudnn.benchmark = True

    # -------------------------
    # LOGGING
    # -------------------------
    make_exp_dirs(opt)

    log_file = os.path.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)

    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # -------------------------
    # DATASET
    # -------------------------
    test_loaders = []

    for _, dataset_opt in sorted(opt['test_datasets'].items()):
        test_set = build_dataset(dataset_opt)

        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt.get('manual_seed', 0)
        )

        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # -------------------------
    # MODEL
    # -------------------------
    model = build_model(opt)

    # -------------------------
    # TEST LOOP
    # -------------------------
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')

        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['test']['save_img']
        )


if __name__ == '__main__':
    test_pipeline()