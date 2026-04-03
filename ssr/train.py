# flake8: noqa
import sys
import os

# ✅ FIX path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import torch
import logging
import time

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import (
    AvgTimer, MessageLogger, get_env_info,
    get_root_logger, get_time_str, make_exp_dirs
)

# lokale imports
import archs
import data
import models


def train_pipeline():
    # -------------------------
    # ARGPARSE + YAML LOAD
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True)
    args = parser.parse_args()

    print("📄 YAML:", args.opt)

    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)

    if opt is None:
        raise ValueError("❌ YAML niet geladen")

    # -------------------------
    # BASIC SETTINGS
    # -------------------------
    opt['dist'] = False
    opt['is_train'] = True
    opt['num_gpu'] = 0

    torch.backends.cudnn.benchmark = True

    # -------------------------
    # LOGGING
    # -------------------------
    make_exp_dirs(opt)

    log_file = os.path.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger('basicsr', logging.INFO, log_file)

    logger.info(get_env_info())
    logger.info(opt)

    # -------------------------
    # DATASET
    # -------------------------
    train_set = build_dataset(opt['datasets']['train'])

    train_loader = build_dataloader(
        train_set,
        opt['datasets']['train'],
        num_gpu=0,
        dist=False,
        sampler=None,
        seed=0
    )

    logger.info(f"Train samples: {len(train_set)}")

    # -------------------------
    # MODEL
    # -------------------------
    model = build_model(opt)

    current_iter = 0
    total_iter = opt['train']['total_iter']

    # -------------------------
    # TRAIN LOOP (SIMPEL)
    # -------------------------
    logger.info("🚀 Start training")

    data_timer = AvgTimer()
    iter_timer = AvgTimer()

    while current_iter < total_iter:
        for train_data in train_loader:
            current_iter += 1

            # ---- train step ----
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)

            # ---- logging ----
            if current_iter % 50 == 0:
                log = model.get_current_log()
                logger.info(f"Iter {current_iter}: {log}")

            # ---- save ----
            if current_iter % 500 == 0:
                logger.info("💾 saving model")
                model.save(epoch=0, current_iter=current_iter)

            if current_iter >= total_iter:
                break

    # -------------------------
    # SAVE FINAL
    # -------------------------
    logger.info("✅ Training klaar")
    model.save(epoch=-1, current_iter=-1)


if __name__ == '__main__':
    train_pipeline()