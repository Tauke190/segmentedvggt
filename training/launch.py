# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import yaml
import os
import sys
import torch
from omegaconf import OmegaConf
from trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, help="YAML config path", required=True)
    p.add_argument("--single_gpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)
    # inject single_gpu flag if trainer expects it
    cfg.single_gpu = args.single_gpu
    Trainer(cfg)


if __name__ == "__main__":
    main()


