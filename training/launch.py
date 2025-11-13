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
import importlib.util
from pathlib import Path

# Add repo root so `import clipseg` works everywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Optional: sanity check
if importlib.util.find_spec("clipseg") is None:
    print(f"[WARN] 'clipseg' not importable. sys.path includes: {REPO_ROOT}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, help="YAML config path", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)
    # inject single_gpu flag if trainer expects it
    cfg.single_gpu = args.single_gpu
    # Trainer(cfg)


if __name__ == "__main__":
    main()


