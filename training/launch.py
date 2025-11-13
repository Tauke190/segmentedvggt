# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from hydra import initialize, compose
from omegaconf import DictConfig
# from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Config name in training/config (no .yaml) OR a path to a .yaml file"
    )
    args = parser.parse_args()

    cfg_name = args.config
    cfg_path = "config"  # relative to this file

    p = Path(cfg_name)
    if p.suffix in {".yaml", ".yml"} or "/" in cfg_name or "\\" in cfg_name:
        if not p.is_absolute():
            p = (Path(__file__).parent / p).resolve()
        cfg_path = str(p.parent)
        cfg_name = p.stem

    with initialize(version_base=None, config_path=cfg_path):
        cfg: DictConfig = compose(config_name=cfg_name)

    # trainer = Trainer(cfg)
    # trainer.run()


if __name__ == "__main__":
    main()