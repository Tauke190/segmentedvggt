# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import gc
import matplotlib.pyplot as plt

import sys
import os
import onnxruntime
from clipseg.multiclass_segmentor import get_multiclass_segmentation_tensor_mask, visualize_tensor
from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
from dataset import COCOSegmentation


TRAIN_PATH = "/home/av354855/data/datasets/coco/train2017"
TRAIN_ANN_FILE = "/home/av354855/data/datasets/coco/annotations/instances_train2017.json"


parser = argparse.ArgumentParser(description="VGGT segmentation head training")
parser.add_argument("--epochs", type=int, default=5, help="Number of finetuning epochs per scene")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for finetuning")
parser.add_argument("--save_path", type=str, default="vggt_seg_finetuned.pt", help="Where to save finetuned weights")
parser.add_argument("--train_path", type=str, default=TRAIN_PATH,required=False, help="Path to COCO training images directory")
parser.add_argument("--annotation_path", type=str,default=TRAIN_ANN_FILE, required=False, help="Path to COCO training annotation file")

# Fixed defaults for training (since CLI args removed)
DEFAULT_CLIPSEG_PROMPT = "vehicle"
DEFAULT_CLIPSEG_CLASS_INDEX = 0

def load_with_strict_false(model, url_or_path: str):
    if os.path.isfile(url_or_path):
        state = torch.load(url_or_path, map_location="cpu")
    else:
        state = torch.hub.load_state_dict_from_url(url_or_path, map_location="cpu")
    # unwrap common wrappers
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strip 'module.' if present
    new_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    msg = model.load_state_dict(new_state, strict=False)
    print("Loaded checkpoint with strict=False")
    print("Missing keys:", msg.missing_keys)       # will include segmentation_head.*
    print("Unexpected keys:", msg.unexpected_keys)
    return msg

def coco_collate_fn(batch):
    # batch: list of (image, target) tuples
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    images = torch.stack(images)
    # Each mask is [H, W] with class indices; stack as [B, H, W]
    masks = torch.stack([m.squeeze().long() for m in masks])
    return images, masks

# -----------------------------
#  Transforms for image + mask
# -----------------------------
def coco_transform(image, mask, size=(256, 256)):
    image = image.resize(size, Image.BILINEAR)
    mask = mask.resize(size, Image.NEAREST)
    image = T.ToTensor()(image)
    mask = torch.from_numpy(np.array(mask)).long()
    return image, mask

def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    model = VGGT(num_seg_classes=81)  # <-- Ensure correct number of classes
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    load_with_strict_false(model, _URL)
    model = model.to(device)

    # Freeze backbone, allow segmentation head
    for p in model.parameters():
        p.requires_grad = False
    if not hasattr(model, "segmentation_head"):
        raise AttributeError("Model missing segmentation_head.")
    for p in model.segmentation_head.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.segmentation_head.parameters(), lr=args.lr)
    print(f"Optimizer initialized (lr={args.lr})")

    train_img_dir = args.train_path
    train_ann_file = args.annotation_path

    train_dataset = COCOSegmentation(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        transforms=lambda img, msk: coco_transform(img, msk, size=(252, 252))
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"COCO train dataset size: {len(train_dataset)}")
    print("Number of classes in dataset:", len(train_dataset.cat_id_to_index))

    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images = images.to(device)
            masks = masks.to(device)  # [B, H, W], dtype long
            optimizer.zero_grad(set_to_none=True)

            out = model(images)
            logits = out["segmentation_logits"]  # [1, 1, 91, 252, 252]
            if logits.dim() == 5 and logits.shape[1] == 1:
                logits = logits.squeeze(1)  # [B, C, H, W]
            elif logits.dim() == 5 and logits.shape[1] > 1:
                # Merge batch and sequence: [B, S, C, H, W] -> [B*S, C, H, W]
                B, S, C, H, W = logits.shape
                logits = logits.view(B * S, C, H, W)

            # Resize masks if needed to match logits
            if logits.shape[-2:] != masks.shape[-2:]:
                masks = F.interpolate(masks.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest")
                masks = masks.squeeze(1).long()

            # Always ensure masks is [B, H, W] before loss
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            # Debugging shapes
            if batch_idx == 0:
                print("images.shape:", images.shape)
                print("logits.shape:", logits.shape)  # should be [B, C, H, W]
                print("masks.shape:", masks.shape)    # should be [B, H, W]

            loss = criterion(logits, masks)

            loss.backward()           # <-- Add this line
            optimizer.step()          # <-- And this line

            # --- Visualization every 500 batches ---
            if batch_idx % 500 == 0:
                with torch.no_grad():
                    # Take the first sample in the batch
                    pred_mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    gt_mask = masks[0].cpu().numpy().astype(np.uint8)
                    img = images[0].cpu()
                    # Convert image tensor to PIL
                    img_pil = T.ToPILImage()(img)
                    # Plot
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(img_pil)
                    plt.title("Image")
                    plt.axis("off")
                    plt.subplot(1, 3, 2)
                    plt.imshow(gt_mask, cmap="nipy_spectral")
                    plt.title("GT Mask")
                    plt.axis("off")
                    plt.subplot(1, 3, 3)
                    plt.imshow(pred_mask, cmap="nipy_spectral")
                    plt.title("Predicted Mask")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(f"seg_pred_vs_gt_batch{batch_idx}_epoch{epoch}.png")
                    plt.close()

        print(f"[epoch {epoch}/{args.epochs}] avg loss={epoch_loss/len(train_loader):.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    torch.save(
        {"segmentation_head": model.segmentation_head.state_dict()},
        args.save_path
    )
    print(f"Saved segmentation_head weights to {args.save_path}")
    print("Training complete.")

if __name__ == "__main__":
    main()
