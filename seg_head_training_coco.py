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

import sys
import os
import onnxruntime
from clipseg.multiclass_segmentor import get_multiclass_segmentation_tensor_mask, visualize_tensor
from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from torchvision.datasets import CocoSegmentation
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO

def save_point_cloud_as_ply(filename, points, colors):
    """
    Save a point cloud to a PLY file.

    Args:
        filename (str): Output file path.
        points (np.ndarray): (N, 3) array of XYZ coordinates.
        colors (np.ndarray): (N, 3) array of RGB colors (uint8).
    """
    assert points.shape[0] == colors.shape[0]
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
# Helper functions for sky segmentation
def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf

parser = argparse.ArgumentParser(description="VGGT segmentation head training")
parser.add_argument("--epochs", type=int, default=5, help="Number of finetuning epochs per scene")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for finetuning")
parser.add_argument("--save_path", type=str, default="vggt_seg_finetuned.pt", help="Where to save finetuned weights")
parser.add_argument("--viz_clipseg_masks", action="store_true", help="Save a visualization of CLIPSeg masks")

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

# --- helper: build GT masks from CLIPSeg for given folder and prompts ---
def build_clipseg_gt_masks(image_folder: str, target_hw, prompt: str, class_index: int, device: str) -> torch.Tensor:
    """
    Returns a tensor of shape [1, S, 1, H, W] with binary masks aligned to model input size.
    """

    # Allow comma-separated prompts -> list
    prompt_arg = [p.strip() for p in prompt.split(",")] if isinstance(prompt, str) else prompt

    # get_multiclass_segmentation_tensor_mask(prompt, image_folder) is expected to return
    # a tensor shaped [S, H, W] for single prompt, or [S, C, H, W] for multiple prompts/classes.
    seg_masks = get_multiclass_segmentation_tensor_mask(prompt_arg, image_folder)  # torch.Tensor
    if not torch.is_tensor(seg_masks):
        seg_masks = torch.as_tensor(seg_masks)

    # Normalize dimensions to [S, C, H, W]
    if seg_masks.ndim == 3:
        # [S, H, W] -> [S, 1, H, W]
        seg_masks = seg_masks.unsqueeze(1)
    elif seg_masks.ndim == 4:
        # choose class/channel if multi-class
        if seg_masks.shape[1] > 1:
            seg_masks = seg_masks[:, class_index:class_index+1, ...]
    else:
        raise ValueError(f"Unexpected CLIPSeg mask shape: {tuple(seg_masks.shape)}")

    # Convert to float, threshold to binary if not already
    seg_masks = seg_masks.float()
    # If values are not {0,1}, threshold at 0.5
    with torch.no_grad():
        if seg_masks.max() > 1.0 or seg_masks.min() < 0.0:
            seg_masks = (seg_masks - seg_masks.min()) / (seg_masks.max() - seg_masks.min() + 1e-6)
        seg_masks = (seg_masks >= 0.5).float()

    # Resize to target HxW using nearest to keep binary labels
    Ht, Wt = target_hw
    seg_masks = F.interpolate(seg_masks, size=(Ht, Wt), mode="nearest")  # [S,1,Ht,Wt]

    # Add batch and return [1,S,1,H,W] on device
    return seg_masks.unsqueeze(0).to(device)



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

# ------------------------------------
#   COCO Semantic Segmentation Dataset
# ------------------------------------
class COCOSegmentation(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        # Load image metadata
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        
        # Load RGB image
        image = Image.open(img_path).convert("RGB")

        # Load all annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        # Create empty mask (H x W)
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

        # Fill mask: category_id per pixel
        for ann in anns:
            m = self.coco.annToMask(ann)
            mask = np.maximum(mask, m * ann["category_id"])

        mask = Image.fromarray(mask)

        # Apply transform
        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask

def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    model = VGGT()
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

    # --- COCO Dataset ---
    train_img_dir = "/home/c3-0/datasets/coco/train2017"
    train_ann_file = "/home/c3-0/datasets/coco/annotations/instances_train2017.json"

    train_dataset = COCOSegmentation(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        transforms=lambda img, msk: coco_transform(img, msk, size=(256, 256))
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print(f"COCO train dataset size: {len(train_dataset)}")

    # Number of classes in COCO (including background)
    num_classes = 91  # COCO has 80 classes, but mask values can go up to 90

    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images = images.to(device)
            masks = masks.to(device)  # [B, H, W], dtype long
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            if "segmentation_logits" not in out:
                raise RuntimeError("Model did not produce 'segmentation_logits'.")
            logits = out["segmentation_logits"]  # [B, num_classes, H, W]
            # Resize masks if needed to match logits
            if logits.shape[-2:] != masks.shape[-2:]:
                masks = F.interpolate(masks.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"  [batch {batch_idx}] loss={loss.item():.4f}")
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
