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

import sys
import os
import onnxruntime
from clipseg.multiclass_segmentor import get_multiclass_segmentation_tensor_mask, visualize_tensor
from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

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


# Fixed dataset root for multiview finetuning
DATASET_ROOT = "/dataset/train"

def find_scene_prompt_and_images(scene_dir: str):
    """
    Locate prompt.txt and images directory for a scene.
    Returns (prompt_string, images_dir_path).
    Accepts:
      scene_dir/prompt.txt + scene_dir/images/
      OR scene_dir/images/prompt.txt + scene_dir/images/
    """
    candidate_prompt_paths = [
        os.path.join(scene_dir, "prompt.txt"),
        os.path.join(scene_dir, "images", "prompt.txt"),
    ]
    prompt_path = next((p for p in candidate_prompt_paths if os.path.isfile(p)), None)
    if prompt_path is None:
        raise FileNotFoundError(f"No prompt.txt found in {scene_dir} or its images/ subfolder.")
    # images directory
    images_dir = os.path.join(scene_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found at {images_dir}")
    with open(prompt_path, "r") as f:
        # Use first non-empty line; allow comma-separated prompts
        lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
            raise ValueError(f"prompt.txt at {prompt_path} is empty.")
        prompt = lines[0]
    return prompt, images_dir

def list_scenes(dataset_root: str):
    """
    Returns list of absolute scene directories under dataset_root.
    """
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    scenes = []
    for name in sorted(os.listdir(dataset_root)):
        full = os.path.join(dataset_root, name)
        if os.path.isdir(full):
            scenes.append(full)
    if not scenes:
        raise ValueError(f"No scene folders found in {dataset_root}")
    return scenes

def get_prompt_for_folder(image_folder: str) -> str:
    # Try image_folder/prompt.txt and parent/prompt.txt
    candidates = [
        os.path.join(image_folder, "prompt.txt"),
        os.path.join(os.path.dirname(image_folder.rstrip("/")), "prompt.txt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        return line
    return DEFAULT_CLIPSEG_PROMPT  # fallback

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

    # Collect scenes
    scenes = list_scenes(DATASET_ROOT)
    print(f"Found {len(scenes)} scenes under {DATASET_ROOT}")

    for scene_idx, scene_dir in enumerate(scenes, 1):
        prompt, images_dir = find_scene_prompt_and_images(scene_dir)
        print(f"[Scene {scene_idx}/{len(scenes)}] prompt='{prompt}' images_dir='{images_dir}'")

        image_names = sorted(glob.glob(os.path.join(images_dir, "*")))
        if not image_names:
            print(f"Skipping empty scene {scene_dir}")
            continue

        images = load_and_preprocess_images(image_names).to(device)
        # Normalize to [1,S,3,H,W]
        if images.dim() == 4:
            images = images.unsqueeze(0)
        elif images.dim() == 5:
            if images.size(0) == 1 and images.size(2) == 3:
                pass
            elif images.size(1) == 1 and images.size(2) == 3:
                images = images.squeeze(1).unsqueeze(0)
            else:
                print(f"Skipping scene (unrecognized shape): {tuple(images.shape)}")
                continue
        else:
            print(f"Skipping scene (unexpected tensor rank): {tuple(images.shape)}")
            continue

        if images.dim() != 5 or images.size(0) != 1:
            print(f"Skipping scene (shape mismatch): {tuple(images.shape)}")
            continue

        _, S, _, H, W = images.shape

        # Build GT masks
        gt_masks = build_clipseg_gt_masks(
            images_dir,
            (H, W),
            prompt,
            DEFAULT_CLIPSEG_CLASS_INDEX,
            device,
        )  # [1,S,1,H,W]

        if gt_masks.shape[1] != S:
            print(f"Skipping scene (mask/image count mismatch {gt_masks.shape[1]} vs {S})")
            continue

        if args.viz_clipseg_masks:
            viz_in = gt_masks[0].cpu().squeeze(1)
            viz_name = f"clipseg_masks_scene_{scene_idx}.png"
            visualize_tensor(viz_in, save_path=viz_name, image_folder=images_dir)
            print(f"Saved CLIPSeg masks visualization to {viz_name}")

        # Train for this scene
        model.train()
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            if "segmentation_logits" not in out:
                raise RuntimeError("Model did not produce 'segmentation_logits'.")
            logits = out["segmentation_logits"]  # [1,S,1,H,W]
            loss = F.binary_cross_entropy_with_logits(logits, gt_masks)
            loss.backward()
            optimizer.step()
            print(f"[scene {scene_idx} epoch {epoch}/{args.epochs}] loss={loss.item():.4f}")

    # Save after all scenes
    torch.save({"model": model.state_dict()}, args.save_path)
    print(f"Saved trained weights to {args.save_path}")

    # Optional final inference on last processed scene (if any)
    print("Training complete.")

if __name__ == "__main__":
    main()
