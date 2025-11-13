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

# Add clipseg folder to Python path
clipseg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../clipseg'))
sys.path.append(clipseg_path)

# Try importing CLIPSeg helpers
try:
    from clipseg.multiclass_segmentor import get_multiclass_segmentation_tensor_mask, visualize_tensor
    CLIPSEG_AVAILABLE = True
except Exception:
    CLIPSEG_AVAILABLE = False

try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

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


parser = argparse.ArgumentParser(description="VGGT demo (inference only, no visualization)")
parser.add_argument(
    "--image_folder", type=str, default="examples/test/images/", help="Path to folder containing images"
)
# --- added args for finetuning with CLIPSeg ---
parser.add_argument("--finetune_seg", action="store_true", help="Fine-tune the segmentation head on masks")
parser.add_argument("--clipseg_prompt", type=str, default="vehicle", help="Comma-separated prompt(s) for CLIPSeg (e.g., 'sky,road')")
parser.add_argument("--clipseg_class_index", type=int, default=0, help="Class index to supervise when CLIPSeg returns multi-class")
parser.add_argument("--epochs", type=int, default=5, help="Number of finetuning epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for finetuning")
parser.add_argument("--save_path", type=str, default="vggt_seg_finetuned.pt", help="Where to save finetuned weights")
parser.add_argument("--viz_clipseg_masks", action="store_true", help="Save a visualization of CLIPSeg masks")

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
    if not CLIPSEG_AVAILABLE:
        raise ImportError("clipseg.multiclass_segmentor not available. Ensure clipseg is on PYTHONPATH.")

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


def main():
    """
    Main function for VGGT inference (no visualization).
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    model = VGGT()  # enable_segmentation=True by default in your VGGT
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    load_with_strict_false(model, _URL)

    model.eval()
    model = model.to(device)

    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*"))
    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape (raw): {images.shape}")

    # Ensure shape [1, S, 3, H, W]; if loader returned [1,3,H,W], add S=1
    if images.dim() == 4:            # [B,3,H,W]
        images = images.unsqueeze(1) # [B,1,3,H,W]
        print("Added sequence dimension S=1.")
    elif images.dim() == 5 and images.size(0) != 1:
        # If loader returned [S,3,H,W], convert to [1,S,3,H,W]
        images = images.unsqueeze(0)
        print("Added batch dimension B=1.")
    elif images.dim() != 5:
        raise ValueError(f"Unexpected images tensor shape {images.shape}")

    print(f"Preprocessed images shape (final): {images.shape}")

    # -------- optional finetuning of segmentation head using CLIPSeg masks --------
    if args.finetune_seg:
        # Expect input images as [B=1,S,3,H,W]
        if images.dim() != 5 or images.size(0) != 1:
            raise ValueError(f"Expected images shape [1,S,3,H,W], got {tuple(images.shape)}")
        _, S, _, H, W = images.shape

        # Build GT masks from CLIPSeg for the provided prompt(s)
        gt_masks = build_clipseg_gt_masks(
            args.image_folder,
            (H, W),
            args.clipseg_prompt,
            args.clipseg_class_index,
            device,
        )  # [1,S,1,H,W]

        if gt_masks.shape[1] != S:
            raise ValueError(f"CLIPSeg produced {gt_masks.shape[1]} masks, but {S} images were loaded.")

        # Optional visualization
        if args.viz_clipseg_masks and CLIPSEG_AVAILABLE:
            try:
                # visualize_tensor expects [S,H,W] or [S,1,H,W]
                viz_in = gt_masks[0].cpu()  # [S,1,H,W]
                visualize_tensor(viz_in, save_path="clip_seg_masks_viz.png", image_folder=args.image_folder)
                print("Saved CLIPSeg masks visualization to clip_seg_masks_viz.png")
            except Exception as e:
                print(f"Failed to visualize CLIPSeg masks: {e}")

        model.train()
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            out = model(images)  # logits at [1,S,1,H,W] expected
            if "segmentation_logits" not in out:
                raise RuntimeError("Model did not produce 'segmentation_logits'.")
            logits = out["segmentation_logits"]  # [1,S,1,H,W]
            loss = F.binary_cross_entropy_with_logits(logits, gt_masks)
            loss.backward()
            optimizer.step()
            print(f"[seg finetune] epoch {epoch}/{args.epochs} loss={loss.item():.4f}")

        # Save finetuned weights
        torch.save({"model": model.state_dict()}, args.save_path)
        print(f"Saved finetuned weights to {args.save_path}")

        model.eval()

    print("Running inference...")
    # Use autocast only on CUDA; avoid CUDA-only calls on CPU
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
    else:
        with torch.no_grad():
            predictions = model(images)

    # Build torch seg mask BEFORE squeezing to numpy
    if "segmentation_logits" in predictions:
        seg_logits = predictions["segmentation_logits"]          # [B,S,1,H,W]
        seg_mask = (torch.sigmoid(seg_logits) > 0.5).float()     # binary
        print("seg_mask (torch) shape:", seg_mask.shape)         # [B,S,1,H,W]
    else:
        seg_mask = None
        print("No segmentation head output found.")

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dim

    if seg_mask is not None:
        seg_mask_np = seg_mask.cpu().numpy().squeeze(0)  # [S,1,H,W]
        predictions["segmentation_logits"] = seg_mask_np
        print("seg_mask (numpy) shape:", seg_mask_np.shape)

    # No visualization; just report and exit
    print("Inference complete.")
    print({
        "images": predictions["images"].shape if "images" in predictions else None,
        "depth": predictions.get("depth", None).shape if "depth" in predictions else None,
        "pose_enc": predictions.get("pose_enc", None).shape if "pose_enc" in predictions else None,
    })

if __name__ == "__main__":
    main()
