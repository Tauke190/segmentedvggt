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
    print(f"Preprocessed images shape: {images.shape}")

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
