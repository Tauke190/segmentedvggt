# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torch.nn as nn

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2

import sys
import os
import onnxruntime


from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map , unproject_depth_map_to_segmented_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import matplotlib.pyplot as plt
import matplotlib



def visualize_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, class_colors=None):
    """
    Overlay a segmentation mask on top of an image.

    Args:
        image (np.ndarray): (H, W, 3) RGB image, values in [0, 1] or [0, 255]
        mask (np.ndarray): (H, W) integer mask (class indices)
        alpha (float): Transparency for the mask overlay.
        class_colors (np.ndarray): (num_classes, 3) array of RGB colors for each class.
    """
    if image.max() > 1.0:
        image = image / 255.0
    if class_colors is None:
        # Use tab20 colormap for up to 81 classes
        cmap = (plt.cm.get_cmap('tab20', 81).colors)
        class_colors = np.array(cmap)[:, :3]
    mask_rgb = class_colors[mask % len(class_colors)]
    mask_rgb = mask_rgb[..., :3]
    mask_rgb = mask_rgb.astype(np.float32)
    mask_rgb = mask_rgb / mask_rgb.max()
    overlay = (1 - alpha) * image + alpha * mask_rgb
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(overlay)
    plt.show()



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

def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
    prompt: str = None,
    seg_threshold: float = 0.5,  # <--- added
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]          # (S,H,W,1)
    depth_conf = pred_dict["depth_conf"]    # (S,H,W)
    S, H, W, _ = depth_map.shape

    # Get segmentation mask (already thresholded) shape [S,1,H,W]
    seg_mask_ch_first = pred_dict.get("segmentation_logits", None)
    if seg_mask_ch_first is None:
        print("Segmentation mask not found; proceeding with depth only.")
        depth_map_seg = depth_map  # fall back
    else:
        # Move channel axis to last to match depth_map (S,H,W,1)
        # seg_mask_ch_first: (S,81,H,W)
        seg_class = np.argmax(seg_mask_ch_first, axis=1)  # (S,H,W)
        seg_class_last = seg_class[..., None].astype(depth_map.dtype)  # (S,H,W,1)

        # Concatenate -> (S,H,W,2): channel 0 depth, channel 1 predicted class
        depth_map_seg = np.concatenate([depth_map, seg_class_last], axis=-1)
        print("depth_map_seg shape:", depth_map_seg.shape)

    # Compute world points
    if not use_point_map:
        if depth_map_seg.shape[-1] == 2: # Depth map with segmentation
            world_points = unproject_depth_map_to_segmented_point_map(
                depth_map_seg, pred_dict["extrinsic"], pred_dict["intrinsic"]
            )
            print("world_points shape after unprojection:", world_points.shape)
        else:
            world_points = unproject_depth_map_to_point_map(
                depth_map, pred_dict["extrinsic"], pred_dict["intrinsic"]
            )
        print("world_points shape after unprojection:", world_points.shape)
        conf = depth_conf
    else:
        world_points = pred_dict["world_points"]
        conf = pred_dict["world_points_conf"]

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    if world_points.shape[-1] > 3:
        points = world_points[..., :3].reshape(-1, 3)  # Only xyz
    else:
        points = world_points.reshape(-1, 3)
    # No mask channel in world_points
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)


    # --- Segmentation class coloring ---
    if seg_mask_ch_first is not None:
        # If seg_mask_ch_first is (S,81,H,W), get class per pixel
        seg_class = np.argmax(seg_mask_ch_first, axis=1)  # (S,H,W)
        seg_class_flat = seg_class.reshape(-1)            # (N,)
        # Colormap for 81 classes
        import matplotlib.pyplot as plt
        cmap = (plt.cm.get_cmap('tab20', 81).colors * 255).astype(np.uint8)  # (81,4)
        cmap = cmap[:, :3]  # Drop alpha
        seg_colors_flat = cmap[seg_class_flat]  # (N,3)
    else:
        seg_colors_flat = colors_flat.copy()

    # Also keep original RGB colors
    rgb_colors_flat = colors_flat.copy()

    # Segmentation probability for thresholding (if available)
    if seg_mask_ch_first is not None:
        # If seg_mask_ch_first is (S,81,H,W), get max prob per pixel
        seg_prob_flat = np.max(seg_mask_ch_first, axis=1).reshape(-1)
    else:
        seg_prob_flat = np.zeros((colors_flat.shape[0],), dtype=np.float32)

    # GUI toggle for color mode
    color_mode_options = ["RGB", "Segmentation"]
    gui_color_mode = server.gui.add_dropdown(
        "Color mode", options=color_mode_options, initial_value="Segmentation"
    )

    def get_current_colors() -> np.ndarray:
        if gui_color_mode.value == "Segmentation":
            return seg_colors_flat
        else:
            return rgb_colors_flat

    print("seg_colors_flat shape:", seg_colors_flat.shape)

    # --- FIX: define cam_to_world, frame_indices, and points_centered ---
    cam_to_world = pred_dict.get("extrinsic")  # (S, 3, 4)

    # Per-point frame indices for filtering (S * H * W,)
    frame_indices = np.repeat(np.arange(S), H * W)

    # Center the scene so points and camera frustums align
    valid_pts = np.isfinite(points).all(axis=1)
    if np.any(valid_pts):
        center = points[valid_pts].mean(axis=0)
    else:
        center = np.zeros(3, dtype=points.dtype)
    points_centered = points - center
    # ---------------------------------------------------------------

    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.2)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=get_current_colors()[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )
    #---------------------->

    # GUI controls for filtering and visualization
    gui_points_conf = server.gui.add_slider(
        "Confidence percentile", 0.0, 100.0, step=1.0, initial_value=float(init_conf_threshold)
    )
    frame_options = ["All"] + [str(i) for i in range(S)]
    gui_frame_selector = server.gui.add_dropdown(
        "Frame", options=frame_options, initial_value="All"
    )
    gui_show_frames = server.gui.add_checkbox(
        "Show camera frames", True
    )
    gui_seg_threshold = server.gui.add_slider(
        "Segmentation threshold", 0.0, 1.0, step=0.01, initial_value=float(seg_threshold)
    )
    gui_filter_by_seg = server.gui.add_checkbox(
        "Filter points by segmentation", False
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis (shifted by the same center)
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation() - center,  # shift to match points_centered
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        # Segmentation-based filtering (optional)
        if gui_filter_by_seg.value:
            seg_mask_ok = seg_prob_flat >= float(gui_seg_threshold.value)
        else:
            seg_mask_ok = np.ones_like(conf_mask, dtype=bool)

        combined_mask = conf_mask & frame_mask & seg_mask_ok

        # Choose color mode
        current_colors = get_current_colors()

        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = current_colors[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value


    @gui_seg_threshold.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_filter_by_seg.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_color_mode.on_update
    def _(_) -> None:
        update_point_cloud()

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    # Save the filtered/confident points (same as used for visualization)
    save_mask = (conf_flat >= np.percentile(conf_flat, init_conf_threshold)) & (conf_flat > 0.2)
    # save_point_cloud_as_ply("output_pointcloud_segmented.ply", points_centered[save_mask], colors_with_mask[save_mask])
    # print("Saved segmented 3D point cloud to output_pointcloud_segmented.ply")

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server

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


parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/test/images/", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
parser.add_argument(
    "--seg_threshold", type=float, default=0.5, help="Threshold for segmentation probability in [0,1]"
)
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path or URL to a custom VGGT checkpoint. If not set, the default pretrained 1B checkpoint is used."
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

def reinit_segmentation_head(model):
    if model.segmentation_head is not None:
        for m in model.segmentation_head.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    model = VGGT(num_seg_classes=81)  # enable_segmentation=True by default in your VGGT
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

    print("Loading default pretrained checkpoint...")
    load_with_strict_false(model, _URL)

    # reinit_segmentation_head(model)

    # Optionally replace only the segmentation head from a custom checkpoint
    if args.checkpoint:
        print(f"Replacing segmentation head from custom checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        result = model.segmentation_head.load_state_dict(checkpoint["segmentation_head"])
        print("Segmentation head loaded. Missing keys:", result.missing_keys)
        print("Segmentation head loaded. Unexpected keys:", result.unexpected_keys)
        if not result.missing_keys and not result.unexpected_keys:
            print("Segmentation head successfully replaced!")
        else:
            print("Segmentation head replacement had issues. See above.")

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*"))
    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)


    # Build torch seg probabilities BEFORE squeezing to numpy
    seg_prob = None
    if "segmentation_logits" in predictions:
        seg_logits = predictions["segmentation_logits"]          # [B,S,81,H,W] or [B,S,1,H,W]
        if seg_logits.shape[2] == 1:
            seg_prob = torch.sigmoid(seg_logits)                 # [B,S,1,H,W]
        else:
            seg_prob = torch.softmax(seg_logits, dim=2)          # [B,S,81,H,W]
        print("seg_prob (torch) shape:", seg_prob.shape)
    else:
        print("No segmentation head output found.")

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dim


    # Also convert seg_prob to numpy (matching squeeze pattern) and store
    if seg_prob is not None:
        seg_prob_np = seg_prob.cpu().numpy().squeeze(0)  # [S,81,H,W] or [S,1,H,W]
        predictions["segmentation_logits"] = seg_prob_np
        print("seg_prob (numpy) shape:", seg_prob_np.shape)

        # Visualize mask on top of the first image
        if seg_prob_np.shape[1] > 1:
            seg_class = np.argmax(seg_prob_np, axis=1)  # (S, H, W)
        else:
            seg_class = (seg_prob_np > 0.5).astype(np.int32).squeeze(1)  # (S, H, W)
        img0 = images[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        mask0 = seg_class[0]
        visualize_mask_on_image(img0, mask0)
    else:
        print("Segmentation probabilities not available.")

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")

    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
        seg_threshold=args.seg_threshold,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()

