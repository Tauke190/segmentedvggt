from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import math


def get_multiclass_segmentation_tensor_mask(prompts, image_folder, threshold=0.5):
    """
    Runs CLIPSeg inference on all images in a folder for given prompts (classes).
    Returns:
        - If num_classes == 1: binary mask (num_images, H, W), values in [0, 1]
        - If num_classes > 1: multiclass mask (num_images, H, W), values in [0, num_classes]
          (0 is background, 1 is first prompt, 2 is second, ...)
    """
    if isinstance(prompts, str):
        prompts = [prompts]  # Make it a list for binary case

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_masks = []

    for img_file in image_files:
        image_path = os.path.join(image_folder, img_file)
        image = Image.open(image_path)
        masks_per_class = []
        for prompt in prompts:
            inputs = processor(text=[prompt], images=[image], padding="max_length", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            preds = torch.sigmoid(outputs.logits)
            orig_size = image.size
            preds_resized = F.interpolate(
                preds.unsqueeze(1),
                size=(orig_size[1], orig_size[0]),
                mode="bilinear",
                align_corners=False
            ).squeeze(1)
            masks_per_class.append(preds_resized.squeeze(0))  # (H, W)
        masks_per_class = torch.stack(masks_per_class, dim=0)  # (num_classes, H, W)
        if len(prompts) == 1:
            # Binary mask: 1 for foreground, 0 for background
            mask_bin = (masks_per_class[0] > threshold).int()  # (H, W)
            all_masks.append(mask_bin)
        else:
            # Multiclass mask: 0 for background, 1 for first prompt, 2 for second, etc.
            mask_int = torch.argmax(masks_per_class, dim=0) + 1  # (H, W), shift indices
            # Set background to 0 where all classes are below threshold
            background = (masks_per_class.max(dim=0).values < threshold)
            mask_int[background] = 0
            all_masks.append(mask_int)
    all_masks = torch.stack(all_masks, dim=0)  # (num_images, H, W)
    return all_masks

def save_tensor(tensor, filename):
    torch.save(tensor, filename+".pt")

def visualize_tensor(tensor, save_path=None):
    seg_masks_np = tensor.cpu().numpy()  # shape: (N, H, W) or (N, H, W, 1)
    if seg_masks_np.ndim == 4:
        seg_masks_np = seg_masks_np.squeeze(-1)

    num_masks = seg_masks_np.shape[0]
    cols = min(3, num_masks)  # up to 3 columns
    rows = math.ceil(num_masks / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    for i in range(num_masks):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.imshow(seg_masks_np[i], cmap='tab20')
        ax.set_title(f"Mask {i}")
        ax.axis('off')

    # Hide any unused subplots
    for j in range(num_masks, rows * cols):
        r, c = divmod(j, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()