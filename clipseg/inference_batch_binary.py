import os
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import torch.nn.functional as F

# Set your folder path and prompts
folder_path = r"C:\Users\avina\Desktop\vggt-repo\vggt\examples\kitchen\images"  # 
prompts = ["vehicle"]  # You can add more prompts

# Load processor and model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Gather all image file paths
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

images = [Image.open(img_path) for img_path in image_files]

# Prepare inputs for batch inference
inputs = processor(
    text=prompts * len(images),
    images=images * len(prompts),
    padding="max_length",
    return_tensors="pt"
)



# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Reshape predictions: (num_images, num_prompts, H, W)
num_images = len(images)
num_prompts = len(prompts)
preds = outputs.logits.reshape(num_images, num_prompts, *outputs.logits.shape[1:])
final_masks = torch.sigmoid(preds[:, 0])

target_sizes = [img.size for img in images]  # (width, height) for each image
# Resize each mask tensor to match its corresponding image size
final_masks_resized = []
for mask_tensor, size in zip(final_masks, target_sizes):
    # torch.nn.functional.interpolate expects (N, C, H, W), so add batch and channel dims
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
    # Interpolate to (height, width)
    mask_resized = F.interpolate(
        mask_tensor,
        size=(size[1], size[0]),  # PIL size is (width, height)
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)  # Remove batch and channel dims
    final_masks_resized.append(mask_resized)

# Stack into a tensor if needed
final_masks_resized = torch.stack(final_masks_resized)  # shape: (num_images, H, W)

print(f"Resized masks shape: {final_masks_resized.shape}")

# --- Visualization Code ---
num_masks = final_masks_resized.shape[0]
if num_masks == 0:
    print("No masks to display.")
else:
    grid_size = math.ceil(math.sqrt(num_masks))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(f'Segmentation Masks for Prompt: "{prompts[0]}"', fontsize=20)
    axes_flat = axes.flatten()
    for i, (mask_tensor, img) in enumerate(zip(final_masks_resized, images)):
        ax = axes_flat[i]
        mask_np = mask_tensor.detach().cpu().numpy()
        ax.imshow(mask_np, cmap='viridis')
        ax.set_title(f"Mask {i+1}", fontsize=10)
        ax.axis('off')
    for j in range(num_masks, len(axes_flat)):
        axes_flat[j].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()