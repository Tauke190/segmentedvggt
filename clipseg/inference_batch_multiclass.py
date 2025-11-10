from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


# Load processor and model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Folder containing images
image_folder = "../vggt/examples/kitchen/images" 
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

prompts = ["vehicle","table"]  # Add more prompts as needed

colors = [
    [255, 0, 0],      # Red for cutlery
    [0, 255, 0],      # Green for pancakes
    [0, 0, 255],      # Blue for blueberries
    [255, 255, 0],    # Yellow for orange juice
]

overlays = []

for img_file in image_files:
    image_path = os.path.join(image_folder, img_file)
    image = Image.open(image_path)

    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.sigmoid(outputs.logits)

    # masks = (preds > 0.2).cpu().numpy()


    orig_size = image.size
    preds_resized = F.interpolate(
        preds.unsqueeze(1),
        size=(orig_size[1], orig_size[0]),
        mode="bilinear",
        align_corners=False
    ).squeeze(1)

    masks = (preds_resized > 0.2).cpu().numpy()


    combined_mask = np.zeros((*masks.shape[1:], 3), dtype=np.uint8)
    for i, mask in enumerate(masks):
        for c in range(3):
            combined_mask[..., c] = np.where(mask, colors[i][c], combined_mask[..., c])

    mask_img = Image.fromarray(combined_mask).convert("RGBA")
    image_rgba = image.convert("RGBA")
    alpha = 128
    mask_img.putalpha(alpha)
    overlay = Image.alpha_composite(image_rgba, mask_img)
    overlays.append(overlay)

# Display all overlays in a grid
num_images = len(overlays)
cols = 5
rows = (num_images + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
for idx, overlay in enumerate(overlays):
    r, c = divmod(idx, cols)
    axes[r, c].imshow(overlay)
    axes[r, c].set_title(f"Image {idx+1}")
    axes[r, c].axis('off')

# Hide any unused subplots
for idx in range(num_images, rows * cols):
    r, c = divmod(idx, cols)
    axes[r, c].axis('off')

plt.tight_layout()
plt.show()