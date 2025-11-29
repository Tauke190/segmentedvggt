import os
import time
from PIL import Image
import torch
import numpy as np
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# --- Configuration ---
input_folder = "examples/elephant/images"  # Change as needed
output_folder = "predicted_masks_mask2former"
os.makedirs(output_folder, exist_ok=True)
num_images = 15

# --- Load Model and Processor ---
# Use a Mask2Former model trained on COCO-Stuff
model_name = "facebook/mask2former-swin-large-coco-panoptic"
processor = Mask2FormerImageProcessor.from_pretrained(model_name)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# --- Gather Images ---
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files = sorted(image_files)[:num_images]
images = [Image.open(os.path.join(input_folder, f)).convert("RGB") for f in image_files]

# --- Inference and Timing ---
start_time = time.time()
for img_name, img in zip(image_files, images):
    # Preprocess
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    # For semantic segmentation, get the segmentation map
    semantic_mask = outputs.segmentation[0].cpu().numpy()  # (H, W)

    # Save mask as PNG
    mask_img = Image.fromarray(semantic_mask.astype(np.uint8))
    mask_img.save(os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_mask.png"))

    # Print unique class IDs as caption
    unique_classes = np.unique(semantic_mask)
    print(f"{img_name}: Predicted class IDs: {unique_classes}")

end_time = time.time()
print(f"Total inference time for {len(images)} images: {end_time - start_time:.2f} seconds")