import os
import time
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='DeepLabV3+ Inference Script')
parser.add_argument('--input_images', type=str, default='examples/cup/images', help='Path to input images folder')
parser.add_argument('--batch_size', type=int, default=60, help='Batch size for inference')
args = parser.parse_args()

input_folder = args.input_images
output_folder = 'predicted_masks'
os.makedirs(output_folder, exist_ok=True)
batch_size = args.batch_size

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load DeepLabv3+ with ResNet-101 backbone
weights = DeepLabV3_ResNet101_Weights.DEFAULT  # pretrained on COCO “things”
model = models.segmentation.deeplabv3_resnet101(weights=weights).to(device)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(520),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Get image paths
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Inference and timing
start_time = time.time()
num_images = len(image_files)
for i in range(0, num_images, batch_size):
    batch_files = image_files[i:i+batch_size]
    batch_imgs = []
    orig_names = []
    for img_name in batch_files:
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        batch_imgs.append(preprocess(img))
        orig_names.append(img_name)
    input_tensor = torch.stack(batch_imgs).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out']
    masks = output.argmax(1).byte().cpu().numpy()
    for mask, img_name in zip(masks, orig_names):
        mask_img = Image.fromarray(mask)
        mask_img.save(os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_mask.png"))
        unique_classes = np.unique(mask)
        print(f"{img_name}: Predicted class IDs: {unique_classes}")
end_time = time.time()

print(f"Total inference time for {num_images} images: {end_time - start_time:.2f} seconds")