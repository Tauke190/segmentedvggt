import os
import time
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights

# Paths
input_folder = 'examples/cup/images'
output_folder = 'predicted_masks'
os.makedirs(output_folder, exist_ok=True)

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

# Get 15 image paths
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Inference and timing
start_time = time.time()
for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()
    # Save mask as PNG
    mask_img = Image.fromarray(mask)
    mask_img.save(os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_mask.png"))
    # Print unique class IDs as caption
    unique_classes = np.unique(mask)
    print(f"{img_name}: Predicted class IDs: {unique_classes}")
end_time = time.time()

print(f"Total inference time for {len(image_files)} images: {end_time - start_time:.2f} seconds")