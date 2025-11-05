import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import torchvision.transforms.v2 as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from vggt.models.vggt import VGGT

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 21  # 20 classes + 1 background for Pascal VOC
# Input dimensions must be divisible by the patch size (14).
# We use the closest values to your request (779x520).
# 770 = 14 * 55, 518 = 14 * 37
INPUT_SIZE = (770, 518)
PATCH_SIZE = 14
LEARNING_RATE = 1e-3
BATCH_SIZE = 4 # Reduce if you get out-of-memory errors
NUM_EPOCHS = 10

# --- 1. Define the Segmentation Head ---
class LinearSegmentationHead(nn.Module):
    """A simple linear head for segmentation."""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return nn.functional.interpolate(
            self.head(x),
            size=INPUT_SIZE,
            mode='bilinear',
            align_corners=False
        )

# --- 2. Load VGGT and Freeze Weights ---
def load_frozen_vggt():
    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

# --- 3. Prepare the Dataset ---
def get_dataloaders(shuffle=True):
    print("Preparing Pascal VOC 2012 dataset from Hugging Face...")
    image_transforms = T.Compose([
        T.Resize(INPUT_SIZE),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transforms = T.Compose([
        T.Resize(INPUT_SIZE, interpolation=T.InterpolationMode.NEAREST),
        T.ToImage(),
        T.ToDtype(torch.long, scale=False)
    ])

    def transform(batch):
        # Apply transforms to a batch of images and masks
        images = [image_transforms(img.convert("RGB")) for img in batch["image"]]
        # Use 'segmentation_mask' for this dataset
        masks = [mask_transforms(m.convert("P")).squeeze(0) for m in batch["segmentation_mask"]]
        return {"pixel_values": images, "label": masks}

    # Load the validation split from the nateraw/pascal-voc-2012 dataset
    ds = load_dataset("nateraw/pascal-voc-2012", split='validation')
    
    # Use map to apply the transform and create new columns
    ds = ds.map(transform, batched=True, remove_columns=["image", "segmentation_mask"])
    ds.set_format(type='torch') # Set the output format to PyTorch tensors

    # The collate function is no longer needed as the dataset now returns tensors directly
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)
    return dataloader

# --- 4. Training Loop ---
def train():
    vggt_model = load_frozen_vggt()
    train_loader = get_dataloaders()
    seg_head = LinearSegmentationHead(in_channels=2048, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(seg_head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    feature_h, feature_w = INPUT_SIZE[0] // PATCH_SIZE, INPUT_SIZE[1] // PATCH_SIZE

    print("\nStarting linear probing...")
    for epoch in range(NUM_EPOCHS):
        seg_head.train()
        total_loss = 0
        # The dataloader now returns a dictionary, so we unpack it
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, masks = batch['pixel_values'].to(DEVICE), batch['label'].to(DEVICE)
            with torch.no_grad():
                features_list, _ = vggt_model.aggregator(images.unsqueeze(0))
                last_features = features_list[-1]
            
            B, C = last_features.shape[1], last_features.shape[3]
            features_2d = last_features.squeeze(0).permute(0, 2, 1).reshape(B, C, feature_h, feature_w)
            
            outputs = seg_head(features_2d)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")

    print("\nTraining finished.")
    torch.save(seg_head.state_dict(), "segmentation_head.pt")
    print("Segmentation head saved to segmentation_head.pt")

# --- 5. Visualization ---
def visualize_predictions():
    print("\nVisualizing segmentation results...")
    vggt_model = load_frozen_vggt()
    seg_head = LinearSegmentationHead(in_channels=2048, num_classes=NUM_CLASSES).to(DEVICE)
    seg_head.load_state_dict(torch.load("segmentation_head.pt"))
    seg_head.eval()

    val_loader = get_dataloaders(shuffle=False)
    # The dataloader now returns a dictionary, so we unpack it
    batch = next(iter(val_loader))
    images, masks = batch['pixel_values'].to(DEVICE), batch['label'].to(DEVICE)

    feature_h, feature_w = INPUT_SIZE[0] // PATCH_SIZE, INPUT_SIZE[1] // PATCH_SIZE

    with torch.no_grad():
        features_list, _ = vggt_model.aggregator(images.unsqueeze(0))
        last_features = features_list[-1]
        B, C = last_features.shape[1], last_features.shape[3]
        features_2d = last_features.squeeze(0).permute(0, 2, 1).reshape(B, C, feature_h, feature_w)
        outputs = seg_head(features_2d)
        preds = torch.argmax(outputs, dim=1)

    # Move data to CPU for plotting
    images, masks, preds = images.cpu(), masks.cpu(), preds.cpu()

    # Un-normalize image for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean

    # Plot
    fig, axes = plt.subplots(images.size(0), 3, figsize=(15, 5 * images.size(0)))
    fig.suptitle("Image / Ground Truth / Prediction", fontsize=16)
    for i in range(images.size(0)):
        ax_img = axes[i, 0] if images.size(0) > 1 else axes[0]
        ax_gt = axes[i, 1] if images.size(0) > 1 else axes[1]
        ax_pred = axes[i, 2] if images.size(0) > 1 else axes[2]

        ax_img.imshow(images[i].permute(1, 2, 0))
        ax_img.set_title("Original Image")
        ax_img.axis('off')

        # Use a consistent colormap for masks
        ax_gt.imshow(masks[i], vmin=0, vmax=NUM_CLASSES-1, cmap='tab20')
        ax_gt.set_title("Ground Truth Mask")
        ax_gt.axis('off')

        ax_pred.imshow(preds[i], vmin=0, vmax=NUM_CLASSES-1, cmap='tab20')
        ax_pred.set_title("Predicted Mask")
        ax_pred.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("segmentation_results.png")
    print("Saved visualization to segmentation_results.png")
    plt.show()


if __name__ == "__main__":
    train()
    visualize_predictions()