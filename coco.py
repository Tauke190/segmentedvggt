import os
import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as T
import matplotlib.pyplot as plt


# -----------------------------
#  Transforms for image + mask
# -----------------------------
def coco_transform(image, mask, size=(256, 256)):
    image = image.resize(size, Image.BILINEAR)
    mask = mask.resize(size, Image.NEAREST)
    image = T.ToTensor()(image)
    mask = torch.from_numpy(np.array(mask)).long()
    return image, mask


# ------------------------------------
#   COCO Semantic Segmentation Dataset
# ------------------------------------
class COCOSegmentation(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        # Load image metadata
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        
        # Load RGB image
        image = Image.open(img_path).convert("RGB")

        # Load all annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        # Create empty mask (H x W)
        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

        # Fill mask: category_id per pixel
        for ann in anns:
            m = self.coco.annToMask(ann)
            mask = np.maximum(mask, m * ann["category_id"])

        mask = Image.fromarray(mask)

        # Apply transform
        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


# ------------------------
#  Create dataset + loader
# ------------------------
if __name__ == "__main__":
    train_img_dir = "/home/av354855/data/datasets/coco/train2017"
    train_ann_file = "/home/av354855/data/datasets/coco/annotations/instances_train2017.json"

    train_dataset = COCOSegmentation(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        transforms=lambda img, msk: coco_transform(img, msk, size=(256, 256))
    )

    # Sample a random index
    idx = random.randint(0, len(train_dataset) - 1)
    image, mask = train_dataset[idx]

    # Save the random image and mask
    img_pil = T.ToPILImage()(image)
    img_pil.save("random_sample_image.png")

    mask_pil = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
    mask_pil.save("random_sample_mask.png")

    print(f"Saved random_sample_image.png and random_sample_mask.png (index {idx})")

    # Visualize side by side
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask.cpu().numpy(), cmap="nipy_spectral")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
