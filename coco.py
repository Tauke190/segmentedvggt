import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as T


# -----------------------------
#  Transforms for image + mask
# -----------------------------
def coco_transform(image, mask):
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
    train_img_dir = "/home/c3-0/datasets/coco/train2017"
    train_ann_file = "/home/c3-0/datasets/coco/annotations"

    train_dataset = COCOSegmentation(
        img_dir=train_img_dir,
        ann_file=train_ann_file,
        transforms=coco_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Test loading one batch
    for images, masks in train_loader:
        print("Images batch:", images.shape)  # [B, 3, H, W]
        print("Masks batch:", masks.shape)    # [B, H, W]
        break
