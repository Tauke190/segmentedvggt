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

        # Build category_id to contiguous index mapping
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_index = {cat['id']: i+1 for i, cat in enumerate(cats)}  # +1 to reserve 0 for background

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

        for ann in anns:
            m = self.coco.annToMask(ann)
            cat_idx = self.cat_id_to_index[ann["category_id"]]
            mask = np.maximum(mask, m * cat_idx)

        mask = Image.fromarray(mask)

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

    img_id = train_dataset.ids[idx]
    ann_ids = train_dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
    print("Annotation count:", len(ann_ids))

    # Convert tensors to displayable formats
    img_pil = T.ToPILImage()(image)
    mask_np = mask.cpu().numpy().astype(np.uint8)

    print(f"Visualizing random sample (index {idx})")

    # Visualize side by side and save
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap="nipy_spectral")
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("random_image_and_mask.png")
    print("Saved visualization as random_image_and_mask.png")
