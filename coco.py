import os
import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from dataset import COCOSegmentation


def coco_transform(image, mask, size=(256, 256)):
    image = image.resize(size, Image.BILINEAR)
    mask = mask.resize(size, Image.NEAREST)
    image = T.ToTensor()(image)
    mask = torch.from_numpy(np.array(mask)).long()
    return image, mask


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
