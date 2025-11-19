import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO



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

