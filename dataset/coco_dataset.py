import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class COCOSegmentation(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None, return_instance_masks=False):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.return_instance_masks = return_instance_masks

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

        height, width = img_info["height"], img_info["width"]

        if self.return_instance_masks:
            # Return a stack of instance masks (N, H, W) and their category ids
            instance_masks = []
            category_ids = []
            for ann in anns:
                mask_instance = self.coco.annToMask(ann)
                if mask_instance.max() > 0:
                    instance_masks.append(mask_instance)
                    category_ids.append(ann["category_id"])
            if instance_masks:
                instance_masks = np.stack(instance_masks, axis=0)  # (N, H, W)
                category_ids = np.array(category_ids, dtype=np.int64)
            else:
                # No objects: return empty arrays
                instance_masks = np.zeros((0, height, width), dtype=np.uint8)
                category_ids = np.zeros((0,), dtype=np.int64)
            mask = (instance_masks, category_ids)
        else:
            # Return a single mask (all objects combined)
            mask = np.zeros((height, width), dtype=np.uint8)
            for ann in anns:
                mask_instance = self.coco.annToMask(ann)
                mask = np.maximum(mask, mask_instance)
            mask = Image.fromarray(mask)

        if self.transforms:
            if self.return_instance_masks:
                # Only apply transforms to image; masks are numpy arrays
                image = self.transforms(image)
                # You may want to add mask transforms here if needed
                return image, mask[0], mask[1]  # image, instance_masks, category_ids
            else:
                image, mask = self.transforms(image, mask)
                return image, mask

        if self.return_instance_masks:
            return image, mask[0], mask[1]
        else:
            return image, mask

