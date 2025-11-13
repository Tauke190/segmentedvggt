import os, sys, glob, hashlib
from typing import List, Optional
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# Add clipseg to path
clipseg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clipseg"))
if clipseg_root not in sys.path:
    sys.path.append(clipseg_root)

from clipseg.multiclass_segmentor import get_multiclass_segmentation_tensor_mask
from vggt.utils.load_fn import load_and_preprocess_images

class ClipSegMultiViewDataset(Dataset):
    def _discover_sequences(self):
        seq_dirs = []
        for d in sorted(glob.glob(os.path.join(self.root, "*"))):
            img_dir = os.path.join(d, "images")
            if os.path.isdir(img_dir) and len(self._list_images(img_dir)) > 0:
                seq_dirs.append(img_dir)   # store path to images folder directly
        if not seq_dirs:
            raise FileNotFoundError(f"No 'images' subfolders with images under {self.root}")
        return seq_dirs

    def __init__(
        self,
        root: str,
        split: str = "train",
        prompts: List[str] = None,
        img_size: int = 518,
        max_views: Optional[int] = None,
        cache_dir: Optional[str] = None,
        cache_masks: bool = True,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.prompts = [p.strip() for p in (prompts or []) if str(p).strip()]
        self.img_size = img_size
        self.max_views = max_views
        self.cache_dir = cache_dir
        self.cache_masks = cache_masks

        # discover sequence folders: any subdir with images
        self.seq_dirs = self._discover_sequences()

        # cache tag per prompt set and img_size
        self._cache_tag = self._make_cache_tag(self.prompts, img_size)

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _list_images(self, img_dir):
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(img_dir, e)))
        return sorted(files)

    def _make_cache_tag(self, prompts, img_size):
        s = "|".join(prompts) + f"|{img_size}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]

    def _mask_cache_path(self, seq_dir):
        base = os.path.basename(seq_dir.rstrip("/"))
        fname = f"{base}_seg_{self._cache_tag}.pt"
        if self.cache_dir:
            return os.path.join(self.cache_dir, fname)
        return os.path.join(seq_dir, fname)

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx):
        img_dir = self.seq_dirs[idx]        # points to .../seqX/images
        images = load_and_preprocess_images(img_dir, target_img_size=self.img_size)
        if isinstance(images, tuple):
            images = images[0]
        S, _, H, W = images.shape
        if self.max_views and S > self.max_views:
            images = images[:self.max_views]
            S = self.max_views

        # CLIPSeg masks
        seg_masks = get_multiclass_segmentation_tensor_mask(self.prompts, img_dir)
        seg_masks = torch.as_tensor(seg_masks)
        if seg_masks.dim() == 4:
            seg_masks = F.interpolate(seg_masks, size=(H, W), mode="nearest")
            seg_target = seg_masks.argmax(1).long()
        else:
            seg_masks = seg_masks.unsqueeze(1).float()
            seg_masks = F.interpolate(seg_masks, size=(H, W), mode="nearest")
            seg_target = seg_masks.squeeze(1).long()

        return {
            "images": images,                 # [S,3,H,W]
            "segmentation_target": seg_target, # [S,H,W]
            "seq_name": os.path.dirname(img_dir),  # parent seq folder
        }