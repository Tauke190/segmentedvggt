from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

from vggt.clipseg.multiclass_segmentor import get_multiclass_segmentation_tensor_mask, save_tensor,visualize_tensor

prompt = ["toy"]
image_folder = r"C:\Users\avina\Desktop\vggt-repo\vggt\examples\test\images"

masks = get_multiclass_segmentation_tensor_mask(prompt, image_folder)

save_tensor(masks, "multiclass_segmentation_masks")

# visualize_tensor(masks)
