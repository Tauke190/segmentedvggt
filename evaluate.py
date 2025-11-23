import torch
import torch.nn.functional as F
from vggt.models.vggt import VGGT
from dataset import COCOSegmentation
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
import argparse
from tqdm.auto import tqdm

TEST_PATH = "/home/av354855/data/datasets/coco/test2017"
TEST_ANN_FILE = "/home/av354855/data/datasets/coco/annotations/image_info_test2017.json"
SEG_HEAD_PATH = "vggt_seg_finetuned.pt"  # Path to your trained segmentation head weights

def coco_transform(image, mask, size=(252, 252), binary=True):
    image = image.resize(size, Image.BILINEAR)
    mask = mask.resize(size, Image.NEAREST)
    image = T.ToTensor()(image)
    mask = torch.from_numpy(np.array(mask)).long()
    if binary:
        mask = (mask > 0).long()
    return image, mask

def load_with_strict_false(model, url_or_path: str):
    if url_or_path.startswith("http"):
        state = torch.hub.load_state_dict_from_url(url_or_path, map_location="cpu")
    else:
        state = torch.load(url_or_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    msg = model.load_state_dict(new_state, strict=False)
    print("Loaded checkpoint with strict=False")
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)
    return msg

def main():
    parser = argparse.ArgumentParser(description="VGGT COCO evaluation")
    parser.add_argument("--mode", choices=["binary", "semantic"], default="binary", help="Evaluation mode: binary or semantic")
    parser.add_argument("--seg_head_path", type=str, default=SEG_HEAD_PATH, help="Path to segmentation head weights")
    parser.add_argument("--test_path", type=str, default=TEST_PATH, help="Path to COCO test images")
    parser.add_argument("--test_ann_file", type=str, default=TEST_ANN_FILE, help="Path to COCO test annotation file")
    args = parser.parse_args()

    binary = args.mode == "binary"
    num_classes = 2 if binary else 81  # 81 for COCO, adjust if needed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading VGGT backbone...")
    model = VGGT(num_seg_classes=num_classes)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    load_with_strict_false(model, _URL)

    print("Attaching segmentation head weights...")
    seg_head_state = torch.load(args.seg_head_path, map_location="cpu")
    model.segmentation_head.load_state_dict(seg_head_state["segmentation_head"])
    model = model.to(device)
    model.eval()

    print("Preparing COCO test dataset...")
    test_dataset = COCOSegmentation(
        img_dir=args.test_path,
        ann_file=args.test_ann_file,
        transforms=lambda img, msk: coco_transform(img, msk, size=(252, 252), binary=binary)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    iou_sum = 0.0
    iou_count = 0

    print(f"Evaluating on COCO test set in {'binary' if binary else 'semantic'} mode...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            masks = masks.to(device)
            out = model(images)
            logits = out["segmentation_logits"]
            if logits.dim() == 5 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            elif logits.dim() == 5 and logits.shape[1] > 1:
                B, S, C, H, W = logits.shape
                logits = logits.view(B * S, C, H, W)
            if logits.shape[-2:] != masks.shape[-2:]:
                masks = F.interpolate(masks.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest")
                masks = masks.squeeze(1).long()
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            loss = criterion(logits, masks)
            test_loss += loss.item()
            pred = logits.argmax(1)
            test_correct += (pred == masks).float().sum().item()
            test_total += masks.numel()
            for cls in range(num_classes):
                pred_inds = (pred == cls)
                target_inds = (masks == cls)
                intersection = (pred_inds & target_inds).sum().item()
                union = (pred_inds | target_inds).sum().item()
                if union > 0:
                    iou_sum += intersection / union
                    iou_count += 1

    avg_test_loss = test_loss / len(test_loader)
    test_acc = test_correct / test_total
    miou = iou_sum / iou_count if iou_count > 0 else 0.0
    print(f"[COCO TEST] avg loss={avg_test_loss:.4f} | pixel acc={test_acc:.4f} | mIoU={miou:.4f}")

if __name__ == "__main__":
    main()