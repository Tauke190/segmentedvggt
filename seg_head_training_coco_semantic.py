# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import os
import argparse
import random
import numpy as np
import torch
from tqdm.auto import tqdm
import gc
import matplotlib.pyplot as plt
import os
from vggt.models.vggt import VGGT
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image
from dataset import COCOSegmentation
from torch.utils.data import Subset
import wandb

TRAIN_PATH = "/home/c3-0/datasets/coco/train2017"
TRAIN_ANN_FILE = "/home/c3-0/datasets/coco/annotations/instances_train2017.json"
VAL_PATH = "/home/c3-0/datasets/coco/val2017"
VAL_ANN_FILE = "/home/c3-0/datasets/coco/annotations/instances_val2017.json"

# TRAIN_PATH = "/home/c3-0/datasets/coco/train201"
# TRAIN_ANN_FILE = "/home/av354855/data/datasets/coco/annotations/instances_train2017.json"
# VAL_PATH = "/home/av354855/data/datasets/coco/val2017"
# VAL_ANN_FILE = "/home/av354855/data/datasets/coco/annotations/instances_val2017.json"

parser = argparse.ArgumentParser(description="VGGT segmentation head training")
parser.add_argument("--epochs", type=int, default=50, help="Number of finetuning epochs per scene")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for finetuning")
parser.add_argument("--save_path", type=str, default="vggt_seg_finetuned.pt", help="Where to save finetuned weights")
parser.add_argument("--train_path", type=str, default=TRAIN_PATH, required=False, help="Path to COCO training images directory")
parser.add_argument("--train_annotation_path", type=str, default=TRAIN_ANN_FILE, required=False, help="Path to COCO training annotation file")
parser.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of training data to use (0 < x <= 1)")
parser.add_argument("--val_path", type=str, default=VAL_PATH, required=False, help="Path to COCO validation images directory")
parser.add_argument("--val_annotation_path", type=str, default=VAL_ANN_FILE, required=False, help="Path to COCO validation annotation file")


def load_with_strict_false(model, url_or_path: str):
    if os.path.isfile(url_or_path):
        state = torch.load(url_or_path, map_location="cpu")
    else:
        state = torch.hub.load_state_dict_from_url(url_or_path, map_location="cpu")
    # unwrap common wrappers
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # strip 'module.' if present
    new_state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    msg = model.load_state_dict(new_state, strict=False)
    print("Loaded checkpoint with strict=False")
    print("Missing keys:", msg.missing_keys)       # will include segmentation_head.*
    print("Unexpected keys:", msg.unexpected_keys)
    return msg

def coco_collate_fn(batch):
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    images = torch.stack(images)
    masks = torch.stack([m.squeeze().long() for m in masks])
    return images, masks

def coco_transform(image, mask, size=(256, 256)):
    image = image.resize(size, Image.BILINEAR)
    mask = mask.resize(size, Image.NEAREST)
    image = T.ToTensor()(image)
    mask = torch.from_numpy(np.array(mask)).long()
    return image, mask

def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Dataset initialization (semantic only) ---
    train_dataset = COCOSegmentation(
        img_dir=args.train_path,
        ann_file=args.train_annotation_path,
        transforms=lambda img, msk: coco_transform(img, msk, size=(252, 252)),
        return_instance_masks=False
    )

    if args.train_fraction < 1.0:
        total_len = len(train_dataset)
        subset_len = int(total_len * args.train_fraction)
        indices = random.sample(range(total_len), subset_len)
        train_dataset = Subset(train_dataset, indices)
        print(f"Using a subset of the training data: {subset_len}/{total_len} samples")

    # --- Split into train/train-eval (90%/10%) ---
    total_len = len(train_dataset)
    train_eval_len = int(0.1 * total_len)
    train_len = total_len - train_eval_len
    indices = list(range(total_len))
    random.shuffle(indices)
    train_indices = indices[:train_len]
    train_eval_indices = indices[train_len:]
    train_subset = Subset(train_dataset, train_indices)
    train_eval_subset = Subset(train_dataset, train_eval_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=8, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    train_eval_loader = DataLoader(
        train_eval_subset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- True validation loader from COCO val2017 ---
    val_dataset = COCOSegmentation(
        img_dir=args.val_path,
        ann_file=args.val_annotation_path,
        transforms=lambda img, msk: coco_transform(img, msk, size=(252, 252))
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- Determine number of classes dynamically ---
    if isinstance(train_dataset, Subset):
        base_dataset = train_dataset.dataset
    else:
        base_dataset = train_dataset
    cat_ids = base_dataset.cat_id_to_index.keys()
    num_seg_classes = len(cat_ids) + 1  # 80 + 1 = 81 for COCO semantic
    print(f"Number of semantic classes (excluding background): {len(cat_ids)}")
    print(f"Category IDs: {sorted(cat_ids)}")
    print(f"Total classes including background: {num_seg_classes}")

    print("Initializing and loading VGGT model...")
    model = VGGT(num_seg_classes=num_seg_classes)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    load_with_strict_false(model, _URL)
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    if not hasattr(model, "segmentation_head"):
        raise AttributeError("Model missing segmentation_head.")
    for p in model.segmentation_head.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.segmentation_head.parameters(), lr=args.lr)
    print(f"Optimizer initialized (lr={args.lr})")

    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    early_stop_patience = 1  # Stop immediately if drop > 5%
    early_stop_counter = 0
    best_model_state = None

    wandb.init(
        project="vggt-segmentation",  # Change this as needed
        config=vars(args)
    )

    # --- Visualize prediction and ground truth before training ---
    model.eval()
    with torch.no_grad():
        images_vis, masks_vis = next(iter(train_loader))
        images_vis = images_vis.to(device)
        out_vis = model(images_vis)
        logits_vis = out_vis["segmentation_logits"]
        if logits_vis.dim() == 5 and logits_vis.shape[1] == 1:
            logits_vis = logits_vis.squeeze(1)
        elif logits_vis.dim() == 5 and logits_vis.shape[1] > 1:
            B, S, C, H, W = logits_vis.shape
            logits_vis = logits_vis.view(B * S, C, H, W)
        pred_mask = logits_vis.argmax(1)[0].cpu().numpy().astype(np.uint8)
        gt_mask = masks_vis[0].cpu().numpy().astype(np.uint8)
        img = images_vis[0].cpu()
        img_pil = T.ToPILImage()(img)
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_mask, cmap="nipy_spectral", vmin=0, vmax=num_seg_classes-1)
        plt.title("GT Mask (before train)")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask, cmap="nipy_spectral", vmin=0, vmax=num_seg_classes-1)
        plt.title("Pred Mask (before train)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("seg_pred_vs_gt_before_training.png")
        plt.close()
    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images = images.to(device)
            masks = masks.to(device)  # [B, H, W], dtype long
            optimizer.zero_grad(set_to_none=True)

            out = model(images)
            logits = out["segmentation_logits"]  # [B, C, H, W] or [B, 1, C, H, W]
            if logits.dim() == 5 and logits.shape[1] == 1:
                logits = logits.squeeze(1)  # [B, C, H, W]
            elif logits.dim() == 5 and logits.shape[1] > 1:
                B, S, C, H, W = logits.shape
                logits = logits.view(B * S, C, H, W)

            # Resize masks if needed to match logits
            if logits.shape[-2:] != masks.shape[-2:]:
                masks = F.interpolate(masks.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest")
                masks = masks.squeeze(1).long()

            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            if batch_idx == 0:
                print("images.shape:", images.shape)
                print("logits.shape:", logits.shape)
                print("masks.shape:", masks.shape)

            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[epoch {epoch}/{args.epochs}] avg train loss={epoch_loss/len(train_loader):.4f}")

        # --- Visualization at the end of each epoch ---
        with torch.no_grad():
            images_vis, masks_vis = next(iter(train_loader))
            images_vis = images_vis.to(device)
            out_vis = model(images_vis)
            logits_vis = out_vis["segmentation_logits"]
            if logits_vis.dim() == 5 and logits_vis.shape[1] == 1:
                logits_vis = logits_vis.squeeze(1)
            elif logits_vis.dim() == 5 and logits_vis.shape[1] > 1:
                B, S, C, H, W = logits_vis.shape
                logits_vis = logits_vis.view(B * S, C, H, W)
            pred_mask = logits_vis.argmax(1)[0].cpu().numpy().astype(np.uint8)
            gt_mask = masks_vis[0].cpu().numpy().astype(np.uint8)
            img = images_vis[0].cpu()
            img_pil = T.ToPILImage()(img)
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img_pil)
            plt.title("Image")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask, cmap="nipy_spectral", vmin=0, vmax=num_seg_classes-1)
            plt.title("GT Mask")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask, cmap="nipy_spectral", vmin=0, vmax=num_seg_classes-1)
            plt.title("Predicted Mask")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"seg_pred_vs_gt_epoch{epoch}.png")
            plt.close()

        # --- Evaluate on train-eval set at the end of each epoch ---
        model.eval()
        val_correct = 0
        val_total = 0
        iou_sum = 0.0
        iou_count = 0
        with torch.no_grad():
            for eval_images, eval_masks in train_eval_loader:
                eval_images = eval_images.to(device)
                eval_masks = eval_masks.to(device)
                eval_out = model(eval_images)
                eval_logits = eval_out["segmentation_logits"]
                if eval_logits.dim() == 5 and eval_logits.shape[1] == 1:
                    eval_logits = eval_logits.squeeze(1)
                elif eval_logits.dim() == 5 and eval_logits.shape[1] > 1:
                    B, S, C, H, W = eval_logits.shape
                    eval_logits = eval_logits.view(B * S, C, H, W)
                if eval_logits.shape[-2:] != eval_masks.shape[-2:]:
                    eval_masks = F.interpolate(eval_masks.unsqueeze(1).float(), size=eval_logits.shape[-2:], mode="nearest")
                    eval_masks = eval_masks.squeeze(1).long()
                if eval_masks.ndim == 4 and eval_masks.shape[1] == 1:
                    eval_masks = eval_masks.squeeze(1)
                eval_pred = eval_logits.argmax(1)
                val_correct += (eval_pred == eval_masks).float().sum().item()
                val_total += eval_masks.numel()
                for cls in range(num_seg_classes):
                    pred_inds = (eval_pred == cls)
                    target_inds = (eval_masks == cls)
                    intersection = (pred_inds & target_inds).sum().item()
                    union = (pred_inds | target_inds).sum().item()
                    if union > 0:
                        iou_sum += intersection / union
                        iou_count += 1
        current_val_acc = val_correct / val_total if val_total > 0 else 0.0
        current_miou = iou_sum / iou_count if iou_count > 0 else 0.0
        print(f"[train-eval][epoch {epoch}] train-eval pixel acc={current_val_acc:.4f} | train-eval mIoU={current_miou:.4f}")
        wandb.log({
            "train_eval_pixel_acc_epoch": current_val_acc,
            "train_eval_mIoU_epoch": current_miou,
            "train_eval_epoch": epoch
        })
        model.train()

        # --- Save best model and early stopping ---
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_epoch = epoch
            best_model_state = model.segmentation_head.state_dict()
            torch.save(
                {"segmentation_head": best_model_state},
                args.save_path
            )
            print(f"New best model saved at epoch {epoch} with val acc {current_val_acc:.4f}")
            early_stop_counter = 0
        elif current_val_acc < best_val_acc * 0.95:
            early_stop_counter += 1
            print(f"Validation accuracy dropped by more than 5% from best. Early stopping triggered.")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    wandb.summary["best_val_acc"] = best_val_acc
    wandb.summary["best_epoch"] = best_epoch
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Best model weights saved to {args.save_path}")
    print("Training complete.")

    # --- Full validation on COCO val2017 at the end ---
    print("Running full validation on COCO val2017...")
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    iou_sum = 0.0
    iou_count = 0
    val_acc = 0.0  # Initialize before use
    with torch.no_grad():
        for images, masks in val_loader:
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
            val_loss += loss.item()
            pred = logits.argmax(1)
            val_correct += (pred == masks).float().sum().item()
            val_total += masks.numel()
            for cls in range(num_seg_classes):
                pred_inds = (pred == cls)
                target_inds = (masks == cls)
                intersection = (pred_inds & target_inds).sum().item()
                union = (pred_inds | target_inds).sum().item()
                if union > 0:
                    iou_sum += intersection / union
                    iou_count += 1

    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    miou = iou_sum / iou_count if iou_count > 0 else 0.0
    print(f"[FINAL VALIDATION] avg val loss={avg_val_loss:.4f} | val pixel acc={val_acc:.4f} | val mIoU={miou:.4f}")
    wandb.log({
        "final_val_loss": avg_val_loss,
        "final_val_pixel_acc": val_acc,
        "final_val_mIoU": miou
    })

    # After creating your dataset
    print(f"Number of semantic classes (excluding background): {len(cat_ids)}")
    print(f"Category IDs: {sorted(cat_ids)}")
    print(f"Total classes including background: {num_seg_classes}")

if __name__ == "__main__":
    main()