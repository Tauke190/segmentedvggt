import os
import glob
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def extract_backbone_features(image_folder, device="cuda"):
    # Load VGGT model
    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    # Load and preprocess images
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
    if not image_paths:
        raise ValueError(f"No images found in {image_folder}")
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Loaded {len(image_paths)} images. Shape: {images.shape}")

    # Add a batch dimension to make the tensor 5D: (S, C, H, W) -> (1, S, C, H, W)
    images = images.unsqueeze(0)

    # Extract backbone features using the aggregator
    with torch.no_grad():
        backbone_features, patch_start_idx = model.aggregator(images)
    print(f"Backbone features shape: {backbone_features.shape}")
    return backbone_features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract VGGT backbone features from images in a folder")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    features = extract_backbone_features(args.image_folder, device=args.device)
    # Optionally, save features to disk
    torch.save(features.cpu(), "backbone_features.pt")
    print("Backbone features saved to backbone_features.pt")