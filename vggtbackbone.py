import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def extract_backbone_features(image_folder, device="cuda"):
    # Load VGGT model
    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    # Convert model to half-precision BEFORE moving to GPU
    model = model.half().to(device)

    # Load and preprocess images
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
    if not image_paths:
        raise ValueError(f"No images found in {image_folder}")
    # Convert images to half-precision to match the model
    images = load_and_preprocess_images(image_paths).to(device, dtype=torch.float16)
    print(f"Loaded {len(image_paths)} images. Shape: {images.shape}")

    # Add a batch dimension to make the tensor 5D: (S, C, H, W) -> (1, S, C, H, W)
    images = images.unsqueeze(0)

    # Extract backbone features using the aggregator
    with torch.no_grad():
        backbone_features, patch_start_idx = model.aggregator(images)
    
    # Inspect the tensors inside the backbone_features list
    print(f"\n--- Inspecting backbone_features (list of {len(backbone_features)} tensors) ---")
    for i, tensor in enumerate(backbone_features):
        print(f"  - Tensor at index {i} has shape: {tensor.shape}")
    print("--------------------------------------------------------\n")

    # --- Visualization of the final layer feature norm ---
    # 1. Get the final layer's features (last tensor in the list)
    print("\n--- Starting Visualization ---")
    final_features = backbone_features[-1].cpu() # Shape: (1, Num_Patches, Feature_Dim)
    print(f"1. Shape of final_features tensor: {final_features.shape}")

    # 2. Calculate the L2 norm across the feature dimension
    # This gives a magnitude for each patch's feature vector.
    feature_norm = torch.linalg.norm(final_features, dim=-1).squeeze(0) # Shape: (Num_Patches,)
    print(f"2. Shape of feature_norm after norm and squeeze: {feature_norm.shape}")

    # 3. Reshape into a 2D grid for visualization
    # The number of patches is not always a perfect square.
    # We find the grid dimensions (H, W) that are closest to a square.
    num_patches = feature_norm.shape[0]
    print(f"3. num_patches calculated from feature_norm.shape[0]: {num_patches}")
    
    # --- Robust grid size calculation ---
    def get_grid_dims(n):
        if n <= 0:
            return 0, 0
        h = int(np.sqrt(n))
        while h > 0:
            if n % h == 0:
                return h, n // h
            h -= 1
        return 1, n # Fallback for prime numbers
        
    h, w = get_grid_dims(num_patches)
    print(f"4. Grid dimensions calculated: h={h}, w={w}")
    # --- End of new calculation ---
    
    print(f"5. Attempting to reshape {num_patches} patches into a {h}x{w} grid.")
    feature_norm_grid = feature_norm.reshape(h, w)
    print(f"6. Reshape successful. New shape: {feature_norm_grid.shape}")

    # 4. Plot the heatmap
    # Adjust figsize to be proportional to the grid dimensions
    fig_h = 8
    fig_w = max(4, fig_h * (w / h)) # Ensure a minimum width
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(feature_norm_grid, cmap='viridis', aspect='auto')
    plt.colorbar(label="Feature L2 Norm")
    plt.title("Feature Norm of Final Layer")
    plt.axis('off')
    plt.savefig("feature_norm_heatmap.png")
    print("Feature norm heatmap saved to feature_norm_heatmap.png")
    # plt.show() # Uncomment to display the plot directly

    # return backbone_features # No longer returning features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract VGGT backbone features from images in a folder")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    extract_backbone_features(args.image_folder, device=args.device)
    # The following lines for saving features have been removed.