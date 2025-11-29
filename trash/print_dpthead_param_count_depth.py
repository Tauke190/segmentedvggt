import torch
from vggt.heads.dpt_head import DPTHead

if __name__ == "__main__":
    # Match your model's config for depth head
    embed_dim = 1024
    dim_in = 2 * embed_dim
    output_dim = 2  # as in vggt.py

    head = DPTHead(
        dim_in=dim_in,
        output_dim=output_dim,
        activation="exp",
        conf_activation="expp1",
    )
    total_params = sum(p.numel() for p in head.parameters())
    print(f"DPTHead (depth) parameter count: {total_params}")
