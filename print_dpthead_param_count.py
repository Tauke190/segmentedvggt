import torch
from vggt.heads.dpt_head import DPTHead

if __name__ == "__main__":
    # Match your model's config
    embed_dim = 1024
    num_seg_classes = 81
    dim_in = 2 * embed_dim
    output_dim = num_seg_classes

    head = DPTHead(
        dim_in=dim_in,
        output_dim=output_dim,
        activation="identity",
        conf_activation="none",
        return_conf=False,
    )
    total_params = sum(p.numel() for p in head.parameters())
    print(f"DPTHead parameter count: {total_params}")
