import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDecoderHead(nn.Module):
    def __init__(self, in_channels, num_classes, features=[512, 256, 128, 64]):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        prev_channels = in_channels
        for feat in features:
            self.up_blocks.append(
                nn.ConvTranspose2d(prev_channels, feat, kernel_size=2, stride=2)
            )
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feat),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feat, feat, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feat),
                    nn.ReLU(inplace=True),
                )
            )
            prev_channels = feat

        self.final_conv = nn.Conv2d(features[-1], num_classes, kernel_size=1)

    def forward(self, x):
        for up, conv in zip(self.up_blocks, self.conv_blocks):
            x = up(x)
            x = conv(x)
        logits = self.final_conv(x)
        return logits
