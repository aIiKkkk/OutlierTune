import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, ori_layer_norm, dev=None) -> None:
        super().__init__()
        self.dev = dev
        # 对称化之后需要对偏置进行处理
        self.layer_norm = ori_layer_norm.to(dev)
    def forward(self, x):
        out = self.layer_norm.forward(x)
        return out
