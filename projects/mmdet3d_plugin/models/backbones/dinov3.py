import os
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class DINOv3Backbone(BaseModule):
    """DINOv3 ViT backbone wrapper for VAD.

    Loads a DINOv3 model from HuggingFace Transformers (requires
    transformers >= 4.56.0) and exposes patch-token features as a 2D
    feature map compatible with the FPN neck.

    Args:
        model_path (str): Path to local HuggingFace model directory
            (downloaded via huggingface-cli or snapshot_download).
        num_register_tokens (int): Number of register tokens prepended
            after the CLS token. DINOv3 ViT-S/16 uses 4.
        patch_size (int): Patch size of the ViT model. Default: 16.
        embed_dim (int): Embedding dimension of the ViT model.
            ViT-S: 384, ViT-B: 768, ViT-L: 1024.
        frozen (bool): If True, freeze all backbone parameters.
        init_cfg (dict, optional): Initialization config.
    """

    def __init__(self,
                 model_path,
                 num_register_tokens=4,
                 patch_size=16,
                 embed_dim=384,
                 frozen=True,
                 init_cfg=None):
        super(DINOv3Backbone, self).__init__(init_cfg=init_cfg)

        from transformers import AutoModel

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # token layout: [CLS] + [num_register_tokens] + [patch_tokens]
        self.num_prefix_tokens = 1 + num_register_tokens

        self.dinov3 = AutoModel.from_pretrained(model_path)

        if frozen:
            for param in self.dinov3.parameters():
                param.requires_grad = False
            self.dinov3.eval()
            print(f'[DINOv3Backbone] Backbone frozen. Loaded from {model_path}')
        else:
            print(f'[DINOv3Backbone] Backbone unfrozen. Loaded from {model_path}')

        self.frozen = frozen

    def train(self, mode=True):
        super(DINOv3Backbone, self).train(mode)
        if self.frozen:
            self.dinov3.eval()
        return self

    def forward(self, x):
        """Extract patch token features and reshape to 2D spatial map.

        Args:
            x (Tensor): Input images, shape (B, 3, H, W).
                H and W must be divisible by patch_size (16).

        Returns:
            tuple[Tensor]: Single-element tuple with feature map
                shape (B, embed_dim, H//patch_size, W//patch_size).
        """
        B, C, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        if self.frozen:
            with torch.no_grad():
                outputs = self.dinov3(pixel_values=x)
        else:
            outputs = self.dinov3(pixel_values=x)

        # last_hidden_state: (B, 1 + num_register_tokens + h*w, embed_dim)
        tokens = outputs.last_hidden_state
        # drop CLS and register tokens, keep only patch tokens
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]  # (B, h*w, embed_dim)
        # reshape to spatial feature map
        feat = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
        return (feat,)
