import os
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES


@BACKBONES.register_module()
class DINOv3Backbone(BaseModule):
    """DINOv3 ViT backbone wrapper for VAD.

    Loads a DINOv3 ViT model from a local clone of facebookresearch/dinov3
    via torch.hub and exposes its patch-token features as a 2D feature map
    compatible with the FPN neck.

    Args:
        model_name (str): Entry point in dinov3 hubconf, e.g. 'dinov3_vits16'.
        repo_path (str): Path to the local dinov3 repo (cloned from GitHub).
        pretrained_weights (str): Path to the .pth checkpoint file.
        frozen (bool): If True, freeze all backbone parameters.
        init_cfg (dict, optional): Initialization config.
    """

    # embed_dim for each model variant
    EMBED_DIMS = {
        'dinov3_vits16': 384,
        'dinov3_vits16plus': 384,
        'dinov3_vitb16': 768,
        'dinov3_vitl16': 1024,
        'dinov3_vitl16plus': 1024,
        'dinov3_vith16plus': 1280,
        'dinov3_vit7b16': 1408,
    }

    def __init__(self,
                 model_name='dinov3_vits16',
                 repo_path='third_party/dinov3',
                 pretrained_weights='ckpts/dinov3_vits16.pth',
                 frozen=True,
                 init_cfg=None):
        super(DINOv3Backbone, self).__init__(init_cfg=init_cfg)

        self.model_name = model_name
        self.patch_size = 16
        self.embed_dim = self.EMBED_DIMS.get(model_name, 384)

        # Load DINOv3 from local repo (no pretrained weights from hub)
        self.dinov3 = torch.hub.load(
            repo_path,
            model_name,
            source='local',
            pretrained=False,
        )

        # Load pretrained weights
        if pretrained_weights and os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            # Handle nested checkpoint formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            missing, unexpected = self.dinov3.load_state_dict(
                state_dict, strict=False)
            if missing:
                print(f'[DINOv3Backbone] Missing keys: {missing[:5]}...')
            if unexpected:
                print(f'[DINOv3Backbone] Unexpected keys: {unexpected[:5]}...')
            print(f'[DINOv3Backbone] Loaded weights from {pretrained_weights}')
        else:
            print(f'[DINOv3Backbone] WARNING: pretrained_weights not found '
                  f'at {pretrained_weights}, using random init.')

        if frozen:
            for param in self.dinov3.parameters():
                param.requires_grad = False
            self.dinov3.eval()
            print('[DINOv3Backbone] Backbone frozen.')

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
            tuple[Tensor]: Single-element tuple containing the feature map
                of shape (B, embed_dim, H//16, W//16).
        """
        B, C, H, W = x.shape
        h = H // self.patch_size
        w = W // self.patch_size

        if self.frozen:
            with torch.no_grad():
                # get_intermediate_layers returns list of spatial feature maps
                # when reshape=True: each element is (B, embed_dim, h, w)
                features = self.dinov3.get_intermediate_layers(
                    x,
                    n=1,
                    reshape=True,
                    return_class_token=False,
                )
        else:
            features = self.dinov3.get_intermediate_layers(
                x,
                n=1,
                reshape=True,
                return_class_token=False,
            )

        # features is a list of length n; take the last layer's output
        feat = features[-1]  # (B, embed_dim, h, w)
        return (feat,)
