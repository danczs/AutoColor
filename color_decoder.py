# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block

from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.ln = nn.LayerNorm((198,dim),eps=1e-6)
        self.ln = nn.LayerNorm(dim,eps=1e-6)

    def forward(self, x):
        return self.ln(x)

class ColorDecoder(nn.Module):
    """  color decoder with VisionTransformer
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, clip_feature_dim=512, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_patches = (img_size // patch_size)**2
        self.patch_size = patch_size
        self.decoder_embed_dim = decoder_embed_dim

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.clip_embed = nn.Linear(clip_feature_dim, decoder_embed_dim, bias=True)

        self.token_num = self.num_patches + 2
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.token_num , decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)#remove qk_scale=None,
            for i in range(decoder_depth)])

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        self.decoder_conv1 = nn.Conv2d(patch_size ** 2 * in_chans, patch_size ** 2 * in_chans, 3, stride=1, padding=(3-1)//2, bias=True)

        self.color_embdding = nn.Linear(patch_size ** 2 * in_chans, decoder_embed_dim, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.num_patches ** .5), cls_token=True)
        extra_token = 1
        decoder_pos_embed = np.concatenate([decoder_pos_embed, np.zeros([extra_token, self.decoder_embed_dim]) + 0.5], axis=0)

        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def feature_unpatchify(self,x):
        p = self.patch_size
        h = w = x.shape[2]

        x = x.reshape(shape=(x.shape[0],p,p,3,h,w))
        x = torch.einsum('npqchw->nchpwq',x)
        x = x.reshape(shape=(x.shape[0],3,h*p,w*p))
        return x

    def forward_loss(self, pred, gray_target, target, alpha=0.):
        """
            loss of pred + gray_target and traget, which have the same shape
        """
        loss_l2 = (pred + gray_target - target) ** 2
        loss_l2 = loss_l2.mean()  # [N, L], mean loss per patch
        loss_l1 = torch.abs((pred + gray_target - target))
        loss_l1 = loss_l1.mean()

        return loss_l2 * (1.0 - alpha) + loss_l1 * alpha

    def forward(self, x, clip_x, color_mask):
        # embed tokens
        x = self.decoder_embed(x)

        color_mask = self.patchify(color_mask)
        x_color = self.color_embdding(color_mask)
        x_color = torch.cat([x[:, 0, :].unsqueeze(1), x_color], dim=1)
        x = x + x_color

        clip_x = clip_x.unsqueeze(1)
        clip_x = self.clip_embed(clip_x)

        x = torch.cat([x, clip_x], dim=1)
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_pred(x)
        x = x[:, 1:-1, :]

        h = w = int(x.shape[1] ** .5)
        dim = x.shape[2]
        x = x.reshape(shape=(x.shape[0], h, w, dim))
        x = x.permute(0, 3, 1, 2)
        x = self.decoder_conv1(x)
        x = self.feature_unpatchify(x)
        return x

def mae_color_decoder_base(**kwargs):
    model = ColorDecoder(
        patch_size=16, embed_dim=768, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=LayerNorm, **kwargs)#partial(nn.LayerNorm, eps=1e-6)
    return model


