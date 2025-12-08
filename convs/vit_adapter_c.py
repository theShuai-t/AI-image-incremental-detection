import logging
import math
from time import sleep

import numpy as np
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import VisionTransformer, _create_vision_transformer, Block, Attention, LayerScale, \
    checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, \
    checkpoint_seq
import torch.nn as nn
import torch
from functools import partial
from convs.xception import SeparableConv2d


def generate_shuffle_idx():
    order = np.arange(196).reshape(14, 14)
    # order = np.arange(195, -1, -1).reshape(14, 14)
    shuffle = np.zeros(order.shape).reshape(14, 14)
    step = 7
    for i in range(step):
        for j in range(step):
            row_idx_begin = 0 + j * 2
            row_idx_end = 2 + j * 2
            column_idx_begin = 0 + i * 2
            column_idx_end = 2 + i * 2
            shuffle[row_idx_begin:row_idx_end, column_idx_begin:column_idx_end] = np.array(
                (order[i][j].item(), order[i][j+7].item(), order[i+7][j].item(),
                 order[i+7][j+7].item())).reshape(2, -1)
    print(shuffle)

    return shuffle


idx = generate_shuffle_idx()


class Xception_Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_gelu=True):
        super(Xception_Block, self).__init__()

        rep = []
        # self.relu = nn.ReLU(inplace=True)
        filters = in_filters
        for i in range(reps - 1):
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            # rep.append(nn.BatchNorm2d(filters))
            # rep.append(self.relu)

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        return x


class Gate(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dim = dim
        self.gate = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU())

    def forward(self, source, variant):
        gate = self.gate(variant)
        return source + gate * source

class Fusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dim = dim
        self.fuse = nn.Sequential(nn.Linear(self.dim * 2, self.dim), nn.ReLU())

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=-1)
        x = self.fuse(x)
        return x

class Shuffle(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.shuffle_conv = Xception_Block(dim, dim, 2, 1, start_with_gelu=False)
        # self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.shuffle_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.shuffle_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv

        self.shuffle_conv_re = Xception_Block(dim, dim, 2, 1, start_with_gelu=False)
        self.shuffle_down_re = nn.Linear(768, dim)
        self.shuffle_up_re = nn.Linear(dim, 768)
        self.shuffle_gate = Gate()
        self.relu = nn.ReLU(inplace=True)
        self.dim = dim
        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def branch(self, x, shuffle_down, shuffle_conv, shuffle_up, rearrange=False):
        B, N, C = x.shape
        x_down = shuffle_down(x)  # equivalent to 1 * 1 Conv
        x = self.relu(x_down)
        x_patch = x[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        if rearrange:
            x_patch = x_patch.reshape(B, self.dim, 14 * 14)
            x_patch = x_patch[:, :, idx].reshape(B, self.dim, 14, 14)

        x_patch = shuffle_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        #
        x_cls = x[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        #

        x_cls = shuffle_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)
        #
        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_up = shuffle_up(x_down)
        return x_up

    def forward(self, x):
        x_shuffle, x_no = x, x
        x_shuffle = self.branch(x_shuffle, self.shuffle_down_re, self.shuffle_conv_re, self.shuffle_up_re,
                                rearrange=True)
        x_no = self.branch(x_no, self.shuffle_down, self.shuffle_conv, self.shuffle_up)
        x = self.shuffle_gate(x_no, x_shuffle)
        return x, x_no, x_shuffle


class BlockWithShuffle(Block):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qkv_norm=False, proj_drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=Mlp, s=0.1):
        super(BlockWithShuffle, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qkv_norm, proj_drop, attn_drop,
                                               init_values, drop_path, act_layer, norm_layer, mlp_layer)
        self.shuffle_mlp = Shuffle(64)
        self.s = s
        # self.adapter_s = torch.nn.parameter.Parameter(torch.tensor(0.0, requires_grad=True))


    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x1, x_no, x_shuffle = self.shuffle_mlp(self.norm2(x))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))) + self.drop_path2(self.ls2(x1)) * self.s
        return x, x_no, x_shuffle

class Adapter(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.adapter_conv = Xception_Block(dim, dim, 2, 1, start_with_gelu=False)

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv

        # self.adapter_gate = Gate()
        self.relu = nn.ReLU(inplace=True)
        self.dim = dim
        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x = self.relu(x_down)
        x_patch = x[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        #
        x_cls = x[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        #
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)
        #
        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_up = self.adapter_up(x_down)
        return x_up


class BlockWithAdapter(Block):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qkv_norm=False, proj_drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_layer=Mlp, s=0.1):
        super(BlockWithAdapter, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qkv_norm, proj_drop, attn_drop,
                                               init_values, drop_path, act_layer, norm_layer, mlp_layer)
        self.adapter_mlp = Adapter(64)
        self.s = s
        # self.adapter_s = torch.nn.parameter.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(self, x):

        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x1 = self.adapter_mlp(self.norm2(x))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))) + self.drop_path2(self.ls2(x1)) * self.s
        return x

class ViTAdapter(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
                 class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.,
                 weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block,
                 *args, **kwargs):
        # Below are copied from as VisionTransformer
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            BlockWithAdapter(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             init_values=init_values, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                             act_layer=act_layer)
            for i in range(depth - 2)])
        self.blocks.extend([BlockWithShuffle(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                             init_values=init_values, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                                             act_layer=act_layer)])
        self.blocks.extend([BlockWithShuffle(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                             init_values=init_values, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                                             act_layer=act_layer)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.iteration = 0

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i in range(0, 10):
            x = self.blocks[i](x)
        x1, x_no_1, x_shuffle_1 = self.blocks[10](x)
        x2, x_no_2, x_shuffle_2 = self.blocks[11](x1)
        x = self.norm(x2)
        return {'features': x,
                'shuffle_net': {'x_1': x1, 'x_2': x2, 'x_no_1': x_no_1, 'x_no_2': x_no_2, 'x_shuffle_1': x_shuffle_1,
                                'x_shuffle_2': x_shuffle_2}}

    def forward(self, x):
        out = self.forward_features(x)
        x = out['features']
        x = self.forward_head(x)
        return x, out


def _create_vit_adapter(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        ViTAdapter, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn, **kwargs)
    return model


def vit_adapter_patch16_224(pretrained: bool = False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vit_adapter('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
