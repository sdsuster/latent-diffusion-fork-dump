
from torch import nn
from torch.nn import LayerNorm
from collections.abc import Sequence

from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock 
from monai.networks.nets.swin_unetr import MERGING_MODE, PatchMergingV2, get_window_size, compute_mask
from ldm.models.swinunet.swin_transformer_v2 import WindowAttention, window_partition, window_reverse
import torch.utils.checkpoint as checkpoint
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.layers import DropPath
from ldm.util import instantiate_from_config
from typing import Optional
from monai.transforms import Activations, AsDiscrete, Compose
from torch import Tensor
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from monai.utils import ensure_tuple_rep, look_up_option, optional_import
rearrange, _ = optional_import("einops", name="rearrange")


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence,
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: LayerNorm = nn.LayerNorm,
        downsample: nn.Module = None,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x, return_skip_connection = True):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            if return_skip_connection:
                skip_connection = rearrange(x, "b d h w c -> b c d h w")
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if return_skip_connection:
                skip_connection = rearrange(x, "b h w c -> b c h w")
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        if return_skip_connection:
            return x, skip_connection
        return x


class PatchExpanding(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: LayerNorm = nn.LayerNorm, spatial_dims: int = 3, padding = (0, 0, 0)) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        self.padding = padding
        if spatial_dims == 3:
            self.expansion = nn.Linear(2 * dim, 8 * dim, bias=False)
            self.norm = norm_layer(2 * dim)
        else:
            raise ValueError("spatial dimension should be 3")

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        x = self.norm(x)
        x_ = self.expansion(x)
        x = torch.zeros((b, 2*d, 2*h, 2*w, c//2), device=x.device)
        x[:, 0::2, 0::2, 0::2, :] = x_[:, :, :, :, :self.dim]
        x[:, 1::2, 0::2, 0::2, :] = x_[:, :, :, :, 1*self.dim:2*self.dim]
        x[:, 0::2, 1::2, 0::2, :] = x_[:, :, :, :, 2*self.dim:3*self.dim]
        x[:, 0::2, 0::2, 1::2, :] = x_[:, :, :, :, 3*self.dim:4*self.dim]
        x[:, 1::2, 0::2, 1::2, :] = x_[:, :, :, :, 4*self.dim:5*self.dim]
        x[:, 0::2, 1::2, 0::2, :] = x_[:, :, :, :, 5*self.dim:6*self.dim]
        x[:, 0::2, 0::2, 1::2, :] = x_[:, :, :, :, 6*self.dim:7*self.dim]
        x[:, 1::2, 1::2, 1::2, :] = x_[:, :, :, :, 7*self.dim:8*self.dim]
        return x
    

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence,
        shift_size: Sequence,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x



class BasicUpLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence,
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: LayerNorm = nn.LayerNorm,
        upsample: nn.Module = None,
        use_checkpoint: bool = False,
        use_skip_connection = False
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            upsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim*(2 if self.use_skip_connection else 1),
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        if self.use_skip_connection:
            self.downsample = nn.Linear(2 * dim, dim, bias=False)
            self.down_norm = norm_layer(2 * dim)
        self.upsample = upsample
        if callable(self.upsample):
            self.upsample = upsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x, skip_connection = None):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            x = rearrange(x, "b c d h w -> b d h w c")
            if self.upsample is not None:
                x = self.upsample(x)
            b, d, h, w, c = x.size()
            if skip_connection is not None:
                x = torch.cat([x, rearrange(skip_connection, "b c d h w -> b d h w c")], dim=-1)

            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
                
            x = x.view(b, d, h, w, -1)
            if self.use_skip_connection:
                x = self.down_norm(x)
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            x = rearrange(x, "b c h w -> b h w c")
            if self.upsample is not None:
                x = self.upsample(x)
            b, h, w, c = x.size()
            if skip_connection is not None:
                x = torch.cat([x, rearrange(skip_connection, "b c h w -> b h w c")], dim=-1)
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)

            x = x.view(b, h, w, -1)
            if self.use_skip_connection:
                x = self.down_norm(x)
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x

class SwinUnet(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence,
        patch_size: Sequence,
        depths: Sequence,
        num_heads: Sequence,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()

        self.layers_up1 = nn.ModuleList()
        self.layers_up2 = nn.ModuleList()
        self.layers_up3 = nn.ModuleList()
        self.layers_up4 = nn.ModuleList()

        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)

        
        layer = BasicLayer(
            dim=int(embed_dim * 2**3),
            depth=1,
            num_heads=num_heads[self.num_layers],
            window_size=self.window_size,
            drop_path=dpr[sum(depths[:self.num_layers-1]) : sum(depths[: self.num_layers-1 + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
        )
        self.layers4.append(layer)

        for i_layer in reversed(range(self.num_layers)):
            layer = BasicUpLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                upsample=PatchExpanding,
                use_checkpoint=use_checkpoint,
                use_skip_connection=True
            )
            if i_layer == 0:
                self.layers_up1.append(layer)
            elif i_layer == 1:
                self.layers_up2.append(layer)
            elif i_layer == 2:
                self.layers_up3.append(layer)

        self.layers_up0 = BasicUpLayer(
            dim=embed_dim//2,
            depth=2,
            num_heads=num_heads[0],
            window_size=self.window_size,
            drop_path=dpr[sum(depths[:0]) : sum(depths[: 0 + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            upsample=PatchExpanding,
            use_checkpoint=use_checkpoint,
        )
        self.head = nn.Conv3d(in_channels=embed_dim//2,out_channels=3,kernel_size=1,bias=False)
        # self.head = nn.Linear(embed_dim//2, 3)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.shape
            # Force trace() to generate a constant by casting to int
            ch = int(x_shape[1])
            if len(x_shape) == 5:
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        # x0_out = self.proj_out(x0, normalize)
        
        x1, x1_skip = self.layers1[0](x0.contiguous())
        # x1_out = self.proj_out(x1, normalize)
        
        x2, x2_skip = self.layers2[0](x1.contiguous())
        # x2_out = self.proj_out(x2, normalize)
        
        x3, x3_skip = self.layers3[0](x2.contiguous())
        # x3_out = self.proj_out(x3, normalize)
        # print(x3.shape)
        
        x4 = self.layers4[0](x3.contiguous(), return_skip_connection = False)
        # print(x4.shape)
        
        x3_ = self.layers_up3[0](x4.contiguous(), skip_connection=x3_skip)
        # print(x3_.shape)

        x2_ = self.layers_up2[0](x3_.contiguous(), skip_connection=x2_skip)

        x1_ = self.layers_up1[0](x2_.contiguous(), skip_connection=x1_skip)
        # # print(x1_.shape)
        x0_ = self.layers_up0(x1_.contiguous())
        head = self.head(x0_.contiguous())

        return head

    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

def l1(x, y):
    return torch.abs(x-y)

class Vit_Seg_Trainer(pl.LightningModule):

    def __init__(self,
                 modelconfig,
                 lossconfig,
                 metricconfig,
                 image_key="image",
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 monitor=None,
                 first_stage_weights: Optional[str] = None,
                 ckpt_path = None,
                 ignore_keys=[],
                 ):
        super().__init__()
        self.model = instantiate_from_config(modelconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.dice_acc = instantiate_from_config(metricconfig)
        # self.loss = l1

        if monitor is not None:
            self.monitor = monitor
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor
        self.image_key = image_key
        self.first_stage_weights = first_stage_weights
        
        self.post_pred = AsDiscrete(argmax=False, logit_thresh=0.5)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        if first_stage_weights is not None:
            self.load_encoder_weights_frozen()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = self.get_input(batch, self.image_key)
        pred= self(x) #, qloss, ind 

        loss = self.loss(pred.contiguous(), target.contiguous())
        loss = torch.mean(loss)
        self.log(f"train/dice_loss", loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        with torch.no_grad():
            pred_post = self.post_pred(torch.sigmoid(pred))
            self.dice_acc.reset()
            self.dice_acc(y_pred=pred_post, y=target)
            acc, not_nans = self.dice_acc.aggregate()
            self.log(f"train/dice_acc0", acc[0],
                    prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train/dice_acc1", acc[1],
                    prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log(f"train/dice_acc2", acc[2],
                    prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss
        
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.model.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def get_input(self, batch, k):
        return batch[k], batch['label']
    

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x, y = self.get_input(batch, self.image_key)

        # x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x[:, :, :, :, 30]
            return log
        xrec= self(x) #, _ 
        class_labels = y[:, :, :, :, 30]
        # Step 1: Convert logits to class labels using argmax
        # prob = torch.sigmoid(class_labels)
        prob = class_labels
        class_labels = torch.argmax(class_labels, dim=1)  # Shape: [batch_size, height, width]
        seg = (prob > 0.5).astype(np.int8)
        
        color_images = np.zeros(seg.shape)

        color_images[:, 2][seg[:, 2] == 1] = 1.
        color_images[:, 1][seg[:, 1] == 1] = 1.
        color_images[:, 0][seg[:, 0] == 1] = 1.
        log["labels"] = torch.tensor(color_images)

        class_labels = xrec[:, :, :, :, 30]
        prob = torch.sigmoid(class_labels)
        prob = class_labels
        class_labels = torch.argmax(class_labels, dim=1)  # Shape: [batch_size, height, width]
        seg = (prob > 0.5).astype(np.int8)
        
        color_images = np.zeros(seg.shape)

        color_images[:, 2][seg[:, 2] == 1] = 1.
        color_images[:, 1][seg[:, 1] == 1] = 1.
        color_images[:, 0][seg[:, 0] == 1] = 1.
        # color_images[:, 0][seg[:, 0] == 0] = 0.
        # color_images[:, 1][seg[:, 1] == 0] = 0.

        log["inputs"] = x[:, :, :, :, 30]
        log["prediction"] = torch.tensor(color_images)

        return log

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        
        x, target = self.get_input(batch, self.image_key)
        pred= self(x) #, qloss, ind 

        loss = self.loss(pred.contiguous(), target.contiguous())
        loss = torch.mean(loss)
        self.log(f"val/dice_loss", loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        pred_post = self.post_trans(pred)
            
        self.dice_acc.reset()
        self.dice_acc(y_pred=pred_post, y=target)
        acc, not_nans = self.dice_acc.aggregate()
        # print(target, pred_post, torch.mean(pred_post - target))
        print(acc)
        self.log(f"val/dice_acc0", acc[0],
                prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val/dice_acc1", acc[1],
                prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val/dice_acc2", acc[2],
                prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def load_encoder_weights_frozen(self):
        checkpoint = torch.load(self.first_stage_weights)
        model_state_dict = self.state_dict()
        partial_checkpoint = {k: v for k, v in checkpoint.items() if '_encoder' in k}
        model_state_dict.update(partial_checkpoint)
        self.load_state_dict(model_state_dict)
        # Freeze the '_encoder' layers
        for name, param in self.named_parameters():
            if '_encoder' in name:
                param.requires_grad = False  # Freeze the parameters of the '_encoder' layers
                print(name)
    