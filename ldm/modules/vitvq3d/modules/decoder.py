import torch

from einops import rearrange
from torch import nn, Tensor
from torch.nn import LayerNorm, Linear, ModuleList

from .modules import Block, no_grad_trunc_normal_
from .embedding import SinCosPositionalEmbedding


class MarlinDecoder(nn.Module):

    def __init__(self, img_size=[80, 80, 64], patch_size=16, embed_dim=384, depth=8,
        num_heads=6, mlp_hidden_dim=1024, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=1., n_channels = 1, expand_channel_dim = False, patch_strides = 8
    ):
        super().__init__()
        self.n_channels = n_channels
        output_dim = self.n_channels * patch_size * patch_size * patch_size
        self.patch_size = patch_size
        self.patch_size = patch_size
        self.n_patch_t = (img_size[0] - patch_size) // patch_strides + 1
        self.n_patch_h = (img_size[1] - patch_size) // patch_strides + 1
        self.n_patch_w = (img_size[2] - patch_size) // patch_strides + 1
        self.embed_dim = embed_dim
        self.expand_channel_dim = expand_channel_dim
        if norm_layer == "LayerNorm":
            self.norm_layer = LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")

        # sine-cosine positional embeddings
        self.pos_embedding = SinCosPositionalEmbedding(
            (self.n_patch_h * self.n_patch_w * self.n_patch_t, embed_dim), dropout_rate=0.)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=self.norm_layer,
                init_values=init_values
            ) for _ in range(depth)])

        self.head = Linear(embed_dim, output_dim)
        self.apply(self._init_weights)
        # no_grad_trunc_normal_(self.mask_token, mean=0., std=0.02, a=-0.02, b=0.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatch_to_img(self, x: Tensor) -> Tensor:
        # x: (Batch, No. batches, Prod of cube size * C)
        x = rearrange(x, "b n (c p) -> b n p c", c=self.n_channels)
        # x: (Batch, No. batches, Prod of cube size, C)
        x = rearrange(x, "b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)", p0=self.patch_size,
            p1=self.patch_size, p2=self.patch_size, h=self.n_patch_h, w=self.n_patch_w)
        # x: (B, C, T, H, W)
        return x

    def forward_features(self, x, return_token_num=0):
        for block in self.blocks:
            x = block(x)

        # if return_token_num > 0:
        #     x = x[:, -return_token_num:]

        x = self.norm(x)
        x = self.head(x)
        # x: (B, N_mask, C)
        return x

    def forward(self, x):
        # mask: 0 -> masked, 1 -> visible
        if self.expand_channel_dim:
            x = torch.squeeze(x, 1)
        b, n, c = x.shape
        expand_pos_embed = self.pos_embedding.emb.data.expand(b, -1, -1)
        # pos_emb_vis = expand_pos_embed[mask].view(b, -1, c)
        # pos_emb_mask = expand_pos_embed[~mask].view(b, -1, c)
        x = x + expand_pos_embed

        # mask_num = pos_emb_mask.shape[1]
        x = self.forward_features(x, return_token_num=0)
        return x
    
    