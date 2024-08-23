from torch import nn, Tensor
from torch.nn import ModuleList, LayerNorm
import torch

from .embedding import PatchEmbedding3d, SinCosPositionalEmbedding
from .modules import Block

class MarlinEncoder(nn.Module):

    def __init__(self, img_size=[80, 80, 64], patch_size=16, embed_dim=768, depth=12,
        num_heads=12, mlp_hidden_dim=1024., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        norm_layer="LayerNorm", init_values=0., n_channel=1, expand_channel_dim = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding3d(
            input_size=(n_channel, img_size[0], img_size[1], img_size[2]),
            patch_size=(patch_size, patch_size, patch_size),
            embedding=embed_dim
        )
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.expand_channel_dim = expand_channel_dim

        # sine-cosine positional embeddings
        self.pos_embedding = SinCosPositionalEmbedding((num_patches, embed_dim), dropout_rate=0.)

        if norm_layer == "LayerNorm":
            self.norm_layer = LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")

        self.blocks = ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=self.norm_layer,
                init_values=init_values)
            for _ in range(depth)
        ])

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        # mask: (B, T, N) with boolean values, 0 -> masked, 1 -> visible
        assert len(x.shape) == 5, "x must be 5D"
        emb = self.patch_embedding(x)
        emb = self.pos_embedding(emb)
        emb = self.forward_features(emb)
        if self.expand_channel_dim:
            emb = torch.unsqueeze(emb, 1)
        return emb

    def extract_features(self, x: Tensor, seq_mean_pool: bool) -> Tensor:
        x = self.patch_embedding(x)
        x = self.pos_embedding(x)
        for block in self.blocks:
            x = block(x)

        if seq_mean_pool:
            x = x.mean(dim=1)
        x = self.norm(x)
        return x