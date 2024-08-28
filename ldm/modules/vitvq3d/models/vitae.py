import os.path
import shutil
import random
from collections import deque
from pathlib import Path
from typing import Generator, Optional
from urllib.request import urlretrieve

import pytorch_lightning as pl
import cv2
from abc import ABC, abstractmethod
import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.nn import Linear, Module
from ldm.util import instantiate_from_config
from taming.modules.vqvae.quantize import VectorQuantizer2
from packaging import version

# from ..config import resolve_config, Downloadable
# from ..face_detector import FaceXZooFaceDetector

from ..modules.decoder import MarlinDecoder
from ..modules.encoder import MarlinEncoder
# from ..util import read_video, padding_video, DownloadProgressBar


class Vit_VQ_Trainer(pl.LightningModule):

    def __init__(self,
                 modelconfig,
                 lossconfig,
                 image_key="image",
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 monitor=None,
                 ):
        super().__init__()
        self.model = instantiate_from_config(modelconfig)
        self.loss = instantiate_from_config(lossconfig)

        if monitor is not None:
            self.monitor = monitor
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor
        self.image_key = image_key
        self.automatic_optimization = False

    def forward(self, x: Tensor, return_pred_indices = False) -> Tensor:
        return self.model(x, return_pred_indices)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        opt1, opt2 = self.optimizers()
        xrec= self(x, return_pred_indices=True) #, qloss, ind 

        aeloss, log_dict_ae = self.loss(0, x, xrec, 0, self.global_step,
                                        last_layer=self.model.get_last_layer(), split="train",
                                        predicted_indices=None
                                        )

        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        opt1.zero_grad()
        self.manual_backward(aeloss)
        opt1.step()


        # if optimizer_idx == 1:
        # discriminator
        discloss, log_dict_disc = self.loss(0, x, xrec, 1, self.global_step,
                                        last_layer=self.model.get_last_layer(), split="train")
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        opt2.zero_grad()
        self.manual_backward(discloss)
        opt2.step()
        return discloss
        
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.model.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_input(self, batch, k):
        x = batch[k]
        return x
    
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        # x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x[:, :, :, :, 30]
            return log
        xrec= self(x) #, _ 
        
        log["inputs"] = x[:, :, :, :, 30]
        log["reconstructions"] = xrec[:, :, :, :, 30]

        return log

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec = self(x, return_pred_indices=True) #, qloss, ind
        aeloss, log_dict_ae = self.loss(0, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.model.get_last_layer(),
                                        split="val"+suffix,
                                        predicted_indices=None
                                        )

        discloss, log_dict_disc = self.loss(0, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.model.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=None
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict
    
class VQ_GanBase(ABC, torch.nn.Module):
    
    def __init__(self,
                 n_embed,
                 embed_dim,
                 z_channels,
                 remap=None,
                 sane_index_shape=False, ) -> None:
        super().__init__()

        self.quantize = VectorQuantizer2(n_embed, embed_dim, beta=0.25, )
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(
            embed_dim, z_channels, 1)
        
    @property
    @abstractmethod
    def encoder(self):
        pass

    @property
    @abstractmethod
    def decoder(self):
        pass

    def encode(self, x):
        h = self.encoder()(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return (quant, emb_loss, info)

    def encode_to_prequant(self, x):
        h = self.encoder()(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder()(quant)
        return dec

    @abstractmethod
    def get_last_layer(self):
        pass

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant = self.encoder()(input) #, diff, (_,_,ind)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec #, diff, ind
        return dec #, diff
    # def forward(self, x: Tensor, return_pred_indices = False) -> Tensor:
    #     if self.as_feature_extractor:
    #         raise RuntimeError(
    #             "For feature extraction, please use `extract_features` or `extract_video`.")
    #     else:
    #         x = self.encoder(x)
    #         print('encoder shape', x.shape)
    #         x = self.enc_dec_proj(x)
    #         x = self.decoder(x)
    #         print('decoder shape', x.shape)
    #     return x




class ViT_VQ_Model(VQ_GanBase):

    def __init__(self,
                 n_embed: int,
                 embed_dim: int,
                 img_size: int,
                 patch_size: int,
                 enc_dec_embed_dim: int,
                 encoder_depth: int,
                 encoder_num_heads: int,
                 decoder_depth: int,
                 decoder_num_heads: int,
                 mlp_hidden_dim: int,
                 qkv_bias: bool,
                 qk_scale: Optional[float],
                 drop_rate: float,
                 attn_drop_rate: float,
                 norm_layer: str,
                 init_values: float,
                 patch_strides: int,
                 as_feature_extractor: bool = True,
                 ):
        super().__init__(n_embed, embed_dim, 1)
        self._encoder = MarlinEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=enc_dec_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            patch_strides=patch_strides
        )
        self.as_feature_extractor = as_feature_extractor
        if as_feature_extractor:
            self.enc_dec_proj = None
            self._decoder = None
        else:
            self._decoder = MarlinDecoder(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=enc_dec_embed_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                patch_strides=patch_strides
            )
    
    def get_last_layer(self):
        return self.decoder().head.weight
    
    def decode(self, quant):
        # quant = self.post_quant_conv(quant)
        dec = self.decoder()(quant)
        dec = self.decoder().unpatch_to_img(dec)
        return dec
    # def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
    #     if self.as_feature_extractor:
    #         raise RuntimeError(
    #             "For feature extraction, please use `extract_features` or `extract_video`.")
    #     else:
    #         x = self.encoder(x)
    #         print('encoder shape', x.shape)
    #         x = self.enc_dec_proj(x)
    #         x = self.decoder(x)
    #         print('decoder shape', x.shape)
    #     return x

    def encoder(self):
        return self._encoder
    
    def decoder(self):
        return self._decoder

    def extract_features(self, x: Tensor, keep_seq: bool = True):
        """Extract features for one video clip (v)"""
        if self.training:
            return self.encoder.extract_features(x, seq_mean_pool=not keep_seq)
        else:
            with torch.no_grad():
                return self.encoder.extract_features(x, seq_mean_pool=not keep_seq)

    # @classmethod
    # def from_file(cls, model_name: str, path: str) -> "Marlin":
    #     if path.endswith(".pt"):
    #         state_dict = torch.load(path, map_location="cpu")
    #     elif path.endswith(".ckpt"):
    #         state_dict = torch.load(path, map_location="cpu")["state_dict"]

    #         discriminator_keys = [k for k in state_dict.keys() if k.startswith("discriminator")]
    #         for key in discriminator_keys:
    #             del state_dict[key]
    #     else:
    #         raise ValueError(f"Unsupported file type: {path.split('.')[-1]}")
    #     # determine if the checkpoint is full model or encoder only.
    #     for key in state_dict.keys():
    #         if key.startswith("decoder."):
    #             as_feature_extractor = False
    #             break
    #     else:
    #         as_feature_extractor = True

    #     config = resolve_config(model_name)
    #     model = cls(
    #         img_size=config.img_size,
    #         patch_size=config.patch_size,
    #         n_frames=config.n_frames,
    #         encoder_embed_dim=config.encoder_embed_dim,
    #         encoder_depth=config.encoder_depth,
    #         encoder_num_heads=config.encoder_num_heads,
    #         decoder_embed_dim=config.decoder_embed_dim,
    #         decoder_depth=config.decoder_depth,
    #         decoder_num_heads=config.decoder_num_heads,
    #         mlp_ratio=config.mlp_ratio,
    #         qkv_bias=config.qkv_bias,
    #         qk_scale=config.qk_scale,
    #         drop_rate=config.drop_rate,
    #         attn_drop_rate=config.attn_drop_rate,
    #         norm_layer=config.norm_layer,
    #         init_values=config.init_values,
    #         tubelet_size=config.tubelet_size,
    #         as_feature_extractor=as_feature_extractor
    #     )
    #     model.load_state_dict(state_dict)
    #     return model

    # @classmethod
    # def from_online(cls, model_name: str, full_model: bool = False) -> "Marlin":
    #     config = resolve_config(model_name)
    #     if not isinstance(config, Downloadable):
    #         raise ValueError(f"Model {model_name} is not downloadable.")

    #     url = config.full_model_url if full_model else config.encoder_model_url
    #     path = Path(".marlin")
    #     path.mkdir(exist_ok=True)
    #     file = path / f"{model_name}.{'full' if full_model else 'encoder'}.pt"
    #     if not file.exists():
    #         with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading Marlin model") as pb:
    #             urlretrieve(url, filename=file, reporthook=pb.update_to)
    #     return cls.from_file(model_name, str(file))

    # @classmethod
    # def clean_cache(cls, verbose: bool = True) -> None:
    #     path = Path(".marlin")
    #     if path.exists():
    #         shutil.rmtree(path)
    #         if verbose:
    #             print("Marlin checkpoints cache cleaned.")
