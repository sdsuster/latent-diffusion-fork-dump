from typing import Optional

import pytorch_lightning as pl
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from ldm.util import instantiate_from_config
from taming.modules.vqvae.quantize import VectorQuantizer2
from packaging import version

# from ..config import resolve_config, Downloadable
# from ..face_detector import FaceXZooFaceDetector

from functools import partial
from monai.inferers.utils import sliding_window_inference
import time

from ..optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from ..modules.decoder import ViTDecoder
from ..modules.encoder import VitEncoder
from monai.transforms import Activations, AsDiscrete, Compose
from monai.data import decollate_batch
import numpy as np
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

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        opt1, opt2 = self.optimizers()
        xrec= self(x) #, qloss, ind 

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
        xrec = self(x) #, qloss, ind
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
    
class AE_GanBase(ABC, torch.nn.Module):
    
    def __init__(self,) -> None:
        super().__init__()
        
    @property
    @abstractmethod
    def encoder(self):
        pass

    @property
    @abstractmethod
    def decoder(self):
        pass

    def encode(self, x):
        return self.encoder()(x)

    def encode_to_prequant(self, x):
        h = self.encoder()(x)
        return h

    def decode(self, quant):
        dec = self.decoder()(quant)
        return dec

    @abstractmethod
    def get_last_layer(self):
        pass

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        latent = self.encode(input) #, diff, (_,_,ind)
        dec = self.decode(latent)
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
                 activation_fn = 'sigmoid',
                 to_one_hot = False,
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
        self.activation_fn = activation_fn
        self.to_one_hot = to_one_hot
        
        self.model_inferer = partial(
        sliding_window_inference,
        roi_size=modelconfig['params']['img_size'],
        sw_batch_size=4,
        predictor=self,
        overlap=0.1,
    )
        if to_one_hot:
            self.post_label = AsDiscrete(to_onehot=modelconfig['params']['out_channels'])
            self.post_trans = Compose([
                                        Activations(sigmoid= self.activation_fn == 'sigmoid', softmax= self.activation_fn == 'softmax'),
                                        AsDiscrete(threshold=0.5, to_onehot=modelconfig['params']['out_channels'])
                                       ])
        else:
            self.post_label = None
            self.post_trans = Compose([
                                        Activations(sigmoid= self.activation_fn == 'sigmoid', softmax= self.activation_fn == 'softmax'),
                                       AsDiscrete(threshold=0.5)])

        if first_stage_weights is not None:
            self.load_encoder_weights_frozen()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = self.get_input(batch, self.image_key)
        model_start_time = time.time()  # Start the timer
        pred= self(x) #, qloss, ind 
        model_end_time = time.time()  # End the timer
        model_time = model_end_time - model_start_time  # Calculate elapsed time

        loss = self.loss(pred.contiguous(), target.contiguous())
        loss = torch.mean(loss)
        self.log(f"train/model_time", model_time,
                   prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"train/dice_loss", loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        # with torch.no_grad():
        #     pred_post = self.post_trans(pred)
                
        #     self.dice_acc.reset()
        #     self.dice_acc(y_pred=pred_post, y=target)
        #     acc, not_nans = self.dice_acc.aggregate()
        #     for i, v in enumerate(acc):
        #         self.log(f"train/dice_acc{i}", acc[0],
        #                 prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        model_total_time = time.time()
        self.log(f"train/model_total_time", model_total_time - model_start_time,
                   prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
        
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.model.parameters()),
                                  lr=lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            opt_ae, warmup_epochs=30, max_epochs=self.trainer.max_epochs
        )
        return {
            "optimizer": opt_ae,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Update every epoch
                "frequency": 1,       # Step every epoch
            },
        }

    def get_input(self, batch, k):
        
        if isinstance(batch, list):
            imgs = []
            labels = []
            for i in range(len(batch)):
                imgs.append(batch[i][k])
                labels.append(batch[i]['label'])
            return torch.concat(imgs), torch.concat(labels)
        
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
    
    def log_images(self, batch, only_inputs=False, plot_ema=False, split='train', **kwargs):
        log = dict()
        x, y = self.get_input(batch, self.image_key)
        y = y.to(torch.float)

        # x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x[:, :, :, :, 30]
            return log
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                if split != 'train':
                    xrec= self.model_inferer(x)
                else:
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
        prob = torch.sigmoid(class_labels).cpu().numpy()
        # prob = class_labels
        # class_labels = torch.argmax(class_labels, dim=1)  # Shape: [batch_size, height, width]
        seg = (prob > 0.5).astype(np.int8)
        
        color_images = np.zeros(seg.shape)

        color_images[:, 2][seg[:, 2] == 1] = 1.
        color_images[:, 1][seg[:, 1] == 1] = 1.
        color_images[:, 0][seg[:, 0] == 1] = 1.
        # color_images[:, 0][seg[:, 0] == 0] = 0.
        # color_images[:, 1][seg[:, 1] == 0] = 0.
        if split != 'train':
            log["original"] = batch['original_image'][:, 1, :, :, 30]
        log["inputs"] = x[:, :, :, :, 30]
        log["prediction"] = torch.tensor(color_images)

        return log

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        
        x, target = self.get_input(batch, self.image_key)
        pred= self.model_inferer(x) #, qloss, ind 

        loss = self.loss(pred.contiguous(), target.contiguous())
        loss = torch.mean(loss)
        self.log(f"val/dice_loss", loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        pred_post = self.post_trans(pred)

        if self.post_label is not None:
            label_post = self.post_label(target)  
        else: 
            label_post = target
            
        self.dice_acc.reset()
        self.dice_acc(y_pred=pred_post, y=label_post)
        acc, not_nans = self.dice_acc.aggregate()
        # print(target, pred_post, torch.mean(pred_post - target))
        for i, v in enumerate(acc):
            self.log(f"train/dice_acc{i}", v,
                    prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val/dice_acc", torch.mean(acc),
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
    
class ViT_Seg(AE_GanBase):  
    def __init__(self,
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
                 ):
    
        super().__init__()
        self._encoder = VitEncoder(
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
            patch_strides=patch_strides,
            expand_channel_dim=True,
            n_channel=4
        )
        self._decoder = ViTDecoder(
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
            patch_strides=patch_strides,
            n_channels=3,
            expand_channel_dim=True
            )


    def get_last_layer(self):
        return self.decoder().head.weight
    
    def decode(self, latent):
        dec = self.decoder()(latent)
        dec = self.decoder().unpatch_to_img(dec)
        return dec

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

class ViT_AE_Model(AE_GanBase):

    def __init__(self,
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
        super().__init__()
        self._encoder = VitEncoder(
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
            patch_strides=patch_strides,
            expand_channel_dim=True
        )
        self.as_feature_extractor = as_feature_extractor
        if as_feature_extractor:
            self.enc_dec_proj = None
            self._decoder = None
        else:
            self._decoder = ViTDecoder(
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
                patch_strides=patch_strides,
                expand_channel_dim=True
            )
    
    def get_last_layer(self):
        return self.decoder().head.weight
    
    def decode(self, latent):
        dec = self.decoder()(latent)
        dec = self.decoder().unpatch_to_img(dec)
        return dec

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
