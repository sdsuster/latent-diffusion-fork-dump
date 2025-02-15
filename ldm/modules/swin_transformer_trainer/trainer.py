from ldm.util import instantiate_from_config
import pytorch_lightning as pl
from typing import Optional
from monai.transforms import Activations, AsDiscrete, Compose
import torch
import numpy as np
from torch import Tensor

# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import List

from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


class VisionTransformerTrainer(pl.LightningModule):

    def __init__(self,
                 modelconfig,
                 lossconfig,
                 metricconfig,
                 image_key="image",
                 scheduler_config=None,
                 monitor=None,
                 ckpt_path = None,
                 ignore_keys=[],
                 ):
        super().__init__()
        self.model = instantiate_from_config(modelconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.metric = instantiate_from_config(metricconfig)
        # self.loss = l1

        if monitor is not None:
            self.monitor = monitor
        self.scheduler_config = scheduler_config
        self.image_key = image_key
        
        # self.post_pred = AsDiscrete(argmax=False, logit_thresh=0.5)
        # self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
    def on_train_start(self):
        # print(self.trainer.train_dataloader.train_dataloader.obtain_train_property('weight'))
        w = self.trainer.datamodule.obtain_train_property('weight')
        if w is not None:
            nw = torch.Tensor(np.array(w))
            self.loss.weight = nw.to(self.device)
            print("\nUpdating training weight loss: ", self.loss.weight)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        x, target = self.get_input(batch, self.image_key)
        pred= self(x) #, qloss, ind 

        loss = self.loss(pred.contiguous(), target.contiguous())
        loss = torch.mean(loss)
        self.log(f"train/loss", loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        with torch.no_grad():
            m = self.metric(pred, target)
            self.log(f"train/metric", m,
                    prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            

        return loss
        
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.model.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        scheduler = LinearWarmupCosineAnnealingLR(opt_ae, warmup_epochs=self.trainer.max_epochs//10, max_epochs=self.trainer.max_epochs)

        # Return optimizer and scheduler
        return {
            "optimizer": opt_ae,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # "step" updates every batch, "epoch" updates every epoch
                "frequency": 1,  # Apply the scheduler every epoch
            }
        }

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
    
    # def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
    #     log = dict()
    #     x, y = self.get_input(batch, self.image_key)

    #     # x = x.to(self.device)
    #     if only_inputs:
    #         log["inputs"] = x[:, :, :, :]
    #         return log
    #     xrec= self(x) #, _ 
    #     class_labels = y[:, :, :, :, 30]
    #     # Step 1: Convert logits to class labels using argmax
    #     # prob = torch.sigmoid(class_labels)
    #     prob = class_labels
    #     class_labels = torch.argmax(class_labels, dim=1)  # Shape: [batch_size, height, width]
    #     seg = (prob > 0.5).astype(np.int8)
        
    #     color_images = np.zeros(seg.shape)

    #     color_images[:, 2][seg[:, 2] == 1] = 1.
    #     color_images[:, 1][seg[:, 1] == 1] = 1.
    #     color_images[:, 0][seg[:, 0] == 1] = 1.
    #     log["labels"] = torch.tensor(color_images)

    #     class_labels = xrec[:, :, :, :, 30]
    #     prob = torch.sigmoid(class_labels).cpu().numpy()
    #     # prob = class_labels
    #     # class_labels = torch.argmax(class_labels, dim=1)  # Shape: [batch_size, height, width]
    #     seg = (prob > 0.5).astype(np.int8)
        
    #     color_images = np.zeros(seg.shape)

    #     color_images[:, 2][seg[:, 2] == 1] = 1.
    #     color_images[:, 1][seg[:, 1] == 1] = 1.
    #     color_images[:, 0][seg[:, 0] == 1] = 1.
    #     # color_images[:, 0][seg[:, 0] == 0] = 0.
    #     # color_images[:, 1][seg[:, 1] == 0] = 0.

    #     log["inputs"] = x[:, :, :, :, 30]
    #     log["prediction"] = torch.tensor(color_images)

    #     return log

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        
        x, target = self.get_input(batch, self.image_key)
        pred= self(x) #, qloss, ind 

        loss = self.loss(pred.contiguous(), target.contiguous())
        loss = torch.mean(loss)
        self.log(f"val/loss", loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            
        m = self.metric(pred, target)
        # print(target, pred_post, torch.mean(pred_post - target))
        self.log(f"val/metric", m,
                prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    