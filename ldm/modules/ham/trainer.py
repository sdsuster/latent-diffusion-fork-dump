from ldm.util import instantiate_from_config
import pytorch_lightning as pl
from typing import Optional
from monai.transforms import Activations, AsDiscrete, Compose
import torch
import numpy as np
from torch import Tensor

from functools import partial
from monai.inferers.utils import sliding_window_inference
import time
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

from torch.optim import Optimizer
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


class HAM_Seg_Trainer(pl.LightningModule):

    def __init__(self,
                 modelconfig,
                 lossconfig,
                 metricconfig,
                 image_key="image",
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 monitor=None,
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
        
        self.model_inferer = partial(
        sliding_window_inference,
        roi_size=modelconfig['params']['img_size'],
        sw_batch_size=4,
        predictor=self,
        overlap=0.1,
    )

        self.post_pred = AsDiscrete(argmax=False, logit_thresh=0.5)
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
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
        
        with torch.no_grad():
            pred_post = self.post_trans(pred)
                
            self.dice_acc.reset()
            self.dice_acc(y_pred=pred_post, y=target)
            acc, not_nans = self.dice_acc.aggregate()
            self.log(f"train/dice_acc", acc[0],
                    prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        #     self.log(f"train/dice_acc1", acc[1],
        #             prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        #     self.log(f"train/dice_acc2", acc[2],
        #             prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        model_total_time = time.time()
        self.log(f"train/model_total_time", model_total_time - model_start_time,
                   prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
        
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.model.parameters()),
                                  lr=lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            opt_ae, warmup_epochs=self.trainer.max_epochs//10, max_epochs=self.trainer.max_epochs
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
        return batch[k], batch['seg']
    

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
            log["inputs"] = x[:, :, :, :]
            return log
        with torch.no_grad():
            if split != 'train':
                xrec= self.model_inferer(x)
            else:
                xrec= self(x) #, _ 
        class_labels = y[:, :, :, :]
        # Step 1: Convert logits to class labels using argmax
        # prob = torch.sigmoid(class_labels)
        prob = class_labels
        class_labels = torch.argmax(class_labels, dim=1)  # Shape: [batch_size, height, width]
        seg = (prob > 0.5).astype(np.int8)
        
        color_images = np.zeros(seg.shape)

        color_images[:][seg[:] == 1] = 1.
        log["labels"] = torch.tensor(color_images)

        class_labels = xrec[:, :, :, :]
        prob = torch.sigmoid(class_labels).cpu().numpy()
        # prob = class_labels
        # class_labels = torch.argmax(class_labels, dim=1)  # Shape: [batch_size, height, width]
        seg = (prob > 0.5).astype(np.int8)
        
        color_images = np.zeros(seg.shape)

        color_images[:][seg[:] == 1] = 1.
        # color_images[:, 0][seg[:, 0] == 0] = 0.
        # color_images[:, 1][seg[:, 1] == 0] = 0.

        log["inputs"] = x[:, :, :, :]
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
            
        self.dice_acc.reset()
        self.dice_acc(y_pred=pred_post, y=target)
        acc, not_nans = self.dice_acc.aggregate()
        # print(target, pred_post, torch.mean(pred_post - target))
        self.log(f"val/dice_acc", acc[0],
                prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    