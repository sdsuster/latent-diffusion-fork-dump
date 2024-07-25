import argparse, os, sys, glob
import clip
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from itertools import islice
import time
from multiprocessing import cpu_count

from ldm.util import instantiate_from_config, parallel_data_prefetch

DATABASES = [
    "openimages",
    "artbench-art_nouveau",
    "artbench-baroque",
    "artbench-expressionism",
    "artbench-impressionism",
    "artbench-post_impressionism",
    "artbench-realism",
    "artbench-romanticism",
    "artbench-renaissance",
    "artbench-surrealism",
    "artbench-ukiyo_e",
]


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

from monai import transforms

brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image"], allow_missing_keys=True),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAI", allow_missing_keys=True),
        transforms.CropForegroundd(keys=["image"], allow_smaller=True, source_key="image", allow_missing_keys=True),
        transforms.SpatialPadd(keys=["image"], spatial_size=(160, 160, 126), allow_missing_keys=True),
        transforms.RandSpatialCropd( keys=["image"],
            roi_size=(80, 80, 60),
            random_center=True, 
            random_size=False,
        ),
        # transforms.Resized( keys=["image"],
        #     spatial_size=(80, 80, 60),
        #     anti_aliasing=True, 
        # ),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99.75, b_min=0, b_max=1),
    ]
)

import PIL.Image as PILImage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: add n_neighbors and modes (text-only, text-image-retrieval, image-image retrieval etc)
    # TODO: add 'image variation' mode when knn=0 but a single image is given instead of a text prompt?
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )

    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval-augmented-diffusion/768x768.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/rdm/rdm768x768/model.ckpt",
        help="path to checkpoint of model",
    )



    # opt = parser.parse_args()

    # config = OmegaConf.load(f"{opt.config}")
    # model = load_model_from_config(config, f"{opt.ckpt}")

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = model.to(device)

    a = brats_transforms({'image':'D:\\Datasets\\BraTS2020\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\BraTS20_Training_001\\BraTS20_Training_001_t2.nii'})['image']
    # a = a.to(device)
    a = a.unsqueeze(0)
    with torch.no_grad():
        # with model.ema_scope():
            PILImage.fromarray((a[0, 0, :, :, 30].cpu().detach().numpy()*255.).astype(np.uint8)).show()
            # h = model.encoder(a)
            # h = model.quant_conv(h)

            # quant = model.post_quant_conv(h)
            # dec = model.decoder(quant)
            # print(dec.shape)
            # PILImage.fromarray(dec[0, 0, :, :, 30].cpu().detach().numpy()*200.).show()
            # return dec

    # print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")
