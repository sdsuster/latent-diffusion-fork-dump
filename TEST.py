from ldm.models.autoencoder import VQModel3dInterface
import torch
print(torch.cuda.is_available())
config = {
    "double_z": False,
    "z_channels": 3,
    "resolution": 192,
    "in_channels": 1,
    "out_ch": 1,
    "ch": 128,
    "ch_mult": [1, 1, 2],
    "num_res_blocks": 2,
    "attn_resolutions": [8],
    "dropout": 0.0,
    'attn_type': 'vanilla-3d'
}
model = VQModel3dInterface(3, n_embed= 8192, ddconfig=config, lossconfig={'target': 'torch.nn.Identity'})
model.to(torch.device('cuda'))
a = model.encode((torch.zeros(size=(1, 1, 80, 80, 80, )).to(torch.device('cuda'))))
print(a.shape)
a = model.decode(a)
print(a.shape)