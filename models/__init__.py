from .unet2d import UNet2D
from .unet3d import UNet3D, ResidualUNet3D


def get_network(opt):
    network = opt.network

    if network == 'UNet2D':
        return UNet2D(opt)
    elif network == 'UNet3D':
        return UNet3D(opt)
    elif network == 'ResidualUNet3D':
        ResidualUNet3D(opt)
    else:
        raise RuntimeError(f'Unrecognized network ({network})')
