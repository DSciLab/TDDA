from .unet3d import UNet3DTrainer


def get_trainer(opt):
    if opt.arch == 'unet3d':
        return UNet3DTrainer(opt)
    else:
        raise RuntimeError(f'Unrecognized arch ({opt.arch}).')
