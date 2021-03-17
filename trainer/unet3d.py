from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .base import BaseTrainer
from utils.losses import DiceLoss
from models.unet3d import UNet3D


class UNet3DTrainer(BaseTrainer):
    def __init__(self, opt):
        super().__init__(opt)
        net = UNet3D(opt)
        self.net = self.to_gpu(net)
        self.optimizer = Adam(net.parameters(),
                              lr=opt.lr,
                              betas=tuple(opt.betas),
                              weight_decay=opt.weight_decay)
        # self.lr_scheduler = ReduceLROnPlateau()
        self.loss_fn = DiceLoss(opt)

    def train_step(self, item):
        vol, gt = item
        vol = self.to_gpu(vol)
        gt = self.to_gpu(gt)

        self.optimizer.zero_grad()
        logit = self.net(vol)
        loss = self.loss_fn(logit, gt)
        loss.backward()
        self.optimizer.step()
        return loss.detach(), logit, gt

    def eval_step(self, item):
        vol, gt = item
        vol = self.to_gpu(vol)
        gt = self.to_gpu(gt)

        logit = self.net(vol)
        loss = self.loss_fn(logit, gt)
        return loss.detach(), logit, gt

    def inference(self, vol):
        logit = self.net(vol)
        return logit

    def on_epoch_begin(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError

    def on_training_begin(self):
        raise NotImplementedError

    def on_training_end(self):
        raise NotImplementedError