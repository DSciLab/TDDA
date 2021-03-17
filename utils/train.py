from torch.utils.data.dataloader import DataLoader
from dataset import get_dataset
from trainer import get_trainer
from mlutils import init


def train(opt):
    init(opt)
    trainer = get_trainer(opt)
    for train_dataset, val_dataset in get_dataset(opt):
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=opt.batch_size,
                                      shuffle=opt.shuffle,
                                      num_workers=opt.num_worker,
                                      pin_memory=opt.pin_memory)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=False,
                                    num_workers=opt.num_worker,
                                    pin_memory=opt.pin_memory)

        trainer.train(train_dataloader, val_dataloader)
