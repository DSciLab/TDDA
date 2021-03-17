from .acdc import ACDCDataset
from .utils import get_data_source


def get_dataset(opt):
    for train_data_source, val_data_source in get_data_source(opt):
        train_dataset = ACDCDataset(opt, train_data_source,
                                    train=True, labeled=True)
        val_dataset = ACDCDataset(opt, val_data_source,
                                    train=False, labeled=True)
        yield train_dataset, val_dataset
