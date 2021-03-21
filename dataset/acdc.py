import torch
import numpy as np
from vox import transform as voxtf
from torch.utils.data.dataset import Dataset


class ACDCDataset(Dataset):
    def __init__(self, opt, data_source, train=True, labeled=True):
        super().__init__()
        self.opt = opt
        self.train = train
        self.labeled = labeled
        self.data_source = data_source
        self.transform = voxtf.Sequantial([
            voxtf.FixChannels(),
            voxtf.PadAndRandomSampling(opt.least_shape, opt.input_shape),
            voxtf.ToTensor()
        ])
        # TODO data transform
        # TODO gt transform

        self.unlabeled_data = []
        for item in self.data_source:
            self.unlabeled_data += item['unlabeled_path_lst']

        self.labeled_data = []
        for item in self.data_source:
            self.labeled_data.append({'data_path': item['es_data_path'],
                                      'data_affine_path': item['es_data_affine_path'],
                                      'gt_path': item['es_gt_path'],
                                      'gt_affine_path': item['es_gt_affine_path']})

            self.labeled_data.append({'data_path': item['ed_data_path'],
                                      'data_affine_path': item['ed_data_affine_path'],
                                      'gt_path': item['ed_gt_path'],
                                      'gt_affine_path': item['ed_gt_affine_path']})

    def __len__(self):
        if self.labeled is True:
            return len(self.labeled_data)
        else:
            return len(self.unlabeled_data)

    @staticmethod
    def read_npy(path):
        return np.load(path)

    def get_labeled(self, index):
        data = self.labeled_data[index]

        data_path = data['data_path']
        data_affine_path = data['data_affine_path']
        gt_path = data['gt_path']
        gt_affine_path = data['gt_affine_path']

        data = self.read_npy(data_path)
        gt = self.read_npy(gt_path)
        data, gt = self.transform(data, gt)
        return data, gt

    def get_unlabeled(self, index):
        data = self.unlabeled_data[index]

        unlabeled_path = data['unlabeled_path_lst']
        frame_path, frame_affine_path = unlabeled_path
        frame = self.read_npy(frame_path)
        # frame_affine = self.read_npy(frame_affine_path)
        frame = self.transform(frame)
        return frame

    def __getitem__(self, index):
        if self.labeled:
            return self.get_labeled(index)
        else:
            return self.unlabeled_data(index)
