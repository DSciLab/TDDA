import os
import pickle
import random
import numpy as np
import nibabel as nib


class PatientData(object):
    INFO = 'info.pickle'
    ED = 'ED'
    ES = 'ES'

    ed_data_name = 'ed_data.npy'
    ed_data_affine_name = 'ed_data_affine.npy'
    es_data_name = 'es_data.npy'
    es_data_affine_name = 'es_data_affine.npy'

    ed_gt_name = 'ed_gt.npy'
    ed_gt_affine_name = 'ed_gt_affine.npy'
    es_gt_name = 'es_gt.npy'
    es_gt_affine_name = 'es_gt_affine.npy'

    unlabeled_frame_name = 'frame_{}.npy'
    unlabeled_frame_affine_name = 'frame_{}_affine.npy'

    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'

        self.id = self.path.split('/')[-2]
        self.ed_vol = None
        self.ed_gt = None
        self.es_vol = None
        self.es_gt = None
        self.vol_4d = None
        self.info = {}

        self.read_info()
        self.read_ed()
        self.read_es()
        self.read_4d()

    def read_info(self):
        info_path = os.path.join(self.path, 'Info.cfg')
        with open(info_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line_splited = line.split(':')
            key = line_splited[0].strip()
            val = line_splited[1].strip()

            try:
                if '.' in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass

            setattr(self, key, val)
            self.info[key] = val

    def read_ed(self):
        ed = self.ED
        if ed < 10:
            vol_path = self.path + f'{self.id}_frame0{ed}.nii.gz'
            gt_path = self.path + f'{self.id}_frame0{ed}_gt.nii.gz'
        else:
            vol_path = self.path + f'{self.id}_frame{ed}.nii.gz'
            gt_path = self.path + f'{self.id}_frame{ed}_gt.nii.gz'

        self.ed_vol = read_nii_gz(vol_path)
        self.ed_gt = read_nii_gz(gt_path)

    def read_es(self):
        es = self.ES
        if es < 10:
            vol_path = self.path + f'{self.id}_frame0{es}.nii.gz'
            gt_path = self.path + f'{self.id}_frame0{es}_gt.nii.gz'
        else:
            vol_path = self.path + f'{self.id}_frame{es}.nii.gz'
            gt_path = self.path + f'{self.id}_frame{es}_gt.nii.gz'

        self.es_vol = read_nii_gz(vol_path)
        self.es_gt = read_nii_gz(gt_path)

    def read_4d(self):
        vol_path = self.path + f'{self.id}_4d.nii.gz'
        self.vol_4d = read_nii_gz(vol_path)

    @staticmethod
    def _write_npy(data, path):
        np.save(path, data)

    @staticmethod
    def _write_pickle(data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def write_info(self, path):
        path = os.path.join(path, self.INFO)
        self._write_pickle(self.info, path)

    def write_labeled(self, path):
        # write vol
        ed_nib_vol, ed_data, ed_affine = self.ed_vol
        es_nib_vol, es_data, es_affine = self.es_vol

        self.ed_data_path = os.path.join(path, self.ed_data_name)
        self.ed_data_affine_path = os.path.join(path, self.ed_data_affine_name)
        self.es_data_path = os.path.join(path, self.es_data_name)
        self.es_data_affine_path = os.path.join(path, self.es_data_affine_name)

        self._write_npy(ed_data, os.path.join(path, self.ed_data_name))
        self._write_npy(ed_affine, os.path.join(path, self.ed_data_affine_name))
        self._write_npy(es_data, os.path.join(path, self.es_data_name))
        self._write_npy(es_affine, os.path.join(path, self.es_data_affine_name))

        # write gt
        ed_nib_gt, ed_data_gt, ed_affine_gt = self.ed_gt
        es_nib_gt, es_data_gt, es_affine_gt = self.es_gt

        self.ed_gt_path = os.path.join(path, self.ed_gt_name)
        self.ed_gt_affine_path = os.path.join(path, self.ed_gt_affine_name)
        self.es_gt_path = os.path.join(path, self.es_gt_name)
        self.es_gt_affine_path = os.path.join(path, self.es_gt_affine_name)

        self._write_npy(ed_data_gt, os.path.join(path, self.ed_gt_name))
        self._write_npy(ed_affine_gt, os.path.join(path, self.ed_gt_affine_name))
        self._write_npy(es_data_gt, os.path.join(path, self.es_gt_name))
        self._write_npy(es_affine_gt, os.path.join(path, self.es_gt_affine_name))

    def write_unlabeled(self, path):
        # write vol
        nib_vol, data, affine = self.vol_4d
        frames_arr = nib_t_slice(nib_vol)
        self.unlabeled_path_lst = []

        for i, (data, affine) in enumerate(frames_arr):
            frame_path = os.path.join(path, self.unlabeled_frame_name.format(i))
            frame_affine_path = os.path.join(path, self.unlabeled_frame_affine_name.format(i))
            self._write_npy(data, frame_path)
            self._write_npy(affine, frame_affine_path)
            self.unlabeled_path_lst.append((frame_path, frame_affine_path))

    def get_meta(self):
        return {
            'es_data_path': self.es_data_path,
            'es_data_affine_path': self.es_data_affine_path,
            'es_gt_path': self.es_gt_path,
            'es_gt_affine_path': self.es_gt_affine_path,

            'ed_data_path': self.ed_data_path,
            'ed_data_affine_path': self.ed_data_affine_path,
            'ed_gt_path': self.ed_gt_path,
            'ed_gt_affine_path': self.ed_gt_affine_path,

            'unlabeled_path_lst': self.unlabeled_path_lst
        }


def read_nii_gz(path):
    nib_vol = nib.load(path)
    data = nib_vol.get_fdata()
    affine = nib_vol.affine
    return nib_vol, data, affine


def nib_t_slice(nib_vol):
    frames = nib.funcs.four_to_three(nib_vol)
    frames_arr = []
    for frame in frames:
        data = frame.get_fdata()
        affine = frame.affine
        frames_arr.append((data, affine))
    return frames_arr


def n4_bias_correct(vol, factor):
    pass


def train_val_split(opt, data_source):
    num_data = len(data_source)
    num_val = int(opt.val_proportion * num_data)
    num_train = num_data - num_val

    random.shuffle(data_source)
    train_data = data_source[:num_train]
    val_data = data_source[num_train:]

    yield train_data, val_data


def k_fold_data(all_fold_data):
    for i in range(len(all_fold_data)):
        train_data = []
        for j in range(len(all_fold_data)):
            if i != j:
                train_data += all_fold_data[j]
            else:
                val_data = all_fold_data[j]

        yield train_data, val_data


def get_data_source(opt):
    if opt.train_mod == 'split':
        data_source = load_pickle(opt.train_meta_path)
        return train_val_split(opt, data_source)
    elif opt.train_mod == 'k_fold':
        data_source = load_pickle(opt.k_fold_meta_path)
        return k_fold_data(data_source)
    else:
        raise RuntimeError(
            f'Unrecognized train mode ({opt.train_mod})')


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
