import os
import shutil
import glob
import pickle
from cfg import Opts

from dataset.utils import PatientData


class DataGenerator(object):
    LABELED_DIR = 'labeled'
    UNLABELED_DIR = 'unlabeled'

    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

        if os.path.exists(self.opt.generated_path):
            shutil.rmtree(self.opt.generated_path)
        os.mkdir(self.opt.generated_path)

        self.patinet_path_lst = self.read_patinets()
        self.all_meta_data = []

    def read_patinets(self):
        wildcard_path = os.path.join(self.opt.raw_path, 'patient*')
        patinet_path_lst = []
        for path in glob.glob(wildcard_path):
            patinet_path_lst.append(path)
        return patinet_path_lst

    def construct_patinet(self):
        for path in self.patinet_path_lst:
            patinet = PatientData(path)
            self.gen_patinet_data(patinet)

    def gen_patinet_data(self, patinet):
        os.mkdir(self.opt.generated_path, patinet.id)

    def save_meta(self):
        with open(self.opt.meta_path, 'wb') as f:
            pickle.dump(self.all_meta_data, f)

    def start(self):
        self.save_meta()


def main(opt):
    pass


if __name__ == '__main__':
    Opts.add_int('ps', 10)
    opt = Opts('conf/acdc_ds.yml')

    main(opt)
