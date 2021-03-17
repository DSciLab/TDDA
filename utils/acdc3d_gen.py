import os
import shutil
import glob
import pickle
from cfg import Opts
import tqdm

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

        self.patient_path_lst = self.read_patients()
        self.all_meta_data = []

    def read_patients(self):
        wildcard_path = os.path.join(self.opt.raw_path, 'patient*')
        patient_path_lst = []
        for path in glob.glob(wildcard_path):
            patient_path_lst.append(path)
        return patient_path_lst

    def construct_patient(self, patient_path):
        patient = PatientData(patient_path)
        return patient

    def gen_patient_data(self, patient):
        patient_path = os.path.join(self.opt.generated_path, patient.id)
        labeled_path = os.path.join(patient_path, self.LABELED_DIR)
        unlabeled_path = os.path.join(patient_path, self.UNLABELED_DIR)
        os.mkdir(patient_path)
        os.mkdir(labeled_path)
        os.mkdir(unlabeled_path)

        patient.write_info(patient_path)
        patient.write_labeled(labeled_path)
        patient.write_unlabeled(unlabeled_path)

    def save_meta(self):
        with open(self.opt.meta_path, 'wb') as f:
            pickle.dump(self.all_meta_data, f)

    def start(self):
        for patient_path in tqdm.tqdm(self.patient_path_lst):
            patient = self.construct_patient(patient_path)
            self.gen_patient_data(patient)
            meta = patient.get_meta()
            self.all_meta_data.append(meta)

        self.save_meta()


def gen(opt):
    generator = DataGenerator(opt)
    generator.start()


if __name__ == '__main__':
    Opts.add_int('ps', 10)
    opt = Opts('conf/acdc_ds.yml')

    gen(opt)
