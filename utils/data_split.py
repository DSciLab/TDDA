import os
import pickle
import random
from mlutils import Log


def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def meta_data_source(opt):
    with open(opt.meta_path, 'rb') as f:
        data = pickle.load(f)
    return data


def train_test_split(opt):
    data_source = meta_data_source(opt)

    num_data = len(data_source)
    num_test = int(opt.test_proportion * num_data)
    num_train = num_data - num_test

    Log.info(f'num data: {num_data}')
    Log.info(f'num train data: {num_train}')
    Log.info(f'num test data: {num_test}')

    random.shuffle(data_source)
    train_data = data_source[:num_train]
    test_data = data_source[num_train:]

    train_meta_path = opt.train_meta_path
    test_meta_path = opt.test_meta_path

    save_pickle(train_data, train_meta_path)
    save_pickle(test_data, test_meta_path)


def k_fold_cross_val(opt):
    data_source = meta_data_source(opt)

    num_data = len(data_source)
    num_per_fold = num_data // opt.k_fold
    Log.info(f'num data per fold: {num_per_fold}')

    folds = []
    for i in range(opt.k_fold):
        if i == opt.k_fold - 1:
            folds.append(data_source)
        else:
            curr_fold_data = data_source[:num_per_fold]
            data_source = data_source[num_per_fold:]
            folds.append(curr_fold_data)

    k_fold_meta_path = opt.k_fold_meta_path
    save_pickle(folds, k_fold_meta_path)


def split_data(opt):
    # generate data split
    train_test_split(opt)
    k_fold_cross_val(opt)
