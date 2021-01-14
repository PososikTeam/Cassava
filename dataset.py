import math
import os
from typing import Tuple, List

import albumentations as A
import cv2
import numpy as np
import pandas as pd
# from pytorch_toolbelt.utils import fs
# from pytorch_toolbelt.utils.fs import id_from_fname
# from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import compute_sample_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from augmentation import get_train_transform, get_test_transform
from prepare import get_prepare_function

UNLABELED_CLASS = -100



class TaskDataset(Dataset):
    def __init__(self, images, targets, img_size, prepare_function_name,
                 transform: A.Compose,
                 target_as_array=False,
                 dtype=int):
        if targets is not None:
            targets = np.array(targets)
            unique_targets = set(targets)
            if len(unique_targets.difference({0, 1, 2, 3, 4, UNLABELED_CLASS})):
                raise ValueError('Unexpected targets in Y ' + str(unique_targets))

        self.images = np.array(images)
        self.targets = targets
        self.transform = transform
        self.target_as_array = target_as_array
        self.dtype = dtype
        self.img_size = img_size
        self.prepare_function_name = prepare_function_name

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = cv2.imread(self.images[item])  # Read with OpenCV instead PIL. It's faster
        if image is None:
            raise FileNotFoundError(self.images[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prep = get_prepare_function(self.prepare_function_name)
        image = prep(image, self.img_size)
        height, width = image.shape[:2]
        label = UNLABELED_CLASS

        if self.targets is not None:
            label = self.targets[item]

        #Плохо тут это делать но пока так
        # image = prepare_img(image, self.img_size)
        data = self.transform(image=image, label=label)
        label = data['label']

        data = {'image': data['image']}

        label = self.dtype(label)
        if self.target_as_array:
            data['targets'] = np.array([label])
        else:
            data['targets'] = label

        return data





def split_train_valid(x, y, fold=None, folds=4, random_state=42):
    """
    Common train/test split function
    :param x:
    :param y:
    :param fold:
    :param folds:
    :param random_state:
    :return:
    """
    train_x, train_y = [], []
    valid_x, valid_y = [], []

    if fold is not None:
        assert 0 <= fold < folds
        skf = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)

        for fold_index, (train_index, test_index) in enumerate(skf.split(x, y)):
            if fold_index == fold:
                train_x = x[train_index]
                train_y = y[train_index]
                valid_x = x[test_index]
                valid_y = y[test_index]
                break
    else:
        train_x, valid_x, train_y, valid_y = train_test_split(x, y,
                                                              random_state=random_state,
                                                              test_size=1.0 / folds,
                                                              shuffle=True,
                                                              stratify=y)

    assert len(train_x) and len(train_y) and len(valid_x) and len(valid_y)
    assert len(train_x) == len(train_y)
    assert len(valid_x) == len(valid_y)
    return train_x, valid_x, train_y, valid_y


def get_current_train(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'train_images.csv'))
    x = np.array(df['image_id'].apply(lambda x: os.path.join(data_dir, 'train_images', f'{x}')))
    y = np.array(df['label'], dtype=int)
    return x, y


def get_extra_train(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'extra_images.csv'))
    x = np.array(df['id_code'].apply(lambda x: os.path.join(data_dir, 'extra_data', f'{x}')))
    y = np.array(df['label'], dtype=int)
    return x, y


def get_unlabeled(dataset_dir, healthy_eye_fraction=1):
    df= pd.read_csv(os.path.join(dataset_dir, 'unlabeled.csv'))
    x = np.array(df['image_id'].apply(lambda x: os.path.join(data_dir, 'unlabeled', f'{x}')))
    y = np.array([UNLABELED_CLASS] * len(x), dtype=int)
    return x, y






def append_train_test(existing, to_add):
    train_x, train_y, valid_x, valid_y = existing
    tx, vx, ty, vy = to_add
    train_x.extend(tx)
    train_y.extend(ty)
    valid_x.extend(vx)
    valid_y.extend(vy)
    return train_x, train_y, valid_x, valid_y


def get_dataset(datasets: List[str], data_dir='data', random_state=42):
    """
    :param datasets: List "aptos2015-train/fold0", "messidor",
    :param fold:
    :param folds:
    :return:
    """

    all_x = []
    all_y = []
    sizes = []

    for ds in datasets:

        dataset_name = ds

        if dataset_name == 'current_train':
            x, y = get_current_train(data_dir)
        elif dataset_name == 'extra_train':
            x, y = get_extra_train(data_dir)
        elif dataset_name == 'unlabeled':
            x, y = get_aptos2015_test(data_dir)
        else:
            raise ValueError(dataset_name)

        all_x.extend(x)
        all_y.extend(y)
        sizes.append(len(x))

    return all_x, all_y, sizes


def get_datasets_universal(
        train_on: List[str],
        valid_on: List[str],
        data_dir='data',
        prep_function = 'reshape',
        image_size=(512, 512),
        augmentation='medium',
        preprocessing=None,
        target_dtype=int,
        random_state=42,
        coarse_grading=False,
        folds=4) -> Tuple[TaskDataset, TaskDataset, List]:
    train_x, train_y, sizes = get_dataset(train_on, data_dir=data_dir, random_state=random_state)
    valid_x, valid_y, _ = get_dataset(valid_on, data_dir=data_dir, random_state=random_state)

    train_transform = get_train_transform(augmentation = augmentation, image_size = image_size)
    valid_transform = get_test_transform(image_size = image_size)

    train_ds = TaskDataset(train_x, train_y, image_size, prep_function,
                                  transform=train_transform,
                                  dtype=target_dtype)

    valid_ds = TaskDataset(valid_x, valid_y, image_size, prep_function,
                                  transform=valid_transform,
                                  dtype=target_dtype)

    return train_ds, valid_ds, sizes


def get_datasets(
        data_dir='data',
        prep_function = 'reshape',
        image_size=(512, 512),
        augmentation='medium',
        use_current=True,
        use_extra=False,
        use_unlabeled=False,
        target_dtype=int,
        random_state=42,
        fold=None,
        folds=4) -> Tuple[TaskDataset, TaskDataset, List]:
    assert use_current or use_extra or use_unlabeled

    trainset_sizes = []
    data_split = [], [], [], []

  

    if use_current:
        x, y = get_current_train(data_dir)
        split = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(len(split[0]))

    if use_extra:
        x, y = get_extra_train(data_dir)
        split = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(len(split[0]))

    if use_unlabeled:
        x, y = get_unlabeled(data_dir)
        split = split_train_valid(x, y, fold=fold, folds=folds, random_state=random_state)
        data_split = append_train_test(data_split, split)
        trainset_sizes.append(len(split[0]))

    train_x, train_y, valid_x, valid_y = data_split



    train_transform = get_train_transform(augmentation = augmentation, image_size = image_size)
    valid_transform = get_test_transform(image_size = image_size)

    train_ds = TaskDataset(train_x, train_y, image_size, prep_function,
                                      transform=train_transform,
                                      dtype=target_dtype)

    valid_ds = TaskDataset(valid_x, valid_y, image_size, prep_function,
                                  transform=valid_transform,
                                  dtype=target_dtype)

    return train_ds, valid_ds, trainset_sizes


def get_dataloaders(train_ds, valid_ds,
                    batch_size,
                    num_workers = 1,
                    fast=False,
                    train_sizes=None,
                    balance=False,
                    balance_datasets=False,
                    balance_unlabeled=False,
                    ):
    sampler = None
    weights = None
    num_samples = 0

    if balance_unlabeled:
        labeled_mask = (train_ds.targets != UNLABELED_CLASS).astype(np.uint8)
        weights = compute_sample_weight('balanced', labeled_mask)
        num_samples = int(np.mean(train_sizes))

    if balance:
        weights = compute_sample_weight('balanced', train_ds.targets)
        hist = np.bincount(train_ds.targets)
        min_class_counts = int(min(hist))
        num_classes = len(np.unique(train_ds.targets))
        num_samples = min_class_counts * num_classes

    if balance_datasets:
        assert train_sizes is not None
        dataset_balancing_term = []

        for subset_size in train_sizes:
            full_dataset_size = float(sum(train_sizes))
            dataset_balancing_term.extend([full_dataset_size / subset_size] * subset_size)

        dataset_balancing_term = np.array(dataset_balancing_term)
        if weights is None:
            weights = np.ones(len(train_ds.targets))

        weights = weights * dataset_balancing_term
        num_samples = int(np.mean(train_sizes))

    # If we do balancing, let's go for fixed number of batches (half of dataset)
    if weights is not None:
        sampler = WeightedRandomSampler(weights, num_samples)

    if fast:
        weights = np.ones(len(train_ds))
        sampler = WeightedRandomSampler(weights, 16)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=sampler is None, sampler=sampler,
                          pin_memory=True, drop_last=True, num_workers= num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          pin_memory=True, drop_last=False, num_workers= num_workers)

    return train_dl, valid_dl

