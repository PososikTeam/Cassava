import argparse
import collections
import json
import multiprocessing
import os
import numpy as np
from datetime import datetime

import torch
import catalyst
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback
from catalyst.utils import load_checkpoint, unpack_checkpoint

from pytorch_toolbelt.utils.random import set_manual_seed, get_random_name
from dataset import get_datasets, get_dataloaders, get_datasets_universal
from model import get_model
from loss import LabelSmoothingLoss, TemperedLogLoss, get_loss
from callbacks import CappaScoreCallback, CosineLossCallback
from optimizer import get_optim
from catalyst.dl import AccuracyCallback, OptimizerCallback, CheckpointCallback
from torch import nn
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('-dd', '--data-dir', type=str, default='data', help='Data directory')

    parser.add_argument('-l', '--loss', type = str, default = 'label_smooth_cross_entropy')
    parser.add_argument('-t1', '--temper1', type = float, default = 0.2)
    parser.add_argument('-t2', '--temper2', type = float, default = 4.0)
    parser.add_argument('-optim', '--optimizer', type = str, default = 'adam')

    parser.add_argument('-prep', '--prep_function', type = str, default='none')

    parser.add_argument('--train_on_different_datasets', action='store_true')
    parser.add_argument('--use-current', action='store_true')
    parser.add_argument('--use-extra', action='store_true')
    parser.add_argument('--use-unlabeled', action='store_true')

    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--balance-datasets', action='store_true')

    parser.add_argument('--show', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    parser.add_argument('-m', '--model', type=str, default='efficientnet-b4', help='')
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Epoch to run')
    parser.add_argument('-s', '--sizes', default=380, type=int, help='Image size for training & inference')
    parser.add_argument('-f', '--fold', type=int, default=None)
    parser.add_argument('-t', '--transfer', default=None, type=str, help='')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('-a', '--augmentations', default='medium', type=str, help='')
    parser.add_argument('-accum', '--accum-step', type = int, default = 1)
    parser.add_argument('-metric', '--metric', type = str, default = 'accuracy01')
    

    args = parser.parse_args()

    diff_dataset_train = args.train_on_different_datasets

    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size
    seed = args.seed

    loss_name = args.loss
    optim_name = args.optimizer

    prep_function = args.prep_function
    
    model_name = args.model
    size = args.sizes,
    print(size)
    print(size[0])
    image_size = (size[0], size[0])
    print(image_size)
    fast = args.fast
    fold = args.fold
    mixup = args.mixup
    balance = args.balance
    balance_datasets = args.balance_datasets
    show_batches = args.show
    verbose = args.verbose
    use_current = args.use_current
    use_extra = args.use_extra
    use_unlabeled = args.use_unlabeled

    learning_rate = args.learning_rate
    augmentations = args.augmentations
    transfer = args.transfer
    accum_step = args.accum_step

    #cosine_loss    accuracy01
    main_metric = args.metric

    print(data_dir)

    num_classes = 5

    assert use_current or use_extra

    print(fold)

    current_time = datetime.now().strftime('%b%d_%H_%M')
    random_name = get_random_name()



    current_time = datetime.now().strftime('%b%d_%H_%M')
    random_name = get_random_name()

    # if folds is None or len(folds) == 0:
    #     folds = [None]


    torch.cuda.empty_cache()
    checkpoint_prefix = f'{model_name}_{size}_{augmentations}'

    if transfer is not None:
        checkpoint_prefix += '_pretrain_from_'+str(transfer)
    else:
        if use_current:
            checkpoint_prefix += '_current'
        if use_extra:
            checkpoint_prefix += '_extra'
        if use_unlabeled:
            checkpoint_prefix += '_unlabeled'
        if fold is not None:
            checkpoint_prefix += f'_fold{fold}'


    directory_prefix = f'{current_time}_{checkpoint_prefix}'
    log_dir = os.path.join('runs', directory_prefix)
    os.makedirs(log_dir, exist_ok=False)



    set_manual_seed(seed)
    model = get_model(model_name)

    if transfer is not None:
        print("Transfering weights from model checkpoint")
        model.load_state_dict(torch.load(transfer)['model_state_dict'])


    model = model.cuda()

    if diff_dataset_train:
        train_on = ['current_train', 'extra_train']
        valid_on = ['unlabeled'] 
        train_ds, valid_ds, train_sizes = get_datasets_universal(train_on = train_on,
                                                        valid_on = valid_on,
                                                        image_size=image_size,
                                                        augmentation=augmentations,
                                                        target_dtype=int,
                                                        prep_function = prep_function)
    else:
        train_ds, valid_ds, train_sizes = get_datasets(data_dir=data_dir,
                                                    use_current=use_current,
                                                    use_extra=use_extra,
                                                    image_size=image_size,
                                                    prep_function = prep_function,
                                                    augmentation=augmentations,
                                                    target_dtype=int,
                                                    fold=fold,
                                                    folds=5)

    train_loader, valid_loader = get_dataloaders(train_ds, valid_ds,
                                                batch_size=batch_size,
                                                train_sizes=train_sizes,
                                                num_workers = 6,
                                                balance=True,
                                                balance_datasets=True,
                                                balance_unlabeled=False)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    runner = SupervisedRunner(input_key='image')

    criterions = get_loss(loss_name)
    # criterions_tempered = TemperedLogLoss()
    # optimizer = catalyst.contrib.nn.optimizers.radam.RAdam(model.parameters(), lr = learning_rate)
    optimizer = get_optim(optim_name, model, learning_rate)
    # optimizer = catalyst.contrib.nn.optimizers.Adam(model.parameters(), lr = learning_rate)
    # criterions = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.8)
    # cappa = CappaScoreCallback()

    Q = math.floor(len(train_ds)/ batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = Q)
    if main_metric != 'accuracy01':
        callbacks = [AccuracyCallback(num_classes=num_classes), CosineLossCallback(),
                 OptimizerCallback(accumulation_steps=accum_step), CheckpointCallback(save_n_best=epochs)]
    else:
        callbacks = [AccuracyCallback(num_classes=num_classes), OptimizerCallback(accumulation_steps=accum_step),
                     CheckpointCallback(save_n_best=epochs)]


    # main_metric = 'accuracy01'

    runner.train(
        fp16=True,
        model=model,
        criterion=criterions,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        loaders=loaders,
        logdir=log_dir,
        num_epochs=epochs,
        verbose=verbose,
        main_metric=main_metric,
        minimize_metric=False,
    )

       

if __name__ == '__main__':
    main()