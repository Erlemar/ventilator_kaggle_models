import os

import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader

from src.utils.technical_utils import load_obj
import gc

class ImagenetteDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def make_features(self):
        pass

    def setup(self, stage=None):
        if self.cfg.training.debug:
            train = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'train.csv'), nrows=196000)
            test = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'test.csv'), nrows=80000)
        else:
            train = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'train.csv'))
            test = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'test.csv'))

        if self.cfg.datamodule.split == 'GroupKFold':
            gkf = GroupKFold(n_splits=self.cfg.datamodule.n_folds)
        elif self.cfg.datamodule.split == 'GroupShuffleSplit':
            gkf = GroupShuffleSplit(n_splits=self.cfg.datamodule.n_folds, random_state=self.cfg.training.seed)

        splits = list(gkf.split(X=train, y=train, groups=train["breath_id"].values))
        train_idx, valid_idx = splits[self.cfg.datamodule.fold_n]

        train_df = train.iloc[train_idx].copy().reset_index(drop=True)
        valid_df = train.iloc[valid_idx].copy().reset_index(drop=True)

        del train
        gc.collect()

        # train dataset
        dataset_class = load_obj(self.cfg.datamodule.class_name)

        self.train_dataset = dataset_class(train_df, mode='train', normalize=self.cfg.datamodule.normalize)
        self.valid_dataset = dataset_class(valid_df, mode='valid', normalize=self.cfg.datamodule.normalize)
        self.test_dataset = dataset_class(test, mode='test', normalize=self.cfg.datamodule.normalize)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=False,
        )

        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=False,
        )

        return test_loader