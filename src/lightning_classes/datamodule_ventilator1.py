import os

import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

from src.utils.technical_utils import load_obj
import gc

class ImagenetteDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def make_features(self, data):
        print('Making features')
        if "pressure" not in data.columns:
            data['pressure'] = 0

        data['RC_sum'] = data['R'] + data['C']
        data['RC_div'] = data['R'] / data['C']
        data['R'] = data['R'].astype(str)
        data['C'] = data['C'].astype(str)
        data['RC'] = data['R'] + data['C']
        data = pd.get_dummies(data)
        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()
        # data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(1)
        data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(2)
        data['u_in_lag'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_out_lag'] = data.groupby('u_out')['u_in'].shift(1)
        return data.fillna(0)

    def setup(self, stage=None):
        if self.cfg.training.debug:
            train = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'train.csv'), nrows=196000)
            test = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'test.csv'), nrows=80000)
        else:
            train = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'train.csv'))
            test = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'test.csv'))

        train = self.make_features(train)
        test = self.make_features(test)

        targets = train[['pressure']].to_numpy().reshape(-1, 80)
        test_targets = test[['pressure']].to_numpy().reshape(-1, 80)
        train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
        test = test.drop(['id', 'breath_id', 'pressure'], axis=1)

        train_u_out = train[['u_out']].to_numpy().reshape(-1, 80)
        test_u_out = test[['u_out']].to_numpy().reshape(-1, 80)


        RS = RobustScaler()
        train = RS.fit_transform(train)
        test = RS.transform(test)

        train = train.reshape(-1, 80, train.shape[-1])
        test = test.reshape(-1, 80, train.shape[-1])
        gkf = KFold(n_splits=self.cfg.datamodule.n_folds, shuffle=True, random_state=self.cfg.training.seed)
        # if self.cfg.datamodule.split == 'GroupKFold':
        #     gkf = GroupKFold(n_splits=self.cfg.datamodule.n_folds)
        # elif self.cfg.datamodule.split == 'GroupShuffleSplit':
        #     gkf = GroupShuffleSplit(n_splits=self.cfg.datamodule.n_folds, random_state=self.cfg.training.seed)

        splits = list(gkf.split(X=train, y=train))
        train_idx, valid_idx = splits[self.cfg.datamodule.fold_n]

        train_df = train[train_idx].copy()
        valid_df = train[valid_idx].copy()

        train_u_out_ = train_u_out[train_idx].copy()
        valid_u_out_ = train_u_out[valid_idx].copy()

        train_targets = targets[train_idx].copy()
        valid_targets = targets[valid_idx].copy()

        del train
        gc.collect()

        # train dataset
        dataset_class = load_obj(self.cfg.datamodule.class_name)

        self.train_dataset = dataset_class(train_df, mode='train', normalize=self.cfg.datamodule.normalize, u_out=train_u_out_, pressure=train_targets)
        self.valid_dataset = dataset_class(valid_df, mode='valid', normalize=self.cfg.datamodule.normalize, u_out=valid_u_out_, pressure=valid_targets)
        self.test_dataset = dataset_class(test, mode='test', normalize=self.cfg.datamodule.normalize, u_out=test_u_out, pressure=test_targets)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=False,
            drop_last=True
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