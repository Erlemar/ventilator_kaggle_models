import gc
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

from src.utils.technical_utils import load_obj


class VentilatorDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def make_features(self, data):
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

    def make_features1(self, data):
        data['area'] = data['time_step'] * data['u_in']
        data['area'] = data.groupby('breath_id')['area'].cumsum()

        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()

        data['u_in_lag1'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_out_lag1'] = data.groupby('breath_id')['u_out'].shift(1)
        data['u_in_lag_back1'] = data.groupby('breath_id')['u_in'].shift(-1)
        data['u_out_lag_back1'] = data.groupby('breath_id')['u_out'].shift(-1)
        data['u_in_lag2'] = data.groupby('breath_id')['u_in'].shift(2)
        data['u_out_lag2'] = data.groupby('breath_id')['u_out'].shift(2)
        data['u_in_lag_back2'] = data.groupby('breath_id')['u_in'].shift(-2)
        data['u_out_lag_back2'] = data.groupby('breath_id')['u_out'].shift(-2)
        data['u_in_lag3'] = data.groupby('breath_id')['u_in'].shift(3)
        data['u_out_lag3'] = data.groupby('breath_id')['u_out'].shift(3)
        data['u_in_lag_back3'] = data.groupby('breath_id')['u_in'].shift(-3)
        data['u_out_lag_back3'] = data.groupby('breath_id')['u_out'].shift(-3)
        data['u_in_lag4'] = data.groupby('breath_id')['u_in'].shift(4)
        data['u_out_lag4'] = data.groupby('breath_id')['u_out'].shift(4)
        data['u_in_lag_back4'] = data.groupby('breath_id')['u_in'].shift(-4)
        data['u_out_lag_back4'] = data.groupby('breath_id')['u_out'].shift(-4)
        data = data.fillna(0)

        data['breath_id__u_in__max'] = data.groupby(['breath_id'])['u_in'].transform('max')
        data['breath_id__u_out__max'] = data.groupby(['breath_id'])['u_out'].transform('max')

        data['u_in_diff1'] = data['u_in'] - data['u_in_lag1']
        data['u_out_diff1'] = data['u_out'] - data['u_out_lag1']
        data['u_in_diff2'] = data['u_in'] - data['u_in_lag2']
        data['u_out_diff2'] = data['u_out'] - data['u_out_lag2']

        data['breath_id__u_in__diffmax'] = data.groupby(['breath_id'])['u_in'].transform('max') - data['u_in']
        data['breath_id__u_in__diffmean'] = data.groupby(['breath_id'])['u_in'].transform('mean') - data['u_in']

        data['breath_id__u_in__diffmax'] = data.groupby(['breath_id'])['u_in'].transform('max') - data['u_in']
        data['breath_id__u_in__diffmean'] = data.groupby(['breath_id'])['u_in'].transform('mean') - data['u_in']

        data['u_in_diff3'] = data['u_in'] - data['u_in_lag3']
        data['u_out_diff3'] = data['u_out'] - data['u_out_lag3']
        data['u_in_diff4'] = data['u_in'] - data['u_in_lag4']
        data['u_out_diff4'] = data['u_out'] - data['u_out_lag4']
        data['cross'] = data['u_in'] * data['u_out']
        data['cross2'] = data['time_step'] * data['u_out']

        data['R'] = data['R'].astype(str)
        data['C'] = data['C'].astype(str)
        data['R__C'] = data["R"].astype(str) + '__' + data["C"].astype(str)
        data = pd.get_dummies(data)
        data.drop(['id', 'breath_id'], axis=1, inplace=True)
        if 'pressure' in data.columns:
            data.drop('pressure', axis=1, inplace=True)

        return data

    def make_features2(self, data):
        data['area'] = data['time_step'] * data['u_in']
        data['area'] = data.groupby('breath_id')['area'].cumsum()

        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()

        data['u_in_lag1'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_out_lag1'] = data.groupby('breath_id')['u_out'].shift(1)
        data['u_in_lag_back1'] = data.groupby('breath_id')['u_in'].shift(-1)
        data['u_out_lag_back1'] = data.groupby('breath_id')['u_out'].shift(-1)
        data['u_in_lag2'] = data.groupby('breath_id')['u_in'].shift(2)
        data['u_out_lag2'] = data.groupby('breath_id')['u_out'].shift(2)
        data['u_in_lag_back2'] = data.groupby('breath_id')['u_in'].shift(-2)
        data['u_out_lag_back2'] = data.groupby('breath_id')['u_out'].shift(-2)
        data['u_in_lag3'] = data.groupby('breath_id')['u_in'].shift(3)
        data['u_out_lag3'] = data.groupby('breath_id')['u_out'].shift(3)
        data['u_in_lag_back3'] = data.groupby('breath_id')['u_in'].shift(-3)
        data['u_out_lag_back3'] = data.groupby('breath_id')['u_out'].shift(-3)
        data['u_in_lag4'] = data.groupby('breath_id')['u_in'].shift(4)
        data['u_out_lag4'] = data.groupby('breath_id')['u_out'].shift(4)
        data['u_in_lag_back4'] = data.groupby('breath_id')['u_in'].shift(-4)
        data['u_out_lag_back4'] = data.groupby('breath_id')['u_out'].shift(-4)
        data = data.fillna(0)

        data['breath_id__u_in__max'] = data.groupby(['breath_id'])['u_in'].transform('max')
        data['breath_id__u_out__max'] = data.groupby(['breath_id'])['u_out'].transform('max')

        data['u_in_diff1'] = data['u_in'] - data['u_in_lag1']
        data['u_out_diff1'] = data['u_out'] - data['u_out_lag1']
        data['u_in_diff2'] = data['u_in'] - data['u_in_lag2']
        data['u_out_diff2'] = data['u_out'] - data['u_out_lag2']

        data['breath_id__u_in__diffmax'] = data.groupby(['breath_id'])['u_in'].transform('max') - data['u_in']
        data['breath_id__u_in__diffmean'] = data.groupby(['breath_id'])['u_in'].transform('mean') - data['u_in']

        data['breath_id__u_in__diffmax'] = data.groupby(['breath_id'])['u_in'].transform('max') - data['u_in']
        data['breath_id__u_in__diffmean'] = data.groupby(['breath_id'])['u_in'].transform('mean') - data['u_in']

        data['u_in_diff1'] = data['u_in'] - data['u_in_lag1']
        data['u_out_diff1'] = data['u_out'] - data['u_out_lag1']
        data['u_in_diff2'] = data['u_in'] - data['u_in_lag2']
        data['u_out_diff2'] = data['u_out'] - data['u_out_lag2']
        data['u_in_diff3'] = data['u_in'] - data['u_in_lag3']
        data['u_out_diff3'] = data['u_out'] - data['u_out_lag3']
        data['u_in_diff4'] = data['u_in'] - data['u_in_lag4']
        data['u_out_diff4'] = data['u_out'] - data['u_out_lag4']
        data['cross'] = data['u_in'] * data['u_out']
        data['cross2'] = data['time_step'] * data['u_out']

        data['one'] = 1
        data['count'] = (data['one']).groupby(data['breath_id']).cumsum()
        data['u_in_cummean'] = data['u_in_cumsum'] / data['count']

        data['breath_id_lag'] = data['breath_id'].shift(1).fillna(0)
        data['breath_id_lag2'] = data['breath_id'].shift(2).fillna(0)
        data['breath_id_lagsame'] = np.select([data['breath_id_lag'] == data['breath_id']], [1], 0)
        data['breath_id_lag2same'] = np.select([data['breath_id_lag2'] == data['breath_id']], [1], 0)
        data['breath_id__u_in_lag'] = data['u_in'].shift(1).fillna(0)
        data['breath_id__u_in_lag'] = data['breath_id__u_in_lag'] * data['breath_id_lagsame']
        data['breath_id__u_in_lag2'] = data['u_in'].shift(2).fillna(0)
        data['breath_id__u_in_lag2'] = data['breath_id__u_in_lag2'] * data['breath_id_lag2same']

        data['R'] = data['R'].astype(str)
        data['C'] = data['C'].astype(str)
        data['R__C'] = data["R"].astype(str) + '__' + data["C"].astype(str)
        data = pd.get_dummies(data)
        data.drop(['id', 'breath_id', 'one', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame',
                   'breath_id_lag2same'], axis=1, inplace=True)

        if 'pressure' in data.columns:
            data.drop('pressure', axis=1, inplace=True)

        return data

    def make_features3(self, data):
        data['area'] = data['time_step'] * data['u_in']
        data['area'] = data.groupby('breath_id')['area'].cumsum()

        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()

        data['u_in_lag1'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_out_lag1'] = data.groupby('breath_id')['u_out'].shift(1)
        data['u_in_lag_back1'] = data.groupby('breath_id')['u_in'].shift(-1)
        data['u_out_lag_back1'] = data.groupby('breath_id')['u_out'].shift(-1)
        data['u_in_lag2'] = data.groupby('breath_id')['u_in'].shift(2)
        data['u_out_lag2'] = data.groupby('breath_id')['u_out'].shift(2)
        data['u_in_lag_back2'] = data.groupby('breath_id')['u_in'].shift(-2)
        data['u_out_lag_back2'] = data.groupby('breath_id')['u_out'].shift(-2)
        data['u_in_lag3'] = data.groupby('breath_id')['u_in'].shift(3)
        data['u_out_lag3'] = data.groupby('breath_id')['u_out'].shift(3)
        data['u_in_lag_back3'] = data.groupby('breath_id')['u_in'].shift(-3)
        data['u_out_lag_back3'] = data.groupby('breath_id')['u_out'].shift(-3)
        data['u_in_lag4'] = data.groupby('breath_id')['u_in'].shift(4)
        data['u_out_lag4'] = data.groupby('breath_id')['u_out'].shift(4)
        data['u_in_lag_back4'] = data.groupby('breath_id')['u_in'].shift(-4)
        data['u_out_lag_back4'] = data.groupby('breath_id')['u_out'].shift(-4)
        data = data.fillna(0)

        data['breath_id__u_in__max'] = data.groupby(['breath_id'])['u_in'].transform('max')
        data['breath_id__u_out__max'] = data.groupby(['breath_id'])['u_out'].transform('max')

        data['u_in_diff1'] = data['u_in'] - data['u_in_lag1']
        data['u_out_diff1'] = data['u_out'] - data['u_out_lag1']
        data['u_in_diff2'] = data['u_in'] - data['u_in_lag2']
        data['u_out_diff2'] = data['u_out'] - data['u_out_lag2']

        data['breath_id__u_in__diffmax'] = data.groupby(['breath_id'])['u_in'].transform('max') - data['u_in']
        data['breath_id__u_in__diffmean'] = data.groupby(['breath_id'])['u_in'].transform('mean') - data['u_in']

        data['breath_id__u_in__diffmax'] = data.groupby(['breath_id'])['u_in'].transform('max') - data['u_in']
        data['breath_id__u_in__diffmean'] = data.groupby(['breath_id'])['u_in'].transform('mean') - data['u_in']

        data['u_in_diff1'] = data['u_in'] - data['u_in_lag1']
        data['u_out_diff1'] = data['u_out'] - data['u_out_lag1']
        data['u_in_diff2'] = data['u_in'] - data['u_in_lag2']
        data['u_out_diff2'] = data['u_out'] - data['u_out_lag2']
        data['u_in_diff3'] = data['u_in'] - data['u_in_lag3']
        data['u_out_diff3'] = data['u_out'] - data['u_out_lag3']
        data['u_in_diff4'] = data['u_in'] - data['u_in_lag4']
        data['u_out_diff4'] = data['u_out'] - data['u_out_lag4']
        data['cross'] = data['u_in'] * data['u_out']
        data['cross2'] = data['time_step'] * data['u_out']

        data['one'] = 1
        data['count'] = (data['one']).groupby(data['breath_id']).cumsum()
        data['u_in_cummean'] = data['u_in_cumsum'] / data['count']

        data['breath_id_lag'] = data['breath_id'].shift(1).fillna(0)
        data['breath_id_lag2'] = data['breath_id'].shift(2).fillna(0)
        data['breath_id_lagsame'] = np.select([data['breath_id_lag'] == data['breath_id']], [1], 0)
        data['breath_id_lag2same'] = np.select([data['breath_id_lag2'] == data['breath_id']], [1], 0)
        data['breath_id__u_in_lag'] = data['u_in'].shift(1).fillna(0)
        data['breath_id__u_in_lag'] = data['breath_id__u_in_lag'] * data['breath_id_lagsame']
        data['breath_id__u_in_lag2'] = data['u_in'].shift(2).fillna(0)
        data['breath_id__u_in_lag2'] = data['breath_id__u_in_lag2'] * data['breath_id_lag2same']

        data['R_sum_c'] = (data['R'] + data['C']).astype(str)
        data['R_mult_c'] = (data['R'] * data['C']).astype(str)
        data['R'] = data['R'].astype(str)
        data['C'] = data['C'].astype(str)
        data['R__C'] = data["R"].astype(str) + '__' + data["C"].astype(str)
        data = pd.get_dummies(data)

        data[["15_in_sum", "15_in_min", "15_in_max", "15_in_mean"]] = (data \
                                                                       .groupby('breath_id')['u_in'] \
                                                                       .rolling(window=15, min_periods=1) \
                                                                       .agg({"15_in_sum": "sum",
                                                                             "15_in_min": "min",
                                                                             "15_in_max": "max",
                                                                             "15_in_mean": "mean"
                                                                             # "15_in_std":"std"
                                                                             }) \
                                                                       .reset_index(level=0, drop=True))
        data['u_in_lagback_diff1'] = data['u_in'] - data['u_in_lag_back1']
        data['u_out_lagback_diff1'] = data['u_out'] - data['u_out_lag_back1']
        data['u_in_lagback_diff2'] = data['u_in'] - data['u_in_lag_back2']
        data['u_out_lagback_diff2'] = data['u_out'] - data['u_out_lag_back2']

        data['ewm_u_in_mean'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).mean().reset_index(level=0, drop=True)

        data['ewm_u_in_std'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0, drop=True)
        data['ewm_u_in_corr'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).corr().reset_index(level=0, drop=True)
    
        data['rolling_10_mean'] = data.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).mean().reset_index(
            level=0, drop=True)
        data['rolling_10_max'] = data.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).max().reset_index(level=0,
                                                                                                                   drop=True)
        data['rolling_10_std'] = data.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).std().reset_index(level=0,
                                                                                                                   drop=True)
    
        data['expand_mean'] = data.groupby('breath_id')['u_in'].expanding(2).mean().reset_index(level=0, drop=True)
        data['expand_max'] = data.groupby('breath_id')['u_in'].expanding(2).max().reset_index(level=0, drop=True)
        data['expand_std'] = data.groupby('breath_id')['u_in'].expanding(2).std().reset_index(level=0, drop=True)
    
        data["u_in_rolling_mean2"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(2).mean()[
            "u_in"].reset_index(drop=True)
        data["u_in_rolling_mean4"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(4).mean()[
            "u_in"].reset_index(drop=True)
        data["u_in_rolling_mean10"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(10).mean()[
            "u_in"].reset_index(drop=True)

        data["u_in_rolling_max2"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(2).max()[
            "u_in"].reset_index(drop=True)
        data["u_in_rolling_max4"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(4).max()[
            "u_in"].reset_index(drop=True)
        data["u_in_rolling_max10"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(10).max()[
            "u_in"].reset_index(drop=True)
        data["u_in_rolling_min2"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(2).min()[
            "u_in"].reset_index(drop=True)
        data["u_in_rolling_min4"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(4).min()[
            "u_in"].reset_index(drop=True)
        data["u_in_rolling_min10"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(10).min()[
            "u_in"].reset_index(drop=True)
        data["u_in_rolling_std2"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(2).std()[
            "u_in"].reset_index(drop=True)
        data["u_in_rolling_std4"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(4).std()[
            "u_in"].reset_index(drop=True)
        data["u_in_rolling_std10"] = data[["breath_id", "u_in"]].groupby("breath_id").rolling(10).std()[
            "u_in"].reset_index(drop=True)

        g = data.groupby('breath_id')['u_in']
        data['ewm_u_in_mean'] = g.ewm(halflife=10).mean() \
            .reset_index(level=0, drop=True)
        data['ewm_u_in_std'] = g.ewm(halflife=10).std() \
            .reset_index(level=0, drop=True)
        data['ewm_u_in_corr'] = g.ewm(halflife=10).corr() \
            .reset_index(level=0, drop=True)

        data['rolling_10_mean'] = g.rolling(window=10, min_periods=1).mean() \
            .reset_index(level=0, drop=True)
        data['rolling_10_max'] = g.rolling(window=10, min_periods=1).max() \
            .reset_index(level=0, drop=True)
        data['rolling_10_std'] = g.rolling(window=10, min_periods=1).std() \
            .reset_index(level=0, drop=True)

        data['expand_mean'] = g.expanding(2).mean() \
            .reset_index(level=0, drop=True)
        data['expand_max'] = g.expanding(2).max() \
            .reset_index(level=0, drop=True)
        data['expand_std'] = g.expanding(2).std() \
            .reset_index(level=0, drop=True)

        data['u_in_lag_back10'] = data.groupby('breath_id')['u_in'].shift(-10)
        data['u_out_lag_back10'] = data.groupby('breath_id')['u_out'].shift(-10)

        data['time_step_diff'] = data.groupby('breath_id')['time_step'].diff().fillna(0)
        ### rolling window ts feats
        data['ewm_u_in_mean'] = data.groupby('breath_id')['u_in'].ewm(halflife=9).mean().reset_index(level=0,
                                                                                                       drop=True)
        data['ewm_u_in_std'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0,
                                                                                                      drop=True)  ## could add covar?
        data['ewm_u_in_corr'] = data.groupby('breath_id')['u_in'].ewm(halflife=15).corr().reset_index(level=0,
                                                                                                        drop=True)  # self umin corr
        # data['ewm_u_in_corr'] = data.groupby('breath_id')['u_in'].ewm(halflife=6).corr(data.groupby('breath_id')["u_out"]).reset_index(level=0,drop=True) # corr with u_out # error
        ## rolling window of 15 periods
        data[["15_in_sum", "15_in_min", "15_in_max", "15_in_mean", "15_out_std"]] = data.groupby('breath_id')[
            'u_in'].rolling(window=15, min_periods=1).agg(
            {"15_in_sum": "sum", "15_in_min": "min", "15_in_max": "max", "15_in_mean": "mean",
             "15_in_std": "std"}).reset_index(level=0, drop=True)
        #     data[["45_in_sum","45_in_min","45_in_max","45_in_mean","45_out_std"]] = data.groupby('breath_id')['u_in'].rolling(window=45,min_periods=1).agg({"45_in_sum":"sum","45_in_min":"min","45_in_max":"max","45_in_mean":"mean","45_in_std":"std"}).reset_index(level=0,drop=True)
        data[["45_in_sum", "45_in_min", "45_in_max", "45_in_mean", "45_out_std"]] = data.groupby('breath_id')[
            'u_in'].rolling(window=45, min_periods=1).agg(
            {"45_in_sum": "sum", "45_in_min": "min", "45_in_max": "max", "45_in_mean": "mean",
             "45_in_std": "std"}).reset_index(level=0, drop=True)

        data[["15_out_mean"]] = data.groupby('breath_id')['u_out'].rolling(window=15, min_periods=1).agg(
            {"15_out_mean": "mean"}).reset_index(level=0, drop=True)

        data.drop(['id', 'breath_id', 'one', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame',
                   'breath_id_lag2same'], axis=1, inplace=True)

        if 'pressure' in data.columns:
            data.drop('pressure', axis=1, inplace=True)

        return data.fillna(0)


    def setup(self, stage=None):
        if self.cfg.training.debug:
            train = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'train.csv'), nrows=196000)
            test = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'test.csv'), nrows=80000)
        else:
            train = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'train.csv'))
            test = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'test.csv'))

        targets = train[['pressure']].to_numpy().reshape(-1, 80)
        test_targets = np.zeros(len(test)).reshape(-1, 80)

        x_col = [
            'time_step',
            'u_in',
            'u_out',
            'R_',
            'C_',
        ]
        train['R_'] = train['R'].map({5: 1, 20: 2, 50: 3})
        train['C_'] = train['C'].map({10: 1, 20: 2, 50: 3})
        test['R_'] = test['R'].map({5: 1, 20: 2, 50: 3})
        test['C_'] = test['C'].map({10: 1, 20: 2, 50: 3})
        features_train = train[x_col].values.reshape(-1, 80, 5)
        test_features = test[x_col].values.reshape(-1, 80, 5)
        train.drop(['R_', 'C_'], axis=1, inplace=True)
        test.drop(['R_', 'C_'], axis=1, inplace=True)

        print('Making features')
        if self.cfg.datamodule.make_features_style == 0:
            train = self.make_features(train)
            test = self.make_features(test)
        elif self.cfg.datamodule.make_features_style == 1:
            train = self.make_features1(train)
            test = self.make_features1(test)
        elif self.cfg.datamodule.make_features_style == 2:
            train = self.make_features2(train)
            test = self.make_features2(test)
        elif self.cfg.datamodule.make_features_style == 3:
            train = self.make_features3(train)
            test = self.make_features3(test)
        elif self.cfg.datamodule.make_features_style == 4:
            train = self.make_features4(train)
            test = self.make_features4(test)
        elif self.cfg.datamodule.make_features_style == 5:
            train = self.make_features5(train)
            test = self.make_features5(test)

        train_u_out = train[['u_out']].to_numpy().reshape(-1, 80)
        test_u_out = test[['u_out']].to_numpy().reshape(-1, 80)



        if self.cfg.datamodule.normalize:
            RS = RobustScaler()
            train = RS.fit_transform(train)
            test = RS.transform(test)
        else:
            train = train.values
            test = test.values

        train = train.reshape(-1, 80, train.shape[-1])
        test = test.reshape(-1, 80, train.shape[-1])
        gkf = KFold(n_splits=self.cfg.datamodule.n_folds, shuffle=True, random_state=self.cfg.training.seed)
        # if self.cfg.datamodule.split == 'GroupKFold':
        #     gkf = GroupKFold(n_splits=self.cfg.datamodule.n_folds)
        # elif self.cfg.datamodule.split == 'GroupShuffleSplit':
        #     gkf = GroupShuffleSplit(n_splits=self.cfg.datamodule.n_folds, random_state=self.cfg.training.seed)

        splits = list(gkf.split(X=train, y=targets))
        train_idx, valid_idx = splits[self.cfg.datamodule.fold_n]

        train_data = train[train_idx].copy()
        valid_data = train[valid_idx].copy()

        train_u_out_ = train_u_out[train_idx].copy()
        valid_u_out_ = train_u_out[valid_idx].copy()

        train_targets = targets[train_idx].copy()
        valid_targets = targets[valid_idx].copy()

        train_features = features_train[train_idx].copy()
        valid_features = features_train[valid_idx].copy()

        del train, features_train
        gc.collect()

        # train dataset
        dataset_class = load_obj(self.cfg.datamodule.class_name)

        self.train_dataset = dataset_class(train_data, mode='train', u_out=train_u_out_, pressure=train_targets, feats=train_features)
        self.valid_dataset = dataset_class(valid_data, mode='valid', u_out=valid_u_out_, pressure=valid_targets, feats=valid_features)
        self.test_dataset = dataset_class(test, mode='test', u_out=test_u_out, pressure=test_targets, feats=test_features)

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
            drop_last=False
        )

        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=False
        )

        return test_loader
