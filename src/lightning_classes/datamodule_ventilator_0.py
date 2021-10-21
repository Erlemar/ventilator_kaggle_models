import gc
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import KFold, GroupKFold
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

    def make_features4(self, data):
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

        data['time_delta'] = data['time_step'].diff()
        data['time_delta'].fillna(0, inplace=True)
        data['time_delta'].mask(data['time_delta'] < 0, 0, inplace=True)
        data['tmp'] = data['time_delta'] * data['u_in']
        data['area_true'] = data.groupby('breath_id')['tmp'].cumsum()

        # u_in_max_dict = data.groupby('breath_id')['u_in'].max().to_dict()
        # data['u_in_max'] = data['breath_id'].map(u_in_max_dict)
        # u_in_min_dict = data.groupby('breath_id')['u_in'].min().to_dict()
        # data['u_in_min'] = data['breath_id'].map(u_in_min_dict)
        u_in_mean_dict = data.groupby('breath_id')['u_in'].mean().to_dict()
        data['u_in_mean'] = data['breath_id'].map(u_in_mean_dict)
        del u_in_mean_dict
        u_in_std_dict = data.groupby('breath_id')['u_in'].std().to_dict()
        data['u_in_std'] = data['breath_id'].map(u_in_std_dict)
        del u_in_std_dict

        # u_in_half is time:0 - time point of u_out:1 rise (almost 1.0s)
        data['tmp'] = data['u_out'] * (-1) + 1  # inversion of u_out
        data['u_in_half'] = data['tmp'] * data['u_in']

        # u_in_half: max, min, mean, std
        u_in_half_max_dict = data.groupby('breath_id')['u_in_half'].max().to_dict()
        data['u_in_half_max'] = data['breath_id'].map(u_in_half_max_dict)
        del u_in_half_max_dict
        u_in_half_min_dict = data.groupby('breath_id')['u_in_half'].min().to_dict()
        data['u_in_half_min'] = data['breath_id'].map(u_in_half_min_dict)
        del u_in_half_min_dict
        u_in_half_mean_dict = data.groupby('breath_id')['u_in_half'].mean().to_dict()
        data['u_in_half_mean'] = data['breath_id'].map(u_in_half_mean_dict)
        del u_in_half_mean_dict
        u_in_half_std_dict = data.groupby('breath_id')['u_in_half'].std().to_dict()
        data['u_in_half_std'] = data['breath_id'].map(u_in_half_std_dict)
        del u_in_half_std_dict

        gc.collect()

        # All entries are first point of each breath_id
        first_data = data.loc[0::80, :]
        # All entries are first point of each breath_id
        last_data = data.loc[79::80, :]

        # The Main mode DataFrame and flag
        main_data = last_data[(last_data['u_in'] > 4.8) & (last_data['u_in'] < 5.1)]
        main_mode_dict = dict(zip(main_data['breath_id'], [1] * len(main_data)))
        data['main_mode'] = data['breath_id'].map(main_mode_dict)
        data['main_mode'].fillna(0, inplace=True)
        del main_data
        del main_mode_dict

        # u_in: first point, last point
        u_in_first_dict = dict(zip(first_data['breath_id'], first_data['u_in']))
        data['u_in_first'] = data['breath_id'].map(u_in_first_dict)
        del u_in_first_dict
        u_in_last_dict = dict(zip(first_data['breath_id'], last_data['u_in']))
        data['u_in_last'] = data['breath_id'].map(u_in_last_dict)
        del u_in_last_dict
        # time(sec) of end point
        time_end_dict = dict(zip(last_data['breath_id'], last_data['time_step']))
        data['time_end'] = data['breath_id'].map(time_end_dict)
        del time_end_dict
        del last_data

        # u_out1_timing flag and DataFrame: speed up
        # 高速版 uout1_data 作成
        data['u_out_diff'] = data['u_out'].diff()
        data['u_out_diff'].fillna(0, inplace=True)
        data['u_out_diff'].replace(-1, 0, inplace=True)
        uout1_data = data[data['u_out_diff'] == 1]

        gc.collect()

        # main_uout1 = uout1_data[uout1_data['main_mode']==1]
        # nomain_uout1 = uout1_data[uout1_data['main_mode']==1]

        # Register Area when u_out becomes 1
        uout1_area_dict = dict(zip(first_data['breath_id'], first_data['u_in']))
        data['area_uout1'] = data['breath_id'].map(uout1_area_dict)
        del uout1_area_dict

        # time(sec) when u_out becomes 1
        uout1_dict = dict(zip(uout1_data['breath_id'], uout1_data['time_step']))
        data['time_uout1'] = data['breath_id'].map(uout1_dict)
        del uout1_dict

        # u_in when u_out becomes1
        u_in_uout1_dict = dict(zip(uout1_data['breath_id'], uout1_data['u_in']))
        data['u_in_uout1'] = data['breath_id'].map(u_in_uout1_dict)
        del u_in_uout1_dict

        # Dict that puts 0 at the beginning of the 80row cycle
        first_0_dict = dict(zip(first_data['id'], [0] * len(uout1_data)))

        del first_data
        del uout1_data

        gc.collect()

        # Faster version u_in_diff creation, faster than groupby
        data['u_in_diff'] = data['u_in'].diff()
        data['tmp'] = data['id'].map(first_0_dict)  # put 0, the 80row cycle
        data.iloc[0::80, data.columns.get_loc('u_in_diff')] = data.iloc[0::80, data.columns.get_loc('tmp')]

        # Create u_in vibration
        data['diff_sign'] = np.sign(data['u_in_diff'])
        data['sign_diff'] = data['diff_sign'].diff()
        data['tmp'] = data['id'].map(first_0_dict)  # put 0, the 80row cycle
        data.iloc[0::80, data.columns.get_loc('sign_diff')] = data.iloc[0::80, data.columns.get_loc('tmp')]
        del first_0_dict

        # Count the number of inversions, so take the absolute value and sum
        data['sign_diff'] = abs(data['sign_diff'])
        sign_diff_dict = data.groupby('breath_id')['sign_diff'].sum().to_dict()
        data['diff_vib'] = data['breath_id'].map(sign_diff_dict)



        data.drop(['id', 'breath_id', 'one', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame',
                   'breath_id_lag2same'], axis=1, inplace=True)

        if 'pressure' in data.columns:
            data.drop('pressure', axis=1, inplace=True)

        return data.fillna(0)

    def make_features5(self, data):
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

        data['ewm_u_in_mean'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).mean().reset_index(level=0,
                                                                                                      drop=True)

        data['ewm_u_in_std'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0, drop=True)
        data['ewm_u_in_corr'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).corr().reset_index(level=0,
                                                                                                      drop=True)

        data['rolling_10_mean'] = data.groupby('breath_id')['u_in'].rolling(window=10,
                                                                            min_periods=1).mean().reset_index(
            level=0, drop=True)
        data['rolling_10_max'] = data.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).max().reset_index(
            level=0,
            drop=True)
        data['rolling_10_std'] = data.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).std().reset_index(
            level=0,
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

        data['time_delta'] = data['time_step'].diff()
        data['time_delta'].fillna(0, inplace=True)
        data['time_delta'].mask(data['time_delta'] < 0, 0, inplace=True)
        data['tmp'] = data['time_delta'] * data['u_in']
        data['area_true'] = data.groupby('breath_id')['tmp'].cumsum()

        # u_in_max_dict = data.groupby('breath_id')['u_in'].max().to_dict()
        # data['u_in_max'] = data['breath_id'].map(u_in_max_dict)
        # u_in_min_dict = data.groupby('breath_id')['u_in'].min().to_dict()
        # data['u_in_min'] = data['breath_id'].map(u_in_min_dict)
        u_in_mean_dict = data.groupby('breath_id')['u_in'].mean().to_dict()
        data['u_in_mean'] = data['breath_id'].map(u_in_mean_dict)
        del u_in_mean_dict
        u_in_std_dict = data.groupby('breath_id')['u_in'].std().to_dict()
        data['u_in_std'] = data['breath_id'].map(u_in_std_dict)
        del u_in_std_dict

        # u_in_half is time:0 - time point of u_out:1 rise (almost 1.0s)
        data['tmp'] = data['u_out'] * (-1) + 1  # inversion of u_out
        data['u_in_half'] = data['tmp'] * data['u_in']

        # u_in_half: max, min, mean, std
        u_in_half_max_dict = data.groupby('breath_id')['u_in_half'].max().to_dict()
        data['u_in_half_max'] = data['breath_id'].map(u_in_half_max_dict)
        del u_in_half_max_dict
        u_in_half_min_dict = data.groupby('breath_id')['u_in_half'].min().to_dict()
        data['u_in_half_min'] = data['breath_id'].map(u_in_half_min_dict)
        del u_in_half_min_dict
        u_in_half_mean_dict = data.groupby('breath_id')['u_in_half'].mean().to_dict()
        data['u_in_half_mean'] = data['breath_id'].map(u_in_half_mean_dict)
        del u_in_half_mean_dict
        u_in_half_std_dict = data.groupby('breath_id')['u_in_half'].std().to_dict()
        data['u_in_half_std'] = data['breath_id'].map(u_in_half_std_dict)
        del u_in_half_std_dict

        gc.collect()

        # All entries are first point of each breath_id
        first_data = data.loc[0::80, :]
        # All entries are first point of each breath_id
        last_data = data.loc[79::80, :]

        # The Main mode DataFrame and flag
        main_data = last_data[(last_data['u_in'] > 4.8) & (last_data['u_in'] < 5.1)]
        main_mode_dict = dict(zip(main_data['breath_id'], [1] * len(main_data)))
        data['main_mode'] = data['breath_id'].map(main_mode_dict)
        data['main_mode'].fillna(0, inplace=True)
        del main_data
        del main_mode_dict

        # u_in: first point, last point
        u_in_first_dict = dict(zip(first_data['breath_id'], first_data['u_in']))
        data['u_in_first'] = data['breath_id'].map(u_in_first_dict)
        del u_in_first_dict
        u_in_last_dict = dict(zip(first_data['breath_id'], last_data['u_in']))
        data['u_in_last'] = data['breath_id'].map(u_in_last_dict)
        del u_in_last_dict
        # time(sec) of end point
        time_end_dict = dict(zip(last_data['breath_id'], last_data['time_step']))
        data['time_end'] = data['breath_id'].map(time_end_dict)
        del time_end_dict
        del last_data

        # u_out1_timing flag and DataFrame: speed up
        # 高速版 uout1_data 作成
        data['u_out_diff'] = data['u_out'].diff()
        data['u_out_diff'].fillna(0, inplace=True)
        data['u_out_diff'].replace(-1, 0, inplace=True)
        uout1_data = data[data['u_out_diff'] == 1]

        gc.collect()

        # main_uout1 = uout1_data[uout1_data['main_mode']==1]
        # nomain_uout1 = uout1_data[uout1_data['main_mode']==1]

        # Register Area when u_out becomes 1
        uout1_area_dict = dict(zip(first_data['breath_id'], first_data['u_in']))
        data['area_uout1'] = data['breath_id'].map(uout1_area_dict)
        del uout1_area_dict

        # time(sec) when u_out becomes 1
        uout1_dict = dict(zip(uout1_data['breath_id'], uout1_data['time_step']))
        data['time_uout1'] = data['breath_id'].map(uout1_dict)
        del uout1_dict

        # u_in when u_out becomes1
        u_in_uout1_dict = dict(zip(uout1_data['breath_id'], uout1_data['u_in']))
        data['u_in_uout1'] = data['breath_id'].map(u_in_uout1_dict)
        del u_in_uout1_dict

        # Dict that puts 0 at the beginning of the 80row cycle
        first_0_dict = dict(zip(first_data['id'], [0] * len(uout1_data)))

        del first_data
        del uout1_data

        gc.collect()

        # Faster version u_in_diff creation, faster than groupby
        data['u_in_diff'] = data['u_in'].diff()
        data['tmp'] = data['id'].map(first_0_dict)  # put 0, the 80row cycle
        data.iloc[0::80, data.columns.get_loc('u_in_diff')] = data.iloc[0::80, data.columns.get_loc('tmp')]

        # Create u_in vibration
        data['diff_sign'] = np.sign(data['u_in_diff'])
        data['sign_diff'] = data['diff_sign'].diff()
        data['tmp'] = data['id'].map(first_0_dict)  # put 0, the 80row cycle
        data.iloc[0::80, data.columns.get_loc('sign_diff')] = data.iloc[0::80, data.columns.get_loc('tmp')]
        del first_0_dict

        # Count the number of inversions, so take the absolute value and sum
        data['sign_diff'] = abs(data['sign_diff'])
        sign_diff_dict = data.groupby('breath_id')['sign_diff'].sum().to_dict()
        data['diff_vib'] = data['breath_id'].map(sign_diff_dict)


        data.drop(['id', 'breath_id', 'one', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame',
                   'breath_id_lag2same'], axis=1, inplace=True)

        if 'pressure' in data.columns:
            data.drop('pressure', axis=1, inplace=True)

        return data.fillna(0)

    def make_features6(self, data):
        # CATE_FEATURES = ['R_cate', 'C_cate', 'RC_dot', 'RC_sum']
        CONT_FEATURES = ['u_in', 'u_out', 'time_step'] + ['u_in_cumsum', 'u_in_cummean', 'area', 'cross', 'cross2'] + [
            'R_cate', 'C_cate']
        LAG_FEATURES = ['breath_time']
        LAG_FEATURES += [f'u_in_lag_{i}' for i in range(1, self.cfg.datamodule.use_lag + 1)]
        # LAG_FEATURES += [f'u_in_lag_{i}_back' for i in range(1, USE_LAG+1)]
        LAG_FEATURES += [f'u_in_time{i}' for i in range(1, self.cfg.datamodule.use_lag + 1)]
        # LAG_FEATURES += [f'u_in_time{i}_back' for i in range(1, USE_LAG+1)]
        LAG_FEATURES += [f'u_out_lag_{i}' for i in range(1, self.cfg.datamodule.use_lag + 1)]
        # LAG_FEATURES += [f'u_out_lag_{i}_back' for i in range(1, USE_LAG+1)]
        # ALL_FEATURES = CATE_FEATURES + CONT_FEATURES + LAG_FEATURES
        ALL_FEATURES = CONT_FEATURES + LAG_FEATURES
        if 'fold' in data.columns:
            ALL_FEATURES.append('fold')

        data['time_delta'] = data.groupby('breath_id')['time_step'].diff().fillna(0)
        data['delta'] = data['time_delta'] * data['u_in']
        data['area'] = data.groupby('breath_id')['delta'].cumsum()

        data['cross'] = data['u_in'] * data['u_out']
        data['cross2'] = data['time_step'] * data['u_out']

        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()
        data['one'] = 1
        data['count'] = (data['one']).groupby(data['breath_id']).cumsum()
        data['u_in_cummean'] = data['u_in_cumsum'] / data['count']

        data = data.drop(['count', 'one'], axis=1)

        for lag in range(1, self.cfg.datamodule.use_lag + 1):
            data[f'breath_id_lag{lag}'] = data['breath_id'].shift(lag).fillna(0)
            data[f'breath_id_lag{lag}same'] = np.select([data[f'breath_id_lag{lag}'] == data['breath_id']], [1], 0)

            # u_in
            data[f'u_in_lag_{lag}'] = data['u_in'].shift(lag).fillna(0) * data[f'breath_id_lag{lag}same']
            # data[f'u_in_lag_{lag}_back'] = data['u_in'].shift(-lag).fillna(0) * data[f'breath_id_lag{lag}same']
            data[f'u_in_time{lag}'] = data['u_in'] - data[f'u_in_lag_{lag}']
            # data[f'u_in_time{lag}_back'] = data['u_in'] - data[f'u_in_lag_{lag}_back']
            data[f'u_out_lag_{lag}'] = data['u_out'].shift(lag).fillna(0) * data[f'breath_id_lag{lag}same']
            # data[f'u_out_lag_{lag}_back'] = data['u_out'].shift(-lag).fillna(0) * data[f'breath_id_lag{lag}same']

        # breath_time
        data['time_step_lag'] = data['time_step'].shift(1).fillna(0) * data[f'breath_id_lag{lag}same']
        data['breath_time'] = data['time_step'] - data['time_step_lag']

        drop_columns = ['time_step_lag']
        drop_columns += [f'breath_id_lag{i}' for i in range(1, self.cfg.datamodule.use_lag + 1)]
        drop_columns += [f'breath_id_lag{i}same' for i in range(1, self.cfg.datamodule.use_lag + 1)]
        data = data.drop(drop_columns, axis=1)

        # fill na by zero
        data = data.fillna(0)

        c_dic = {10: 0, 20: 1, 50: 2}
        r_dic = {5: 0, 20: 1, 50: 2}
        rc_sum_dic = {v: i for i, v in enumerate([15, 25, 30, 40, 55, 60, 70, 100])}
        rc_dot_dic = {v: i for i, v in enumerate([50, 100, 200, 250, 400, 500, 2500, 1000])}

        data['C_cate'] = data['C'].map(c_dic)
        data['R_cate'] = data['R'].map(r_dic)
        data['RC_sum'] = (data['R'] + data['C']).map(rc_sum_dic)
        data['RC_dot'] = (data['R'] * data['C']).map(rc_dot_dic)

        norm_features = CONT_FEATURES + LAG_FEATURES
        if 'fold' in data.columns:
            norm_features.append('fold')
        assert norm_features == ALL_FEATURES, 'something went wrong'

        print('data.columns', data.columns)
        print('ALL_FEATURES', ALL_FEATURES)

        return data[ALL_FEATURES].fillna(0)

    def make_features7(self, data):
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

        data['u_in_lagback_diff1'] = data['u_in'] - data['u_in_lag_back1']
        data['u_out_lagback_diff1'] = data['u_out'] - data['u_out_lag_back1']
        data['u_in_lagback_diff2'] = data['u_in'] - data['u_in_lag_back2']
        data['u_out_lagback_diff2'] = data['u_out'] - data['u_out_lag_back2']
        data['u_in_lagback_diff3'] = data['u_in'] - data['u_in_lag_back3']
        data['u_out_lagback_diff3'] = data['u_out'] - data['u_out_lag_back3']
        data['u_in_lagback_diff4'] = data['u_in'] - data['u_in_lag_back4']
        data['u_out_lagback_diff4'] = data['u_out'] - data['u_out_lag_back4']

        ######
        data['u_in_lag_back10'] = data.groupby('breath_id')['u_in'].shift(-10)
        data['u_out_lag_back10'] = data.groupby('breath_id')['u_out'].shift(-10)
        data['u_in_lagback_diff10'] = data['u_in'] - data['u_in_lag_back10']
        data['u_out_lagback_diff10'] = data['u_out'] - data['u_out_lag_back10']

        data['time_step_diff'] = data['time_step'] - data.groupby('breath_id')['time_step'].shift().fillna(0)

        data[["15_out_sum", "15_out_min", "15_out_max", "15_out_mean", "15_out_std"]] = (data \
                                                                       .groupby('breath_id')['u_out'] \
                                                                       .rolling(window=15, min_periods=1) \
                                                                       .agg({"15_out_sum": "sum",
                                                                             "15_out_min": "min",
                                                                             "15_out_max": "max",
                                                                             "15_out_mean": "mean",
                                                                             "15_out_std": "std"
                                                                             }).reset_index(level=0, drop=True))

        for window in [2, 4, 5, 10, 15, 20, 30, 45]:
            data[[f"{window}_in_sum", f"{window}_in_min", f"{window}_in_max",
                  f"{window}_in_mean", f"{window}_in_std"]] = (data.groupby('breath_id')['u_in'].rolling(window=window,
                                                                                                         min_periods=1) \
                                                                                        .agg({f"{window}_in_sum": "sum",
                                                                                              f"{window}_in_min": "min",
                                                                                              f"{window}_in_max": "max",
                                                                                              f"{window}_in_mean": "mean",
                                                                                              f"{window}_in_std": "std"
                                                                                              }).reset_index(level=0,
                                                                                                             drop=True))

        for halflife in [5, 9, 10, 15, 20]:

            data[f'ewm_u_in_mean_{halflife}'] = data.groupby('breath_id')['u_in'].ewm(halflife=halflife).mean().reset_index(level=0,
                                                                                                          drop=True)

            data[f'ewm_u_in_std_{halflife}'] = data.groupby('breath_id')['u_in'].ewm(halflife=halflife).std().reset_index(level=0, drop=True)
            data[f'ewm_u_in_corr_{halflife}'] = data.groupby('breath_id')['u_in'].ewm(halflife=halflife).corr().reset_index(level=0,
                                                                                                          drop=True)


        data['expand_mean'] = data.groupby('breath_id')['u_in'].expanding(2).mean().reset_index(level=0, drop=True)
        data['expand_max'] = data.groupby('breath_id')['u_in'].expanding(2).max().reset_index(level=0, drop=True)
        data['expand_std'] = data.groupby('breath_id')['u_in'].expanding(2).std().reset_index(level=0, drop=True)

        data.drop(['id', 'breath_id', 'one', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame',
                   'breath_id_lag2same'], axis=1, inplace=True)

        if 'pressure' in data.columns:
            data.drop('pressure', axis=1, inplace=True)

        return data.fillna(0)

    def make_features8(self, data):
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
        c_dic = {10: 0, 20: 1, 50: 2}
        r_dic = {5: 0, 20: 1, 50: 2}
        rc_sum_dic = {v: i for i, v in enumerate([15, 25, 30, 40, 55, 60, 70, 100])}
        rc_dot_dic = {v: i for i, v in enumerate([50, 100, 200, 250, 400, 500, 2500, 1000])}

        data['C_cate'] = data['C'].map(c_dic)
        data['R_cate'] = data['R'].map(r_dic)
        data['RC_sum'] = (data['R'] + data['C']).map(rc_sum_dic)
        data['RC_dot'] = (data['R'] * data['C']).map(rc_dot_dic)

        data['R_sum_c'] = (data['R'] + data['C']).astype(str)
        data['R_mult_c'] = (data['R'] * data['C']).astype(str)
        data['R'] = data['R'].astype(str)
        data['C'] = data['C'].astype(str)
        data['R__C'] = data["R"].astype(str) + '__' + data["C"].astype(str)
        data = pd.get_dummies(data)

        data['u_in_lagback_diff1'] = data['u_in'] - data['u_in_lag_back1']
        data['u_out_lagback_diff1'] = data['u_out'] - data['u_out_lag_back1']
        data['u_in_lagback_diff2'] = data['u_in'] - data['u_in_lag_back2']
        data['u_out_lagback_diff2'] = data['u_out'] - data['u_out_lag_back2']
        data['u_in_lagback_diff3'] = data['u_in'] - data['u_in_lag_back3']
        data['u_out_lagback_diff3'] = data['u_out'] - data['u_out_lag_back3']
        data['u_in_lagback_diff4'] = data['u_in'] - data['u_in_lag_back4']
        data['u_out_lagback_diff4'] = data['u_out'] - data['u_out_lag_back4']

        ######
        data['u_in_lag_back10'] = data.groupby('breath_id')['u_in'].shift(-10)
        data['u_out_lag_back10'] = data.groupby('breath_id')['u_out'].shift(-10)
        data['u_in_lagback_diff10'] = data['u_in'] - data['u_in_lag_back10']
        data['u_out_lagback_diff10'] = data['u_out'] - data['u_out_lag_back10']

        data['time_step_diff'] = data['time_step'] - data.groupby('breath_id')['time_step'].shift().fillna(0)

        data[["15_out_sum", "15_out_min", "15_out_max", "15_out_mean", "15_out_std"]] = (data \
                                                                       .groupby('breath_id')['u_out'] \
                                                                       .rolling(window=15, min_periods=1) \
                                                                       .agg({"15_out_sum": "sum",
                                                                             "15_out_min": "min",
                                                                             "15_out_max": "max",
                                                                             "15_out_mean": "mean",
                                                                             "15_out_std": "std"
                                                                             }).reset_index(level=0, drop=True))

        for window in [2, 4, 5, 10, 15, 20, 30, 45]:
            data[[f"{window}_in_sum", f"{window}_in_min", f"{window}_in_max",
                  f"{window}_in_mean", f"{window}_in_std"]] = (data.groupby('breath_id')['u_in'].rolling(window=window,
                                                                                                         min_periods=1) \
                                                                                        .agg({f"{window}_in_sum": "sum",
                                                                                              f"{window}_in_min": "min",
                                                                                              f"{window}_in_max": "max",
                                                                                              f"{window}_in_mean": "mean",
                                                                                              f"{window}_in_std": "std"
                                                                                              }).reset_index(level=0,
                                                                                                             drop=True))

        for halflife in [5, 9, 10, 15, 20]:

            data[f'ewm_u_in_mean_{halflife}'] = data.groupby('breath_id')['u_in'].ewm(halflife=halflife).mean().reset_index(level=0,
                                                                                                          drop=True)

            data[f'ewm_u_in_std_{halflife}'] = data.groupby('breath_id')['u_in'].ewm(halflife=halflife).std().reset_index(level=0, drop=True)
            data[f'ewm_u_in_corr_{halflife}'] = data.groupby('breath_id')['u_in'].ewm(halflife=halflife).corr().reset_index(level=0,
                                                                                                          drop=True)


        data['expand_mean'] = data.groupby('breath_id')['u_in'].expanding(2).mean().reset_index(level=0, drop=True)
        data['expand_max'] = data.groupby('breath_id')['u_in'].expanding(2).max().reset_index(level=0, drop=True)
        data['expand_std'] = data.groupby('breath_id')['u_in'].expanding(2).std().reset_index(level=0, drop=True)

        data.drop(['id', 'breath_id', 'one', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame',
                   'breath_id_lag2same'], axis=1, inplace=True)

        if 'pressure' in data.columns:
            data.drop('pressure', axis=1, inplace=True)

        return data.fillna(0)

    def make_features9(self, data):
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

        c_dic = {10: 0, 20: 1, 50: 2}
        r_dic = {5: 0, 20: 1, 50: 2}
        rc_sum_dic = {v: i for i, v in enumerate([15, 25, 30, 40, 55, 60, 70, 100])}
        rc_dot_dic = {v: i for i, v in enumerate([50, 100, 200, 250, 400, 500, 2500, 1000])}

        data['C_cate'] = data['C'].map(c_dic)
        data['R_cate'] = data['R'].map(r_dic)
        data['RC_sum'] = (data['R'] + data['C']).map(rc_sum_dic)
        data['RC_dot'] = (data['R'] * data['C']).map(rc_dot_dic)

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

        data['ewm_u_in_mean'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).mean().reset_index(level=0,
                                                                                                      drop=True)

        data['ewm_u_in_std'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0, drop=True)
        data['ewm_u_in_corr'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).corr().reset_index(level=0,
                                                                                                      drop=True)

        data['rolling_10_mean'] = data.groupby('breath_id')['u_in'].rolling(window=10,
                                                                            min_periods=1).mean().reset_index(
            level=0, drop=True)
        data['rolling_10_max'] = data.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).max().reset_index(
            level=0,
            drop=True)
        data['rolling_10_std'] = data.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).std().reset_index(
            level=0,
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

        gkf = GroupKFold(n_splits=self.cfg.datamodule.n_folds).split(train, train.pressure, groups=train.breath_id)
        for fold, (_, valid_idx) in enumerate(gkf):
            train.loc[valid_idx, 'fold'] = fold

        train_targets = train.loc[train['fold'] != self.cfg.datamodule.fold_n, 'pressure'].copy().values.reshape(-1, 80)
        valid_targets = train.loc[train['fold'] == self.cfg.datamodule.fold_n, 'pressure'].copy().values.reshape(-1, 80)

        train_u_out_ = train.loc[train['fold'] != self.cfg.datamodule.fold_n, 'u_out'].copy().values.reshape(-1, 80)
        valid_u_out_ = train.loc[train['fold'] == self.cfg.datamodule.fold_n, 'u_out'].copy().values.reshape(-1, 80)
        test_targets = np.zeros(len(test)).reshape(-1, 80)

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
        elif self.cfg.datamodule.make_features_style == 6:
            train = self.make_features6(train)
            test = self.make_features6(test)
        elif self.cfg.datamodule.make_features_style == 7:
            train = self.make_features7(train)
            test = self.make_features7(test)
        elif self.cfg.datamodule.make_features_style == 8:
            train = self.make_features8(train)
            test = self.make_features8(test)
        elif self.cfg.datamodule.make_features_style == 9:
            train = self.make_features9(train)
            test = self.make_features9(test)
        else:
            raise ValueError


        print('f{train.columns = }')
        test_u_out = test[['u_out']].to_numpy().reshape(-1, 80)

        if self.cfg.datamodule.normalize:
            if not self.cfg.datamodule.normalize_all:
                RS = RobustScaler()
                RS.fit(train.drop(['fold'], axis=1))
                # train = RS.transform(train)
                train_data = RS.transform(train.loc[train['fold'] != self.cfg.datamodule.fold_n].drop(['fold'], axis=1))
                valid_data = RS.transform(train.loc[train['fold'] == self.cfg.datamodule.fold_n].drop(['fold'], axis=1))
                test = RS.transform(test)
            else:
                RS = RobustScaler()
                RS.fit(np.vstack([train.drop(['fold'], axis=1), test]))
                # train = RS.transform(train)
                train_data = RS.transform(train.loc[train['fold'] != self.cfg.datamodule.fold_n].drop(['fold'], axis=1))
                valid_data = RS.transform(train.loc[train['fold'] == self.cfg.datamodule.fold_n].drop(['fold'], axis=1))
                test = RS.transform(test)

        else:
            train_data = train.loc[train['fold'] != self.cfg.datamodule.fold_n].drop(['fold'], axis=1).values
            valid_data = train.loc[train['fold'] == self.cfg.datamodule.fold_n].drop(['fold'], axis=1).values
            test = test.values

        # train = train.reshape(-1, 80, train.shape[-1])
        train_data = train_data.reshape(-1, 80, train_data.shape[-1])
        valid_data = valid_data.reshape(-1, 80, valid_data.shape[-1])
        test = test.reshape(-1, 80, train_data.shape[-1])
        # gkf = KFold(n_splits=self.cfg.datamodule.n_folds, shuffle=True, random_state=self.cfg.training.seed)
        # if self.cfg.datamodule.split == 'GroupKFold':
        #     gkf = GroupKFold(n_splits=self.cfg.datamodule.n_folds)
        # elif self.cfg.datamodule.split == 'GroupShuffleSplit':
        #     gkf = GroupShuffleSplit(n_splits=self.cfg.datamodule.n_folds, random_state=self.cfg.training.seed)

        # splits = list(gkf.split(X=train, y=targets))
        # train_idx, valid_idx = splits[self.cfg.datamodule.fold_n]

        # train_data = train[train_idx].copy()
        # valid_data = train[valid_idx].copy()

        # train_u_out_ = train_u_out[train_idx].copy()
        # valid_u_out_ = train_u_out[valid_idx].copy()
        #
        # train_targets = targets[train_idx].copy()
        # valid_targets = targets[valid_idx].copy()

        del train
        gc.collect()

        # train dataset
        dataset_class = load_obj(self.cfg.datamodule.class_name)
        print('train_data', train_data.shape)
        print('train_u_out_', train_u_out_.shape)
        self.train_dataset = dataset_class(train_data, mode='train', u_out=train_u_out_, pressure=train_targets)
        self.valid_dataset = dataset_class(valid_data, mode='valid', u_out=valid_u_out_, pressure=valid_targets)
        self.test_dataset = dataset_class(test, mode='test', u_out=test_u_out, pressure=test_targets)

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
