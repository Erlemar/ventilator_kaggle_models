from typing import Dict

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset


class VentilatorDataset10(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 mode: str = '',
                 normalize: bool = False,
                 ):
        """
        Image classification dataset.

        Args:
            data: dataframe with image id and bboxes
            mode: train/val/test
            img_path: path to images
            transforms: albumentations
        """
        print(f'{mode=}')
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

        self.u_outs = data['u_out'].values.reshape(-1, 80)
        self.pressures = data['pressure'].values.reshape(-1, 80)
        # rs = data['R'].values.reshape(-1, 80)
        # cs = data['C'].values.reshape(-1, 80)

        print('data.columns', data.columns)
        columns = ['time_step', 'u_in', 'u_out', 'RC_sum', 'RC_div', 'R_20', 'R_5', 'R_50', 'C_10', 'C_20', 'C_50',
                   'RC_2010', 'RC_2020', 'RC_2050', 'RC_5010', 'RC_5020', 'RC_5050', 'RC_510', 'RC_520', 'RC_550',
                   'u_in_cumsum', 'time_step_lag', 'u_in_lag', 'u_out_lag']

        print(f'{columns=}')

        data = data[columns].fillna(0)
        cols_to_scale = [col for col in columns if 'R_' not in col and 'C_' not in col and 'RC' not in col]
        print(f'{cols_to_scale=}')
        rs = RobustScaler()
        if normalize:
            if mode == 'train':
                print('Fitting scaler')
                data[cols_to_scale] = rs.fit_transform(data[cols_to_scale])
                with open('rs.joblib', 'wb') as f:
                    joblib.dump(rs, f)
            else:
                print('Applying scaler')
                with open('rs.joblib', 'rb') as f:
                    rs = joblib.load(f)
                data[cols_to_scale] = rs.transform(data[cols_to_scale])

        self.data = data[columns].fillna(0).values
        del data
        self.data = self.data.reshape(-1, 80, len(columns))

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.data[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)

class VentilatorDataset11(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 mode: str = '',
                 normalize: bool = False,
                 ):
        """
        Image classification dataset.

        Args:
            data: dataframe with image id and bboxes
            mode: train/val/test
            img_path: path to images
            transforms: albumentations
        """
        print(f'{mode=}')
        if "pressure" not in data.columns:
            data['pressure'] = 0

        data['RC_sum'] = data['R'] + data['C']
        data['RC_div'] = data['R'] / data['C']
        data['R'] = data['R'].astype(str)
        data['C'] = data['C'].astype(str)
        data['RC'] = data['R'] + data['C']
        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()
        # data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(1)
        data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(2)
        data['u_in_lag'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_out_lag'] = data.groupby('u_out')['u_in'].shift(1)

        self.u_outs = data['u_out'].values.reshape(-1, 80)
        self.pressures = data['pressure'].values.reshape(-1, 80)

        data['area'] = data['time_step'] * data['u_in']
        data['area'] = data.groupby('breath_id')['area'].cumsum()

        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()

        data = pd.get_dummies(data)

        # THESE WERE added
        data['u_in_lag2'] = data['u_in'].shift(2).fillna(0)
        data['u_in_lag4'] = data['u_in'].shift(4).fillna(0)
        data['ewm_u_in_mean'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).mean().reset_index(level=0,
                                                                                                      drop=True)
        data['ewm_u_in_std'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).std().reset_index(level=0, drop=True)
        data['ewm_u_in_corr'] = data.groupby('breath_id')['u_in'].ewm(halflife=10).corr().reset_index(level=0,
                                                                                                      drop=True)

        data['rolling_10_mean'] = data.groupby('breath_id')['u_in'].rolling(window=10,
                                                                            min_periods=1).mean().reset_index(level=0,
                                                                                                              drop=True)
        data['rolling_10_max'] = data.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).max().reset_index(
            level=0, drop=True)
        data['rolling_10_std'] = data.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).std().reset_index(
            level=0, drop=True)

        data['expand_mean'] = data.groupby('breath_id')['u_in'].expanding(2).mean().reset_index(level=0, drop=True)
        data['expand_max'] = data.groupby('breath_id')['u_in'].expanding(2).max().reset_index(level=0, drop=True)
        data['expand_std'] = data.groupby('breath_id')['u_in'].expanding(2).std().reset_index(level=0, drop=True)


        print('data.columns', data.columns)
        columns = ['time_step', 'u_in', 'u_out', 'RC_sum', 'RC_div', 'R_20', 'R_5', 'R_50', 'C_10', 'C_20', 'C_50',
                   'RC_2010', 'RC_2020', 'RC_2050', 'RC_5010', 'RC_5020', 'RC_5050', 'RC_510', 'RC_520', 'RC_550',
                   'u_in_cumsum', 'time_step_lag', 'u_in_lag', 'u_out_lag',
                   'area', 'u_in_lag2', 'u_in_lag4', 'ewm_u_in_std', 'ewm_u_in_corr', 'rolling_10_mean', 'rolling_10_max', 'rolling_10_std',
                   'expand_mean', 'expand_max', 'expand_std']

        print(f'{columns=}')

        data = data[columns].fillna(0)
        cols_to_scale = [col for col in columns if 'R_' not in col and 'C_' not in col and 'RC' not in col]
        print(f'{cols_to_scale=}')
        rs = RobustScaler()
        if normalize:
            if mode == 'train':
                print('Fitting scaler')
                data[cols_to_scale] = rs.fit_transform(data[cols_to_scale])
                with open('rs.joblib', 'wb') as f:
                    joblib.dump(rs, f)
            else:
                print('Applying scaler')
                with open('rs.joblib', 'rb') as f:
                    rs = joblib.load(f)
                data[cols_to_scale] = rs.transform(data[cols_to_scale])

        self.data = data[columns].fillna(0).values
        del data
        self.data = self.data.reshape(-1, 80, len(columns))

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.data[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)

class VentilatorDataset12(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 mode: str = '',
                 normalize: bool = False,
                 ):
        """
        Image classification dataset.

        Args:
            data: dataframe with image id and bboxes
            mode: train/val/test
            img_path: path to images
            transforms: albumentations
        """
        print(f'{mode=}')
        if "pressure" not in data.columns:
            data['pressure'] = 0

        data['RC_sum'] = data['R'] + data['C']
        data['RC_div'] = data['R'] / data['C']
        data['R'] = data['R'].astype(str)
        data['C'] = data['C'].astype(str)
        data['RC'] = data['R'] + data['C']
        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()
        # data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(1)
        data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(2)
        data['u_in_lag'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_out_lag'] = data.groupby('u_out')['u_in'].shift(1)

        self.u_outs = data['u_out'].values.reshape(-1, 80)
        self.pressures = data['pressure'].values.reshape(-1, 80)

        data['area'] = data['time_step'] * data['u_in']
        data['area'] = data.groupby('breath_id')['area'].cumsum()

        data['cross'] = data['u_in'] * data['u_out']
        data['cross2'] = data['time_step'] * data['u_out']

        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()
        data['one'] = 1
        data['count'] = (data['one']).groupby(data['breath_id']).cumsum()
        data['u_in_cummean'] = data['u_in_cumsum'] / data['count']

        data['breath_id_lag'] = data['breath_id'].shift(1).fillna(0)

        data['breath_id_lag2'] = data['breath_id'].shift(2).fillna(0)
        data['breath_id_lagsame'] = np.select([data['breath_id_lag'] == data['breath_id']], [1], 0)
        data['breath_id_lag2same'] = np.select([data['breath_id_lag2'] == data['breath_id']], [1], 0)
        data['u_in_lag_'] = data['u_in'].shift(1).fillna(0)
        data['u_in_lag_'] = data['u_in_lag_'] * data['breath_id_lagsame']
        data['u_in_lag2_'] = data['u_in'].shift(2).fillna(0)
        data['u_in_lag2_'] = data['u_in_lag2_'] * data['breath_id_lag2same']

        data = pd.get_dummies(data)


        print('data.columns', data.columns)
        columns = ['time_step', 'u_in', 'u_out', 'RC_sum', 'RC_div', 'R_20', 'R_5', 'R_50', 'C_10', 'C_20', 'C_50',
                   'RC_2010', 'RC_2020', 'RC_2050', 'RC_5010', 'RC_5020', 'RC_5050', 'RC_510', 'RC_520', 'RC_550',
                   'u_in_cumsum', 'time_step_lag', 'u_in_lag', 'u_out_lag',
                   'area', 'cross', 'cross2', 'u_in_cummean', 'u_in_lag_', 'u_in_lag2_']

        print(f'{columns=}')

        data = data[columns].fillna(0)

        cols_to_scale = [col for col in columns if 'R_' not in col and 'C_' not in col and 'RC' not in col]
        print(f'{cols_to_scale=}')
        rs = RobustScaler()
        if normalize:
            if mode == 'train':
                print('Fitting scaler')
                data[cols_to_scale] = rs.fit_transform(data[cols_to_scale])
                with open('rs.joblib', 'wb') as f:
                    joblib.dump(rs, f)
            else:
                print('Applying scaler')
                with open('rs.joblib', 'rb') as f:
                    rs = joblib.load(f)
                data[cols_to_scale] = rs.transform(data[cols_to_scale])

        self.data = data[columns].fillna(0).values
        del data
        self.data = self.data.reshape(-1, 80, len(columns))

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.data[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)

class VentilatorDataset13(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 mode: str = '',
                 normalize: bool = False,
                 ):
        """
        Image classification dataset.

        Args:
            data: dataframe with image id and bboxes
            mode: train/val/test
            img_path: path to images
            transforms: albumentations
        """
        print(f'{mode=}')
        if "pressure" not in data.columns:
            data['pressure'] = 0

        data['RC_sum'] = data['R'] + data['C']
        data['RC_div'] = data['R'] / data['C']
        data['R'] = data['R'].astype(str)
        data['C'] = data['C'].astype(str)
        data['RC'] = data['R'] + data['C']
        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()
        # data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(1)
        data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(2)
        data['u_in_lag'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_out_lag'] = data.groupby('u_out')['u_in'].shift(1)

        self.u_outs = data['u_out'].values.reshape(-1, 80)
        self.pressures = data['pressure'].values.reshape(-1, 80)

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

        data = pd.get_dummies(data)


        print('data.columns', data.columns)
        columns = ['time_step', 'u_in', 'u_out', 'RC_sum', 'RC_div', 'R_20', 'R_5', 'R_50', 'C_10', 'C_20', 'C_50',
                   'RC_2010', 'RC_2020', 'RC_2050', 'RC_5010', 'RC_5020', 'RC_5050', 'RC_510', 'RC_520', 'RC_550',
                   'u_in_cumsum', 'time_step_lag', 'u_in_lag', 'u_out_lag',
                   'area',
'u_in_cumsum',
'u_in_lag1',
'u_out_lag1',
'u_in_lag_back1',
'u_out_lag_back1',
'u_in_lag2',
'u_out_lag2',
'u_in_lag_back2',
'u_out_lag_back2',
'u_in_lag3',
'u_out_lag3',
'u_in_lag_back3',
'u_out_lag_back3',
'u_in_lag4',
'u_out_lag4',
'u_in_lag_back4',
'u_out_lag_back4',
'breath_id__u_in__max',
'breath_id__u_out__max',
'u_in_diff1',
'u_out_diff1',
'u_in_diff2',
'u_out_diff2',
'breath_id__u_in__diffmax',
'breath_id__u_in__diffmean',
'breath_id__u_in__diffmax',
'breath_id__u_in__diffmean',
'u_in_diff3',
'u_out_diff3',
'u_in_diff4',
'u_out_diff4',
'cross',
'cross2',]
        columns = list(set(columns))

        print(f'{columns=}')
        data = data[columns].fillna(0)

        cols_to_scale = [col for col in columns if 'R_' not in col and 'C_' not in col and 'RC' not in col]
        print(f'{cols_to_scale=}')
        rs = RobustScaler()
        if normalize:
            if mode == 'train':
                print('Fitting scaler')
                data[cols_to_scale] = rs.fit_transform(data[cols_to_scale])
                with open('rs.joblib', 'wb') as f:
                    joblib.dump(rs, f)
            else:
                print('Applying scaler')
                with open('rs.joblib', 'rb') as f:
                    rs = joblib.load(f)
                data[cols_to_scale] = rs.transform(data[cols_to_scale])

        self.data = data[columns].fillna(0).values
        # print('data', data.shape, len(columns), self.data.shape)
        del data
        self.data = self.data.reshape(-1, 80, len(columns))

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.data[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)
