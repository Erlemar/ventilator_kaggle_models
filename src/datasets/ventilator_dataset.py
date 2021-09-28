from typing import Dict

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset


class VentilatorDataset(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 ):
        """
        Image classification dataset.

        Args:
            data: dataframe with image id and bboxes
            mode: train/val/test
            img_path: path to images
            transforms: albumentations
        """
        if "pressure" not in data.columns:
            data['pressure'] = 0

        self.data = data.groupby('breath_id').agg(list).reset_index()

        self.pressures = np.array(self.data['pressure'].values.tolist())

        rs = np.array(self.data['R'].values.tolist())
        cs = np.array(self.data['C'].values.tolist())
        u_ins = np.array(self.data['u_in'].values.tolist())

        self.u_outs = np.array(self.data['u_out'].values.tolist())

        self.inputs = np.concatenate([
            rs[:, None],
            cs[:, None],
            u_ins[:, None],
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None]
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)


class VentilatorDataset1(Dataset):

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
        if "pressure" not in data.columns:
            data['pressure'] = 0

        r_map = {5: 0, 20: 1, 50: 2}
        c_map = {10: 0, 20: 1, 50: 2}

        data['R'] = data['R'].map(r_map)
        data['C'] = data['C'].map(c_map)

        data['breath_time'] = data['time_step'] - data.groupby('breath_id')['time_step'].shift(1)
        data['u_in_lag'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_in_diff1'] = data['u_in'] - data['u_in_lag']

        self.data = data.fillna(0).groupby('breath_id').agg(list).reset_index()

        self.pressures = np.array(self.data['pressure'].values.tolist())
        u_ins = np.array(self.data['u_in'].values.tolist())

        self.u_outs = np.array(self.data['u_out'].values.tolist())
        self.rs = np.array(self.data['R'].values.tolist())
        self.cs = np.array(self.data['C'].values.tolist())

        self.inputs = np.concatenate([
            u_ins[:, None],
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None],
            np.array(self.data['breath_time'].values.tolist())[:, None],
            np.array(self.data['u_in_diff1'].values.tolist())[:, None],
            np.array(self.data['time_step'].values.tolist())[:, None],
            np.array(self.data['u_in_lag'].values.tolist())[:, None]
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "r": torch.tensor(self.rs[:, None][idx], dtype=torch.long),
            "c": torch.tensor(self.cs[:, None][idx], dtype=torch.long),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)


class VentilatorDataset2(Dataset):

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
        if "pressure" not in data.columns:
            data['pressure'] = 0

        # data['RC_sum'] = data['R'] + data['C']
        # data['RC_div'] = data['R'] / data['C']
        r_map = {5: 0, 20: 1, 50: 2}
        c_map = {10: 0, 20: 1, 50: 2}

        data['R'] = data['R'].map(r_map)
        data['C'] = data['C'].map(c_map)

        data['breath_time'] = data['time_step'] - data.groupby('breath_id')['time_step'].shift(1)
        data['u_in_lag'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_out_lag'] = data.groupby('breath_id')['u_out'].shift(1)
        data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(1)
        data['time_step_lag2'] = data.groupby('breath_id')['time_step'].shift(2)
        data['u_in_diff1'] = data['u_in'] - data['u_in_lag']

        self.data = data.fillna(0).groupby('breath_id').agg(list).reset_index()

        self.pressures = np.array(self.data['pressure'].values.tolist())
        u_ins = np.array(self.data['u_in'].values.tolist())

        self.u_outs = np.array(self.data['u_out'].values.tolist())
        self.rs = np.array(self.data['R'].values.tolist())
        self.cs = np.array(self.data['C'].values.tolist())

        self.inputs = np.concatenate([
            u_ins[:, None],
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None],
            np.array(self.data['breath_time'].values.tolist())[:, None],
            np.array(self.data['u_in_diff1'].values.tolist())[:, None],
            np.array(self.data['time_step'].values.tolist())[:, None],
            np.array(self.data['time_step_lag'].values.tolist())[:, None],
            np.array(self.data['time_step_lag2'].values.tolist())[:, None],
            np.array(self.data['u_in_lag'].values.tolist())[:, None],
            np.array(self.data['u_out_lag'].values.tolist())[:, None]
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "r": torch.tensor(self.rs[:, None][idx], dtype=torch.long),
            "c": torch.tensor(self.cs[:, None][idx], dtype=torch.long),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)


class VentilatorDataset3(Dataset):

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
        if "pressure" not in data.columns:
            data['pressure'] = 0

        data['u_in_lag'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_in_lag2'] = data.groupby('breath_id')['u_in'].shift(2)
        data['u_out_lag'] = data.groupby('breath_id')['u_out'].shift(1)
        data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(1)
        data['breath_time'] = data['time_step'] - data['time_step_lag']
        data['time_step_lag2'] = data.groupby('breath_id')['time_step'].shift(2)
        data['u_in_diff1'] = data['u_in'] - data['u_in_lag']

        self.data = data.fillna(0).groupby('breath_id').agg(list).reset_index()

        self.pressures = np.array(self.data['pressure'].values.tolist())
        u_ins = np.array(self.data['u_in'].values.tolist())

        self.u_outs = np.array(self.data['u_out'].values.tolist())
        self.rs = np.array(self.data['R'].values.tolist())
        self.cs = np.array(self.data['C'].values.tolist())

        self.inputs = np.concatenate([
            u_ins[:, None],
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None],
            np.array(self.data['breath_time'].values.tolist())[:, None],
            np.array(self.data['u_in_diff1'].values.tolist())[:, None],
            np.array(self.data['time_step'].values.tolist())[:, None],
            np.array(self.data['time_step_lag'].values.tolist())[:, None],
            np.array(self.data['time_step_lag2'].values.tolist())[:, None],
            np.array(self.data['u_in_lag'].values.tolist())[:, None],
            np.array(self.data['u_in_lag2'].values.tolist())[:, None],
            np.array(self.data['u_out_lag'].values.tolist())[:, None],
            self.rs[:, None],
            self.cs[:, None],
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)


class VentilatorDataset5(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 mode: str = '',
                 normalize: bool = False,
                 ):
        """
        from 1

        Args:
            data: dataframe with image id and bboxes
            mode: train/val/test
            img_path: path to images
            transforms: albumentations
        """
        if "pressure" not in data.columns:
            data['pressure'] = 0

        r_map = {5: 0, 20: 1, 50: 2}
        c_map = {10: 0, 20: 1, 50: 2}

        data['R'] = data['R'].map(r_map)
        data['C'] = data['C'].map(c_map)

        data['area'] = data['time_step'] * data['u_in']
        data['area'] = data.groupby('breath_id')['area'].cumsum()

        data['ewm_u_in_mean'] = data.groupby('breath_id')['u_in'].ewm(halflife=8).mean().reset_index(level=0, drop=True)
        data['ewm_u_in_std'] = data.groupby('breath_id')['u_in'].ewm(halflife=9).std().reset_index(level=0, drop=True)
        data['ewm_u_in_corr'] = data.groupby('breath_id')['u_in'].ewm(halflife=14).corr().reset_index(level=0,
                                                                                                      drop=True)
        data[["15_in_max", "15_out_std"]] = data.groupby('breath_id')['u_in'].rolling(window=15, min_periods=1).agg(
            {"15_in_max": "max", "15_in_std": "std"}).reset_index(level=0, drop=True)

        data['breath_time'] = data['time_step'] - data.groupby('breath_id')['time_step'].shift(1)
        data['u_in_lag'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_in_diff1'] = data['u_in'] - data['u_in_lag']

        self.data = data.fillna(0).groupby('breath_id').agg(list).reset_index()

        self.pressures = np.array(self.data['pressure'].values.tolist())
        u_ins = np.array(self.data['u_in'].values.tolist())

        self.u_outs = np.array(self.data['u_out'].values.tolist())
        self.rs = np.array(self.data['R'].values.tolist())
        self.cs = np.array(self.data['C'].values.tolist())

        self.inputs = np.concatenate([
            u_ins[:, None],
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None],
            np.array(self.data['breath_time'].values.tolist())[:, None],
            np.array(self.data['u_in_diff1'].values.tolist())[:, None],
            np.array(self.data['time_step'].values.tolist())[:, None],
            np.array(self.data['u_in_lag'].values.tolist())[:, None],
            np.array(self.data['area'].values.tolist())[:, None],
            np.array(self.data['ewm_u_in_mean'].values.tolist())[:, None],
            np.array(self.data['ewm_u_in_std'].values.tolist())[:, None],
            np.array(self.data['15_in_max'].values.tolist())[:, None],
            np.array(self.data['15_out_std'].values.tolist())[:, None],
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "r": torch.tensor(self.rs[:, None][idx], dtype=torch.long),
            "c": torch.tensor(self.cs[:, None][idx], dtype=torch.long),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)


class VentilatorDataset6(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 mode: str = '',
                 normalize: bool = False,
                 ):
        """
        from 5

        Args:
            data: dataframe with image id and bboxes
            mode: train/val/test
            img_path: path to images
            transforms: albumentations
        """
        if "pressure" not in data.columns:
            data['pressure'] = 0

        r_map = {5: 0, 20: 1, 50: 2}
        c_map = {10: 0, 20: 1, 50: 2}
        data['RC'] = data['R'] / data['C']
        data['R'] = data['R'].map(r_map)
        data['C'] = data['C'].map(c_map)

        data['area'] = data['time_step'] * data['u_in']
        data['area'] = data.groupby('breath_id')['area'].cumsum()

        data['ewm_u_in_mean'] = data.groupby('breath_id')['u_in'].ewm(halflife=8).mean().reset_index(level=0, drop=True)
        data['ewm_u_in_std'] = data.groupby('breath_id')['u_in'].ewm(halflife=9).std().reset_index(level=0, drop=True)
        data['ewm_u_in_corr'] = data.groupby('breath_id')['u_in'].ewm(halflife=14).corr().reset_index(level=0,
                                                                                                      drop=True)
        data[["15_in_max", "15_out_std"]] = data.groupby('breath_id')['u_in'].rolling(window=15, min_periods=1).agg(
            {"15_in_max": "max", "15_in_std": "std"}).reset_index(level=0, drop=True)

        data['breath_time'] = data['time_step'] - data.groupby('breath_id')['time_step'].shift(1)
        data['u_in_lag'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_in_lag2'] = data.groupby('breath_id')['u_in'].shift(2)
        data['u_in_lag-1'] = data.groupby('breath_id')['u_in'].shift(-1)
        data['u_in_lag-2'] = data.groupby('breath_id')['u_in'].shift(-2)
        data['u_out_lag'] = data.groupby('breath_id')['u_out'].shift(1)
        data['u_in_diff1'] = data['u_in'] - data['u_in_lag']

        data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(1)
        data['time_step_lag2'] = data.groupby('breath_id')['time_step'].shift(2)
        data['time_step_lag-1'] = data.groupby('breath_id')['time_step'].shift(-1)
        data['time_step_lag-2'] = data.groupby('breath_id')['time_step'].shift(-2)
        data['breath_time'] = data['time_step'] - data['time_step_lag']

        self.data = data.fillna(0).groupby('breath_id').agg(list).reset_index()

        self.pressures = np.array(self.data['pressure'].values.tolist())
        u_ins = np.array(self.data['u_in'].values.tolist())

        self.u_outs = np.array(self.data['u_out'].values.tolist())
        self.rs = np.array(self.data['R'].values.tolist())
        self.cs = np.array(self.data['C'].values.tolist())

        self.inputs = np.concatenate([
            u_ins[:, None],
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None],
            np.array(self.data['RC'].values.tolist())[:, None],
            np.array(self.data['breath_time'].values.tolist())[:, None],
            np.array(self.data['u_in_diff1'].values.tolist())[:, None],
            np.array(self.data['time_step'].values.tolist())[:, None],
            np.array(self.data['u_in_lag'].values.tolist())[:, None],
            np.array(self.data['u_in_lag2'].values.tolist())[:, None],
            np.array(self.data['u_in_lag-1'].values.tolist())[:, None],
            np.array(self.data['u_in_lag-2'].values.tolist())[:, None],
            np.array(self.data['area'].values.tolist())[:, None],
            np.array(self.data['ewm_u_in_mean'].values.tolist())[:, None],
            np.array(self.data['ewm_u_in_std'].values.tolist())[:, None],
            np.array(self.data['15_in_max'].values.tolist())[:, None],
            np.array(self.data['15_out_std'].values.tolist())[:, None],
            np.array(self.data['time_step_lag'].values.tolist())[:, None],
            np.array(self.data['time_step_lag2'].values.tolist())[:, None],
            np.array(self.data['time_step_lag-1'].values.tolist())[:, None],
            np.array(self.data['time_step_lag-2'].values.tolist())[:, None],
            np.array(self.data['u_out_lag'].values.tolist())[:, None],
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "r": torch.tensor(self.rs[:, None][idx], dtype=torch.long),
            "c": torch.tensor(self.cs[:, None][idx], dtype=torch.long),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)


class VentilatorDataset7(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 mode: str = '',
                 normalize: bool = False,
                 ):
        """
        from 6

        Args:
            data: dataframe with image id and bboxes
            mode: train/val/test
            img_path: path to images
            transforms: albumentations
        """
        if "pressure" not in data.columns:
            data['pressure'] = 0

        r_map = {5: 0, 20: 1, 50: 2}
        c_map = {10: 0, 20: 1, 50: 2}
        data['RC'] = data['R'] / data['C']
        data['RandC'] = data['R'] + data['C']
        data['R'] = data['R'].map(r_map)
        data['C'] = data['C'].map(c_map)

        data['area'] = data['time_step'] * data['u_in']
        data['area'] = data.groupby('breath_id')['area'].cumsum()
        data['cross'] = data['u_in'] * data['u_out']
        data['cross2'] = data['time_step'] * data['u_out']

        data['breath_id_lag'] = data.groupby('breath_id')['breath_id'].shift(1).fillna(0)
        data['breath_id_lag2'] = data.groupby('breath_id')['breath_id'].shift(1).fillna(0)
        data['breath_id_lagsame'] = np.select([data['breath_id_lag'] == data['breath_id']], [1], 0)
        data['breath_id_lag2same'] = np.select([data['breath_id_lag2'] == data['breath_id']], [1], 0)

        data['ewm_u_in_mean'] = data.groupby('breath_id')['u_in'].ewm(halflife=8).mean().reset_index(level=0, drop=True)
        data['ewm_u_in_std'] = data.groupby('breath_id')['u_in'].ewm(halflife=9).std().reset_index(level=0, drop=True)
        data['ewm_u_in_corr'] = data.groupby('breath_id')['u_in'].ewm(halflife=14).corr().reset_index(level=0,
                                                                                                      drop=True)
        data[["15_in_max", "15_out_std"]] = data.groupby('breath_id')['u_in'].rolling(window=15, min_periods=1).agg(
            {"15_in_max": "max", "15_in_std": "std"}).reset_index(level=0, drop=True)
        data[["10_in_max", "10_out_std"]] = data.groupby('breath_id')['u_in'].rolling(window=10, min_periods=1).agg(
            {"15_in_max": "max", "15_in_std": "std"}).reset_index(level=0, drop=True)

        data['expand_mean'] = data.groupby('breath_id')['u_in'].expanding(2).mean().reset_index(level=0, drop=True)
        data['expand_max'] = data.groupby('breath_id')['u_in'].expanding(2).max().reset_index(level=0, drop=True)
        data['expand_std'] = data.groupby('breath_id')['u_in'].expanding(2).std().reset_index(level=0, drop=True)

        data['breath_time'] = data['time_step'] - data.groupby('breath_id')['time_step'].shift(1)
        data['u_in_lag'] = data.groupby('breath_id')['u_in'].shift(1)
        data['u_in_lag'] = data['u_in_lag'] * data['breath_id_lagsame']
        data['u_in_lag2'] = data.groupby('breath_id')['u_in'].shift(2)
        data['u_in_lag2'] = data['u_in_lag2'] * data['breath_id_lag2same']
        data['u_in_lag-1'] = data.groupby('breath_id')['u_in'].shift(-1)
        data['u_in_lag-2'] = data.groupby('breath_id')['u_in'].shift(-2)
        data['u_out_lag'] = data.groupby('breath_id')['u_out'].shift(1)
        data['u_out_lag2'] = data.groupby('breath_id')['u_out'].shift(2)
        data['u_in_diff1'] = data['u_in'] - data['u_in_lag']
        data['u_out_lag2'] = data['u_out_lag2'] * data['breath_id_lag2same']

        data['time_step_lag'] = data.groupby('breath_id')['time_step'].shift(1)
        data['time_step_lag2'] = data.groupby('breath_id')['time_step'].shift(2)
        data['time_step_lag-1'] = data.groupby('breath_id')['time_step'].shift(-1)
        data['time_step_lag-2'] = data.groupby('breath_id')['time_step'].shift(-2)
        data['breath_time'] = data['time_step'] - data['time_step_lag']

        data['u_in_cumsum'] = (data['u_in']).groupby(data['breath_id']).cumsum()
        data['one'] = 1
        data['count'] = (data['one']).groupby(data['breath_id']).cumsum()
        data['u_in_cummean'] = data['u_in_cumsum'] / data['count']

        data = data.drop(['one', 'count', 'breath_id_lag', 'breath_id_lag2', 'breath_id_lagsame', 'breath_id_lag2same',
                          'u_out_lag2'], axis=1)
        cols_to_scale = ['time_step', 'u_in', 'u_out', 'pressure',
       'RC', 'RandC', 'area', 'cross', 'cross2', 'ewm_u_in_mean',
       'ewm_u_in_std', 'ewm_u_in_corr', '15_in_max', '15_out_std', '10_in_max',
       '10_out_std', 'expand_mean', 'expand_max', 'expand_std', 'breath_time',
       'u_in_lag', 'u_in_lag2', 'u_in_lag-1', 'u_in_lag-2', 'u_out_lag',
       'u_in_diff1', 'time_step_lag', 'time_step_lag2', 'time_step_lag-1',
       'time_step_lag-2', 'u_in_cumsum', 'u_in_cummean']
        rs = RobustScaler()
        if normalize:
            if mode == 'train':
                data[cols_to_scale] = rs.fit_transform(data[cols_to_scale])
                with open('rs.joblib', 'wb') as f:
                    joblib.dump(rs, f)
            else:
                with open('rs.joblib', 'rb') as f:
                    rs = joblib.load(f)
                data[cols_to_scale] = rs.transform(data[cols_to_scale])

        self.data = data.fillna(0).groupby('breath_id').agg(list).reset_index()

        self.pressures = np.array(self.data['pressure'].values.tolist())
        u_ins = np.array(self.data['u_in'].values.tolist())

        self.u_outs = np.array(self.data['u_out'].values.tolist())
        self.rs = np.array(self.data['R'].values.tolist())
        self.cs = np.array(self.data['C'].values.tolist())

        self.inputs = np.concatenate([
            u_ins[:, None],
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None],
            np.array(self.data['RC'].values.tolist())[:, None],
            np.array(self.data['RandC'].values.tolist())[:, None],
            np.array(self.data['breath_time'].values.tolist())[:, None],
            np.array(self.data['u_in_diff1'].values.tolist())[:, None],
            np.array(self.data['time_step'].values.tolist())[:, None],
            np.array(self.data['u_in_lag'].values.tolist())[:, None],
            np.array(self.data['u_in_lag2'].values.tolist())[:, None],
            np.array(self.data['u_in_lag-1'].values.tolist())[:, None],
            np.array(self.data['u_in_lag-2'].values.tolist())[:, None],
            np.array(self.data['area'].values.tolist())[:, None],
            np.array(self.data['ewm_u_in_mean'].values.tolist())[:, None],
            np.array(self.data['ewm_u_in_std'].values.tolist())[:, None],
            np.array(self.data['15_in_max'].values.tolist())[:, None],
            np.array(self.data['15_out_std'].values.tolist())[:, None],
            np.array(self.data['10_in_max'].values.tolist())[:, None],
            np.array(self.data['10_out_std'].values.tolist())[:, None],
            np.array(self.data['time_step_lag'].values.tolist())[:, None],
            np.array(self.data['time_step_lag2'].values.tolist())[:, None],
            np.array(self.data['time_step_lag-1'].values.tolist())[:, None],
            np.array(self.data['time_step_lag-2'].values.tolist())[:, None],
            np.array(self.data['u_out_lag'].values.tolist())[:, None],
            np.array(self.data['cross'].values.tolist())[:, None],
            np.array(self.data['cross2'].values.tolist())[:, None],
            np.array(self.data['u_in_cummean'].values.tolist())[:, None],
            np.array(self.data['expand_mean'].values.tolist())[:, None],
            np.array(self.data['expand_max'].values.tolist())[:, None],
            np.array(self.data['expand_std'].values.tolist())[:, None],
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "r": torch.tensor(self.rs[:, None][idx], dtype=torch.long),
            "c": torch.tensor(self.cs[:, None][idx], dtype=torch.long),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)
