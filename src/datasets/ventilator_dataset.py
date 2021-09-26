from typing import Dict

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset


class VentilatorDataset(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 ):
        """
        Image classification dataset.

        Args:
            df: dataframe with image id and bboxes
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
                 ):
        """
        Image classification dataset.

        Args:
            df: dataframe with image id and bboxes
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
