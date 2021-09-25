from typing import List, Dict, Optional

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from albumentations.core.composition import Compose
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
