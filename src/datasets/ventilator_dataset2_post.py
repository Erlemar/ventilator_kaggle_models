from typing import Dict

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset


class VentilatorDataset(Dataset):

    def __init__(self,
                 data: pd.DataFrame,
                 mode: str = '',
                 u_out: np.array = None,
                 pressure: np.array = None,
                 pressure1: np.array = None,
                 pressure2: np.array = None,

                 ):
        """
        Image classification dataset.

        Args:
            data: dataframe with image id and bboxes
            mode: train/val/test

        """
        print(f'{mode=}')
        self.data = data
        del data
        self.u_outs = u_out
        self.pressures = pressure
        self.pressures1 = pressure1
        self.pressures2 = pressure2

    def __getitem__(self, idx: int) -> Dict[str, npt.ArrayLike]:
        data = {
            "input": torch.tensor(self.data[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
            "p1": torch.tensor(self.pressures1[idx], dtype=torch.float),
            "p2": torch.tensor(self.pressures2[idx], dtype=torch.float),
        }

        return data

    def __len__(self) -> int:
        return len(self.data)
