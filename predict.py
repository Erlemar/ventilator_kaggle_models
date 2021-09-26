import argparse
import glob
import os

import pandas as pd
import torch
import yaml
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.utils.technical_utils import load_obj
from src.utils.utils import set_seed


def make_prediction(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model inference

    Args:
        cfg: hydra config

    Returns:
        None
    """
    set_seed(cfg.training.seed)
    test = pd.read_csv(os.path.join(cfg.datamodule.path, 'test.csv'))
    dataset_class = load_obj(cfg.datamodule.class_name)
    test_dataset = dataset_class(test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        pin_memory=cfg.datamodule.pin_memory,
        shuffle=False,
    )

    model_path = glob.glob(f'outputs/{cfg.inference.run_name}/saved_models/best*.pth')[0]
    print(model_path)

    sub = pd.read_csv(os.path.join(cfg.datamodule.path, 'sample_submission.csv'))

    predictions = []
    device = 'cuda'
    model = load_obj(cfg.model.class_name)(**cfg.model.params)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():

        for ind, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            prediction = model(batch).squeeze(-1)
            for pred in prediction:
                predictions.extend(pred.reshape(-1, ).detach().cpu().numpy())

    sub['pressure'] = predictions
    sub.to_csv(f'outputs/{cfg.inference.run_name}_prediction.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference in Melanoma competition')
    parser.add_argument('--run_name', help='folder_name', type=str, default='2021-09-26_07-24-47')
    parser.add_argument('--mode', help='valid or test', type=str, default='test')
    args = parser.parse_args()

    initialize(config_path='conf')
    inference_cfg = compose(config_name='config.yaml')
    inference_cfg['inference']['run_name'] = args.run_name
    inference_cfg['inference']['mode'] = args.mode
    print(inference_cfg.inference.run_name)
    path = f'outputs/{inference_cfg.inference.run_name}/.hydra/config.yaml'

    with open(path) as cfg:
        cfg_yaml = yaml.safe_load(cfg)

    cfg_yaml['inference'] = inference_cfg['inference']
    cfg = OmegaConf.create(cfg_yaml)
    make_prediction(cfg)
