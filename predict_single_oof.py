import argparse
import glob
import os

import pandas as pd
import torch
import yaml
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from src.metrics.ventilator_mae import VentilatorMAE
from src.utils.technical_utils import load_obj
from src.utils.utils import set_seed
import os
import warnings
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.technical_utils import load_obj
from src.utils.utils import set_seed, save_useful_info
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')


def make_prediction(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model inference

    Args:
        cfg: hydra config

    Returns:
        None
    """
    # set_seed(cfg.training.seed)
    run_name = os.path.basename(os.getcwd())
    cfg.callbacks.model_checkpoint.params.dirpath = Path(
        os.getcwd(), cfg.callbacks.model_checkpoint.params.dirpath
    ).as_posix()
    callbacks = []
    for callback in cfg.callbacks.other_callbacks:
        if callback.params:
            callback_instance = load_obj(callback.class_name)(**callback.params)
        else:
            callback_instance = load_obj(callback.class_name)()
        callbacks.append(callback_instance)

    loggers = []
    if cfg.logging.log:
        for logger in cfg.logging.loggers:
            if 'experiment_name' in logger.params.keys():
                logger.params['experiment_name'] = run_name
            elif 'name' in logger.params.keys():
                logger.params['name'] = run_name

            loggers.append(load_obj(logger.class_name)(**logger.params))

    callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping.params))
    callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint.params))

    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        **cfg.trainer,
    )

    model = load_obj(cfg.training.lightning_module_name)(cfg=cfg)
    dm = load_obj(cfg.datamodule.data_module_name)(cfg=cfg)
    # best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
    # extract file name without folder
    model_path = glob.glob(f'outputs/{cfg.inference.run_name}/saved_models/best*.pth')[0]
    print(model_path)
    model.model.load_state_dict(torch.load(model_path))

    oof = pd.read_csv(os.path.join(cfg.datamodule.path, 'train.csv'))
    gkf = GroupKFold(n_splits=cfg.datamodule.n_folds).split(oof, oof.pressure, groups=oof.breath_id)
    for fold, (_, valid_idx) in enumerate(gkf):
        oof.loc[valid_idx, 'fold'] = fold
    oof = oof[['id', 'breath_id', 'pressure', 'fold', 'u_out']]
    print(f'{cfg.datamodule.fold_n=}')
    y_true = oof.loc[oof['fold'] == cfg.datamodule.fold_n, 'pressure']
    oof['pressure'] = 0

    dm.setup()

    prediction = trainer.predict(model, dm.val_dataloader())
    print(len(prediction))
    predictions = []
    for pred in prediction:
        predictions.extend(pred.reshape(-1, ).detach().cpu().numpy())

    print('predictions', len(predictions))
    print('valids', oof.loc[oof['fold'] == cfg.datamodule.fold_n, 'pressure'].shape)
    oof.loc[oof['fold'] == cfg.datamodule.fold_n, 'pressure'] = predictions
    print('sklearn MAE', mean_absolute_error(y_true, oof.loc[oof['fold'] == cfg.datamodule.fold_n, 'pressure']))
    print('ventilator mae', VentilatorMAE()(torch.tensor(oof.loc[oof['fold'] == cfg.datamodule.fold_n, 'pressure'].values),
                                      torch.tensor(y_true.values),
                                      torch.tensor(oof.loc[oof['fold'] == cfg.datamodule.fold_n, 'u_out'].values)))
    oof.to_csv(f'outputs/{cfg.inference.run_name}_oof_fold_{cfg.datamodule.fold_n}.csv', index=False)
    ####

    #
    # model_path = glob.glob(f'outputs/{cfg.inference.run_name}/saved_models/best*.pth')[0]
    # print(model_path)
    #
    #
    #
    #
    # predictions = []
    # device = 'cuda'
    # model = load_obj(cfg.model.class_name)(**cfg.model.params)
    # model.load_state_dict(torch.load(model_path))
    # model.to(device)
    # model.eval()
    #
    # with torch.no_grad():
    #
    #     for ind, batch in enumerate(test_loader):
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         prediction = model(batch).squeeze(-1)
    #         for pred in prediction:
    #             predictions.extend(pred.reshape(-1, ).detach().cpu().numpy())
    #
    # sub['pressure'] = predictions
    # sub.to_csv(f'outputs/{cfg.inference.run_name}_prediction.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference in Ventilator competition')
    parser.add_argument('--run_name', help='folder_name', type=str, default='2021-10-20_16-33-02')
    parser.add_argument('--mode', help='valid or test', type=str, default='valid')
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
    cfg['training']['debug'] = False
    print(OmegaConf.to_yaml(cfg))
    make_prediction(cfg)
