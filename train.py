import os
import warnings
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GroupKFold

from src.metrics.ventilator_mae import VentilatorMAE
from src.utils.technical_utils import load_obj
from src.utils.utils import set_seed, save_useful_info

warnings.filterwarnings('ignore')


def run(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model

    # TODO: check their f1

    Args:
        cfg: hydra config

    """
    set_seed(cfg.training.seed)
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
    trainer.fit(model, dm)

    if cfg.general.save_pytorch_model and cfg.general.save_best:
        if os.path.exists(trainer.checkpoint_callback.best_model_path):  # type: ignore
            best_path = trainer.checkpoint_callback.best_model_path  # type: ignore
            # extract file name without folder
            save_name = os.path.basename(os.path.normpath(best_path))
            model = model.load_from_checkpoint(best_path, cfg=cfg, strict=False)
            model_name = Path(
                cfg.callbacks.model_checkpoint.params.dirpath, f'best_{save_name}'.replace('.ckpt', '.pth')
            ).as_posix()
            torch.save(model.model.state_dict(), model_name)
        else:
            os.makedirs('saved_models', exist_ok=True)
            model_name = 'saved_models/last.pth'
            torch.save(model.model.state_dict(), model_name)

    if cfg.general.predict:
        if cfg.training.debug:
            sub = pd.read_csv(os.path.join(cfg.datamodule.path, 'sample_submission.csv'), nrows=80000)
        else:
            sub = pd.read_csv(os.path.join(cfg.datamodule.path, 'sample_submission.csv'))
        prediction = trainer.predict(model, dm.test_dataloader())
        predictions = []
        for pred in prediction:
            predictions.extend(pred.reshape(-1, ).detach().cpu().numpy())
        sub['pressure'] = predictions
        sub.to_csv(f'submission_{run_name}.csv', index=False)

        if cfg.training.debug:
            oof = pd.read_csv(os.path.join(cfg.datamodule.path, 'train.csv'), nrows=196000)
        else:
            oof = pd.read_csv(os.path.join(cfg.datamodule.path, 'train.csv'))

        gkf = GroupKFold(n_splits=cfg.datamodule.n_folds).split(oof, oof.pressure, groups=oof.breath_id)
        for fold, (_, valid_idx) in enumerate(gkf):
            oof.loc[valid_idx, 'fold'] = fold
        oof = oof[['id', 'breath_id', 'pressure', 'fold', 'u_out']]
        print(f'{cfg.datamodule.fold_n=}')
        y_true = oof.loc[oof['fold'] == cfg.datamodule.fold_n, 'pressure']
        oof['pressure'] = 0

        prediction = trainer.predict(model, dm.val_dataloader())
        predictions = []
        for pred in prediction:
            predictions.extend(pred.reshape(-1, ).detach().cpu().numpy())

        oof.loc[oof['fold'] == cfg.datamodule.fold_n, 'pressure'] = predictions
        print('ventilator mae',
              VentilatorMAE()(torch.tensor(oof.loc[oof['fold'] == cfg.datamodule.fold_n, 'pressure'].values),
                              torch.tensor(y_true.values),
                              torch.tensor(oof.loc[oof['fold'] == cfg.datamodule.fold_n, 'u_out'].values)))
        oof.to_csv(f'{run_name}_oof_fold_{cfg.datamodule.fold_n}.csv', index=False)

    print(run_name)


@hydra.main(config_path='conf', config_name='config_0')
def run_model(cfg: DictConfig) -> None:
    os.makedirs('logs', exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    if cfg.general.log_code:
        save_useful_info()
    run(cfg)


if __name__ == '__main__':
    run_model()
