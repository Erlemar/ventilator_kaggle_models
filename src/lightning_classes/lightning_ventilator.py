import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import os
from omegaconf import DictConfig

from src.utils.technical_utils import load_obj

class VentilatorRegression(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(VentilatorRegression, self).__init__()
        self.cfg = cfg
        self.model = load_obj(cfg.model.class_name)(**self.cfg.model.params)
        print(self.model)
        if 'params' in self.cfg.loss:
            self.loss = load_obj(cfg.loss.class_name)(**self.cfg.loss.params)
        else:
            self.loss = load_obj(cfg.loss.class_name)()
        self.metrics = torch.nn.ModuleDict(
            {
                self.cfg.metric.metric.metric_name: load_obj(self.cfg.metric.metric.class_name)()
            }
        )
        if 'other_metrics' in self.cfg.metric.keys():
            for metric in self.cfg.metric.other_metrics:
                self.metrics.update({metric.metric_name: load_obj(metric.class_name)()})

        print(f'{self.metrics=}')
        self.best_ventilator_mae = torch.tensor(1000)
        if self.cfg.training.pp_for_loss:
            train = pd.read_csv(os.path.join(self.cfg.datamodule.path, 'train.csv'))
            all_pressure = sorted(train['pressure'].unique())
            self.pressure_min = torch.tensor(all_pressure[0], device=self.device)
            self.pressure_max = torch.tensor(all_pressure[-1], device=self.device)
            self.pressure_step = all_pressure[1] - all_pressure[0]

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        if 'decoder_lr' in self.cfg.optimizer.params.keys():
            params = [
                {'params': self.model.decoder.parameters(), 'lr': self.cfg.optimizer.params.lr},
                {'params': self.model.encoder.parameters(), 'lr': self.cfg.optimizer.params.decoder_lr},
            ]
            optimizer = load_obj(self.cfg.optimizer.class_name)(params)

        else:
            optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return (
            [optimizer],
            [{'scheduler': scheduler, 'interval': self.cfg.scheduler.step, 'monitor': self.cfg.scheduler.monitor,
              'name': self.cfg.scheduler.class_name}],
        )

    def training_step(self, batch, *args, **kwargs):  # type: ignore
        # data = batch['input']
        # pred = self(data).squeeze(-1)
        pred = self(batch).squeeze(-1)
        # print('pred', pred)
        # print('train_batch', batch['input'].shape)
        if self.cfg.training.pp_for_loss:
            pred1 = torch.round((pred - self.pressure_min) / self.pressure_step) * self.pressure_step + self.pressure_min
            pred1 = torch.clip(pred1, self.pressure_min, self.pressure_max)

            if self.cfg.loss.class_name == 'torch.nn.L1Loss' or self.cfg.loss.class_name == 'torch.nn.HuberLoss':
                loss = self.loss(pred1, batch['p']).mean()
            else:
                loss = self.loss(pred1, batch['p'], batch['u_out']).mean()
        else:
            if self.cfg.loss.class_name == 'torch.nn.L1Loss' or self.cfg.loss.class_name == 'torch.nn.HuberLoss':
                loss = self.loss(pred, batch['p']).mean()
            else:
                loss = self.loss(pred, batch['p'], batch['u_out']).mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        for metric in self.metrics:
            if metric == 'mae':
                score = self.metrics[metric](pred, batch['p']).mean()
            else:
                score = self.metrics[metric](pred, batch['p'], batch['u_out']).mean()
            self.log(f'train_{metric}', score, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, *args, **kwargs):  # type: ignore
        # data = batch['input']
        # pred = self(data).squeeze(-1)
        pred = self(batch).squeeze(-1)
        # print('valid_batch', batch['input'].shape)
        if self.cfg.loss.class_name == 'torch.nn.L1Loss' or self.cfg.loss.class_name == 'torch.nn.HuberLoss':
            loss = self.loss(pred, batch['p']).mean()
        else:
            loss = self.loss(pred, batch['p'], batch['u_out']).mean()
        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for metric in self.metrics:
            if metric == 'mae':
                score = self.metrics[metric](pred, batch['p']).mean()
            else:
                score = self.metrics[metric](pred, batch['p'], batch['u_out']).mean()
                if metric == 'ventilator_mae':
                    self.best_ventilator_mae = score if score < self.best_ventilator_mae else self.best_ventilator_mae

                    self.log('best_ventilator_mae', self.best_ventilator_mae, on_step=False, on_epoch=True, prog_bar=True,
                             logger=True)

            self.log(f'valid_{metric}', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, *args, **kwargs):  # type: ignore
        # data = batch['input']
        return self(batch).squeeze(-1)

    # def training_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     y_true = torch.cat([x['target'] for x in outputs])
    #     y_pred = torch.cat([x['logits'] for x in outputs])
    #     score = self.metric(y_pred.argmax(1), y_true)
    #
    #     # score = torch.tensor(1.0, device=self.device)
    #
    #     logs = {'train_loss': avg_loss, f'train_{self.cfg.training.metric}': score}
    #     return {'log': logs, 'progress_bar': logs}