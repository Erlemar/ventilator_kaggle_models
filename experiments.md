```shell

python train.py datamodule.num_workers=0 datamodule.batch_size=256 callbacks.early_stopping.params.patience=50 general.log_code=False datamodule.fold_n=0 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset10 model=ventilator_model_9 model.params.input_dim=24 model.params.dense_dim=24 model.params.lstm_dim=400 model.params.logit_dim=128 scheduler=step trainer.gpus=1 trainer.max_epochs=10

python train.py datamodule.num_workers=0 datamodule.batch_size=256 callbacks.early_stopping.params.patience=50 general.log_code=False datamodule.fold_n=0 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset10 model=ventilator_model_9 model.params.input_dim=24 model.params.dense_dim=24 model.params.lstm_dim=400 model.params.logit_dim=50 scheduler=step trainer.gpus=1 trainer.max_epochs=10 loss=mae

python train.py datamodule.num_workers=0 datamodule.batch_size=256 callbacks.early_stopping.params.patience=50 general.log_code=False datamodule.fold_n=0 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset10 model=ventilator_model_9 model.params.input_dim=24 model.params.dense_dim=24 model.params.lstm_dim=400 model.params.logit_dim=128 scheduler=exponential scheduler.params.gamma=0.9999 trainer.gpus=1 trainer.max_epochs=10

python train.py datamodule.num_workers=0 datamodule.batch_size=256 callbacks.early_stopping.params.patience=50 general.log_code=False datamodule.fold_n=0 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset10 model=ventilator_model_9 model.params.input_dim=24 model.params.dense_dim=24 model.params.lstm_dim=400 model.params.logit_dim=128 scheduler=exponential scheduler.params.gamma=0.9999 optimizer=adamw optimizer.params.weight_decay: 0.0001 trainer.gpus=1 trainer.max_epochs=10

python train.py datamodule.num_workers=0 datamodule.batch_size=256 callbacks.early_stopping.params.patience=50 general.log_code=False datamodule.fold_n=0 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset10 model=ventilator_model_9 model.params.input_dim=24 model.params.dense_dim=24 model.params.lstm_dim=400 model.params.logit_dim=50 scheduler=step trainer.gpus=1 trainer.max_epochs=10 loss=mae

python train.py datamodule.num_workers=0 datamodule.batch_size=256 callbacks.early_stopping.params.patience=50 general.log_code=False datamodule.fold_n=0 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset10 model=ventilator_model_9 model.params.input_dim=24 model.params.dense_dim=24 model.params.lstm_dim=400 model.params.logit_dim=128 scheduler=step trainer.gpus=1 trainer.max_epochs=1 model.params.initialize=True model.params.init_style=1 general.predict=False

python train.py datamodule.num_workers=0 datamodule.batch_size=256 callbacks.early_stopping.params.patience=50 general.log_code=False datamodule.fold_n=0 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset10 model=ventilator_model_9 model.params.input_dim=24 model.params.dense_dim=24 model.params.lstm_dim=400 model.params.logit_dim=128 scheduler=step trainer.gpus=1 trainer.max_epochs=1 model.params.initialize=True model.params.init_style=2 general.predict=False
3, 4, 5, 6, 7


initialize

```