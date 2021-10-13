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

python train.py datamodule.num_workers=0 datamodule.batch_size=256 callbacks.early_stopping.params.patience=50 general.log_code=False datamodule.fold_n=0 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset10 model=ventilator_model_10 model.params.input_dim=24 model.params.dense_dim=128 model.params.lstm_dim=512 model.params.logit_dim=128 scheduler=step trainer.gpus=1 trainer.max_epochs=1 model.params.initialize=True model.params.init_style=9 general.predict=False

trainer.gradient_clip_val=0.5

model.params.use_mlp=True
model.params.use_mlp=False
model.params.use_mlp=True model.params.simpler_mlp=True

model.params.use_layer_norm=True model.params.layer_norm_style=0
model.params.use_layer_norm=True model.params.layer_norm_style=1
model.params.use_layer_norm=True model.params.layer_norm_style=2
model.params.use_layer_norm=True model.params.layer_norm_style=3
model.params.use_layer_norm=True model.params.layer_norm_style=4


model.params.use_layer_norm=True model.params.layer_norm_style=5 model.params.use_mlp=True model.params.simpler_mlp=True

model.params.dropout=0.1
model.params.lstm_dropout=0.2

model.params.init_style=0
model.params.init_style=8
model.params.init_style=9

model.params.layer_norm_logits=True

optimizer=adamw
optimizer=adabelief
optimizer=rmsprop


scheduler=cosinewarm
scheduler=cyclic
scheduler=linearwithwarmup

lr=0.01, 0.001, 0.0001


activation=torch.nn.PReLU
activation=torch.nn.Tanh
activation=torch.nn.SELU

activation2=torch.nn.SELU
activation2=torch.nn.PReLU

model.params.dense_dim=512
model.params.lstm_dim=1024
model.params.num_layers=2


datamodule.data_module_name=src.lightning_classes.datamodule_ventilator1.ImagenetteDataModule datamodule.class_name=src.datasets.ventilator_dataset2.VentilatorDataset

=====
```shell
0.2090
ventilator_kaggle_models/train.py datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset12 model=ventilator_model_8 model.params.use_mlp=False model.params.nhead=5 general.log_code=False model.params.input_dim=30 datamodule.fold_n=0 scheduler=exponential logging=wandb datamodule.normalize=True

0.1996
ventilator_kaggle_models/train.py datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset10 model=ventilator_model_8 general.log_code=False model.params.input_dim=24 model.params.use_mlp=False model.params.nhead=4 datamodule.fold_n=0 scheduler=exponential logging=wandb datamodule.normalize=True


0.2003
ventilator_kaggle_models/train.py datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset13 model=ventilator_model_8 model.params.use_mlp=False model.params.nhead=5 general.log_code=False model.params.input_dim=55 datamodule.fold_n=0 scheduler=exponential logging=wandb

0.1967
ventilator_kaggle_models/train.py datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset13 model=ventilator_model_8 model.params.use_mlp=False model.params.nhead=5 general.log_code=False model.params.input_dim=55 datamodule.fold_n=0 scheduler=exponential logging=wandb datamodule.normalize=True

0.1954
ventilator_kaggle_models/train.py datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=30 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset8 model=ventilator_model_5 general.log_code=False model.params.input_dim=35 datamodule.fold_n=0 logging=wandb scheduler=exponential datamodule.split=GroupShuffleSplit

0.1925
ventilator_kaggle_models/train.py datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential model.params.use_mlp=True model.params.nhead=8 model.params.use_only_transformer_encoder=True model.params.use_transformer=True model.params.use_lstm=True logging=wandb

0.1923
ventilator_kaggle_models/train.py datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential model.params.use_mlp=True model.params.nhead=8 model.params.use_only_transformer_encoder=True model.params.use_transformer=True model.params.use_lstm=True logging=wandb model.params.activation=torch.nn.LeakyReLU

0.1913
ventilator_kaggle_models/train.py datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential model.params.use_mlp=True model.params.nhead=8 model.params.use_only_transformer_encoder=True model.params.use_transformer=True model.params.use_lstm=True logging=wandb model.params.activation=torch.nn.GELU

0.1908
ventilator_kaggle_models/train.py datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset8 model=ventilator_model_8 general.log_code=False model.params.input_dim=35 datamodule.fold_n=0 scheduler=exponential model.params.use_mlp=False model.params.nhead=5 logging=wandb

-------------------------------
python train.py datamodule.num_workers=0 datamodule.batch_size=256 callbacks.early_stopping.params.patience=50 general.log_code=False datamodule.fold_n=0 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset8 model=ventilator_model_8 model.params.use_mlp=False model.params.nhead=5 model.params.input_dim=35 scheduler=step trainer.gpus=1 trainer.max_epochs=3 general.predict=False trainer.gradient_clip_val=0.5 optimizer=adabelief scheduler=cosinewarm loss=mae metric=metric_manager

```
------

python train.py datamodule.num_workers=0 datamodule.batch_size=32 callbacks.early_stopping.params.patience=50 general.log_code=False datamodule.fold_n=0 datamodule=ventilator_datamodule_notebook model=ventilator_model_notebook model.params.input_dim=50 trainer.gpus=1 trainer.max_epochs=3 general.predict=False loss=mae metric=metric_manager training.debug=True optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 trainer.gradient_clip_val=1000


--------
datamodule=ventilator_datamodule_notebook datamodule.data_module_name=src.lightning_classes.datamodule_ventilator_notebook1.VentilatorDataModule datamodule.make_features_style=3 

model=ventilator_model_notebook model.class_name=src.models.ventilator_model_notebook1.VentilatorNet model.params.input_dim=50

=======
srun --mem=64gb --gres=gpu:1 --nodes=1 --container-image=artgor/cv_nvidia:latest --no-container-entrypoint --container-mount-home --container-mounts=/home/mtsml_006:/workspace python ventilator_kaggle_models/train.py \
callbacks.early_stopping.params.patience=50 general.log_code=False logging=wandb \
model=ventilator_model_notebook model.class_name=src.models.ventilator_model_notebook1.VentilatorNet model.params.input_dim=50 \
trainer.gpus=1 trainer.max_epochs=1000 trainer.gradient_clip_val=1000 \
loss=mae metric=metric_manager \
optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 \
datamodule.num_workers=0 datamodule.batch_size=1024 datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule=ventilator_datamodule_notebook datamodule.data_module_name=src.lightning_classes.datamodule_ventilator_notebook1.VentilatorDataModule datamodule.make_features_style=3 datamodule.normalize=False datamodule.fold_n=0

======
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model_notebook model.class_name=src.models.ventilator_model_notebook1.VentilatorNet model.params.input_dim=50 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=mae metric=metric_manager optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_notebook datamodule.data_module_name=src.lightning_classes.datamodule_ventilator_notebook1.VentilatorDataModule datamodule.make_features_style=3 datamodule.normalize=False datamodule.fold_n=0
--
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=50 model.params.init_style=10 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=mae metric=metric_manager optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=1 datamodule.normalize=False datamodule.fold_n=0

BatchNorm1d(80) after LSTM/GRU