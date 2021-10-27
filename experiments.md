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
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=50 model.params.init_style=3 model.params.lstm_layers=6 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=1 datamodule.normalize=False datamodule.fold_n=0

--
1
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__1 model.class_name=src.models.ventilator_model__1.VentilatorNet model.params.input_dim=50 model.params.init_style=3 model.params.nhead=25 model.params.transformer_num_layers=1 model.params.use_transformer_encoder=True trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=1 datamodule.normalize=False datamodule.fold_n=0

2
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__1 model.class_name=src.models.ventilator_model__1.VentilatorNet model.params.input_dim=50 model.params.init_style=3 model.params.nhead=25 model.params.transformer_num_layers=1 model.params.use_transformer=True trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=1 datamodule.normalize=False datamodule.fold_n=0

3
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__1 model.class_name=src.models.ventilator_model__1.VentilatorNet model.params.input_dim=50 model.params.init_style=3 model.params.nhead=25 model.params.transformer_num_layers=1 model.params.use_transformer=True model.params.use_transformer_encoder=True trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=1 datamodule.normalize=False datamodule.fold_n=0


init style: 3, 5, 6
try without normalizing data
try without initializing

python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__1 model.class_name=src.models.ventilator_model__1.VentilatorNet model.params.input_dim=50 model.params.init_style=6 model.params.nhead=8 model.params.transformer_num_layers=4 model.params.use_transformer_encoder=True model.params.use_mlp=True trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=1 datamodule.normalize=False datamodule.fold_n=0

2 - 53
3 - 108
4 - 75
5 - 130 
6 - 145
7 - 149
8 - 149
9 - 112
10 - 276
11 - 158
12 - 153
13 - 326

15 - 168

3: 103
8: 139
10: 124
===
conv

python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__heng model.class_name=src.models.ventilator_model__heng.VentilatorNet model.params.input_dim=53 model.params.init_style=3 model.params.lstm_layers=6 model.params.double_conv=True model.params.conv_out_channels=64 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=128 datamodule=ventilator_datamodule_heng datamodule.make_features_style=2 datamodule.normalize=True datamodule.fold_n=0

conv_out_channels 16, 32, 64
conv_style 1, 2
double_conv False, True

python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__heng1 model.class_name=src.models.ventilator_model__heng1.VentilatorNet model.params.input_dim=53 model.params.init_style=3 model.params.lstm_layers=6 model.params.conv_out_channels=64 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=128 datamodule=ventilator_datamodule_heng datamodule.make_features_style=2 datamodule.normalize=True datamodule.fold_n=0 model.params.conv_layers=5

===
datamodule.normalize=True
datamodule.normalize_all=True
datamodule.make_features_style=6
datamodule.use_lag=4
model.params.init_style=12
model.params.lstm_dim=256
model.params.num_layers=4

python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__2 model.class_name=src.models.ventilator_model__2.VentilatorNet model.params.input_dim=23 model.params.init_style=12 model.params.lstm_layers=1 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=6 datamodule.normalize=True datamodule.fold_n=0 datamodule.normalize_all=True datamodule.use_lag=4 model.params.lstm_dim=256 model.params.num_layers=4


python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__3 model.class_name=src.models.ventilator_model__3.VentilatorNet model.params.input_dim=23 model.params.init_style=12 model.params.lstm_layers=1 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=6 datamodule.normalize=True datamodule.fold_n=0 datamodule.normalize_all=True datamodule.use_lag=4 model.params.lstm_dim=256 model.params.num_layers=4 training.lightning_module_name=src.lightning_classes.lightning_ventilator__0.VentilatorRegression loss=smoothl1loss training.loss_calc_style=2


python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__3 model.class_name=src.models.ventilator_model__3.VentilatorNet model.params.input_dim=29 model.params.init_style=12 model.params.lstm_layers=1 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adamw training.lr=0.005 optimizer.params.weight_decay=0.001 scheduler=cosinewithwarmup datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=6 datamodule.normalize=True datamodule.fold_n=0 datamodule.normalize_all=True datamodule.use_lag=6 model.params.lstm_dim=1024 model.params.num_layers=6 training.lightning_module_name=src.lightning_classes.lightning_ventilator__0.VentilatorRegression loss=mae training.loss_calc_style=1

==
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=50 model.params.init_style=3 model.params.lstm_layers=6 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=1 datamodule.normalize=False datamodule.fold_n=0

# stacked layers in original
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=50 model.params.init_style=3 model.params.lstm_layers=6 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=1 datamodule.normalize=False datamodule.fold_n=0 model.params.single_lstm=True model.params.num_layers=4


# original
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=50 model.params.init_style=3 model.params.lstm_layers=6 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=1 datamodule.normalize=False datamodule.fold_n=0 model.params.single_lstm=False model.params.num_layers=1

MAKE OOF
# ПОЧЕМУ СТАЛО ХУЖЕ?
фолды
фичи
модель

!!!!во-первых, найти лучшую версию.
!!!!во-вторых, понять в чем разница.
==
анализировать предсказания моделей и их ошибки
мои нововведения
взять как старт ту новую на 20 фолдах и добавлять к ней все
- dropout for regularization
- try make_features2
- taking the cut of the sequence from 80 to 35 or 32
- фичи
- нормализация всего
- модель
- 2 головы
- лоссы
- adamw
- scheduler LESS AGRESSIVE
- 950 classes? or postprosess predictions?!!!!!!!!!!
- голова на предсказание следующего значения
+ swa
- 3 best models
- cnn in header 
- pretrain on predicting next?
- https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/280356#1552537
          self.cnn1 = nn.Conv1d(config.EMBED_SIZE, config.HIDDEN_SIZE, kernel_size=2, padding=1)
          self.cnn2 = nn.Conv1d(config.HIDDEN_SIZE, config.HIDDEN_SIZE, kernel_size=2, padding=0)

multihead attention?
positional embedding?



ventilator_kaggle_models/train.py callbacks.early_stopping.params.patience=50 general.log_code=False logging=wandb model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=108 model.params.init_style=3 model.params.lstm_layers=6 model.params.num_layers=1 trainer.gpus=1 trainer.max_epochs=1000 trainer.gradient_clip_val=1000 loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=1024 datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule=ventilator_datamodule_0 datamodule.make_features_style=3 datamodule.n_folds=20 datamodule.fold_n=0

--
new best!
0.1516
ventilator_kaggle_models/train.py callbacks.early_stopping.params.patience=50 general.log_code=False logging=wandb model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=149 model.params.init_style=3 model.params.lstm_layers=6 model.params.num_layers=1 trainer.gpus=1 trainer.max_epochs=1000 trainer.gradient_clip_val=1000 loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=1024 datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule=ventilator_datamodule_0 datamodule.make_features_style=8 datamodule.n_folds=20 datamodule.fold_n=0


0.1534
ventilator_kaggle_models/train.py callbacks.early_stopping.params.patience=50 general.log_code=False logging=wandb model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=112 model.params.init_style=3 model.params.lstm_layers=6 model.params.num_layers=1 trainer.gpus=1 trainer.max_epochs=1000 trainer.gradient_clip_val=1000 loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=1024 datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule=ventilator_datamodule_0 datamodule.make_features_style=9 datamodule.n_folds=20 datamodule.fold_n=0 datamodule.normalize_all=True

0.1567
ventilator_kaggle_models/train.py callbacks.early_stopping.params.patience=50 general.log_code=False logging=wandb model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=108 model.params.init_style=3 model.params.lstm_layers=6 model.params.num_layers=1 trainer.gpus=1 trainer.max_epochs=1000 trainer.gradient_clip_val=1000 loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=1024 datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule=ventilator_datamodule_0 datamodule.make_features_style=3 datamodule.normalize_all=True datamodule.n_folds=20 datamodule.fold_n=0

0.1569
ventilator_kaggle_models/train.py callbacks.early_stopping.params.patience=50 general.log_code=False logging=wandb model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=108 model.params.init_style=3 model.params.lstm_layers=6 model.params.num_layers=1 trainer.gpus=1 trainer.max_epochs=1000 trainer.gradient_clip_val=1000 loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=1024 datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule=ventilator_datamodule_0 datamodule.make_features_style=3 datamodule.n_folds=30 datamodule.fold_n=0
=====


1
model.class_name=src.models.ventilator_model__0_0.VentilatorNet

2

model.class_name=src.models.ventilator_model__0_1.VentilatorNet

3
model.class_name=src.models.ventilator_model__0_2.VentilatorNet

4
feature_style 14
feats 151

5
trainer.gradient_clip_val=10000
trainer.gradient_clip_val=100

6
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__1 model.class_name=src.models.ventilator_model__1.VentilatorNet model.params.input_dim=149 model.params.init_style=3 model.params.nhead=149 model.params.transformer_num_layers=1 model.params.use_transformer_encoder=True trainer.gpus=1 trainer.max_epochs=4 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=8 datamodule.fold_n=0

7
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__1 model.class_name=src.models.ventilator_model__1.VentilatorNet model.params.input_dim=149 model.params.init_style=3 model.params.nhead=149 model.params.transformer_num_layers=1 model.params.use_transformer=True trainer.gpus=1 trainer.max_epochs=4 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=8 datamodule.fold_n=0

8
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__1 model.class_name=src.models.ventilator_model__1.VentilatorNet model.params.input_dim=149 model.params.init_style=3 model.params.nhead=149 model.params.transformer_num_layers=1 model.params.use_transformer=True model.params.use_transformer_encoder=True  trainer.gpus=1 trainer.max_epochs=4 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=8 datamodule.fold_n=0

9
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__1 model.class_name=src.models.ventilator_model__1.VentilatorNet model.params.input_dim=149 model.params.init_style=3 model.params.nhead=8 model.params.transformer_num_layers=4 model.params.use_transformer_encoder=True model.params.use_mlp=True trainer.gpus=1 trainer.max_epochs=4 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=8 datamodule.fold_n=0

init style: 3, 5, 6

10
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__2 model.class_name=src.models.ventilator_model__2.VentilatorNet model.params.input_dim=149 model.params.init_style=3 model.params.lstm_layers=1 model.params.lstm_dim=256 model.params.num_layers=4 trainer.gpus=1 trainer.max_epochs=4 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=8 datamodule.fold_n=0


```shell
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=149 model.params.init_style=3 model.params.lstm_layers=6 model.params.num_layers=1 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=8 datamodule.n_folds=20 datamodule.fold_n=0
```

11
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__3 model.class_name=src.models.ventilator_model__3.VentilatorNet model.params.input_dim=149 model.params.init_style=3 model.params.lstm_layers=1 model.params.num_layers=4 model.params.lstm_dim=256 model.params.num_layers=4 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True training.lightning_module_name=src.lightning_classes.lightning_ventilator__0.VentilatorRegression loss=smoothl1loss training.loss_calc_style=2 metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=8 datamodule.n_folds=20 datamodule.fold_n=0
and mae loss

12
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__3 model.class_name=src.models.ventilator_model__3.VentilatorNet model.params.input_dim=149 model.params.init_style=3 model.params.lstm_layers=1 model.params.num_layers=6 model.params.lstm_dim=1024 model.params.num_layers=4 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True training.lightning_module_name=src.lightning_classes.lightning_ventilator__0.VentilatorRegression loss=mae training.loss_calc_style=1 metric=metric_manager1 optimizer=adamw training.lr=0.005 optimizer.params.weight_decay=0.001 scheduler=cosinewithwarmup datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=8 datamodule.n_folds=20 datamodule.fold_n=0

==
training.pp_for_loss=True

===
new

