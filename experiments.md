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