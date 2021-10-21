Current best version:
```shell
ventilator_kaggle_models/train.py callbacks.early_stopping.params.patience=50 general.log_code=False logging=wandb model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=108 model.params.init_style=3 model.params.lstm_layers=6 model.params.num_layers=1 trainer.gpus=1 trainer.max_epochs=1000 trainer.gradient_clip_val=1000 loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=1024 datamodule.path=/workspace/data/ventilator_pressure_prediction datamodule=ventilator_datamodule_0 datamodule.make_features_style=3 datamodule.n_folds=20 datamodule.fold_n=0
```
REPLACE THE datamodule.path with your path




```shell
python train.py datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset1 model=ventilator_model_1

python train.py datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset1 model=ventilator_model_2

python train.py datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset2 model=ventilator_model_3

python train.py datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset3 model=ventilator_model_4

python train.py datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset7 model=ventilator_model_6 model.params.input_dim=30 datamodule.batch_size=64

python train.py datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset7 model=ventilator_model_6 model.params.input_dim=30 model.params.norm=True datamodule.batch_size=64

python train.py datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset7 model=ventilator_model_7 model.params.input_dim=30 datamodule.batch_size=64

datamodule.split=GroupShuffleSplit

python train.py datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset7 model=ventilator_model_7 model.params.input_dim=30 datamodule.batch_size=64 datamodule.split=GroupShuffleSplit datamodule.normalize=True
==================================================================
python train.py datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True

python train.py datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=1000 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.logit_dim=128

python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=False model.params.nhead=5

=====
python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=False model.params.nhead=5 trainer.gpus=0 model.params.use_only_transformer_encoder=True


python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=True model.params.nhead=5 trainer.gpus=0 model.params.use_transformer=False

python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=True model.params.nhead=8 trainer.gpus=0 model.params.use_only_transformer_encoder=False model.params.use_transformer=True model.params.use_lstm=False

python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=True model.params.nhead=8 trainer.gpus=0 model.params.use_only_transformer_encoder=False model.params.use_transformer=False model.params.use_lstm=False

python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=False model.params.nhead=5 trainer.gpus=0 model.params.use_only_transformer_encoder=False model.params.use_transformer=True model.params.use_lstm=False

python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=False model.params.nhead=5 trainer.gpus=0 model.params.use_only_transformer_encoder=False model.params.use_transformer=False model.params.use_lstm=False

python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=False model.params.nhead=5 trainer.gpus=0 model.params.use_only_transformer_encoder=True model.params.use_transformer=True model.params.use_lstm=False

python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset9 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=True model.params.nhead=8 trainer.gpus=0 model.params.use_only_transformer_encoder=True model.params.use_transformer=True model.params.use_lstm=True

# and now previous dataset
python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset8 model=ventilator_model_8 general.log_code=False model.params.input_dim=35 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=True model.params.nhead=8 trainer.gpus=0 model.params.use_only_transformer_encoder=True model.params.use_transformer=True model.params.use_lstm=True

And now SELU
python train.py datamodule.num_workers=0 datamodule.batch_size=2 trainer.max_epochs=1 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset.VentilatorDataset8 model=ventilator_model_8 general.log_code=False model.params.input_dim=35 datamodule.fold_n=0 scheduler=exponential training.debug=True model.params.use_mlp=True model.params.nhead=8 trainer.gpus=0 model.params.use_only_transformer_encoder=True model.params.use_transformer=True model.params.use_lstm=True model.params.activation=torch.nn.SELU

torch.nn.LeakyReLU
torch.nn.GELU


```

Ideas:
https://www.kaggle.com/snnclsr/a-dummy-approach-to-improve-your-score-postprocess/notebook
Since there are discrete number of pressure values (950), we are rounding the predictions to the nearest neighbors of that target values (pressures).
```python
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)

def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        # If the predicted value is bigger than the highest pressure in the train dataset,
        # return the max value.
        return sorted_pressures[-1]
    elif insert_idx == 0:
        # Same control but for the lower bound.
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return lower_val if abs(lower_val - prediction) < abs(upper_val - prediction) else upper_val
```

```shell
python train.py datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=10 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset10 model=ventilator_model_9 general.log_code=False model.params.input_dim=24 model.params.dense_dim=24 model.params.lstm_dim=400 model.params.logit_dim=50 datamodule.fold_n=0 scheduler=step training.debug=False trainer.gpus=1

python train.py datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=10 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset11 model=ventilator_model_9 general.log_code=False model.params.input_dim=35 model.params.dense_dim=35 model.params.lstm_dim=400 model.params.logit_dim=50 datamodule.fold_n=0 scheduler=step training.debug=False trainer.gpus=1

python train.py datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=10 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset12 model=ventilator_model_9 general.log_code=False model.params.input_dim=30 model.params.dense_dim=30 model.params.lstm_dim=400 model.params.logit_dim=50 datamodule.fold_n=0 scheduler=step training.debug=False trainer.gpus=1


python train.py datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=10 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset13 model=ventilator_model_9 general.log_code=False model.params.input_dim=58 model.params.dense_dim=58 model.params.lstm_dim=400 model.params.logit_dim=50 datamodule.fold_n=0 scheduler=step training.debug=False trainer.gpus=1

#! normalize
python train.py datamodule.num_workers=0 datamodule.batch_size=256 trainer.max_epochs=2 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset13 model=ventilator_model_9 general.log_code=False model.params.input_dim=55 model.params.dense_dim=55 model.params.lstm_dim=400 model.params.logit_dim=50 datamodule.fold_n=0 scheduler=step training.debug=False trainer.gpus=1 datamodule.normalize=True

# all datasets with this model. NHEAD
python train.py datamodule.num_workers=0 datamodule.batch_size=32 trainer.max_epochs=2 callbacks.early_stopping.params.patience=50 datamodule.class_name=src.datasets.ventilator_dataset1.VentilatorDataset13 model=ventilator_model_8 general.log_code=False model.params.input_dim=50 model.params.use_mlp=False model.params.nhead=2 datamodule.fold_n=0 scheduler=step training.debug=False trainer.gpus=1 datamodule.normalize=True

plateau?
simple mae?
lstm bias false

lstm dropout

small decay 0.9999
adamw
initialize

```