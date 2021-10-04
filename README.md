# tempest

[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/Erlemar/pytorch_tempest/?ref=repository-badge)

This repository has my pipeline for training neural nets.

Main frameworks used:

* [hydra](https://github.com/facebookresearch/hydra)
* [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)

The main ideas of the pipeline:

* all parameters and modules are defined in configs;
* prepare configs beforehand for different optimizers/schedulers and so on, so it is easy to switch between them;
* have templates for different deep learning tasks. Currently, image classification and named entity recognition are supported;

Examples of running the pipeline:
This will run training on MNIST (data will be downloaded):
```shell
>>> python train.py --config-name mnist_config model.encoder.params.to_one_channel=True
```

The default run:

```shell
>>> python train.py
```

The default version of the pipeline is run on imagenette dataset. To do it, download the data from this repository:
https://github.com/fastai/imagenette
unzip it and define the path to it in conf/datamodule/image_classification.yaml path

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