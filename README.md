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