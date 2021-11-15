old:
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__0 model.class_name=src.models.ventilator_model__0.VentilatorNet model.params.input_dim=149 model.params.init_style=3 model.params.lstm_layers=6 model.params.num_layers=1 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adam scheduler=plateau scheduler.params.patience=10 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0 datamodule.make_features_style=8 datamodule.n_folds=20 datamodule.fold_n=0
===
python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__0_post model.class_name=src.models.ventilator_model__0_post.VentilatorNet model.params.input_dim=73 model.params.init_style=3 model.params.lstm_layers=4 model.params.num_layers=1 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True loss=ventilator metric=metric_manager1 optimizer=adamw scheduler=plateau scheduler.params.patience=30 scheduler.params.factor=0.5 datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0_post datamodule.make_features_style=20 datamodule.n_folds=20 datamodule.fold_n=0

python train.py callbacks.early_stopping.params.patience=50 general.log_code=False model=ventilator_model__0_post model.class_name=src.models.ventilator_model__0_post.VentilatorNet model.params.input_dim=73 model.params.init_style=3 model.params.lstm_layers=4 model.params.num_layers=1 trainer.gpus=1 trainer.max_epochs=3 trainer.gradient_clip_val=1000 training.debug=True training.lr=3e-4 loss=ventilator metric=metric_manager1 optimizer=adamw scheduler=cosinewarm datamodule.num_workers=0 datamodule.batch_size=32 datamodule=ventilator_datamodule_0_post datamodule.make_features_style=20 datamodule.n_folds=20 datamodule.fold_n=0


batch_size 128

batch_size 32