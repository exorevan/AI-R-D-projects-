import warnings

warnings.filterwarnings('ignore')
import random

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner import Tuner
from matplotlib.gridspec import GridSpec
from pytorch_forecasting import (Baseline, DeepAR, NBeats,
                                 TemporalFusionTransformer, TimeSeriesDataSet)
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, MASE, RMSE, SMAPE, MultiLoss
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from torchmetrics.functional import r2_score
#utils
from tqdm.notebook import tqdm


def fix_random_seeds(seed=42):
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed for CPU and CUDA devices
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(predictions, answers):
    rmse = RMSE()(predictions, answers).item()
    smape = SMAPE()(predictions, answers).item()
    mae = MAE()(predictions, answers).item()
    r2 = r2_score(predictions, answers).item()

    print(f"Evaluation Metrics:")
    print(f"RMSE: {rmse:.3f}")
    print(f"SMAPE: {smape:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R-squared: {r2:.3f}")

if __name__ == '__main__':
    fix_random_seeds()


    df = pd.read_csv('Frameworks/ETNA/full_data_sum.csv')


    df['date'] = pd.to_datetime(df['date'])
    df['dt'] = pd.to_datetime(df.date) + df.hour.astype('timedelta64[h]')

    # Для создания датасета нужно перейти от действительных значений времени к временным шагам
    dates_transformer = LabelEncoder()
    df['time_idx'] = dates_transformer.fit_transform(df.dt)
    df['time_idx'] += 1

    # Категориальные переменные должны быть в строковом типе
    df['square_id'] = df['square_id'].astype(str)


    dates = sorted(df.date.unique())
    train_dates = dates[:49]
    val_dates = dates[49:-1]
    test_dates = dates[-1:]

    train = df[df.date.isin(train_dates)]
    val = df[df.date.isin(val_dates)]
    test = df[df.date.isin(test_dates)]

    train_cutoff = train['time_idx'].max()
    validation_cutoff = test['time_idx'].max() - 25


    to_select = ['6974', '6554', '8966', '7445', '9361', '5557', '8847', '6875', '3862', '3527', '6274', '7155', '8087', '6071', '5445', '6745', '9986', '4349', '4972', '3234', '8045', '8380', '9877', '4639', '7987', '4657', '9398', '6077', '4478', '1684', '3958', '3258', '3554', '6754', '8176', '5473', '6846', '8502', '4144', '2247', '4664', '3957', '4973', '6472', '8046', '8242', '2145', '9778', '7552', '9869', '8958', '7055', '6836', '5451', '4040', '3532', '8891', '8757', '7167', '9150', '6467', '9474', '7154', '4345', '8558', '3928', '5443', '9785', '7424', '6664', '6546', '4830', '2654', '5245', '4128', '5368', '5447', '48', '7161', '8152', '4446', '4341', '4667', '5646', '7271', '6766', '6250', '6662', '7173', '7769', '6050', '4061', '8273', '4367', '3755', '3334', '8561', '4578', '3462', '8311', '7085', '3966', '3551', '6473', '8464', '8283', '8247', '7672', '8148', '4464', '7786', '4563', '3945', '6252', '4874', '6466', '5047', '3929', '8673', '6465', '5649', '7562', '3674', '8534', '3558', '7722', '4550', '8056', '3353', '7389', '5344', '4047', '6470', '3967', '6053', '4279', '8332', '6854', '7226', '8854', '6852', '3233', '6054', '7144', '4658', '7980', '6270', '5848', '6585', '7306', '5457', '7468', '6645', '5145', '6055', '5357', '4343', '4029', '3118', '3553', '4058', '6052', '4129', '3836', '5851', '3868', '4362', '6953', '6451', '4443', '3360', '4444', '5544', '4351', '9483', '9684', '6775', '3659', '4669', '7822', '851', '4642', '7341', '8776', '4452', '5847', '4056', '2960', '5957', '3632', '4876', '4350', '6444', '3759', '6447', '7754', '7240', '8286', '6972', '8177', '3863', '8459', '4742', '5636', '3954', '8278', '6297', '6461', '6869', '3871', '7073', '9770', '6695', '7375', '8011', '6647', '7143', '7753', '7540', '6833', '8885', '8934', '4889', '1150', '4158', '4526', '4378', '9598', '8856', '5669', '4354', '3655', '6961', '4748', '1147', '8004', '3763', '8274', '1349', '8562', '6976', '9155', '8751', '5865', '7172', '6648', '4826', '3748', '8791', '4948', '4542', '3285', '4428', '4951', '5437', '7684', '7354', '6355', '5551', '4147', '7163', '7156', '8744', '4057', '7724', '3960', '6459', '4327', '9143', '4777', '5458', '3255', '1555', '4743', '5953', '4066', '7324', '4851', '6154', '4666', '6769', '5450', '7273', '7075', '4678', '8746', '5755', '4453', '8013', '7551', '3432', '5947', '7046', '4871', '3661', '3854', '3767', '8374', '5948', '7556', '4852', '3845', '6663', '4243', '8688', '9351', '6260', '6371', '4027', '4565', '5543', '3962', '3828', '4344', '5372', '7256', '5977', '8049', '6877', '4726', '6572', '8441', '5550', '4379', '6966', '8410', '7305', '4143', '8051', '8833', '5741', '3427', '8085', '7453', '7353', '8435', '6872', '5273', '6756', '4068', '3347', '7852', '5750', '6862', '6046', '5653', '4276', '5753', '7813', '9266', '5146', '8835', '8271', '3671', '6150', '8071', '6671', '3765', '3654', '6152', '7065', '8635', '6153', '3733', '7265', '7064', '5028', '6971', '3970', '4733', '6151', '4228', '6533', '8312', '8756', '3548', '7623', '6372', '4553', '3959', '3673', '4226', '7342', '6755', '6771', '6367', '8047', '6853']

    df = df[df.square_id.isin(to_select)]
    train = train[train.square_id.isin(to_select)]
    val = val[val.square_id.isin(to_select)]
    test = test[test.square_id.isin(to_select)]


    max_encoder_length = 60
    max_prediction_length = 24


    context_length = max_encoder_length
    prediction_length = max_prediction_length

    training = TimeSeriesDataSet(
        train,
        time_idx="time_idx",
        target="internet",
        categorical_encoders={"square_id": NaNLabelEncoder().fit(train.square_id)},
        group_ids=["square_id"],
        static_categoricals=["square_id"], # Deep AR позволяет учитывать статичные
                                        # категориальные признаки, в нашем случае это id станции
        time_varying_unknown_reals=["internet"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
        allow_missing_timesteps=False
    )

    validation = TimeSeriesDataSet.from_dataset(training,
                                                df,
                                                min_prediction_idx=train_cutoff + 1)


    batch_size = 128
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        batch_sampler="synchronized"
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=4, batch_sampler="synchronized"
    )


    random_bs = np.random.choice(df.square_id.unique(), 150)

    train_sample = TimeSeriesDataSet.from_dataset(training, train[train.square_id.isin(random_bs)])
    validation_sample = TimeSeriesDataSet.from_dataset(training, val[val.square_id.isin(random_bs)], min_prediction_idx=train_cutoff + 1)


    train_subset_loader = train_sample.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        batch_sampler="synchronized")

    val_subset_loader = validation_sample.to_dataloader(
        train=False, batch_size=batch_size, num_workers=4, batch_sampler="synchronized"
    )


    pl.seed_everything(42)

    trainer = pl.Trainer(accelerator='cpu', gradient_clip_val=1e-1)
    net = DeepAR.from_dataset(
        training,
        learning_rate=3e-2,
        hidden_size=60,
        rnn_layers=5,
        optimizer="Adam",
    )


    logger = TensorBoardLogger('DeepAR_Fit', name='result_deepAR_model')

    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        min_delta=1e-4,
                                        patience=10,
                                        verbose=False,
                                        mode="min")
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator='cpu',
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        # limit_train_batches=200,
        limit_val_batches=200,
        enable_checkpointing=True,
        logger=logger
    )

    trainer.fit(
        net,
        train_dataloaders=train_subset_loader,
        val_dataloaders=val_subset_loader
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = DeepAR.load_from_checkpoint(best_model_path)

    predictions = best_model.predict(val_subset_loader,
                                 trainer_kwargs=dict(accelerator=DEVICE), )
    
    print(evaluate(predictions, val_answers))

    test_dataset = TimeSeriesDataSet.from_dataset(training,
                                      df,
                                      min_prediction_idx=validation_cutoff + 1,
                                      predict_mode=True)

    test_loader = test_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=4, batch_sampler="synchronized"
    )

    test_answers = torch.cat([y[0] for x, y in iter(test_loader)]).to('cpu')

    raw_predictions = net.predict(test_loader, mode="prediction", return_x=True)
    test_preds = raw_predictions[0].to('cpu')
    x = raw_predictions[1]['encoder_target'].to('cpu')

    print(evaluate(test_preds, test_answers))

    # Номер базовой станции для визуализации
    base_station_num = 60

    # Можно визуализировать отдельные прогнозы
    base_station_num = 60
    plt.plot(np.concatenate([x[base_station_num], test_preds[base_station_num]]), color='red', label='Прогноз')
    plt.plot(np.concatenate([x[base_station_num], test_answers[base_station_num]]), color='blue', label='Реальные значения')


    # Добавляем подписи к осям и легенду
    plt.xlabel('Временные шаги')
    plt.ylabel('Значения')
    plt.legend()
    plt.title(f'Пример прогноза')

    # Отображаем график
    plt.show()
