import pytorch_forecasting as ptf
import pandas as pd

df = pd.read_csv("mars_temp.csv", delimiter=';')
df["time"] = df["time"].div(60).astype(int)
df["id"] = 'temp'
df["target"] = df["temp"].astype(float)
df.head(), len(df)

training = ptf.TimeSeriesDataSet(
   df,
   time_idx="time",  
   target="target",
   group_ids=["id"],
   allow_missing_timesteps=True,      
   min_encoder_length=60,  
   max_encoder_length=60,
   min_prediction_length=30,   
   max_prediction_length=30   
)

train_dataloader = training.to_dataloader(batch_size=128)
validation = ptf.TimeSeriesDataSet.from_dataset(training, df, ...)
val_dataloader = validation.to_dataloader(batch_size=128)

model = ptf.DeepAR.from_dataset(
   training,
   hidden_size=30,
   rnn_layers=2,
   learning_rate=0.01
)