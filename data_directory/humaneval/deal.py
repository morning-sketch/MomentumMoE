import pandas as pd
import math

train_df = pd.read_parquet('train.parquet',columns=['solution'])
num_rows = len(train_df)
train_size = math.ceil(num_rows * 0.2)
train_df = train_df.iloc[:train_size]
train_df.to_parquet('train.parquet', engine='pyarrow')
train_df.to_parquet('validation.parquet', engine='pyarrow')


test_df = pd.read_parquet('test.parquet',columns=['solution'])
test_df=test_df.iloc[train_size:]
test_df.to_parquet('test.parquet', engine='pyarrow')
print(test_df.head())

