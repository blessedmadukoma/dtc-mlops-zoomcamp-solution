import pickle
import pandas as pd

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# year = 2023
# month = 3

year = 2023
# # April
# month = 4

# May
month = 5

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-0{month}.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


"""Question 1:"""
print(f"The standard deviation of the predicted duration: {y_pred.std()}")


"""Question 2:"""

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df['predictions'] = y_pred


df_result = df[['ride_id', 'predictions']]


# output_file = "output.parquet"

# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )

"""Question 3:"""
# jupyter nbconvert --to script starter.ipynb

"""Question 4:"""
# sha256:0650e730afb87402baa88afbf31c07b84c98272622aaba002559b614600ca691

"""Question 5:"""
print(f"The mean predicted duration: {y_pred.mean():.2f}")