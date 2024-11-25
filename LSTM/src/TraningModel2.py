from datetime import datetime
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def remove_oulier_from_data():
  data = './data/processed_dataset_new.csv'
  subway_df = pd.read_csv(data)

  # clean data
  # Remove outliers
  station_ID_list = pd.unique(subway_df["station_complex_id"])
  from scipy import stats
  subway_df_outliers = pd.DataFrame()
  for i, station_ID in enumerate(station_ID_list):
    subway_df_temp = subway_df[subway_df["station_complex_id"] == station_ID].copy()
    
    # Calculate Z-score
    subway_df_temp["z_score"] = stats.zscore(subway_df_temp["ridership"])
    # Remove outliers (Z-score method)
    subway_df_temp_outliers = subway_df_temp[np.abs(subway_df_temp['z_score']) > 3]
    if len(subway_df_temp_outliers) > 0:
      subway_df_outliers = pd.concat([subway_df_outliers, subway_df_temp_outliers])
  subway_df_outliers

  subway_df_cleaned = subway_df.drop(subway_df_outliers.index)
  subway_df_cleaned
  print(subway_df_cleaned)
  subway_df_cleaned.to_csv("./data/subway_df_cleaned.csv", index=False)

# time feature enginering
def time_feature_enginering():
  subway_df_cleaned = pd.read_csv('./data/subway_df_cleaned.csv')
  timestamp_s = list(subway_df_cleaned.index.map(subway_df_cleaned['transit_timestamp']))
  day_sin_list = []
  day_cos_list = []
  hour_list = []
  dayofweek_list = []
  day_second = 24 * 60 * 60
  transit_timestamp_new = []
  for i in range(len(timestamp_s)):
    timeArrayTemp = time.strptime(timestamp_s[i], '%Y-%m-%d %H:%M:%S')
    transit_timestamp_new.append(datetime.strptime(timestamp_s[i], "%Y-%m-%d %H:%M:%S"))
    dayofweek_list.append(timeArrayTemp.tm_wday)
    hour_list.append(timeArrayTemp.tm_hour)
    timestamp_s[i] = int(time.mktime(timeArrayTemp))
    day_sin_list.append(np.sin(timestamp_s[i] * (2 * np.pi / day_second)))
    day_cos_list.append(np.cos(timestamp_s[i] * (2 * np.pi / day_second)))

  subway_df_cleaned['transit_timestamp'] = transit_timestamp_new
  subway_df_cleaned['dayofweek'] = dayofweek_list
  subway_df_cleaned['day_hour'] = hour_list
  subway_df_cleaned['Day sin'] = day_sin_list
  subway_df_cleaned['Day cos'] = day_cos_list

  print(len(pd.unique(subway_df_cleaned['Georeference'])), len(pd.unique(subway_df_cleaned['station_complex_id'])))

  subway_df_cleaned.drop(['transit_mode', 'station_complex',
                          'borough', 'Georeference'], axis='columns', inplace=True)
  
  return subway_df_cleaned

# Creates time-series data using subsequences
def create_dataset(X, y, time_steps):
  Xs, ys = [], []

  # Pivot X so there is one timestep for all combined station/variable combos
  X = X.pivot_table(index='transit_timestamp', columns='station_complex_id', values=X.columns[2:], fill_value=0)
  X.columns = ['%s%s' % (b, '|%s' % a if b else '') for a, b in X.columns]
  X.reset_index(inplace=True)
  # Pivot y so there is one timestep for all combined stations
  y = y.pivot_table(index='transit_timestamp', columns='station_complex_id', values=['ridership'], fill_value=0)
  y.columns = ['%s%s' % (b, '|%s' % a if b else '') for a, b in y.columns]
  y.reset_index(inplace=True)

  # print(X)
  # print(y)

  # Determine total number of hours
  min_datetime = X.transit_timestamp.min()
  max_datetime = X.transit_timestamp.max()
  diff = max_datetime - min_datetime
  total_hours = int(((diff.days)*24) + (diff.seconds / 3600))

  # print(X)
  # print(total_hours)
  X_temp = X.copy()
  y_temp = y.copy()
  X_temp.drop(columns=['transit_timestamp'], inplace=True)
  y_temp.drop(columns=['transit_timestamp'], inplace=True)

  i = 0
  for i in range(total_hours - time_steps):
    start_datetime = min_datetime + pd.Timedelta(hours=i)
    end_datetime = start_datetime + pd.Timedelta(hours=time_steps)
    # Get all X values >= start_datetime and < end_datetime
    indices = X.index[(X['transit_timestamp'] >= start_datetime) & (X['transit_timestamp'] < end_datetime)]
    vX = X_temp.loc[indices]

    Xs.append(vX.values)
    
    # Append sequence to ys
    # vy = y.loc[(y['transit_timestamp'] == end_datetime)]
    vy = y_temp.loc[indices]

    ys.append(vy.values[0])

    i += 1
        
  return np.asarray(Xs), np.asarray(ys)


def model_training_all_station():
  subway_df_cleaned = time_feature_enginering()
  # print(subway_df_cleaned.info())
  # print(subway_df_cleaned.describe().T)

  print('Full dataset:\t', subway_df_cleaned.shape[0])

  # Normalize the data
  from sklearn.preprocessing import MinMaxScaler
  subway_df_norm = subway_df_cleaned.copy()
  # Columns to exclude from scaling
  columns_to_exclude = ['transit_timestamp', 'station_complex_id', 'ridership']  # Replace with actual column names
  # Get the list of columns to scale
  columns_to_scale = [col for col in subway_df_norm.columns if col not in columns_to_exclude]
  scaler = MinMaxScaler()
  subway_df_norm[columns_to_scale] = scaler.fit_transform(subway_df_norm[columns_to_scale])
  # print(subway_df_norm)

  # split Data
  y = subway_df_norm[['ridership', 'transit_timestamp','station_complex_id']]
  X = subway_df_norm.copy()
  # print(y)
  X.drop(['ridership'], axis='columns', inplace=True)
  X, y = create_dataset(X, y, 72)
  # print(X)
  # print(y)

  LSTM_training(X, y)

def LSTM_training (X, y):
  # split the data into k folds using TimSeriesSplit
  from sklearn.model_selection import TimeSeriesSplit
  k = 5
  tcsv = TimeSeriesSplit(n_splits = k)

  for (train_index, test_index) in tcsv.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # print(X_train)
    # print(y_train)
    # break

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    # Create LSTM
    model = Sequential()
    number_of_stimeseries = X_train.shape[1]
    number_of_features = X_train.shape[2]
    model.add(LSTM(units=50, activation='sigmoid', input_shape=(number_of_stimeseries, number_of_features), return_sequences=True))
    model.add(LSTM(units=50))
    # model.add(LSTM(units=50, return_sequences=False))
    # model.add(Dense(units=1))
    # Add dropout layer to penalize more complex models
    model.add(Dropout(rate=0.2))
    # Output layer with 428 neurons as we are predicting 428 stations
    model.add(Dense(488))
    # model.compile(loss=root_mean_squared_error, optimizer=optimizer)
    # Compile LSTM
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train Model
    model.fit(X_train, y_train, epochs=100)

    # Evaluate the model on the test set
    from sklearn.metrics  import mean_squared_error, mean_absolute_error, r2_score
    y_pred = model.predict(X_test)
    print(y_pred)
    # break
    y_pred_reshape = y_pred.reshape(-1,1)
    y_pred_reshape[y_pred_reshape < 0] = 0
    y_test_reshape = y_test.reshape(-1,1)
    print(y_pred_reshape)
    print(y_test_reshape)
    mse = mean_squared_error(y_test_reshape, y_pred_reshape)
    mae = mean_absolute_error(y_test_reshape, y_pred_reshape)
    r2 = r2_score(y_test_reshape, y_pred_reshape)

    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'R²: {r2}')

    # save model and its architecture 
    model.save('model.h5')

# remove_oulier_from_data()
model_training_all_station()