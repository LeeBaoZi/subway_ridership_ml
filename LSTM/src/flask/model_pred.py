from tensorflow.keras.models import load_model
import joblib
import pandas as pd

def predict_by_model(X_test_scaled, timeStep):
  # Load model and scaler
  model = None
  if timeStep == 6:
    model = load_model('./ml_model/model.h5')
  elif timeStep == 24:
    model = load_model('./ml_model/model_24.h5')
  scaler = joblib.load('./ml_model/scaler.pkl')

  print(X_test_scaled)
  standardized_predictions = model.predict(X_test_scaled)
  print(standardized_predictions)

  # Get ridership std and mean
  subway_df_norm = pd.read_csv('./data/subway_df_norm.csv')
  mean = scaler.mean_[subway_df_norm.columns.get_loc('ridership') - 2]
  scale = scaler.scale_[subway_df_norm.columns.get_loc('ridership') - 2]
  # reverse from standarlization
  original_predictions = standardized_predictions[0] * scale + mean
  original_predictions_frame = pd.DataFrame({
    'station_complex_id': sorted(pd.unique(subway_df_norm['station_complex_id'])),
    'ridership': [int(value) for value in original_predictions]
  })

  print(original_predictions_frame)

  return original_predictions_frame