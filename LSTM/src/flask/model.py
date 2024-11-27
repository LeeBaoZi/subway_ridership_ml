from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model('./ml_moswl/model.h5')
scaler = joblib.load('./ml_moswl/scaler.pkl')

X_test_scaled = scaler.transform(X_test)

standardized_predictions = model.predict(X_test_scaled)

original_predictions = scaler.inverse_transform(standardized_predictions)