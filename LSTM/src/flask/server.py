from flask import Flask,jsonify, request
from flask_restful import Resource, Api
from flask_cors import CORS
import pandas as pd
from model_pred import predict_by_model
import numpy as np

app = Flask(__name__)
CORS(app)
api = Api(app)

subway_df = pd.read_csv('./data/subway_df_cleaned.csv')
X = pd.read_csv('./data/X_24.csv')

class Ridership(Resource):
  def get(self):
    # original data
    timeTemp = request.args.get('timeTemp')
    print('Time', timeTemp)
    select_data_by_time = subway_df[subway_df["transit_timestamp"] == timeTemp].copy()
    # Check if data is found
    if select_data_by_time.empty:
      return {"message": "No data found for the provided timestamp"}, 404

    # predict data
    timeStep = int(request.args.get('timeStep'))
    print('Time Step', timeStep)
    timeTemp = pd.to_datetime(timeTemp)
    startTime = timeTemp - pd.Timedelta(hours=timeStep)
    endTime = timeTemp - pd.Timedelta(hours=1)
    X['transit_timestamp'] = pd.to_datetime(X['transit_timestamp'])
    privious_data_by_time = X.loc[(X['transit_timestamp'] >= startTime) & (X['transit_timestamp'] <= endTime)]
    # Check if data is found
    if privious_data_by_time.empty:
      return {"message": "No data found for the provided timestamp"}, 404
    privious_data_by_time.drop(columns=['transit_timestamp'], inplace=True)
    print(privious_data_by_time)
    test_data = []
    test_data.append(privious_data_by_time.values)
    pred_data = predict_by_model(np.asarray(test_data), timeStep)
    
    # Convert to JSON
    result = {
      "originalData": select_data_by_time.to_dict(orient="records"),
      "predictData": pred_data.to_dict(orient="records")
    }

    return jsonify(result)

api.add_resource(Ridership, '/getRidershipByTime')
