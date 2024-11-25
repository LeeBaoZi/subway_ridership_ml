from flask import Flask,jsonify, request
from flask_restful import Resource, Api
from flask_cors import CORS
import pandas as pd


app = Flask(__name__)
CORS(app)
api = Api(app)

subway_df = pd.read_csv('./data/subway_df_cleaned.csv')

class Ridership(Resource):
  def get(self):
    timeTemp = request.args.get('timeTemp')
    print('Time', timeTemp)
    select_data_by_time = subway_df[subway_df["transit_timestamp"] == timeTemp].copy()

    # Check if data is found
    if select_data_by_time.empty:
      return {"message": "No data found for the provided timestamp"}, 404
    
    # Convert to JSON
    result = select_data_by_time.to_dict(orient="records")
    return jsonify(result)

api.add_resource(Ridership, '/getRidershipByTime')
