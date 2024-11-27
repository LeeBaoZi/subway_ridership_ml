import pandas as pd

data = './data/MTA_Subway_Hourly_Ridership.csv'
subway_df = pd.read_csv(data)

subway_df['transit_timestamp'] = pd.to_datetime(subway_df['transit_timestamp'], format="%m/%d/%Y %I:%M:%S %p")
subway_df['station_complex_id'] = subway_df['station_complex_id'].astype(str)

processed_df = (
    subway_df.groupby(['transit_timestamp', 'transit_mode', 'station_complex_id', 'station_complex', 'borough', 'latitude', 'longitude', 'Georeference'])
    .agg({'ridership': 'sum'})
    .reset_index()
)

processed_df['weekend'] = processed_df['transit_timestamp'].dt.dayofweek.apply(lambda x: 1 if x in [5, 6] else 0)

print(processed_df)
processed_df.to_csv("./data/processed_dataset_new.csv", index=False)

def process_data():
    return processed_df