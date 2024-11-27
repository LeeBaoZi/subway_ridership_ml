import numpy as np


def preprocess_data(df):
    # fix mixed type warning
    df.loc[df['station_complex'] == "RI Tramway (Manhattan)", 'station_complex_id'] = 1000
    df.loc[df['station_complex'] == "RI Tramway (Roosevelt)", 'station_complex_id'] = 2000

    df['station_complex_id'] = df['station_complex_id'].astype(int)

    # drop unnecessary columns
    columns_to_drop = ['transit_mode', 'station_complex', 'borough', 'longitude', 'latitude', 'Georeference']
    df = df.drop(columns=columns_to_drop)

    df = cap_outliers(df, "ridership")

    return df

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound,
                            np.where(df[column] > upper_bound, upper_bound, df[column]))
    return df

