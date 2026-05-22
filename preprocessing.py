import pandas as pd
import numpy as np

def preprocess_data(df):
    print("datasetin ilk 5 satırı:  ",df.head())
    print("son beş:",df.tail())
    print("describe:",df.describe())
    print("info:",df.info())
    print(f"veri set {df.shape[0]} satır ve {df.shape[1]} sütundan olusuyor")
    print(" eksik deger kontrolu:\n", df.isnull().sum())

    df['datetime'] = pd.to_datetime(df['time'], format='%d:%m:%Y:%H:%M')
    df = df.sort_values('datetime').reset_index(drop=True)

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    kur = {
        2018: 4.83,
        2019: 5.67,
        2020: 7.01,
        2021: 8.89,
        2022: 16.57,
        2023: 23.15
    }
    df['usd_rate'] = df['year'].map(kur)
    df['USD/MWh'] = df['TRY/MWh'] / df['usd_rate']
    df['price_usd_lag_24h'] = df['USD/MWh'].shift(24)
    df['consumption_lag_24h'] = df['consumption_MWh'].shift(24)
    df['consumption_lag_1w'] = df['consumption_MWh'].shift(168)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df_final = df.dropna()
    return df_final



if __name__=='__main__':
        import pandas as pd
        df_deneme=pd.read_csv('full_data.csv')
        print("*************************************")
        print(len(df_deneme))
        df_dene=preprocess_data(df_deneme)
        print(df_dene.info())
        print("*************************************")

        print(len(df_dene))











