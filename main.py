import pandas as pd
from model import train_and_evaluate_model, train_price_model_usd
from preprocessing import preprocess_data

if __name__ == '__main__':
    df = pd.read_csv("full_data.csv")

    clean_df = preprocess_data(df)

    clean_df.to_csv('clean_data.csv', index=False)
    print("\n Data preprocessed and saved as 'clean_data.csv'")

    print("\nInitiating Electricity Price Forecasting Model")
    price_model, price_predictions = train_price_model_usd(clean_df)

    print("\n Initiating Energy Consumption Forecasting Model")
    cons_model, cons_predictions = train_and_evaluate_model(clean_df)

    print("\n Whole Machine Learning pipeline executed successfully!")