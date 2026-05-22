import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV


def train_and_evaluate_model(df):
    features = ['hour', 'day_of_week', 'month', 'is_weekend',
                'consumption_lag_24h', 'consumption_lag_1w',
                'price_usd_lag_24h',
                 'hour_sin', 'hour_cos']
    target = 'consumption_MWh'

    train = df[df['year'] < 2023]
    test = df[df['year'] == 2023]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    print("\n Tüketim modeli eğitiliyor")
    base_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    base_model.fit(X_train, y_train)

    base_preds = base_model.predict(X_test)
    base_mae = mean_absolute_error(y_test, base_preds)
    base_mape = mean_absolute_percentage_error(y_test, base_preds)
    base_r2 = r2_score(y_test, base_preds)

    print(f"-> Base MAE: {base_mae:.2f} | Base MAPE: %{base_mape * 100:.2f}")

    print("\n Hiperparametre optimizasyonu başlatılıyor")
    tscv = TimeSeriesSplit(n_splits=3)

    param_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9]
    }

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=10,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_

    tuned_preds = tuned_model.predict(X_test)
    tuned_mae = mean_absolute_error(y_test, tuned_preds)
    tuned_mape = mean_absolute_percentage_error(y_test, tuned_preds)
    tuned_r2 = r2_score(y_test, tuned_preds)

    print("\nTüketim modeli sonuçları:")
    print(f"Base MAE : {base_mae:.2f}  | Base MAPE : %{base_mape * 100:.2f}  | Base R² : %{base_r2 * 100:.2f}")
    print(f"Tuned MAE: {tuned_mae:.2f} | Tuned MAPE: %{tuned_mape * 100:.2f} | Tuned R²: %{tuned_r2 * 100:.2f}")

    improvement = ((base_mae - tuned_mae) / base_mae) * 100
    print(f"İyileşme Oranı: %{improvement:.2f}")
    joblib.dump(tuned_model, 'consumption_model.pkl')
    return tuned_model, tuned_preds


def train_price_model_usd(df):

    df = df.copy()
    df['trend_index'] = range(len(df))

    df['price_rolling_24h'] = df['USD/MWh'].rolling(window=24).mean()
    df['price_rolling_1w'] = df['USD/MWh'].rolling(window=168).mean()

    df['price_usd_lag_24h'] = df['USD/MWh'].shift(24)
    df['price_usd_lag_48h'] = df['USD/MWh'].shift(48)

    df.fillna(method='bfill', inplace=True)
    features = ['hour', 'day_of_week', 'month', 'is_weekend',
                'consumption_MWh', 'consumption_lag_24h',
                'hour_sin', 'hour_cos', 'price_usd_lag_24h',
                'trend_index', 'price_rolling_24h', 'price_rolling_1w', 'price_usd_lag_48h']

    target = 'USD/MWh'

    train = df[df['year'] < 2023]
    test = df[df['year'] == 2023]

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    print("\nFiyat modeli eğitiliyor")
    base_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5)
    base_model.fit(X_train, y_train)

    base_preds = base_model.predict(X_test)
    base_mae = mean_absolute_error(y_test,base_preds)
    base_wape = (np.sum(np.abs(y_test - base_preds)) / np.sum(y_test)) * 100
    base_r2 = r2_score(y_test, base_preds)

    print(f"-> USD Bazlı MAE: {base_mae:.4f} USD | Base WAPE: %{base_wape:.2f}")

    print("\n[Hiperparametre optimizasyonu başlatılıyor")
    tscv = TimeSeriesSplit(n_splits=3)

    param_grid = {
        'n_estimators': [500, 800, 1000],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 5, 10]
    }

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=15,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)
    tuned_price_model = search.best_estimator_


    tuned_preds = tuned_price_model.predict(X_test)
    tuned_mae = mean_absolute_error(y_test, tuned_preds)
    tuned_wape = (np.sum(np.abs(y_test - tuned_preds)) / np.sum(y_test)) * 100
    tuned_r2 = r2_score(y_test, tuned_preds)

    print("\nFiyat modeli sonuçları:")
    print(f"Base MAE : {base_mae:.4f}  | Tuned MAE : {tuned_mae:.4f}")
    print(f"Base WAPE: %{base_wape:.2f} | Tuned WAPE: %{tuned_wape :.2f}")
    print(f"Base R²  : %{base_r2 * 100:.2f}  | Tuned R²  : %{tuned_r2 * 100:.2f}")
    improvement = ((base_mae - tuned_mae) / base_mae) * 100
    print(f"İyileşme Oranı: %{improvement:.2f}")


    joblib.dump(tuned_price_model, 'price_model_usd.pkl')
    return tuned_price_model, tuned_preds

if __name__=='__main__':
    import pandas as pd
    clean_df = pd.read_csv('clean_data.csv')
    train_price_model_usd(clean_df)
    train_and_evaluate_model(clean_df)



