import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit


TICKER      = "AAPL"
START_DATE  = "2015-01-01"
END_DATE    = "2024-01-01"
WINDOW_SIZE = 30
BATCH_SIZE  = 64
N_SPLITS    = 5 


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.dropna(how="all", inplace=True)          # drop fully-empty rows
    print(f"Downloaded {len(df)} rows for {ticker}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use Close, Volume, and simple technical indicators as model inputs.
    The TARGET is next-day Close price.
    """
    data = pd.DataFrame(index=df.index)

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    data["Close"] = close
    data["Volume"] = volume
    data["Return"] = close.pct_change()           # daily % return
    data["MA7"] = close.rolling(7).mean()      # 7-day moving avg
    data["MA21"] = close.rolling(21).mean()     # 21-day moving avg
    data["Volatility"] = close.rolling(7).std()       # short-term volatility
    data["Momentum"] = close - close.shift(5)       # 5-day price momentum

    # Median imputation for any NaNs produced by rolling calculations
    for col in data.columns:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    return data


def normalize(data: pd.DataFrame):
    """
    MinMaxScaler → each feature scaled to [0, 1].
    We keep the Close scaler separately so we can inverse-transform
    predictions back to real dollar values later.
    """
    feature_scaler = MinMaxScaler()
    close_scaler   = MinMaxScaler()

    scaled_features = feature_scaler.fit_transform(data.values)
    close_scaler.fit(data[["Close"]].values)   # only fit on Close column

    return scaled_features, feature_scaler, close_scaler


def create_sequences(scaled: np.ndarray, steps: int, close_col_idx: int = 0):
    """
    Convert time series → supervised learning pairs.    
    """
    X, y = [], []
    for i in range(steps, len(scaled)):
        X.append(scaled[i - steps : i])          # past `steps` days
        y.append(scaled[i, close_col_idx])        # next day's Close
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_data(X: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS):
    # This mirrors real-world forecasting: you never peek at the future.
    tss = TimeSeriesSplit(n_splits=n_splits) # always trains on older data, tests on newer data.
    splits = list(tss.split(X))
    train_idx, test_idx = splits[-1]             # use the final fold (largest possible training set)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import models
    from torch.utils.data import DataLoader


    TICKER      = "AAPL"
    START_DATE  = "2015-01-01"
    END_DATE    = "2024-01-01"
    WINDOW_SIZE = 30
    BATCH_SIZE  = 64
    N_SPLITS    = 5 

    df = download_data(TICKER, START_DATE, END_DATE)
    data = build_features(df)

    scaled, f_scaler, c_scaler = normalize(data)

    X, y = create_sequences(scaled, WINDOW_SIZE)

    X_train, X_test, y_train, y_test = split_data(X, y)

    train_loader = DataLoader(models.StockDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(models.StockDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False)