import pandas as pd
import os
import yfinance as yf
import requests
import numpy as np

MODULE_DIR = os.path.dirname(__file__)
STORAGE_DIR = os.path.join(MODULE_DIR, "storage")
SPOT_STORAGE_DIR = os.path.join(STORAGE_DIR, "spot")
FUTURE_STORAGE_DIR = os.path.join(STORAGE_DIR, "future")
FUNDING_RATE_DIR = os.path.join(STORAGE_DIR, "funding-rate")
LONG_SHORT_RATIO_DIR = os.path.join(STORAGE_DIR, "long-short-ratio")
STABLECOIN_DIR = os.path.join(STORAGE_DIR, "stablecoin")



class Loader:
    start_date = "2023-11-01"
    end_date = "2024-11-04"    
    @classmethod
    def make(cls, target="BTCUSDT", delay=100, hours:int=24):
        '''
        maker
        ---------
        Processes financial time series data and returns\n
        input features (train) and target values (test)\n
        for model training.
        
        Parameters
        ---------
        - target (str): The asset symbol to predict (default: "BTC").
        - delay (int): The look-back window size for input data (default: 100).
        - hours (int): The period over which the percentage return is calculated (default: 24 hours).
        
        Returns
        ---------
        - train (DataDrame): Input feature sets of `delay` time steps.
        - test (list): Target values (future returns) for each input set.
        '''
        
        target_path = os.path.join(SPOT_STORAGE_DIR, f"{target}.csv")
        
        # init dataframe
        X = pd.read_csv(target_path, usecols=[
            "timestamp",
            "close",
            "volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", 
            "open",
            "high",
            "low"
            ]).set_index("timestamp")
        
        X.index = pd.to_datetime(X.index)
        
        for file in os.listdir(SPOT_STORAGE_DIR):
            path = os.path.join(SPOT_STORAGE_DIR, file)
            file_df = pd.read_csv(path, usecols=["timestamp","close","volume"]).set_index("timestamp")
            file_df.index = pd.to_datetime(file_df.index)
            file_df.columns = [f"{file[:-4]}_{col}" for col in file_df.columns]
            X = pd.concat([X, file_df], axis=1)
            
        X["return"] = X["close"].pct_change(hours//4).shift(-hours//4)
        X = X.dropna()
        y = X["return"]
        
        index = y.index[delay:]
        train = [X[i:i + delay] for i in range(len(X) - delay)]
        test = [y[i + delay] for i in range(len(X) - delay)]
        
        return train, pd.Series(test, index=index)
    
    
    @classmethod
    def make_not_overlap(cls, target="BTCUSDT", delay=100, hours:int=24, dtype="backtest"):
        '''
        maker
        ---------
        Processes financial time series data and returns\n
        input features (train) and target values (test)\n
        for model training.
        
        Parameters
        ---------
        - target (str): The asset symbol to predict (default: "BTC").
        - delay (int): The look-back window size for input data (default: 100).
        - hours (int): The period over which the percentage return is calculated (default: 24 hours).
        
        Returns
        ---------
        - train (DataDrame): Input feature sets of `delay` time steps.
        - test (list): Target values (future returns) for each input set.
        '''
        
        target_path = os.path.join(SPOT_STORAGE_DIR, f"{target}.csv")
        
        # init dataframe
        X = pd.read_csv(target_path, usecols=[
            "timestamp",
            "close",
            "volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", 
            "open",
            "high",
            "low"
            ]).set_index("timestamp")

        X["funding_rate"] = Loader.load_funding_rate(symbol=target)["fundingRate"]
        X["funding_rate"] = X["funding_rate"].bfill()
        X["funding_rate_ema0.05"] = X["funding_rate"].ewm(alpha=0.05).mean()
        X["funding_rate_ema0.2"] = X["funding_rate"].ewm(alpha=0.2).mean()

        future_data = Loader.load_future_data(symbol=target)
        future_data = future_data.drop_duplicates(keep='first')

        X["future_open"] = future_data["open"]
        X["future_high"] = future_data["high"]
        X["future_low"] = future_data["low"]
        X["future_close"] = future_data["close"]
        X["future_volume"] = future_data["volume"]

        X["total_circulating_usd"] = Loader.load_stablecoin()["totalCirculatingUSD"]

        # # FIXME:
        # X["long_short_ratio"] = Loader.load_long_short_ratio(symbol=target)["longShortRatio"]
        # X["long_short_ratio"] = X["long_short_ratio"].bfill()
        
        X.index = pd.to_datetime(X.index)
        
        for file in os.listdir(SPOT_STORAGE_DIR):
            path = os.path.join(SPOT_STORAGE_DIR, file)
            file_df = pd.read_csv(path, usecols=["timestamp","close","volume"]).set_index("timestamp")
            file_df.index = pd.to_datetime(file_df.index)
            file_df.columns = [f"{file[:-4]}_{col}" for col in file_df.columns]
            X = pd.concat([X, file_df], axis=1)
        
        step = hours//4
        X["return"] = X["close"].pct_change(step).shift(-step)
        X = X.dropna(axis=0)
        y = X["return"]
        if dtype != "backtest":
            return X, y
    
        # X should contains the last value having the same index as y 
        start_index = (len(X) - delay) % step
        end_index = len(X) - delay
        train = [X.iloc[i:i + delay] for i in range(start_index, end_index, step)]
        test = y[delay + start_index - 1: delay + end_index - 1: step]
        
        
        return train, test
    
    
    
    @classmethod
    def make_spy(cls, delay=100, hours:int=24):
        '''
        maker
        ---------
        Processes financial time series data and returns\n
        input features (train) and target values (test)\n
        for model training.
        
        Parameters
        ---------
        - target (str): The asset symbol to predict (default: "BTC").
        - delay (int): The look-back window size for input data (default: 100).
        - hours (int): The period over which the percentage return is calculated (default: 24 hours).
        
        Returns
        ---------
        - train (DataDrame): Input feature sets of `delay` time steps.
        - test (list): Target values (future returns) for each input set.
        '''
        X = cls.load_stock_data()
        X.columns = ["close" if col == "Close" else col for col in X.columns]

        X["return"] = X["close"].pct_change(hours//4).shift(-hours//4)
        X = X.dropna()
        y = X["return"]
        
        index = y.index[delay:]
        train = np.array([X[i:i + delay] for i in range(len(X) - delay)])
        test = np.array([y[i + delay] for i in range(len(X) - delay)])
        
        return train, test, index

    @classmethod
    def load_price(cls, symbol:str="BTCUSDT"):
        file_path = os.path.join(SPOT_STORAGE_DIR, f"{symbol}.csv")
        coin_data = pd.read_csv(file_path)
        coin_data = coin_data.set_index("timestamp")
        coin_data.index = pd.to_datetime(coin_data.index)
        return coin_data

    @classmethod
    def load_future_data(cls, symbol:str="BTCUSDT"):
        file_path = os.path.join(FUTURE_STORAGE_DIR, f"{symbol}.csv")
        future_data = pd.read_csv(file_path)
        future_data = future_data.set_index("timestamp")
        future_data.index = pd.to_datetime(future_data.index)
        return future_data
    
    @classmethod
    def load_funding_rate(cls, symbol:str="BTCUSDT"):
        file_path = os.path.join(FUNDING_RATE_DIR, f"{symbol}.csv")
        fr_data = pd.read_csv(file_path)
        fr_data = fr_data.set_index("fundingTime")
        fr_data.index = pd.to_datetime(fr_data.index)
        fr_data = fr_data.drop_duplicates(keep='first')
        fr_data.index = fr_data.index.round('s')
        return fr_data

    @classmethod
    def load_long_short_ratio(cls, symbol:str="BTCUSDT"):
        file_path = os.path.join(LONG_SHORT_RATIO_DIR, f"{symbol}.csv")
        lsr_data = pd.read_csv(file_path)
        lsr_data = lsr_data.set_index("timestamp")
        lsr_data.index = pd.to_datetime(lsr_data.index)
        lsr_data = lsr_data.drop_duplicates(keep='first')
        lsr_data.index = lsr_data.index.round('s')
        return lsr_data 
    
    @classmethod
    def load_stablecoin(cls, symbol:str="USDT"):
        file_path = os.path.join(STABLECOIN_DIR, "stablecoin.csv")
        sc_data = pd.read_csv(file_path)
        sc_data = sc_data.set_index("timestamp")
        sc_data.index = pd.to_datetime(sc_data.index)
        return sc_data

    @classmethod
    def load_stock_data(cls, ticker:str="SPY"):
        return yf.download(ticker,start=cls.start_date,end=cls.end_date, interval="1h") 
    
    