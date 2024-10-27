import pandas as pd
import os
import yfinance as yf
import requests

MODULE_DIR = os.path.dirname(__file__)
STORAGE_DIR = os.path.join(MODULE_DIR, "storage")
BASE_STORAGE_DIR = os.path.join(STORAGE_DIR, "base")
FUNDING_RATE_DIR = os.path.join(STORAGE_DIR, "funding_rate")



class Loader:
    start_date = "2022-11-01"
    end_date = "2024-10-01"    
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
        
        target_path = os.path.join(BASE_STORAGE_DIR, f"{target}.csv")
        
        # init dataframe
        X = pd.read_csv(target_path, usecols=["timestamp","close","volume"]).set_index("timestamp")
        X.index = pd.to_datetime(X.index)
        
        for file in os.listdir(BASE_STORAGE_DIR):
            path = os.path.join(BASE_STORAGE_DIR, file)
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
        
        return train, test, index
    
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
        train = [X[i:i + delay] for i in range(len(X) - delay)]
        test = [y[i + delay] for i in range(len(X) - delay)]
        
        return train, test, index

    @classmethod
    def load_price(cls, symbol:str="BTCUSDT"):
        file_path = os.path.join(BASE_STORAGE_DIR, f"{symbol}.csv")
        coin_data = pd.read_csv(file_path)
        coin_data = coin_data.set_index("timestamp")
        coin_data.index = pd.to_datetime(coin_data.index)
        return coin_data
    
    @classmethod
    def load_funding_rate(cls, symbol:str="BTCUSDT"):
        file_path = os.path.join(FUNDING_RATE_DIR, f"{symbol}.csv")
        fr_data = pd.read_csv(file_path)
        fr_data = fr_data.set_index("timestamp")
        fr_data.index = pd.to_datetime(fr_data.index)
        return fr_data
    
    @classmethod
    def load_stock_data(cls, ticker:str="SPY"):
        return yf.download(ticker,start=cls.start_date,end=cls.end_date, interval="1h") 