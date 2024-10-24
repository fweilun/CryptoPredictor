import pandas as pd
import os



def maker(target="BTC", delay=100, hours:int=24):
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
    - train (list): Input feature sets of `delay` time steps.
    - test (list): Target values (future returns) for each input set.
    '''
    
    module_dir = os.path.dirname(__file__)
    storage_dir = os.path.join(module_dir, "storage")
    target_path = os.path.join(storage_dir, f"{target}USDT.csv")
    
    # init dataframe
    X = pd.DataFrame()
    X["close"] = pd.read_csv(target_path)["close"]
    X["volume"] = pd.read_csv(target_path)["volume"]
    
    for file in os.listdir(storage_dir):
        path = os.path.join(storage_dir, file)
        X[file[:-3] + "_close"] = pd.read_csv(path)["close"]
        X[file[:-3] + "_volume"] = pd.read_csv(path)["volume"]
    
    X["return"] = X["close"].pct_change(hours//4).shift(-hours//4)
    X = X.dropna()
    y = X["return"]
    
    
    train = []
    test = []
    for i in range(len(X) - delay):
        start, end = i, i + delay
        predict = i + delay
        train.append(X[start:end])
        test.append(y[predict])
    
    return train, test

