import pandas as pd
import os


TARGET = "BTC"
DELAY = 100
HOUR = 4


def maker(X, delay=100, hours:int=24):
    train = []
    test = []
    X["return"] = X["close"].pct_change(hours//4).shift(-hours//4)
    X = X.dropna()
    y = X["return"]
    Colums = ["close", "volume"]
    for i in range(len(X) - delay):
        start, end = i, i + delay
        predict = i + delay
        train.append(X[Colums][start:end])
        test.append(y[predict])
    
    return train, test




def maker_group(X, delay=100, hours:int=24):
    train = []
    test = []
    X["return"] = X["close"].pct_change(hours//4).shift(-hours//4)
    for file in os.listdir("data"):
        path = os.path.join(data_file_path, file)
        X[file + "_close"] = pd.read_csv(path)["close"]
    
    X = X.dropna()
    y = X["return"]
    
    for i in range(len(X) - delay):
        start, end = i, i + delay
        predict = i + delay
        train.append(X[start:end])
        test.append(y[predict])
    
    return train, test, pd.to_datetime(X['timestamp'])[:len(X) - delay]

