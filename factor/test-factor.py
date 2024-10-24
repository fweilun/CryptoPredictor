from abc import ABC
import numpy as np
import pandas as pd
import importlib, json
from factor.factors import Factor
from factor.signal.Cross import Cross
import matplotlib.pyplot as plt
import os, shutil
from data.maker import maker


# def maker(X, delay=100, hours:int=24):
#     train = []
#     test = []
#     X["return"] = X["close"].pct_change(hours//4).shift(-hours//4)
#     X = X.dropna()
#     y = X["return"]
#     Colums = ["close", "volume"]
#     for i in range(len(X) - delay):
#         start, end = i, i + delay
#         predict = i + delay
#         train.append(X[Colums][start:end])
#         test.append(y[predict])
    
#     return train, test

# def maker_group(X, delay=100, hours:int=24):
#     train = []
#     test = []
#     X["return"] = X["close"].pct_change(hours//4).shift(-hours//4)
#     for file in os.listdir("data"):
#         path = os.path.join(data_file_path, file)
#         X[file + "_close"] = pd.read_csv(path)["close"]
    
#     X = X.dropna()
#     y = X["return"]
    
#     for i in range(len(X) - delay):
#         start, end = i, i + delay
#         predict = i + delay
#         train.append(X[start:end])
#         test.append(y[predict])
    
#     return train, test, pd.to_datetime(X['timestamp'])[:len(X) - delay]

TARGET = "BTC"
DELAY = 1000
HOURS = 24
SPLIT_RATE = 0.8


if __name__ == "__main__":
    signal_name = input("Input signal name: ")
    base_factor:Factor = None
    module = importlib.import_module(f'factor.signal.{signal_name}')
    BaseClass = getattr(module, signal_name)
    f = open(f"factor/cfg/{signal_name}.json", 'r')
    jfile = json.load(f)
    
    data_path = "data/storage/BTCUSDT.csv"
    df = pd.read_csv(data_path)
    train, test = maker(target=TARGET, delay=DELAY, hours=HOURS)
    
    print(f"start time: {df['timestamp'].iloc[0]}")
    print(f"end time: {df['timestamp'].iloc[-1]}")
    print(f"number of cases: {len(train)}")
    print(f"predict interval: {HOURS}hrs")
    
    assert("signal_list" in jfile), "wrong config.."
    for config in jfile["signal_list"]:
        assert("argument" in config), "wrong config.."
        arg = config["argument"]
        base_factor = BaseClass(**arg)
        
        res = []
        for (x, y) in zip(train, test):
            res.append(base_factor.Gen(x))

        print(f"{str(base_factor):<20}: {np.corrcoef(res, test)[0][1]:0.3f}")
        
        directory_path = f"factor/result/{signal_name}/{str(base_factor).replace(' ','_')}/"
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        os.makedirs(directory_path)
        
        plt.figure(figsize=(16,5))
        plt.plot(res)
        save_path = directory_path+"value.png"
        plt.savefig(save_path)