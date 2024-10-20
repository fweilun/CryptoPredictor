from abc import ABC
import numpy as np
import pandas as pd
import importlib, json
from factor.factors import Factor
from factor.signal.Cross import Cross

'''
1. load data
2. load program
3. load result
'''

def maker(X, delay=100):
    train = []
    test = []
    X["return"] = X["close"].pct_change().shift(-1)
    X = X.dropna()
    y = X["return"]
    Colums = ["close", "volume"]
    for i in range(len(X) - delay):
        start, end = i, i + delay
        predict = i + delay
        train.append(X[Colums][start:end])
        test.append(y[predict])
    
    return train, test



if __name__ == "__main__":
    signal_name = input("Input signal name: ")
    base_factor:Factor = None
    module = importlib.import_module(f'factor.signal.{signal_name}')
    BaseClass = getattr(module, signal_name)
    f = open(f"factor/cfg/{signal_name}.json", 'r')
    jfile = json.load(f)
    
    data_path = "factor/data/base-BTCUSDT.csv"
    df = pd.read_csv(data_path)
    train, test = maker(df)
    
    assert("signal_list" in jfile), "wrong config.."
    for config in jfile["signal_list"]:
        assert("argument" in config), "wrong config.."
        
        arg = config["argument"]
        base_factor = BaseClass(**arg)
        
        res = []
        for (x, y) in zip(train, test):
            res.append(base_factor.Gen(x))
            
        print(f"{str(base_factor):<13}: {np.corrcoef(res, test)[0][1]:0.3f}")
