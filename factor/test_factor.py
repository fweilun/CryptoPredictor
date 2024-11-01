from abc import ABC
import numpy as np
import pandas as pd
import importlib, json
from factor.factors import Factor
from factor.signal.Cross import Cross
import matplotlib.pyplot as plt
import os, shutil, tqdm
from data.loader import Loader


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




TARGET = "BTCUSDT"
DELAY = 1000
HOURS = 24
SPLIT_RATE = 0.8


class FactorTest:
    
    @classmethod
    def correlation(cls, signal_output, test):
        return np.corrcoef(signal_output, test)[0][1]
    
    @classmethod
    def correlation_stable(cls, signal_output, test):
        n = 5
        x_ = np.array_split(signal_output, n)
        y_ = np.array_split(test, n)
        result = []
        for i in range(n):
            assert(len(x_[i]) == len(y_[i])), "not same length"
            result.append(cls.correlation(x_[i], y_[i]))
        
        result = pd.Series(result)
        positive = (result > 0.02).sum() / len(result)
        negative = (result < -0.02).sum() / len(result)
        neutral = 1 - positive - negative
        if (positive > negative):
            score = 2*positive + neutral - 4*negative
        else:
            score = 2*negative + neutral - 4*positive
        return score
    
    @classmethod
    def correlation_series(cls, signal_output, test):
        n = 5
        x_ = np.array_split(signal_output, n)
        y_ = np.array_split(test, n)
        result = []
        for i in range(n):
            assert(len(x_[i]) == len(y_[i])), "not same length"
            result.append(cls.correlation(x_[i], y_[i]))
        
        return pd.Series(result)
    
    @classmethod
    def accuracy(cls, signal_output:pd.Series, test):
        counter = 0
        not_equal_0 = 0
        n = len(test)
        
        for i in range(n):
            x_ = signal_output.iloc[i]
            y_ = test[i]
            x_ = int(x_ > 0) - int(x_ < 0)
            y_ = int(y_ > 0) - int(y_ < 0)
            counter += (x_ == y_) & (y_ != 0)
            not_equal_0 += (x_ != 0)
        
        if not_equal_0 == 0: 
            return 0
        acc = counter / not_equal_0
        return max(acc, 1-acc)
    
    @classmethod
    def zero_percent(cls, signal_output):
        return (signal_output == 0).sum()/len(signal_output)
   
   
   
   
   
def make_signal_report(signal_results, test):
    scoring = pd.DataFrame(index=signal_results.keys())
    keys = signal_results.keys()
    scoring["correlation"] = [FactorTest.correlation(signal_results[index], test) for index in keys]
    scoring["correlation_stable"] = [FactorTest.correlation_stable(signal_results[index], test)  for index in keys]
    # scoring["accuracy"] = [FactorTest.accuracy(signal_results[index], test)  for index in keys]
    # scoring["zeros"] = [FactorTest.zero_percent(signal_results[index])  for index in keys]
    return scoring
   
   
   


def main():
    signal_name = input("Input signal name: ")
    base_factor:Factor = None
    module = importlib.import_module(f'factor.signal.{signal_name}')
    BaseClass = getattr(module, signal_name)
    f = open(f"factor/cfg/{signal_name}.json", 'r')
    jfile = json.load(f)
    
    train, test, index = Loader.make(target=TARGET, delay=DELAY, hours=HOURS)
    
    print(f"start time: {index[0]}")
    print(f"end time: {index[-1]}")
    print(f"number of cases: {len(train)}")
    print(f"predict interval: {HOURS}hrs")
    
    assert("signal_list" in jfile), "wrong config.."
    signal_results = pd.DataFrame()
    for config in tqdm.tqdm(jfile["signal_list"], "configs"):
        assert("argument" in config), "wrong config.."
        arg = config["argument"]
        base_factor = BaseClass(**arg)
        
        signal_output = pd.Series([base_factor.Gen(row) for row in train], index=index,name=str(base_factor))
        if (base_factor.exist):
            base_factor.delete_exist_result()
            
        base_factor.make_result_dir()
        base_factor.save_signal_output(signal_output)
        signal_results[base_factor.__str__()]  = signal_output
    
    scoring = make_signal_report(signal_results=signal_results, test=test)
    
    print("factor results:")
    print(scoring)
    print("correlation matrix:")
    print(signal_results.corr())
    

if __name__ == "__main__":
    main()