from factor.factors import Factor
from factor.signal.Cross import Cross
from data.loader import Loader
from config.load import load_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib, json, tqdm



class FactorTest:
    
    @classmethod
    def correlation(cls, signal_output, test):
        return np.corrcoef(signal_output, test)[0][1]
    
    @classmethod
    def correlation_stable(cls, signal_output, test):
        n = 10
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
    scoring = pd.DataFrame(index=signal_results.columns)
    keys = signal_results.columns
    # print(signal_results[keys[0]], test)
    
    scoring["correlation"] = [FactorTest.correlation(signal_results[k], test) for k in keys]
    scoring["correlation_stable"] = [FactorTest.correlation_stable(signal_results[k], test)  for k in keys]
    # scoring["accuracy"] = [FactorTest.accuracy(signal_results[index], test)  for index in keys]
    # scoring["zeros"] = [FactorTest.zero_percent(signal_results[index])  for index in keys]
    return scoring


class FactorRunner:
    @classmethod
    def run1factor(cls, factor:Factor, delay:int=1000, target:str="BTCUSDT", save=True):
        X, y = Loader.make(target=target, delay=delay, hours=72)
        df = pd.Series([factor.Gen(row) for row in X], index=y.index, name=str(factor))
        if (factor.exist):
            factor.delete_exist_result()
            
        factor.make_result_dir()
        if save:
            factor.save_signal_output(df)
        return df


def main():
    config = load_config()
    TARGET = config.get("TARGET")
    DELAY = config.get("DELAY")
    HOURS = config.get("HOURS")
    
    config = load_config()
    start_date = config.get("test_factor").get("start_date")
    end_date = config.get("test_factor").get("end_date")
    
    signal_name = input("Input signal name: ")
    module = importlib.import_module(f'factor.signal.{signal_name}')
    BaseClass:Factor = getattr(module, signal_name)
    
    train, test = Loader.make(target=TARGET, delay=DELAY, hours=HOURS)
    index = test.index
    
    print(f"start time: {index[0]}")
    print(f"end time: {index[-1]}")
    print(f"number of cases: {len(train)}")
    print(f"predict interval: {HOURS}hrs")
    
    signal_results = pd.DataFrame()
    for signal in tqdm.tqdm(BaseClass.load_signals(), "factors"):
        signal.load_target(TARGET)
        FactorRunner.run1factor(signal)
        signal_output = pd.Series([signal.Gen(row) for row in train], index=index,name=str(signal))
        signal_results[signal.__str__()]  = signal_output

    # train by fixed interval, setting.yml
    signal_results = signal_results.loc[start_date:end_date]
    test = test.loc[start_date:end_date]
    # print(signal_results)
    # exit()
    print(f"report generate on")
    print("start date:", start_date)
    print("end date:", end_date)
    scoring = make_signal_report(signal_results=signal_results, test=test)
    
    print("factor results:")
    print(scoring)
    print("correlation matrix:")
    print(signal_results.corr())
    

if __name__ == "__main__":
    main()