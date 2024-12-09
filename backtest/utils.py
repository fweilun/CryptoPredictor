import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
    
    

class FactorTest:
    @classmethod
    def r2_score(cls,x,y):
        return r2_score(x,y)
    
    @classmethod
    def accuracy(cls, signal_output:pd.Series, test):
        counter = 0
        not_equal_0 = 0
        n = len(test)
        
        for i in range(n):
            x_ = signal_output.iloc[i]
            y_ = test.iloc[i]
            x_ = int(x_ > 0) - int(x_ < 0)
            y_ = int(y_ > 0) - int(y_ < 0)
            counter += (x_ == y_) & (y_ != 0)
            not_equal_0 += (x_ != 0)
        
        if not_equal_0 == 0: 
            return 0
        acc = counter / not_equal_0
        return max(acc, 1-acc)
    
    @classmethod
    def zero_percent(cls, signal_output, _):
        return (signal_output == 0).sum()/len(signal_output)
    
    @classmethod
    def mean(cls, signal_output, _):
        return np.mean(signal_output)
    
    @classmethod
    def std(cls, signal_output, _):
        return np.std(signal_output)
    
    @classmethod
    def skewness(cls, signal_output, _):
        return pd.Series(signal_output).skew()
    
    @classmethod
    def correlation(cls,x,y):
        corr = np.corrcoef(x, y)
        if not np.isnan(corr).any():
            return corr[0,1]
        
        if len(x.dropna().unique()) == 1:
            return 0
        
        return pd.DataFrame([x, y]).T.dropna().corr().iloc[0,1]
    
    @classmethod
    def ema_correlation(cls, signal_output, test, alpha=0.1):
        _x = pd.Series(signal_output).ewm(alpha=alpha).mean()
        _y = pd.Series(test).ewm(alpha=alpha).mean()
        if (_x == 0).all():
            # print("exist all zero signal.")
            return 0
        return np.corrcoef(_x, _y)[0,1]
    
    @classmethod
    def correlation_series(cls, signal_output, test, split=5):
        n = split
        x_ = np.array_split(signal_output, n)
        y_ = np.array_split(test, n)
        result = []
        for i in range(n):
            assert(len(x_[i]) == len(y_[i])), "not same length"
            result.append(cls.correlation(x_[i], y_[i]))
        
        return pd.Series(result)
    
    @classmethod
    def confusion_matrix(cls, x, y):
        x_ = (x>0).astype(int)
        y_ = (y>0).astype(int)
        return confusion_matrix(x_, y_)
    
    @classmethod
    def mean_absolute_error(cls, x, y):
        return np.mean(np.abs(x-y))
    
    @classmethod
    def correlation_stable(cls, signal_output, test, split=5):
        x_ = np.array_split(signal_output, split)
        y_ = np.array_split(test, split)
        result = []
        for i in range(split):
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
    def ema_stable(cls, signal_output, test, split=5, alpha=0.2):
        signal_output = pd.Series(signal_output).ewm(alpha=alpha).mean()
        test = pd.Series(test).ewm(alpha=alpha).mean()
        return cls.correlation_stable(signal_output, test, split=split)
    
    @classmethod
    def run(cls, tests, x, y):
        x = pd.Series(x)
        y = pd.Series(y)
        for k, test_func in tests.items():
            if not callable(test_func):
                raise ValueError(f"{k} is not a callable function")
            if test_func == cls.confusion_matrix:
                print(k)
                print(test_func(x,y))
                continue
                
            print(k, test_func(x, y))
        
        
def make_signal_report(signal_results, test, factor_tests=None):
    if factor_tests is None:
        factor_tests = [
            FactorTest.correlation,
            FactorTest.correlation_stable,
            FactorTest.accuracy,
            FactorTest.mean,
            FactorTest.std,
            FactorTest.skewness,
            FactorTest.zero_percent
        ]
    
    scoring = pd.DataFrame(index=signal_results.columns)
    keys = signal_results.columns
    for factor_test in factor_tests:
        scoring[factor_test.__name__] = [
            factor_test(signal_results[k], test) for k in keys
        ]
    
    return scoring
 
test_factor_signal_report = lambda signal_results, test:make_signal_report(signal_results,test, factor_tests=[
    FactorTest.correlation,
    FactorTest.correlation_stable,
    FactorTest.accuracy,
    FactorTest.mean,
    FactorTest.std,
    FactorTest.skewness,
    FactorTest.zero_percent
])

factor_model_signal_report = lambda signal_results, test:make_signal_report(signal_results,test, factor_tests=[
    FactorTest.correlation,
    FactorTest.correlation_stable,
])

        
def index_contains(index1:pd.Index, index2:pd.Index):
    index1 = index2.intersection(index1)
    return index1.equals(index2)




    
class ResultEntry:
    def __init__(self, train, train_y, test, test_y, model):
        self.train:pd.DataFrame = train
        self.test:pd.DataFrame = test
        self.train_y:pd.Series = train_y
        self.test_y:pd.Series = test_y
        self.model = model
    
    def parse(self, test_on_train, test_on_test):
        result = []
        for col in self.train.columns:
            train_score = test_on_train(self.train[col], self.train_y)
            test_score = test_on_test(self.test[col], self.test_y)
            result.append((train_score, test_score))
            
        return pd.DataFrame(result).dropna()
        
        
        
        
