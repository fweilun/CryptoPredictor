from data.loader import Loader
import factor, os
import pandas as pd
import numpy as np
import shutil, time, warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from factor.test_factor import make_signal_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import List
from config.load import load_config
from factor.test_factor import FactorRunner




class Test:
    @classmethod
    def r2_score(cls,x,y):
        return r2_score(x,y)
    
    @classmethod
    def accuracy_score(cls,x,y):
        x_ = (x>0).astype(int)
        y_ = (y>0).astype(int)
        return accuracy_score(x_, y_)
    
    @classmethod
    def correlation(cls,x,y):
        return np.corrcoef(x, y)[0][1]
    
    @classmethod
    def confusion_matrix(cls, x, y):
        x_ = (x>0).astype(int)
        y_ = (y>0).astype(int)
        return confusion_matrix(x_, y_)
    
    @classmethod
    def mean_absolute_error(cls, x, y):
        return np.mean(np.abs(x-y))
    
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
        
        
def reset_results(target):
    folder_path = os.path.join(os.path.dirname(__file__), "factor","result",target)
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    time.sleep(0.3)
    print("remove all results")



def select_columns(x:pd.DataFrame, thresh_hold=1):
    return x[x["correlation_stable"] > thresh_hold].index


class FactorResultCaching:
    def __init__(self, target, factors:List[factor.BaseFactor]):
        self.cached_data:pd.DataFrame = None
        self.target = target
        self.factors = factors
    
    def search(self, index):
        if not self.__indexcmp(self.cached_data.index, index):
            raise Exception("Cached data not include all index.")
        
        return self.cached_data.loc[index:]
    
    def prepare(self, index):
        for f in self.factors: f.load_target(self.target)
        for fctr in self.factors:
            if (fctr.exist and self.__indexcmp(fctr.load_signal_output().index, index)):
                continue
            else:
                print(f"running factor {fctr}")
                FactorRunner.run1factor(fctr, save=True)
                
        self.cached_data = pd.concat([fctr.load_signal_output().loc[index] for fctr in self.factors if fctr.exist], axis=1)
    
    
    def __indexcmp(self, index1, index2):
        index1 = index2.intersection(index1)
        if index1.size != index2.size:
            return False
        return (index1 != index2).sum() == 0



class FactorModel:
    '''
    class FactorModel:
        def __init__(self):
            self.weights = {
                
            }
        
        def load_factors(self):
            self.selected_factors = []
        
        def fit(self, X, y):
            # turn X to feature matrix
            # train on X, y
            # receive model coef_data
            pass
        
        def predict(self):
            pass
    '''
    def __init__(self, thresh_hold=1, target="BTCUSDT") -> None:
        self.all_factors:List[factor.BaseFactor] = []
        self.weights:List[float] = []
        
        self.weights_dict = {}
        
        self.thresh_hold = thresh_hold
        self.factors_cls:List[factor.BaseFactor] = []
        self.target = target
        self.signal_report = None
        
        self.__x_scalar = StandardScaler()
        self.__factor_class = []
        self.__show = True
        self.__debug = False
        
        # model requirements
        self.__model = None
        self.__signal_order = None
        self.__factors = None
        
    def load_factors(self, all_factors:List[factor.BaseFactor]):
        self.__factor_class = all_factors
    
        all_factors:List[factor.BaseFactor] = []
        for fctr_class in self.__factor_class:
            all_factors.extend(fctr_class.load_signals())
        
        # result path is initialized by target
        for fctr in all_factors: 
            fctr.load_target(self.target)
            
        self.all_factors = all_factors
        
        if self.__show:
            print("selected factors:")
            for fctr_class in self.__factor_class:
                number_of_config = len(fctr_class.load_signal_config()['signal_list'])
                print(f"- {fctr_class.__name__:<20}  with number of config: {number_of_config}")
            print(f"total number of factors: {len(all_factors)}")
        
        
    
    def fit(self,X,y):
        
        # feature engineering on X
        if self.__debug: print("make signal output")
        x_train = self.__make_signal_output(X, y.index)
        columns = x_train.columns
        x_train = pd.DataFrame(x_train, columns=columns)
        
        
        # feature selection
        if self.__debug: print("select columns")
        selected_col = self.__select_columns(x_train, y, self.thresh_hold)
        x_train = x_train.loc[:, selected_col]
        
        # scalar
        if self.__debug: print("fit transformation")
        x_train = self.__x_scalar.fit_transform(x_train)
        
        model = RidgeCV(cv=5)
        # print(x_train)
        model.fit(x_train, y)
        
        self.__model = model
        self.__signal_order = self.__x_scalar.feature_names_in_
        self.__factors = [fctr for fctr in self.all_factors if str(fctr) in self.__signal_order]
    
    def fit_on_cache(self, x_train, y):
        # feature selection
        if self.__debug: print("select columns")
        selected_col = self.__select_columns(x_train, y, self.thresh_hold)
        x_train = x_train.loc[:, selected_col]
        # scalar
        if self.__debug: print("fit transformation")
        x_train = self.__x_scalar.fit_transform(x_train)
        
        model = RidgeCV(cv=5)
        model.fit(x_train, y)
        
        self.__model = model
        self.__signal_order = self.__x_scalar.feature_names_in_
        self.__factors = [fctr for fctr in self.all_factors if str(fctr) in self.__signal_order]
    
    def predict(self, x_test):
        all_factor_values = []
        for row in x_test:
            factor_values = [factor.Gen(row) for factor in self.__factors]
            all_factor_values.append(factor_values)
            
        factor_df = pd.DataFrame(all_factor_values, columns=self.__signal_order)
        factor_df_scaled = self.__x_scalar.transform(factor_df[self.__signal_order])
        return self.__model.predict(factor_df_scaled)
    
    def predict_on_cache(self, x_train):
        if self.__debug: print("fetch cached data.")
        factor_df_scaled = self.__x_scalar.transform(x_train[self.__signal_order])
        if self.__debug: print("matrix multiplication.")
        return self.__model.predict(factor_df_scaled)
    
    def __select_columns(self, x:pd.DataFrame, y, thresh_hold=0):
        max_correlation = 0.8
        selected_features = []
        signal_report = make_signal_report(x, y)
        '''
        select conditions
        1. with higher score
        2. correlation filter
        '''
        signal_report = signal_report.sort_values("correlation_stable", ascending=False)
        
        for index, value in signal_report["correlation_stable"].items():
            if value < thresh_hold:
                break
            if not selected_features:
                selected_features.append(index)
                continue
            
            correlations = x[selected_features].corrwith(x[index])
            if all(abs(corr) <= max_correlation for corr in correlations):
                selected_features.append(index)
                
        return selected_features
    
    def __make_signal_output(self, X, index):
        def __indexcmp(index1, index2):
            # if (index1.__contains__(index2.max()) and index1.__contains__(index2.min())):
            #     return True
            # return False
            index1 = index2.intersection(index1)
            if index1.size != index2.size:
                return False
            return (index1 != index2).sum() == 0
        
        packing = None
        if self.__show: packing = tqdm(self.all_factors, 'factors')
        else: packing = self.all_factors
        
        if self.__debug: print("comparing index")
        for fctr in packing:
            if (fctr.exist and __indexcmp(fctr.load_signal_output().index, index)):
                continue
            if not fctr.exist: 
                fctr.make_result_dir()
                
            signal_output = pd.Series([fctr.Gen(row) for row in X], index=index, name=str(fctr))
            fctr.save_signal_output(signal_output)
            if self.__debug: print("loading")
        
        if self.__debug: print("concat signal output")
        x_train = pd.concat([fctr.load_signal_output().loc[index] for fctr in self.all_factors if fctr.exist], axis=1)
        x_train = x_train
        
        if (x_train.isna().values.sum() != 0):
            print("Warning: there are nan values in the train data.")
            
        return x_train.dropna()




class RollingFitter:
    def __init__(self, core, train_period, test_period, gap, jump):
        self.train_period:int = train_period
        self.test_period:int = test_period
        self.jump:int = jump
        self.gap:int = gap
        self.core:FactorModel = core
        self.data_cache = None
        self.pred_record = None
        self.test_record = None
    
    def init_cache(self, cache):
        self.cache = cache
    
    def fit_on_cache(self, X, y):
        self.cache.prepare(y.index)
        
        assert(len(X) == len(y)), "X and y must have the same length."
        n = len(X)
        correlation_record = []
        pred_record = []
        test_record = []
        for start in tqdm(range(1000, n - self.train_period - self.test_period, self.jump), "case"):
            end = start + self.train_period
            
            test_start = end + self.gap
            test_end = test_start + self.test_period
            _, train_y = X[start:end], y[start:end]
            _, test_y = X[test_start:test_end], y[test_start:test_end]
            
            self.core.fit_on_cache(self.cache.cached_data.loc[train_y.index], train_y)
            pred_y = self.core.predict_on_cache(self.cache.cached_data.loc[test_y.index])
            
            pred_record.append(pd.Series(pred_y, index=test_y.index))
            test_record.append(test_y)
            correlation_record.append(Test.accuracy_score(test_y, pred_y))
            
        self.pred_record = pred_record
        self.test_record = test_record
        return correlation_record
    


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    config = load_config()
    TARGET = config.get("TARGET")
    HOURS = config.get("HOURS")
    DELAY = config.get("DELAY")
    JUMP = config.get("JUMP")
    GAP = config.get("GAP")
    train_period = config.get("backtest").get("train_period")
    test_period = config.get("backtest").get("test_period")
    
    X, y = Loader.make(target=TARGET, delay=1000, hours=72)
    factor_model = FactorModel(thresh_hold=1, target=TARGET)
    
    factor_model.load_factors([
        factor.weilun01,
        factor.weilun02,
        factor.weilun03,
        factor.weilun04,
        factor.weilun05,
        # factor.weilun06,
        factor.weilun07,
        factor.Slope,
        factor.Skewness,
        factor.CrossRSI,
        factor.Cross,
    ])
    
    
    cache = FactorResultCaching("BTCUSDT", factor_model.all_factors)
    
    rollingfit = RollingFitter(
        core=factor_model, 
        train_period=train_period, 
        test_period=test_period,
        gap=GAP, 
        jump=JUMP)
    
    rollingfit.init_cache(FactorResultCaching(target=factor_model.target, factors=factor_model.all_factors))
    res = rollingfit.fit_on_cache(X, y)
    
    pred = pd.concat(rollingfit.pred_record, axis=0)
    signal = (pred > 0).astype(int)
    # signal * 
    result = (1 + y.loc[signal.index] * signal*2)[::18].cumprod()
    result2 = (1 + y.loc[signal.index])[::18].cumprod()
    print(Test.accuracy_score(signal, y.loc[signal.index]))
    print(Test.correlation(signal, y.loc[signal.index]))
    # print(Test.correlation(signal, y.loc[signal.index]))
    plt.plot(result, label="strategy")
    plt.plot(result2, label="B&H")
    plt.legend()
    plt.show()
    # print(res)
    