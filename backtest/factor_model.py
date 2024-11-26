from data.loader import Loader
import factor, os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from backtest.test_factor import make_signal_report, test1factor_by_class
from sklearn.preprocessing import StandardScaler
from typing import List
from config.load import load_config
from backtest.test_factor import FactorRunner
from backtest.utils import FactorTest, index_contains
import loggings as log
# import logging

warnings.filterwarnings("ignore")
log.get_logger('matplotlib').setLevel(log.WARNING)

_debug_flag = True

log.add_file_handler("logs/factor_model.log")
if _debug_flag:
    log.set_level(log.DEBUG)
else:
    log.set_level(log.INFO)
_logger = log.get_logger(name=__name__)




config = load_config()
TARGET = config.get("TARGET")
HOURS = config.get("HOURS")
DELAY = config.get("DELAY")
JUMP = config.get("JUMP")
GAP = config.get("GAP")
train_period = config.get("backtest").get("train_period")
test_period = config.get("backtest").get("test_period")





class FactorResultCaching:
    def __init__(self, target, factors:List[factor.BaseFactor]):
        self.cached_data:pd.DataFrame = None
        self.target = target
        self.factors = factors
    
    def search(self, index):
        if not index_contains(self.cached_data.index, index):
            raise Exception("Cached data not include all index.")
        
        return self.cached_data.loc[index:]
    
    def prepare(self, index):
        for f in self.factors: f.load_target(self.target)
        for fctr in self.factors:
            if (fctr.exist and index_contains(fctr.load_signal_output().index, index)):
                continue
            else:
                print(f"running factor {fctr}")
                test1factor_by_class(fctr, rerun=True, plot=True)
                
        self.cached_data = pd.concat([fctr.load_signal_output().loc[index] for fctr in self.factors if fctr.exist], axis=1)
        # self.cached_data = self.cached_data[:"2024-04-01"]
    
    



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
    def __init__(self, threshold=1, target="BTCUSDT") -> None:
        self.all_factors:List[factor.BaseFactor] = []
        self.weights:List[float] = []
        
        self.weights_dict:Dict[str, float] = {}
        self.threshold = threshold
        self.target = target
        self.__scalar = StandardScaler()
        self.__factor_class = []
        self.signal_report = None
        self.__show = True
        self.__debug = False
        
        # model requirements
        self.__model = None
        self.__signal_order = None
        self.__factors = None
        
    def load_factors(self, all_factors: List[factor.BaseFactor]):
        self.__factor_class = all_factors
        self.all_factors = [factor for fctr in self.__factor_class for factor in fctr.load_signals()]
        
        for fctr in self.all_factors: 
            fctr.load_target(self.target)

        if self.__show:
            print("Selected factors:")
            for fctr_class in self.__factor_class:
                signal_config = fctr_class.load_signal_config()
                print(f"- {fctr_class.__name__:<20} with number of configs: {len(signal_config['signal_list'])}")
            print(f"Total number of factors: {len(self.all_factors)}")
    
    def fit(self,X,y):
        
        # feature engineering on X
        if self.__debug: print("make signal output")
        x_train = self.__make_signal_output(X, y.index)
        columns = x_train.columns
        x_train = pd.DataFrame(x_train, columns=columns)
        
        
        # feature selection
        if self.__debug: print("select columns")
        selected_col = self.__select_columns(x_train, y, self.threshold)
        x_train = x_train.loc[:, selected_col]
        
        # scalar
        if self.__debug: print("fit transformation")
        x_train = self.__scalar.fit_transform(x_train)
        
        model = RidgeCV(cv=5)
        model.fit(x_train, y)
        
        
        self.__model = model
        self.__signal_order = self.__scalar.feature_names_in_
        self.__factors = [fctr for fctr in self.all_factors if str(fctr) in self.__signal_order]
        self.weights_dict = dict(zip(self.__factors, model.coef_))
        
    def fit_on_features(self, x_train, y):
        # feature selection
        if self.__debug: print("select columns")
        selected_col = self.__select_columns(x_train, y, self.threshold)
            
        x_train = x_train.loc[:, selected_col]
        
        # scalar
        if self.__debug: print("fit transformation")
        x_train = self.__scalar.fit_transform(x_train)
        
        model = LassoCV(cv=5)
        model.fit(x_train, y)
        
        self.__model = model
        self.__signal_order = self.__scalar.feature_names_in_
        self.__factors = [fctr for fctr in self.all_factors if str(fctr) in self.__signal_order]
        self.weights_dict = dict(zip(self.__factors, model.coef_))
    
    def predict(self, x_test):
        all_factor_values = []
        for row in x_test:
            factor_values = [factor.Gen(row) for factor in self.__factors]
            all_factor_values.append(factor_values)
            
        factor_df = pd.DataFrame(all_factor_values, columns=self.__signal_order)
        factor_df_scaled = self.__scalar.transform(factor_df[self.__signal_order])
        return self.__model.predict(factor_df_scaled)
    
    def predict_on_features(self, x_train):
        if self.__debug: print("fetch cached data.")
        factor_df_scaled = self.__scalar.transform(x_train[self.__signal_order])
        if self.__debug: print("matrix multiplication.")
        return self.__model.predict(factor_df_scaled)
    
    def __select_columns(self, x:pd.DataFrame, y, threshold=0):
        max_correlation = 0.8
        selected_features = []
        signal_report = make_signal_report(x, y)
        '''
        select conditions
        1. with higher score
        2. correlation filter
        '''
        signal_report = signal_report.sort_values("correlation_stable", ascending=False)
        
        for index, row in signal_report[["correlation", "correlation_stable"]].iterrows():
            if row["correlation_stable"] < threshold or abs(row["correlation"]) < 0.03:
                break
            
            if not selected_features:
                # print(f"Signal: {index:<20} corr: {row['correlation']} stable:{row['correlation_stable']}")
                selected_features.append(index)
                continue
            
            correlations = x[selected_features].corrwith(x[index])
            if all(abs(corr) <= max_correlation for corr in correlations):
                # print(f"Signal: {index:<20} corr: {row['correlation']} stable:{row['correlation_stable']}")
                selected_features.append(index)
        
        if len(selected_features) == 0:
            selected_features = signal_report[signal_report["correlation"] > 0.01]
            # if len(selected_features) == 0:
                # return signal_report.index[0]
            return [selected_features.sort_values("correlation_stable", ascending=False).index[0]]
            
        return selected_features
    
    def __make_signal_output(self, X, index):
        packing = tqdm(self.all_factors, desc='factors') if self.__show else self.all_factors
        for fctr in packing:
            if fctr.exist and index_contains(fctr.load_signal_output().index, index):
                continue
            if not fctr.exist:
                fctr.make_result_dir()
            signal_output = pd.Series([fctr.Gen(row) for row in X], index=index, name=str(fctr))
            fctr.save_signal_output(signal_output)

        x_train = pd.concat([fctr.load_signal_output().loc[index] for fctr in self.all_factors if fctr.exist], axis=1)
        if x_train.isna().values.any():
            print("Warning: NaN values detected in training data.")
        return x_train.dropna()




class RollingFitter:
    def __init__(self, core, train_period, test_period, gap, jump):
        self.train_period:int = train_period
        self.test_period:int = test_period
        self.jump:int = jump
        self.gap:int = gap
        self.core:FactorModel = core
        self.cache:FactorResultCaching = None
        self.pred_record = None
        self.test_record = None
    
    def init_cache(self, cache):
        self.cache = cache
    
    def fit_on_cache(self, y):
        self.cache.prepare(y.index)
        n = len(y)
        
        pred_record = []
        test_record = []
        weights_record = []
        for start in range(0, n - self.train_period - self.test_period, self.jump):
            end = start + self.train_period
            
            test_start = end + self.gap
            test_end = test_start + self.test_period
            
            train_y = y[start:end]
            test_y = y[test_start:test_end]
            
            _logger.debug(
                "Training period: %s-%s, Testing period: %s-%s", 
                train_y.index.min(), train_y.index.max(), 
                test_y.index.min(), test_y.index.max()
            )
            
            # load cached data
            train_x = self.cache.cached_data.loc[train_y.index]
            test_x = self.cache.cached_data.loc[test_y.index]
            assert index_contains(train_x.index, train_y.index), "index contains error"
            assert index_contains(test_x.index, test_y.index), "index contains error"
            
            self.core.fit_on_features(train_x, train_y)
            pred_y = self.core.predict_on_features(test_x)
            
            pred_record.append(pd.Series(pred_y, index=test_y.index))
            test_record.append(test_y)
            
            weights_record.append(self.core.weights_dict)
            for k, v in self.core.weights_dict.items():
                _logger.debug("     %-20s %s", k, v)
            
        return {
            "pred_record": pred_record,
            "test_record": test_record,
            "weights_record": weights_record,
        }


class FactorModelPerformanceEvaluator:
    def __init__(self, ts_result):
        self.ts_result = ts_result
        self.pred_record = self.ts_result["pred_record"]
        self.test_record = self.ts_result["test_record"]
        self.weights_record = self.ts_result["weights_record"]
    
    def mock_trade(self):
        pred = pd.concat(self.pred_record, axis=0)
        y = pd.concat(self.test_record, axis=0)
        signal = (pred > 0).astype(int)
        
        result = (1 + y.loc[signal.index] * signal).cumprod()
        result2 = (1 + y.loc[signal.index]).cumprod()
        
        returns1 = y.loc[signal.index] * signal
        returns2 = y.loc[signal.index]
        sharpe1 = (returns1.mean()/returns1.std())
        sharpe2 = (returns2.mean()/returns2.std())
        print(sharpe1, sharpe2)
        # print(returns1.mean(), returns1.std(), result[-1] / returns1.std())
        # print(returns2.mean(), returns2.std(), result2[-1] / returns2.std())
        
        print(FactorTest.accuracy(signal, y.loc[signal.index]))
        print(FactorTest.correlation(signal, y.loc[signal.index]))
        plt.plot(result, label="strategy")
        plt.plot(result2, label="B&H")
        plt.legend()
        plt.show()
    
        
        
    



def main():
    X, y = Loader.make_not_overlap(target=TARGET, delay=DELAY, hours=HOURS)
    factor_model = FactorModel(threshold=1.2, target=TARGET)
    
    factor_model.load_factors([
        factor.weilun01,
        factor.weilun02,
        factor.weilun03,
        factor.weilun04,
        factor.weilun05,
        factor.weilun06,
        factor.weilun07,
        factor.weilun08,
        factor.weilun09,
        # factor.Cross,
        factor.CrossRSI,
        factor.Slope,
        factor.Skewness,
    ])
    
    
    rollingfit = RollingFitter(
        core=factor_model, 
        train_period=train_period, 
        test_period=test_period,
        gap=GAP, 
        jump=JUMP)
    
    rollingfit.init_cache(
        FactorResultCaching(target=factor_model.target, factors=factor_model.all_factors))
    
    res = rollingfit.fit_on_cache(y)
    evaluator = FactorModelPerformanceEvaluator(res)
    # print(res)
    evaluator.mock_trade()


if __name__ == "__main__":
    main()