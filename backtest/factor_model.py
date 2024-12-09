from data.loader import Loader
import factor, os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
# from backtest.test_factor import test1tafactor_by_class
from sklearn.preprocessing import StandardScaler
from typing import List
from config.load import load_config
from backtest.test_factor import FactorRunner
from backtest.utils import FactorTest, index_contains, factor_model_signal_report, make_signal_report, ResultEntry
import loggings as log
import talib
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

start_date = config.get("backtest").get("start_date")
end_date = config.get("backtest").get("end_date")

class FactorSelection:
    max_inner_correlation = 0.8
    @classmethod
    def select_columns_by_correlation(cls, x:pd.DataFrame, y, threshold=0):
        max_correlation = 0.8
        selected_features = []
        signal_report = make_signal_report(x, y, factor_tests=[
            FactorTest.correlation
        ])
        signal_report = signal_report[["correlation"]].abs().sort_values("correlation", ascending=False)
        
        for index, _ in signal_report.iterrows():
            if not selected_features:
                selected_features.append(index)
                continue
            
            if len(selected_features) == 8:
                break
            
            inter_correlation = x[selected_features].corrwith(x[index])
            if all(abs(corr) <= max_correlation for corr in inter_correlation):
                selected_features.append(index)
        
        return selected_features
    
    @classmethod
    def select_columns_by_ema_correlation(cls, x:pd.DataFrame, y, threshold=0):
        max_correlation = 0.8
        selected_features = []
        signal_report = make_signal_report(x, y, factor_tests=[
            FactorTest.ema_correlation
        ])
        signal_report = signal_report[["ema_correlation"]].abs().sort_values("ema_correlation", ascending=False)
        
        for index, _ in signal_report.iterrows():
            if not selected_features:
                selected_features.append(index)
                continue
            
            if len(selected_features) == 8:
                break
            
            inter_correlation = x[selected_features].corrwith(x[index])
            if all(abs(corr) <= max_correlation for corr in inter_correlation):
                selected_features.append(index)
        
        return selected_features
    
    @classmethod
    def select_columns_by_stable(cls, x:pd.DataFrame, y, threshold=0):
        max_correlation = 0.7
        threshold = 1.2
        selected_features = []
        signal_report = factor_model_signal_report(x, y)
        '''
        select conditions
        1. with higher score
        2. correlation filter
        '''
        signal_report = signal_report.sort_values("correlation_stable", ascending=False)
        for index, row in signal_report[["correlation", "correlation_stable"]].iterrows():
            if not selected_features:
                selected_features.append(index)
                continue
            
            correlations = x[selected_features].corrwith(x[index])
            if all(abs(corr) <= max_correlation for corr in correlations):
                if len(selected_features) < 10:
                    selected_features.append(index)
                
        return selected_features
    
    @classmethod
    def select_columns_by_winrate(cls, x:pd.DataFrame, y, threshold=0):
        max_correlation = 0.8
        selected_features = []
        signal_report = make_signal_report(x, y, factor_tests=[
            FactorTest.accuracy
        ])
        signal_report = signal_report[["accuracy"]].abs().sort_values("accuracy", ascending=False)
        
        for index, _ in signal_report.iterrows():
            if not selected_features:
                selected_features.append(index)
                continue
            
            if len(selected_features) == 8:
                break
            
            inter_correlation = x[selected_features].corrwith(x[index])
            if all(abs(corr) <= max_correlation for corr in inter_correlation):
                selected_features.append(index)
        
        return selected_features
    
    @classmethod
    def select_columns_by_ema_stable(cls, x:pd.DataFrame, y, threshold=0):
        selected_features = []
        signal_report = make_signal_report(x, y, factor_tests=[
            FactorTest.ema_stable
        ])
        signal_report = signal_report[["ema_stable"]].sort_values("ema_stable", ascending=False)
        
        for index, _ in signal_report.iterrows():
            if not selected_features:
                selected_features.append(index)
                continue
            
            if len(selected_features) == 8:
                break
            
            inter_correlation = x[selected_features].corrwith(x[index])
            if all(abs(corr) <= cls.max_inner_correlation for corr in inter_correlation):
                selected_features.append(index)
        
        return selected_features
    

    @classmethod
    def GetSelection(cls):
        return cls.select_columns_by_stable
    
    

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
            # test on X, ytest            # testve model coef_datatest            pass
        
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
        selected_col = FactorSelection.GetSelection(x_train, y, self.threshold)
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
        self.temp = None
        
    def fit_on_features(self, x_train, y):
        # feature selection
        if self.__debug: print("select columns")
        # selected_col = self.__select_columns(x_train, y, self.threshold)
        selected_col = FactorSelection.GetSelection()(x_train, y, self.threshold)
            
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
        self.temp = selected_col
        
    
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
    def __init__(self, core, train_period, test_period, gap, jump, target):
        self.train_period:int = train_period
        self.test_period:int = test_period
        self.jump:int = jump
        self.gap:int = gap
        self.core:FactorModel = core
        self.target = target
        self.pred_record = None
        self.test_record = None
        
        self.X:pd.DataFrame = None
        self.y:pd.DataFrame = None
    
    def load_y(self, y):
        self.y = y
        
    def fit(self):
        self.__load_factor_result()
        n = len(self.y)
        
        pred_record = []
        test_record = []
        weights_record = []
        temp = []
        for start in range(0, n - self.train_period - self.test_period, self.jump):
            end = start + self.train_period
            
            test_start = end + self.gap
            test_end = test_start + self.test_period
            
            train_y = self.y[start:end]
            test_y = self.y[test_start:test_end]
            
            _logger.debug(
                "Training period: %s-%s, Testing period: %s-%s", 
                train_y.index.min(), train_y.index.max(), 
                test_y.index.min(), test_y.index.max()
            )
            
            # load cached data
            train_x = self.X.loc[train_y.index]
            test_x = self.X.loc[test_y.index]
            
            
            assert index_contains(train_x.index, train_y.index), "index contains error"
            assert index_contains(test_x.index, test_y.index), "index contains error"
            
            train_x = train_x.iloc[::HOURS//4]
            train_y = train_y.iloc[::HOURS//4]
            test_x = test_x.iloc[::HOURS//4]
            test_y = test_y.iloc[::HOURS//4]
            
            self.core.fit_on_features(train_x, train_y)
            pred_y = self.core.predict_on_features(test_x)
            # self.core.all_factors
            # print(self.core.temp)
            # exit()
            # temp.append(ResultEntry(train_x[self.core.temp], train_y, test_x[self.core.temp], test_y, None).parse(FactorTest.correlation_stable, FactorTest.correlation)) 
            
            pred_record.append(pd.Series(pred_y, index=test_y.index))
            test_record.append(test_y)
            
            weights_record.append(self.core.weights_dict)
            for k, v in self.core.weights_dict.items():
                _logger.debug("     %-20s %s", k, v)
            
        return {
            "pred_record": pred_record,
            "test_record": test_record,
            "weights_record": weights_record,
            "temp":temp
        }
        
    def __load_factor_result(self):
        factors = self.core.all_factors
        for f in factors: f.load_target(self.target)
        
        for fctr in factors:
            if (fctr.exist and index_contains(fctr.load_signal_output().index, self.y.index)):
                continue
            else:
                print(f"running factor {fctr}")
                # test1tafactor_by_class(fctr, rerun=True, plot=True)
                     
        self.X = pd.concat([fctr.load_signal_output().loc[self.y.index] for fctr in factors], axis=1)
        print("%d of factor results.", len(self.X.columns))
        self.X = self.X.loc[~self.X.isna().all(axis=1), :]
        print("%d of factor results after drop.", len(self.X.columns))
        first_valid_index = self.X.loc[~self.X.isna().any(axis=1)].index[0]
        print("start time:", self.X.index[0])
        print("remove na, start time:", first_valid_index)
        self.X = self.X.loc[first_valid_index:]
        self.y = self.y.loc[first_valid_index:]


class FactorModelPerformanceEvaluator:
    def __init__(self, ts_result):
        self.ts_result = ts_result
        self.pred_record = self.ts_result["pred_record"]
        self.test_record = self.ts_result["test_record"]
        self.weights_record = self.ts_result["weights_record"]
        self.temp = self.ts_result["temp"]
    
    def mock_trade(self):
        pred = pd.concat(self.pred_record, axis=0)
        y = pd.concat(self.test_record, axis=0)
        signal = (pred > 0).astype(int)
        
        returns1 = y.loc[signal.index] * signal
        returns2 = y.loc[signal.index]
        
        result = (1 + returns1).cumprod()
        result2 = (1 + returns2).cumprod()
        
        sharpe = lambda mean, std:mean/std * (252*24/HOURS)**0.5
        sharpe1 = sharpe(returns1.mean(), returns1.std())
        sharpe2 = sharpe(returns2.mean(), returns2.std())
        
        print("sharpe:",sharpe1,"B&H sharpe:", sharpe2)
        print("accuracy: ", FactorTest.accuracy(signal, y.loc[signal.index]))
        print("correlation: ", FactorTest.correlation(signal, y.loc[signal.index]))
        plt.plot(result, label="strategy")
        plt.plot(result2, label="B&H")
        plt.legend()
        plt.grid(axis='x')
        plt.show()
    
    def scoring(self):
        result = self.ts_result["temp"]
        result = pd.concat(result)
        print(result.corr())
        plt.scatter(x=result[result.columns[0]], y=result[result.columns[1]])
        plt.show()
        
    
    def perform_maxtrix(self):
        def winrate(dir_diff):
            return (dir_diff==0).astype(int).mean()
        def returns(x):
            return (1+x).prod()
        def sharpe(x):
            return (x.mean()/x.std())*((252*24/HOURS)**0.5)
        def holdrate(x):
            return (x>0).astype(int).mean()
                
        pred = pd.concat(self.pred_record, axis=0)
        y = pd.concat(self.test_record, axis=0)
        
        dir_diff = (pred>0).astype(int) - (y>0).astype(int)
        strat = y * (pred>0).astype(int)
        
        print("strategy perform matrix:")
        print(strat.groupby(strat.index.year).agg([returns, sharpe, holdrate]))
        print("B&H perform matrix:")
        print(y.groupby(y.index.year).agg([returns, sharpe, holdrate]))
        print("strategy win rate:")
        print(dir_diff.groupby(dir_diff.index.year).apply(winrate))
        
        
        
        
        
        
    
    
        
        
    



def main():
    _, y = Loader.make_not_overlap(target=TARGET, delay=DELAY, hours=HOURS, dtype="continuous")
    y = y[start_date:end_date]
    # print(y)
    # exit()
    factor_model = FactorModel(threshold=1.2, target=TARGET)
    factor_model.load_factors([
        factor.Cross,
        factor.weilun01,
        factor.weilun02,
        factor.weilun03,
        factor.bang01,
        factor.bang02,
        factor.bang03,
        # factor.DPO,
        factor.weilun04,
        factor.Momentum2,
        factor.VwapSignal,
        # factor.weilunta,
        factor.FundingRate,
        factor.StablecoinFlow,
        # factor.MarketParticipation,
        factor.Momentum,
        factor.Skewness,
        factor.Slope,
        factor.SpotFutureSpread,
        # factor.VolatilityCrossMarket,
        factor.VolumeAnomaly,
    ])
    
    
    rollingfit = RollingFitter(
        core=factor_model, 
        train_period=train_period, 
        test_period=test_period,
        gap=GAP, 
        jump=JUMP,
        target=TARGET)
    
    # exit()
    rollingfit.load_y(y=y)
    res = rollingfit.fit()
    
    evaluator = FactorModelPerformanceEvaluator(res)
    evaluator.mock_trade()
    evaluator.perform_maxtrix()
    # evaluator.scoring()


if __name__ == "__main__":
    main()

