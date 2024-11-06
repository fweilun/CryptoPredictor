from data.loader import Loader
import factor, os
import pandas as pd
import factor.result
import numpy as np
import shutil, time
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from factor.test_factor import make_signal_report


TARGET = "SOLUSDT"
DELAY = 1000
HOURS = 72
SPLIT_RATE = 0.8


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
        
        
def reset_results():
    folder_path = os.path.join(os.path.join(os.path.dirname(__file__), "factor"),"result")
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    time.sleep(0.3)
    print("remove all results")

def select_columns(x:pd.DataFrame):
    # x.sort_values(by="correlation_stable", ascending=False)
    return x[x["correlation_stable"] > 1.3].index


def backtest(reset=False, target=None):
    if reset:
        reset_results()
        
    selected_factor_cls:List[factor.BaseFactor] = [
        factor.weilun01,
        factor.weilun02,
        factor.weilun03,
        factor.weilun04,
        factor.weilun05,
        factor.weilun06,
        factor.weilun07,
        factor.Cross, 
        factor.Slope,
        factor.Skewness,
        factor.CrossRSI
    ]
    tests = {
        "r2_score":Test.r2_score,
        "accuracy":Test.accuracy_score,
        "correlation":Test.correlation,
        "confusion":Test.confusion_matrix,  # TODO: fix this test case
    }
    
    for fctr_class in selected_factor_cls:
        fctr_class.cfg = factor.cfg[fctr_class.__name__]
    
    selected_factors:List[factor.BaseFactor] = []
    for fctr_class in selected_factor_cls:
        for config in fctr_class.cfg["signal_list"]:
            selected_factors.append(fctr_class(**config["argument"]))
    
    if not target: target = TARGET
    print(f"backtest on {target}")
    X, y, index = Loader.make(target=target, delay=DELAY, hours=HOURS)
    
    print("selected factors:")
    for fctr_class in selected_factor_cls:
        print(f"- {fctr_class.__name__:<20}  with number of config: {len(fctr_class.cfg['signal_list'])}")
    
    print(f"total number of factors: {len(selected_factors)}")
    
    
    for fctr in tqdm(selected_factors, 'factors'):
        if fctr.exist:
            continue
        signal_output = pd.Series([fctr.Gen(row) for row in X], index=index, name=str(fctr))
        fctr.make_result_dir()
        fctr.save_signal_output(signal_output)
    
    x_finished = pd.concat([fctr.load_signal_output() for fctr in selected_factors if fctr.exist], axis=1)
    x_finished = x_finished.dropna(axis=0)
    intersection_index = x_finished.index.intersection(index)

    y = pd.Series(y, index=index)[intersection_index]
    x_finished = x_finished.loc[intersection_index]
    scoring = make_signal_report(signal_results=x_finished, test=y)
    selected_col = select_columns(scoring)
    # selected_col = [True for _ in range(len(x_train.columns))]
    
    
    
    split = int(SPLIT_RATE*len(x_finished))
    x_train, y_train, x_test,y_test = x_finished[:split], y[:split], x_finished[split:], y[split:]
    
    
    # model = LinearRegression()
    
    # print(x_train.loc[:,selected_col])
    # model = LassoCV(cv=5)
    model = RidgeCV(cv=5)
    model.fit(x_train.loc[:,selected_col], y_train)
    
    
    
    re = pd.concat([x_train.corrwith(y_train), x_test.corrwith(y_test)], axis=1)
    pd.set_option("display.max_rows", 50)
    re.columns = ["train corr", "test corr"]
    re = re.sort_values("train corr")
    re["weights"] = pd.Series(model.coef_, index=model.feature_names_in_)
    re.to_csv(f"{target}_{DELAY}_{HOURS}.csv")
    print(re.dropna())
    

    print("\ntrain result:")
    y_pred = model.predict(x_train.loc[:,selected_col])
    Test.run(tests, y_train, y_pred)
    
    print("\ntest result:")
    y_pred = model.predict(x_test.loc[:,selected_col])
    Test.run(tests, y_test, y_pred)
    

if __name__ == "__main__":
    backtest(reset=False)
    
    

