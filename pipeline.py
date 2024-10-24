from data.maker import maker
import factor, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score


TARGET = "BTC"
DELAY = 1000
HOURS = 24
SPLIT_RATE = 0.8


def main():
    selected_factor_cls:List[factor.BaseFactor] = [
        factor.weilun01,
        factor.weilun02,
        factor.weilun03, 
        factor.Cross, 
        factor.Slope,
        factor.Skewness
    ]
    
    for fctr_class in selected_factor_cls:
        fctr_class.cfg = factor.cfg[fctr_class.__name__]
    
    selected_factors:List[factor.BaseFactor] = []
    for fctr_class in selected_factor_cls:
        for config in fctr_class.cfg["signal_list"]:
            selected_factors.append(fctr_class(**config["argument"]))
    
    
    X, y = maker(target=TARGET, delay=DELAY, hours=HOURS)
    train_x = []
    print("selected factors:")
    for fctr_class in selected_factor_cls:
        print(f"- {fctr_class.__name__:<20}  with number of config: {len(fctr_class.cfg['signal_list'])}")
    
    print(f"total number of factors: {len(selected_factors)}")
        
    for row in tqdm(X, "cases"):
        row_add_features = [fctr.Gen(row) for fctr in selected_factors]
        train_x.append(row_add_features)
    
    split = int(SPLIT_RATE*len(train_x))
    
    x_train, y_train, x_test,y_test = train_x[:split], y[:split], train_x[split:], y[split:]
    
    # np.save("x_train.npy", np.array(x_train))
    # np.save("y_train.npy", np.array(y_train))
    
    # np.save("x_test.npy", np.array(x_test))
    # np.save("y_test.npy", np.array(y_test))
    
    model = LassoCV(cv=5)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)  # 使用訓練好的模型進行預測
    r2 = r2_score(y_train, y_pred)   # 計算 R²
    print(f"train set R²: {r2}")
    
    y_pred = model.predict(x_test)  # 使用訓練好的模型進行預測
    r2 = r2_score(y_test, y_pred)   # 計算 R²
    print(f"test set R²: {r2}")

'''
def features_selection():
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")
    
    
    model = LassoCV(cv=10)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_train)  # 使用訓練好的模型進行預測
    r2 = r2_score(y_train, y_pred)   # 計算 R²
    print(f"train set R²: {r2}")
    
    y_pred = model.predict(x_test)  # 使用訓練好的模型進行預測
    r2 = r2_score(y_test, y_pred)   # 計算 R²
    print(f"test set R²: {r2}")
'''

if __name__ == "__main__":
    main()
    
    

