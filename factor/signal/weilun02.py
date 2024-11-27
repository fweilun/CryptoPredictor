from factor.factors import Factor
import pandas as pd
import numpy as np



class weilun02(Factor):
    need = ["close"]
    cfg = {}
    def __init__(self, alpha:float=0.2, c:float=1.6):
        self.alpha = alpha
        self.c = c
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        returns = x["close"].apply(np.log).diff().dropna()
        returns_std = returns.ewm(alpha=self.alpha, adjust=False).std().iloc[-1]
        returns_mean = returns.ewm(alpha=self.alpha, adjust=False).mean().iloc[-1]
        up_val = returns_mean + returns_std * self.c
        down_val = returns_mean -returns_std * self.c
        if returns.iloc[-1] > up_val and returns.iloc[-2] < up_val:
            return 1
        elif returns.iloc[-1] < down_val and returns.iloc[-2] > up_val:
            return -1
        
        return 0
    
    def GenAll(self, x: pd.DataFrame):
        returns = x["close"].apply(np.log).diff()
        returns_std = returns.ewm(alpha=self.alpha, adjust=False).std()
        returns_mean = returns.ewm(alpha=self.alpha, adjust=False).mean()
        up_val = returns_mean + returns_std * self.c
        down_val = returns_mean -returns_std * self.c
        ones = ((returns > up_val) & (returns.shift(1) < up_val)).astype(int)
        neg_ones = ((returns < down_val) & (returns.shift(1) > down_val)).astype(int)
        return ones - neg_ones
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.alpha}_{self.c}"