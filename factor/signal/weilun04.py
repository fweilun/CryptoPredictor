from factor.factors import Factor
import pandas as pd
import numpy as np



class weilun04(Factor):
    need = ["close"]
    def __init__(self, n:int=-1):
        self.n = n
        self.mid = self.n/2
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        coin_close = x[x.columns[x.columns.str.contains("close")]]
        coin_returns = coin_close.iloc[-1].apply(np.log) - coin_close.iloc[-self.n].apply(np.log)
        Q1 = coin_returns.quantile(0.25)
        Q3 = coin_returns.quantile(0.75)
        IQR = Q3 - Q1
        filtered_data = coin_returns[(coin_returns >= Q1 - 1.5 * IQR) & (coin_returns <= Q3 + 1.5 * IQR)]
        return filtered_data.mean()
    
    def GenAll(self, x: pd.DataFrame) -> pd.Series:
        for col in self.need:
            assert col in x.columns, f"{col} not exist"
        
        coin_close = x[x.columns[x.columns.str.contains("close")]]
        coin_returns = np.log(coin_close) - np.log(coin_close.shift(self.n))
        
        Q1 = coin_returns.quantile(0.25, axis=1)
        Q3 = coin_returns.quantile(0.75, axis=1)
        IQR = Q3 - Q1
        coin_returns["lower_bound"] = Q1 - 1.5 * IQR
        coin_returns["upper_bound"] = Q3 + 1.5 * IQR
        mean_returns = coin_returns.T.apply(lambda x:x[(x < x["upper_bound"]) & (x>x["lower_bound"])].mean())
        return mean_returns
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.n}"