from factor.factors import Factor
import pandas as pd
import numpy as np



class weilun01(Factor):
    need = ["close"]
    cfg = {}
    def __init__(self, n:int, c:float):
        self.n = n
        self.c = c
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        returns = x["close"].apply(np.log).diff().dropna()
        mean = returns.iloc[-self.n].mean()
        std = returns.iloc[-self.n].std() * self.c
        last = returns.iloc[-1]
        last2 = returns.iloc[-2]
        
        if last < mean+std and last2 > mean+std:
            return 1
        elif last > mean-std and last2 < mean-std:
            return -1
        
        return 0
    
    def GenAll(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        
        returns = x["close"].apply(np.log).diff()
        mean = returns.rolling(self.n).mean()
        std = returns.rolling(self.n).std()*self.c
        ones = ((returns < mean + std) & (returns.shift(1) > mean+std)).astype(int)
        neg_ones = ((returns > mean - std) & (returns.shift(1) < mean-std)).astype(int)
        return ones - neg_ones
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.n}_{self.c}"