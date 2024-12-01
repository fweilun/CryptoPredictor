from factor.factors import Factor
import pandas as pd


class Cross(Factor):
    need = ["close"]
    def __init__(self, n1:int, n2:int):
        assert(n1 < n2), "n1 should be smaller than n2"
        self.n1 = n1
        self.n2 = n2
        
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["close"]
        assert((price.size > self.n1) & (price.size > self.n2))        
        cond1 = (price[self.n1:].mean() - price[self.n2:].mean()) > 0
        cond2 = (price[self.n1-1:-1].mean() - price[self.n2-1:-1].mean()) < 0
        if cond1 and cond2: return 1
        if not cond1 and not cond2: return -1
        return 0
    
    def GenAll(self, x:pd.DataFrame):
        price = x["close"]
        mean_diff = price.rolling(self.n1).mean() - price.rolling(self.n2).mean()
        signal = ((mean_diff > 0) & (mean_diff.shift(1) < 0)).astype(int)
        neg_signal = ((mean_diff < 0) & (mean_diff.shift(1) > 0)).astype(int)
        return (signal - neg_signal).dropna()
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.n1}_{self.n2}"