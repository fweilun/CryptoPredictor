from factor.factors import Factor
import pandas as pd
import numpy as np

class CrossRSI(Factor):
    need = ["close"]
    def __init__(self, n1:int, n2:int, n3:int):
        assert(n1 < n2), "n1 should be smaller than n2"
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3 #RSI周期
        
        
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["close"]
        delta = price.diff().dropna()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(self.n3).mean()
        avg_loss = loss.rolling(self.n3).mean()
        rs = avg_gain / avg_loss
        assert((price.size > self.n1) & (price.size > self.n2) & (price.size > self.n3))       
        return float((price[self.n1:].mean() - price[self.n2:].mean()) > 0  and (100 - (100 / (1 + rs.iloc[-1])))>60) 
    
    def __str__(self) -> str:
        return f"CrossRSI {self.n1} {self.n2} {self.n3} "
    



