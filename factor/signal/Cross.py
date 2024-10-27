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
        return float((price[self.n1:].mean() - price[self.n2:].mean()) > 0)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.n1}_{self.n2}"