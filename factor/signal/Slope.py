from factor.factors import Factor
import pandas as pd
import numpy as np



class Slope(Factor):
    need = ["close"]
    def __init__(self, n:int=-1):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["close"]
        if (self.n == -1): self.n = price.size
        assert(price.size > self.n), "n is too larget for current data"
        y = np.arange(0,self.n, 1)
        slope, _ = np.polyfit(price[-self.n:], y, 1)
        return slope
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.n}"