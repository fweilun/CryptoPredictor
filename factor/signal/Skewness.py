from factor.factors import Factor
import pandas as pd
import numpy as np



class Skewness(Factor):
    need = ["close"]
    def __init__(self, n:int=-1):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["close"]
        if (self.n == -1): self.n = price.size
        returns = price.apply(np.log).diff().dropna()[-self.n:]
        return returns[-self.n:].skew()
    
    def __str__(self) -> str:
        return f"Skewness {self.n}"