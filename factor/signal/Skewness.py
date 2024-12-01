from factor.factors import Factor
import pandas as pd
import numpy as np



class Skewness(Factor):
    need = ["close"]
    def __init__(self, n:int):
        self.n = n
    
    def GenAll(self, x:pd.DataFrame) -> pd.Series:
        for col in self.need:
            assert col in x.columns, f"{col} not exist"

        price = x["close"]
        if self.n == -1:
            self.n = price.size

        returns = price.apply(np.log).diff()
        skew = returns.rolling(window=self.n).skew()
        return skew
        
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.n}"