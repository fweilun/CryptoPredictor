from factor.factors import Factor
import pandas as pd
import numpy as np



class weilun05(Factor):
    need = ["volume"]
    def __init__(self, n:int=-1):
        self.n = n
        self.mid = self.n/2
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        r = np.log(x["volume"].iloc[-self.n]) - np.log(x["volume"].iloc[-1])
        return int(r>0) - int(r<0)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.n}"