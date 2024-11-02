from factor.factors import Factor
import pandas as pd



class weilun07(Factor):
    need = ["high", "low", "close", "open"]
    def __init__(self, n):
        self.n = n

    
    def Gen(self, x:pd.DataFrame):
        last_bar = x.iloc[-self.n:]
        o = last_bar["open"] 
        c = last_bar["close"]
        h = last_bar["high"]
        l = last_bar["low"]
        s = (c-o)/(h-l+.01)
        return s.mean()

    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.n}"