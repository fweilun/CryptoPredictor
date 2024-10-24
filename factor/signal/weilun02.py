from factor.factors import Factor
import pandas as pd
import numpy as np



class weilun02(Factor):
    need = ["close"]
    cfg = {}
    def __init__(self, n:int=-1):
        self.n = n
        self.mid = self.n/2
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["close"]
        return (sum(price.iloc[-self.n:] < price.iloc[-1]) - self.mid) / self.mid
    
    def __str__(self) -> str:
        return f"price rank {self.n}"