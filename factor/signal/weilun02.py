from factor.factors import Factor
import pandas as pd
import numpy as np



class weilun02(Factor):
    need = ["close"]
    cfg = {}
    def __init__(self, alpha:float=0.2, c:float=1.6):
        self.alpha = alpha
        self.c = c
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        returns = x["close"].apply(np.log).diff().dropna()
        returns_std = returns.ewm(alpha=self.alpha, adjust=False).std().iloc[-1]
        
        if returns.iloc[-1] > returns_std * self.c:
            return 1
        elif returns.iloc[-1] < -returns_std * self.c:
            return -1
        
        return 0
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.alpha}_{self.c}"