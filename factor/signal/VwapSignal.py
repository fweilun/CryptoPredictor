import pandas as pd
import numpy as np
from factor.factors import Factor

class VwapSignal(Factor):
    need = ["high", "low", "close", "volume"]
    
    def __init__(self, n):
        self.n = n
        pass
    
    def GenAll(self, x: pd.DataFrame) -> pd.DataFrame:
        for col in self.need:
            assert col in x.columns, f"{col} not exist"
        x['Typical_Price'] = (x['high'] + x['low'] + x['close']) / 3
        x['Cumulative_TP_Volume'] = (x['Typical_Price'] * x['volume']).rolling(self.n).mean()
        x['Cumulative_Volume'] = x['volume'].rolling(self.n).mean()
        x['VWAP'] = x['Cumulative_TP_Volume'] / x['Cumulative_Volume']
        return  (x['close'] > x['VWAP']).astype(float)-0.5
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.n}"
