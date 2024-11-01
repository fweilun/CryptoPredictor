from factor.factors import Factor
import pandas as pd



class weilun06(Factor):
    need = ["taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
    def __init__(self, alpha:float=0.2):
        self.alpha = alpha
    
    def Gen(self, x:pd.DataFrame):
        r1 = self.__calculate_ema(x["taker_buy_base_asset_volume"], alpha=self.alpha)
        r2 = self.__calculate_ema(x["taker_buy_quote_asset_volume"], alpha=self.alpha)
        return (r1-r2)/(r1+r2)
    
    def __calculate_ema(self, series:pd.Series, alpha:float):
        ema = series.iloc[0]
        for value in series[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.alpha}"