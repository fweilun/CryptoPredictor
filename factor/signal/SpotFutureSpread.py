from factor.factors import Factor
import numpy as np
import pandas as pd

class SpotFutureSpread(Factor):
    need = ["close", "future_close"]
    def __init__(self, periods:list):
        self.periods = periods
        
    def Gen(self, x:pd.DataFrame):
        spot_price = x["close"]
        future_price = x["future_close"]
        
        spread_metrics = []
        for period in self.periods:
            # 計算現貨和合約的價格溢價
            spread = (future_price - spot_price) / spot_price
            
            # 計算價差的統計特徵
            mean_spread = spread.rolling(window=period).mean().iloc[-1]
            std_spread = spread.rolling(window=period).std().iloc[-1]
            
            # 最近的價差
            recent_spread = spread.iloc[-1]
            
            # 評分：正值表示合約相對昂貴，負值表示合約相對便宜
            spread_metrics.append((recent_spread - mean_spread) / (std_spread + 1e-5))
        
        return float(np.mean(spread_metrics))
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}"