from factor.factors import Factor
import pandas as pd
import numpy as np

class FundingRate(Factor):
    need = ["funding_rate"]
    def __init__(self, periods:list):
        self.periods = periods
        
    def Gen(self, x:pd.DataFrame):
        funding_rate = x["funding_rate"]
        
        # 計算資金費率的特徵
        funding_metrics = []
        for period in self.periods:
            # 區間資金費率的平均值和標準差
            mean_rate = funding_rate.rolling(window=period).mean().iloc[-1]
            std_rate = funding_rate.rolling(window=period).std().iloc[-1]
            
            # 最近一個週期的資金費率是否高於平均值
            recent_rate = funding_rate.iloc[-1]
            
            # 綜合評分：正值表示多頭壓力，負值表示空頭壓力
            funding_metrics.append((recent_rate - mean_rate) / (std_rate + 1e-5))
        
        return float(np.mean(funding_metrics))
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}"