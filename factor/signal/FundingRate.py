from factor.factors import Factor
import pandas as pd
import numpy as np

class FundingRate(Factor):
    need = ["funding_rate"]
    
    def __init__(self, periods:list, threshold:float=1.0):
        """
        Args:
            periods: List of periods to calculate funding rate metrics
            threshold: Sensitivity threshold for signal generation
        """
        self.periods = periods
        self.threshold = threshold
        
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
            
            # 標準化分數：考慮資金費率的異常程度
            normalized_score = (recent_rate - mean_rate) / (std_rate + 1e-5)
            funding_metrics.append(normalized_score)
        
        # 平均分數，並根據閾值調整信號強度
        avg_score = float(np.mean(funding_metrics))
        return np.sign(avg_score) * min(abs(avg_score), self.threshold)
    
    def GenAll(self, x:pd.DataFrame):
        funding_rate = x["funding_rate"]
        signals = pd.Series(index=x.index, dtype=float)
        
        for i in range(max(self.periods), len(x)):
            slice_data = x.iloc[i-max(self.periods):i]
            slice_funding_rate = slice_data["funding_rate"]
            
            funding_metrics = []
            for period in self.periods:
                mean_rate = slice_funding_rate.rolling(window=period).mean().iloc[-1]
                std_rate = slice_funding_rate.rolling(window=period).std().iloc[-1]
                recent_rate = slice_funding_rate.iloc[-1]
                
                normalized_score = (recent_rate - mean_rate) / (std_rate + 1e-5)
                funding_metrics.append(normalized_score)
            
            avg_score = float(np.mean(funding_metrics))
            signals.iloc[i] = np.sign(avg_score) * min(abs(avg_score), self.threshold)
        
        signals.iloc[:max(self.periods)] = 0
        return signals
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}_{self.threshold}"