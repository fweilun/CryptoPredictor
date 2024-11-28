from factor.factors import Factor
import numpy as np
import pandas as pd

class SpotFutureSpread(Factor):
    need = ["close", "future_close"]
    
    def __init__(self, periods:list, normalize:bool=True):
        """
        Args:
            periods: List of periods to calculate spread metrics
            normalize: Whether to normalize the spread score
        """
        self.periods = periods
        self.normalize = normalize
    
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
            spread_score = (recent_spread - mean_spread) / (std_spread + 1e-5)
            spread_metrics.append(spread_score)
        
        # 計算平均分數，可選是否標準化
        avg_score = float(np.mean(spread_metrics))
        return avg_score if not self.normalize else np.tanh(avg_score)
    
    def GenAll(self, x:pd.DataFrame):
        spot_price = x["close"]
        future_price = x["future_close"]
        signals = pd.Series(index=x.index, dtype=float)
        
        for i in range(max(self.periods), len(x)):
            slice_spot = spot_price.iloc[i-max(self.periods):i]
            slice_future = future_price.iloc[i-max(self.periods):i]
            
            spread_metrics = []
            for period in self.periods:
                spread = (slice_future - slice_spot) / slice_spot
                mean_spread = spread.rolling(window=period).mean().iloc[-1]
                std_spread = spread.rolling(window=period).std().iloc[-1]
                recent_spread = spread.iloc[-1]
                
                spread_score = (recent_spread - mean_spread) / (std_spread + 1e-5)
                spread_metrics.append(spread_score)
            
            avg_score = float(np.mean(spread_metrics))
            signals.iloc[i] = avg_score if not self.normalize else np.tanh(avg_score)
        
        signals.iloc[:max(self.periods)] = 0
        return signals
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}_{self.normalize}"