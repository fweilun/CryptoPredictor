from factor.factors import Factor
import pandas as pd
import numpy as np

class Momentum(Factor):
    need = ["close"]
    
    def __init__(self, periods:list, weight_type:str='linear'):
        """
        Args:
            periods: List of momentum calculation periods
            weight_type: Weighting method for different periods
        """
        self.periods = periods
        self.weight_type = weight_type
    
    def _calculate_weights(self):
        # 根據不同的權重類型生成權重
        if self.weight_type == 'linear':
            return np.linspace(1, len(self.periods), len(self.periods))
        elif self.weight_type == 'exponential':
            return np.exp(np.linspace(0, 1, len(self.periods)))
        else:
            return np.ones(len(self.periods))
    
    def Gen(self, x:pd.DataFrame):
        price = x["close"]
        momentum_scores = []
        weights = self._calculate_weights()
        
        for period in self.periods:
            # 計算每個週期的動量
            momentum = (price.iloc[-1] - price.iloc[-period]) / price.iloc[-period]
            momentum_scores.append(momentum)
        
        # 加權平均動量
        weighted_momentum = np.dot(momentum_scores, weights) / np.sum(weights)
        return float(weighted_momentum)
    
    def GenAll(self, x:pd.DataFrame):
        price = x["close"]
        signals = pd.Series(index=price.index, dtype=float)
        weights = self._calculate_weights()
        
        for i in range(max(self.periods), len(price)):
            slice_price = price.iloc[i-max(self.periods):i]
            momentum_scores = []
            
            for period in self.periods:
                momentum = (slice_price.iloc[-1] - slice_price.iloc[-period]) / slice_price.iloc[-period]
                momentum_scores.append(momentum)
            
            weighted_momentum = np.dot(momentum_scores, weights) / np.sum(weights)
            signals.iloc[i] = float(weighted_momentum)
        
        signals.iloc[:max(self.periods)] = 0
        return signals
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}_{self.weight_type}"