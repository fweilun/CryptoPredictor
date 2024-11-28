from factor.factors import Factor
import pandas as pd
import numpy as np

class VolatilityCrossMarket(Factor):
    need = ["close", "future_close", "high", "low"]
    
    def __init__(self, periods:list=[5, 10, 20], volatility_type:str='parkinson'):
        """
        波動性跨市場相關性因子
        Args:
            periods: 計算波動性的不同週期
            volatility_type: 波動性計算方法
        """
        self.periods = periods
        self.volatility_type = volatility_type
    
    def _calculate_volatility(self, high, low, close, method='parkinson'):
        # Parkinson波動率
        if method == 'parkinson':
            return np.log(high / low)**2 / (4 * np.log(2))
        
        # Yang-Zhang波動率
        elif method == 'yang_zhang':
            return np.sqrt(np.log(high/close)**2 + np.log(low/close)**2)
    
    def Gen(self, x:pd.DataFrame):
        spot_high = x["high"]
        spot_low = x["low"]
        spot_close = x["close"]
        future_high = x["future_high"]
        future_low = x["future_low"]
        future_close = x["future_close"]
        
        volatility_correlations = []
        
        for period in self.periods:
            # 計算現貨和合約的波動率
            spot_volatility = self._calculate_volatility(
                spot_high.rolling(window=period).max(), 
                spot_low.rolling(window=period).min(), 
                spot_close.rolling(window=period).mean(), 
                method=self.volatility_type
            ).iloc[-1]
            
            future_volatility = self._calculate_volatility(
                future_high.rolling(window=period).max(), 
                future_low.rolling(window=period).min(), 
                future_close.rolling(window=period).mean(), 
                method=self.volatility_type
            ).iloc[-1]
            
            # 計算波動率差異
            volatility_correlation = np.abs(spot_volatility - future_volatility)
            volatility_correlations.append(volatility_correlation)
        
        return float(np.mean(volatility_correlations))
    
    def GenAll(self, x:pd.DataFrame):
        spot_high = x["high"]
        spot_low = x["low"]
        spot_close = x["close"]
        future_high = x["future_high"]
        future_low = x["future_low"]
        future_close = x["future_close"]
        
        signals = pd.Series(index=x.index, dtype=float)
        
        for i in range(max(self.periods), len(x)):
            slice_spot_high = spot_high.iloc[i-max(self.periods):i]
            slice_spot_low = spot_low.iloc[i-max(self.periods):i]
            slice_spot_close = spot_close.iloc[i-max(self.periods):i]
            slice_future_high = future_high.iloc[i-max(self.periods):i]
            slice_future_low = future_low.iloc[i-max(self.periods):i]
            slice_future_close = future_close.iloc[i-max(self.periods):i]
            
            volatility_correlations = []
            
            for period in self.periods:
                spot_volatility = self._calculate_volatility(
                    slice_spot_high.rolling(window=period).max(), 
                    slice_spot_low.rolling(window=period).min(), 
                    slice_spot_close.rolling(window=period).mean(), 
                    method=self.volatility_type
                ).iloc[-1]
                
                future_volatility = self._calculate_volatility(
                    slice_future_high.rolling(window=period).max(), 
                    slice_future_low.rolling(window=period).min(), 
                    slice_future_close.rolling(window=period).mean(), 
                    method=self.volatility_type
                ).iloc[-1]
                
                volatility_correlation = np.abs(spot_volatility - future_volatility)
                volatility_correlations.append(volatility_correlation)
            
            signals.iloc[i] = float(np.mean(volatility_correlations))
        
        signals.iloc[:max(self.periods)] = 0
        return signals
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}_{self.volatility_type}"