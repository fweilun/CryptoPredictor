from factor.factors import Factor
import pandas as pd
import numpy as np

class VolumeAnomaly(Factor):
    need = ["volume", "future_volume"]
    
    def __init__(self, periods:list=[5, 10, 20], method:str='zscore'):
        """
        成交量異常因子
        Args:
            periods: 計算異常的不同週期
            method: 異常檢測方法 ('zscore' or 'mad')
        """
        self.periods = periods
        self.method = method
    
    def _detect_anomaly(self, volume_series, period):
        # Z-score方法
        if self.method == 'zscore':
            mean = volume_series.rolling(window=period).mean()
            std = volume_series.rolling(window=period).std()
            return (volume_series - mean) / (std + 1e-5)
        
        # 中位數絕對偏差方法
        elif self.method == 'mad':
            median = volume_series.rolling(window=period).median()
            mad = np.abs(volume_series - median).rolling(window=period).median()
            return (volume_series - median) / (mad + 1e-5)
    
    def Gen(self, x:pd.DataFrame):
        spot_volume = x["volume"]
        future_volume = x["future_volume"]
        
        spot_anomalies = []
        future_anomalies = []
        
        for period in self.periods:
            spot_anomaly = self._detect_anomaly(spot_volume, period).iloc[-1]
            future_anomaly = self._detect_anomaly(future_volume, period).iloc[-1]
            
            spot_anomalies.append(spot_anomaly)
            future_anomalies.append(future_anomaly)
        
        # 結合現貨和合約的成交量異常
        combined_anomaly = np.mean(spot_anomalies + future_anomalies)
        return float(np.tanh(combined_anomaly))
    
    def GenAll(self, x:pd.DataFrame):
        spot_volume = x["volume"]
        future_volume = x["future_volume"]
        signals = pd.Series(index=x.index, dtype=float)
        
        for i in range(max(self.periods), len(x)):
            slice_spot_volume = spot_volume.iloc[i-max(self.periods):i]
            slice_future_volume = future_volume.iloc[i-max(self.periods):i]
            
            spot_anomalies = []
            future_anomalies = []
            
            for period in self.periods:
                spot_anomaly = self._detect_anomaly(slice_spot_volume, period).iloc[-1]
                future_anomaly = self._detect_anomaly(slice_future_volume, period).iloc[-1]
                
                spot_anomalies.append(spot_anomaly)
                future_anomalies.append(future_anomaly)
            
            combined_anomaly = np.mean(spot_anomalies + future_anomalies)
            signals.iloc[i] = float(np.tanh(combined_anomaly))
        
        signals.iloc[:max(self.periods)] = 0
        return signals
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}_{self.method}"