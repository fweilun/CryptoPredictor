from factor.factors import Factor
import pandas as pd
import numpy as np

class bang01(Factor):
    need = ["funding_rate", "close"]
    
    def __init__(self, periods:list,rsi_period:int, threshold:float=1.0 ):
        """
        Args:
            periods: List of periods to calculate funding rate metrics
            threshold: Sensitivity threshold for signal generation
            rsi_period: Period to calculate RSI
        """
        self.periods = periods
        self.threshold = threshold
        self.rsi_period = rsi_period

    def calculate_rsi(self, close:pd.Series, period:int):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-5)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        
        funding_rate = x["funding_rate"]
        close = x["close"]
        
        # 計算資金費率特徵
        funding_metrics = []
        for period in self.periods:
            mean_rate = funding_rate.rolling(window=period).mean().iloc[-1]
            std_rate = funding_rate.rolling(window=period).std().iloc[-1]
            recent_rate = funding_rate.iloc[-1]
            normalized_score = (recent_rate - mean_rate) / (std_rate + 1e-5)
            funding_metrics.append(normalized_score)
        
        avg_funding_score = float(np.mean(funding_metrics))
        
        # 計算 RSI 指標
        rsi = self.calculate_rsi(close, self.rsi_period).iloc[-1]
        rsi_signal = (rsi - 50) / 50  # Normalize RSI to range [-1, 1]
        
        # 總分數
        combined_score = avg_funding_score + rsi_signal
        
        # 判斷信號：漲是 1，跌是 -1
        if combined_score > 0:
            return 1
        elif combined_score < 0:
            return -1
        return 0
    
    def GenAll(self, x: pd.DataFrame):
        for col in self.need:
            assert (col in x.columns), f"{col} 不存在"
        
        funding_rate = x["funding_rate"]
        close = x["close"]
        signals = pd.Series(index=x.index, dtype=int)
        
        # 计算需要的最大窗口长度
        max_window = max(self.periods + [self.rsi_period])
        
        for i in range(max_window, len(x)):
            # 提取当前窗口的数据
            slice_data = x.iloc[i - max_window:i]
            slice_funding_rate = slice_data["funding_rate"]
            slice_close = slice_data["close"]
            
            # 计算资金费率特征
            funding_metrics = []
            for period in self.periods:
                if i >= period:
                    mean_rate = slice_funding_rate.iloc[-period:].mean()
                    std_rate = slice_funding_rate.iloc[-period:].std()
                    recent_rate = slice_funding_rate.iloc[-1]
                    normalized_score = (recent_rate - mean_rate) / (std_rate + 1e-5)
                    funding_metrics.append(normalized_score)
                else:
                    funding_metrics.append(0)
            
            avg_funding_score = float(np.mean(funding_metrics))
            
            # 计算 RSI 指标
            rsi = self.calculate_rsi(slice_close, self.rsi_period).iloc[-1]
            rsi_signal = (rsi - 50) / 50  # Normalize RSI to range [-1, 1]
            
            # 总分数
            combined_score = avg_funding_score + rsi_signal
            
            # 漲跌信號
            if combined_score > self.threshold:
                signals.iloc[i] = 1
            elif combined_score < -self.threshold:
                signals.iloc[i] = -1
            else:
                signals.iloc[i] = 0
        
        # 将初始部分的信号设为0，避免因数据不足导致的错误信号
        signals.iloc[:max_window] = 0
        return signals

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}_{self.threshold}_RSI_{self.rsi_period}"