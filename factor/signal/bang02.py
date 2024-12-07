from factor.factors import Factor
import pandas as pd
import numpy as  np

class bang02(Factor):
    need = ["funding_rate", "close"]

    def __init__(self, short_window: int, long_window: int):
        """
        Args:
            short_window: 短期移动平均窗口
            long_window: 长期移动平均窗口
        """
        self.short_window = short_window
        self.long_window = long_window

    def Gen(self, x: pd.DataFrame):
        for col in ["funding_rate", "close"]:
            assert col in x.columns, f"{col} 不存在"
        
        short_ma = x["close"].rolling(window=self.short_window).mean()
        long_ma = x["close"].rolling(window=self.long_window).mean()
        funding_rate = x["funding_rate"].iloc[-1]
        last_close = x["close"].iloc[-1]
        
        # 信号逻辑
        if funding_rate > 0.0001 and last_close < short_ma.iloc[-1] and short_ma.iloc[-1] < long_ma.iloc[-1]:
            return -1  # 看空信号
        elif funding_rate < 0.0001 and last_close > short_ma.iloc[-1] and short_ma.iloc[-1] > long_ma.iloc[-1]:
            return 1  # 看多信号
        return 0  # 无明显信号

    def GenAll(self, x: pd.DataFrame):
        for col in ["funding_rate", "close"]:
            assert col in x.columns, f"{col} 不存在"
        
        short_ma = x["close"].rolling(window=self.short_window).mean()
        long_ma = x["close"].rolling(window=self.long_window).mean()
        signals = pd.Series(index=x.index, dtype=int)

        for i in range(max(self.short_window, self.long_window), len(x)):
            funding_rate = x["funding_rate"].iloc[i]
            last_close = x["close"].iloc[i]
            
            if funding_rate > 0.0001 and last_close < short_ma.iloc[i] and short_ma.iloc[i] < long_ma.iloc[i]:
                signals.iloc[i] = -1  # 看空信号
            elif funding_rate < 0.0001 and last_close > short_ma.iloc[i] and short_ma.iloc[i] > long_ma.iloc[i]:
                signals.iloc[i] = 1  # 看多信号
            else:
                signals.iloc[i] = 0  # 无明显信号

        signals.iloc[:max(self.short_window, self.long_window)] = 0
        return signals

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_short{self.short_window}_long{self.long_window}"