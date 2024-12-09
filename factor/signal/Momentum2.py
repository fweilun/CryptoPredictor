from factor.factors import Factor
import pandas as pd
import numpy as np


class Momentum2(Factor):
    need = ["close"]
    
    def __init__(self, fast: int = 32, slow: int = 96):
        """
        :param fast: 快速指數平滑平均的週期
        :param slow: 慢速指數平滑平均的週期
        """
        self.fast = fast
        self.slow = slow
    
    def osc(self, prices: pd.Series) -> pd.Series:
        """
        計算震盪指標 (Oscillator)
        """
        f, g = 1 - 1 / self.fast, 1 - 1 / self.slow
        numerator = prices.ewm(span=2 * self.fast - 1).mean() - prices.ewm(span=2 * self.slow - 1).mean()
        denominator = np.sqrt(
            1.0 / (1 - f * f) - 2.0 / (1 - f * g) + 1.0 / (1 - g * g)
        )
        return numerator / denominator
    
    def GenAll(self, x: pd.DataFrame) -> pd.Series:
        """
        計算震盪指標生成的所有倉位信號
        """
        for col in self.need:
            assert col in x.columns, f"{col} not exist"
        prices = x["close"]
        log_prices = np.log(prices)
        volatility = log_prices.diff().ewm(com=32).std()
        smoothed_prices = (log_prices.diff() / volatility).clip(-4.2, 4.2).cumsum()
        signal = (np.tanh(self.osc(smoothed_prices)) / volatility)
        
        return signal/100
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.fast}_{self.slow}"
