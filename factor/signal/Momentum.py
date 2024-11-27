from factor.factors import Factor
import pandas as pd

class Momentum(Factor):
    need = ["close"]
    def __init__(self, periods:list):
        self.periods = periods
        
    def Gen(self, x:pd.DataFrame):
        price = x["close"]
        momentum_scores = []
        
        for period in self.periods:
            # 計算每個週期的動量
            momentum = (price.iloc[-1] - price.iloc[-period]) / price.iloc[-period]
            momentum_scores.append(momentum)
        
        # 返回動量是否為正的機率
        positive_momentum_count = sum(1 for score in momentum_scores if score > 0)
        return float(positive_momentum_count / len(momentum_scores))
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}"