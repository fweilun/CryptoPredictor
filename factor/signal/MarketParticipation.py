from factor.factors import Factor
import pandas as pd
import numpy as np

class MarketParticipation(Factor):
    need = ["volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
    
    def __init__(self, periods:list=[5, 10, 20]):
        """
        市場參與度因子：衡量主動買入和整體交易量的關係
        Args:
            periods: 計算參與度的不同週期
        """
        self.periods = periods
    
    def Gen(self, x:pd.DataFrame):
        volume = x["volume"]
        taker_buy_base = x["taker_buy_base_asset_volume"]
        taker_buy_quote = x["taker_buy_quote_asset_volume"]
        
        participation_scores = []
        
        for period in self.periods:
            # 主動買入佔總交易量的比例
            buy_base_ratio = taker_buy_base.rolling(window=period).mean().iloc[-1] / volume.rolling(window=period).mean().iloc[-1]
            buy_quote_ratio = taker_buy_quote.rolling(window=period).mean().iloc[-1] / volume.rolling(window=period).mean().iloc[-1]
            
            # 計算參與度分數
            participation_score = (buy_base_ratio + buy_quote_ratio) / 2
            participation_scores.append(participation_score)
        
        return float(np.mean(participation_scores))
    
    def GenAll(self, x:pd.DataFrame):
        volume = x["volume"]
        taker_buy_base = x["taker_buy_base_asset_volume"]
        taker_buy_quote = x["taker_buy_quote_asset_volume"]
        
        signals = pd.Series(index=x.index, dtype=float)
        
        for i in range(max(self.periods), len(x)):
            slice_volume = volume.iloc[i-max(self.periods):i]
            slice_taker_buy_base = taker_buy_base.iloc[i-max(self.periods):i]
            slice_taker_buy_quote = taker_buy_quote.iloc[i-max(self.periods):i]
            
            participation_scores = []
            
            for period in self.periods:
                buy_base_ratio = slice_taker_buy_base.rolling(window=period).mean().iloc[-1] / slice_volume.rolling(window=period).mean().iloc[-1]
                buy_quote_ratio = slice_taker_buy_quote.rolling(window=period).mean().iloc[-1] / slice_volume.rolling(window=period).mean().iloc[-1]
                
                participation_score = (buy_base_ratio + buy_quote_ratio) / 2
                participation_scores.append(participation_score)
            
            signals.iloc[i] = float(np.mean(participation_scores))
        
        signals.iloc[:max(self.periods)] = 0
        return signals
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}"