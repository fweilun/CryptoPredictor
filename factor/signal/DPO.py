from factor.factors import Factor
import pandas as pd
import numpy  as np

class DPO(Factor):
    need = ["close"]
    
    def __init__(self, n:int=20, displacement:int=5):
        self.n = n  # 移动平均的周期数
        self.displacement = displacement  # 偏移的周期数（向前偏移多少周期）
    
    def Gen(self, x: pd.DataFrame):
        # 确保必要的列 'close' 存在
        for col in self.need:
            assert(col in x.columns), f"{col} 不存在"
        
        # 提取收盘价格
        coin_close = x["close"]
        
        # 计算收盘价格的移动平均
        moving_average = coin_close.rolling(window=self.n).mean()
        
        # 将移动平均向前偏移指定的周期数
        displaced_moving_average = moving_average.shift(self.displacement)
        
        # 计算DPO（收盘价与偏移后的移动平均之差）
        dpo = coin_close - displaced_moving_average
        
        # 返回整个DPO序列，保留与原始数据对齐
        return dpo
    
    def GenAll(self, x: pd.DataFrame):
        # 确保必要的列 'close' 存在
        for col in self.need:
            assert(col in x.columns), f"{col} 不存在"
    
        # 提取收盘价格
        coin_close = x["close"]
    
        # 计算收盘价格的移动平均
        moving_average = coin_close.rolling(window=self.n).mean()
    
        # 将移动平均向前偏移指定的周期数
        displaced_moving_average = moving_average.shift(self.displacement)
    
        # 计算DPO（收盘价与偏移后的移动平均之差）
        dpo = coin_close - displaced_moving_average
    
        # 返回整个序列
        return dpo / (1+abs(dpo))
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_n{self.n}_displacement{self.displacement}"