from factor.factors import Factor
import pandas as pd
import talib

class weilunta(Factor):
    need = ["close"]
    cfg = {}
    
    def __init__(self, n:int, rsi_period:int, sma_period:int):
        self.n = n
        self.rsi_period = rsi_period
        self.sma_period = sma_period
    
    def Gen(self, x: pd.DataFrame):
        # 確保需要的欄位存在
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        
        # 計算 RSI 和 SMA 指標
        rsi = talib.RSI(x["close"], timeperiod=self.rsi_period)
        sma = talib.SMA(x["close"], timeperiod=self.sma_period)
        
        # 簡單的交易策略：RSI < 30 看做超賣區，進行買入；RSI > 70 看做超買區，進行賣出
        # 如果 RSI 小於 30，並且當前價格高於移動平均線，返回 1 (買入信號)
        # 如果 RSI 大於 70，並且當前價格低於移動平均線，返回 -1 (賣出信號)
        
        last_rsi = rsi.iloc[-1]
        last_sma = sma.iloc[-1]
        last_price = x["close"].iloc[-1]
        
        if last_rsi < 30 and last_price > last_sma:
            return 1  # 買入信號
        elif last_rsi > 70 and last_price < last_sma:
            return -1  # 賣出信號
        
        return 0  # 無信號
    
    def GenAll(self, x: pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        
        # 計算 RSI 和 SMA 指標
        rsi = talib.RSI(x["close"], timeperiod=self.rsi_period)
        sma = talib.SMA(x["close"], timeperiod=self.sma_period)
        
        # 計算信號：當 RSI 小於 30 且價格高於 SMA 時生成買入信號
        # 當 RSI 大於 70 且價格低於 SMA 時生成賣出信號
        buy_signal = (rsi < 30) & (x["close"] > sma)
        sell_signal = (rsi > 70) & (x["close"] < sma)
        
        # 返回買入和賣出信號的差異 (1 為買入，-1 為賣出)
        return buy_signal.astype(int) - sell_signal.astype(int)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.n}_{self.sma_period}_{self.rsi_period}"
