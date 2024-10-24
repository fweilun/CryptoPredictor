import pandas as pd
from abc import ABC


class Factor(ABC):
    def __init__(self): ...
    def Gen(self, cls, x:pd.Series) -> float: ...
    def __str__(self):
        return "not defined ..."





'''
class Cross(Factor):
    need = ["Adj Close"]
    def __init__(self, n1:int, n2:int):
        assert(n1 < n2), "n1 should be smaller than n2"
        self.n1 = n1
        self.n2 = n2
        
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        assert((price.size > self.n1) & (price.size > self.n2))        
        return float((price[self.n1:].mean() - price[self.n2:].mean()) > 0)
    
    def __str__(self) -> str:
        return f"Cross {self.n1} {self.n2}"
    

class Skewness(Factor):
    need = ["Adj Close"]
    def __init__(self, n:int=-1):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        if (self.n == -1): self.n = price.size
        returns = price.apply(np.log).diff().dropna()[-self.n:]
        return returns[-self.n:].skew()
    
    def __str__(self) -> str:
        return f"Skewness {self.n}"

class Slope(Factor):
    need = ["Adj Close"]
    def __init__(self, n:int=-1):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        if (self.n == -1): self.n = price.size
        assert(price.size > self.n), "n is too larget for current data"
        y = np.arange(0,self.n, 1)
        slope, _ = np.polyfit(price[-self.n:], y, 1)
        return slope
    
    def __str__(self) -> str:
        return f"Slope {self.n}"


class BollingBand(Factor):
    need = ["Adj Close"]
    def __init__(self, n:int=-1, threshold:float=1.6):
        self.n = n
        self.threshold = threshold
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        if (self.n == -1): self.n = price.size
        price = price[-self.n:]
        z = (price[-1] - price.mean())/price.std()
        return (z > 0) * (abs(z) > self.threshold)
    
    def __str__(self) -> str:
        return f"Bolling Band {self.n}"



class Momentum(Factor):
    need = ["Adj Close"]
    def __init__(self, n:int=-1):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        if (self.n == -1): self.n = price.size
        return (price[-1] - price[-self.n]) / price[-self.n]

    def __str__(self) -> str:
        return f"Momentum {self.n}"




class Volatility(Factor):
    need = ["Adj Close"]
    def __init__(self, n:int=-1):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        if (self.n == -1): self.n = price.size
        returns = price.pct_change().dropna()[-self.n:]
        return returns.std()
    
    def __str__(self) -> str:
        return f"Volatility {self.n}"



class MaxDrawdown(Factor):
    need = ["Adj Close"]
    def __init__(self, n:int=-1):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        if (self.n == -1): self.n = price.size
        rolling_max = price.rolling(self.n, min_periods=1).max()
        drawdown = (price - rolling_max) / rolling_max
        return drawdown.min()

class RSI(Factor):
    need = ["Adj Close"]
    def __init__(self, n:int=14):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        delta = price.diff().dropna()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(self.n).mean()
        avg_loss = loss.rolling(self.n).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs[-1]))
    
    def __str__(self) -> str:
        return f"RSI {self.n}"

class SharpeRatio(Factor):
    need = ["Adj Close"]
    def __init__(self, n:int=-1):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]  
        if (self.n == -1): self.n = price.size
        returns = price.pct_change().dropna()[-self.n:]
        return returns.mean() / returns.std()
    
    def __str__(self) -> str:
        return f"Sharpe Ratio {self.n}"

class MACD(Factor):
    need = ["Adj Close"]
    def __init__(self, short:int=12, long:int=26, signal:int=9):
        self.short = short
        self.long = long
        self.signal = signal
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        short_ema = price.ewm(span=self.short).mean()
        long_ema = price.ewm(span=self.long).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=self.signal).mean()
        return macd_line[-1] - signal_line[-1]
    
    def __str__(self):
        return f"MACD {self.short} {self.long} {self.signal}"
    

class Variance(Factor):
    need = ["Adj Close"]
    def __init__(self, n:int=-1):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        if (self.n == -1): self.n = price.size
        returns = price.pct_change().dropna()[-self.n:]
        return returns.var()
    
    def __str__(self) -> str:
        return f"Variance {self.n}"
    
    
class PriceRank(Factor):
    need = ["Adj Close"]
    def __init__(self, n):
        self.n = n
    
    def Gen(self, x: pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        price = x["Adj Close"]
        return sum(price[-self.n:] < price[-1]) / self.n
    def __str__(self):
        return f"Price Rank {self.n}"
    
    
class UnEmployeeRate(Factor):
    need = ["UnEmployRate"]
    def __init__(self, n:int):
        self.n = n
    
    def Gen(self, x:pd.DataFrame):
        for col in self.need: assert(col in x.columns), f"{col} not exist"
        log_unemploy = x["UnEmployRate"].apply(np.log)
        return log_unemploy[-1] - log_unemploy[-self.n]
    
    def __str__(self):
        return f"UnEmploy Rate {self.n}"
    
'''