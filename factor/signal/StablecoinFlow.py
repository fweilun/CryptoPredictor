from factor.factors import Factor
import pandas as pd

class StablecoinFlow(Factor):
    need = ["total_circulating_usd"]

    def __init__(self, threshold_increase: float):
        self.threshold_increase = threshold_increase

    def Gen(self, x: pd.DataFrame):
        for col in self.need: 
            assert(col in x.columns), f"{col} not exist"
        
        stable_supply = x["total_circulating_usd"].dropna()
        assert(stable_supply.size > 1), "Not enough data to compute signals."

        supply_change = stable_supply.pct_change(6).dropna()

        significant_increase = supply_change.iloc[-1] > self.threshold_increase

        if significant_increase:
            return 1
        return -1

    def GenAll(self, x: pd.DataFrame):
        stable_supply = x["total_circulating_usd"].dropna()
        supply_change = stable_supply.pct_change(6)

        signals = supply_change.apply(lambda change: 1 if change > self.threshold_increase else (-1 if change < -self.threshold_increase else 0))
        
        return signals

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_increase_{self.threshold_increase}"