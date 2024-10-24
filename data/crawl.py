import yfinance as yf
import pandas as pd
from datetime import datetime
import requests, json
from abc import ABC
from typing import List
from binance.spot import Spot
import time


class Crawler(ABC):
    def __init__(self): ...
    def Get(self) -> pd.DataFrame: ...
    


class SingleUSStock(Crawler):
    def __init__(self,start_date:str = '2005-01-01', end_date:str='2024-10-04', stock_id:str="QQQ"):
        self.start_date = start_date
        self.end_date = end_date
        self.stock_id = stock_id
    
    def Get(self) -> pd.DataFrame:
        # etfs = ['SPY', 'DIA', 'QQQ']
        data = yf.download(self.stock_id, start=self.start_date, end=self.end_date, interval='1d')
        return data





class UnEmployRate(Crawler):
    def __init__(self, start_date="2005", end_date="2024"):
        assert(start_date.isdigit()), "start_date should be a year."
        assert(end_date.isdigit()), "end_date should be a year."
        assert((int(start_date) < 2050) & (int(start_date) > 2000)), "out of range 2000-2050."
        assert(int(start_date) < int(end_date)), "start year should be less than end year."
        
        self.start_date = start_date
        self.end_date = end_date
    
    def Get(self) -> pd.DataFrame:
        headers = {'Content-type': 'application/json'}
        
        # partition
        partitions = list(range(int(self.start_date), int(self.end_date),10))
        partitions.append(int(self.end_date))
        unemploy_list = []
        for i in range(1, len(partitions)):
            start_date, end_date = partitions[i-1], partitions[i]
        
            data = json.dumps({"seriesid": ['LNS14000000'],"startyear":start_date, "endyear":end_date, "registrationkey":""})
            p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
            json_data = json.loads(p.text)

            df = pd.DataFrame(json_data['Results']['series'][0]['data'])

            latest_year = int(df['year'].iloc[0])
            latest_month = int(df['period'].iloc[0].replace('M',''))
            start_year = int(df['year'].iloc[-1])
            start_month = int(df['period'].iloc[-1].replace('M',''))

            def process_date(year,month):
                if month == 12:
                    month = 1
                    year += 1
                else:
                    month += 1
                return year,month

            latest_year, latest_month = process_date(latest_year, latest_month)
            start_year, start_month = process_date(start_year, start_month)
            issue_date = pd.date_range(f'{start_year}-{start_month}-1', f'{latest_year}-{latest_month}-1',
                                    freq='MS') + pd.tseries.offsets.DateOffset(days=9)
            df_new = pd.DataFrame({'value': df['value'].values[::-1]}, index=issue_date)
            df_new['value'] = df_new['value'].astype(float)
            df_new.index.name = 'date'
            unemploy_list.append(df_new)
        
        unemploy_result = pd.concat(unemploy_list)
        unemploy_result.Name = "UnEmployRate"
        unemploy_result
        return unemploy_result
    
    from binance.spot import Spot




class BinaceData(Crawler):
    def __init__(self,symbol="BTCUSDT",start_date="2005", end_date="2024", interval="4h"):
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol
        self.interval = interval
    
    def Get(self) -> pd.DataFrame:
        client = Spot()
        start_time = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
        end_time = int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
        tmp_time = start_time
        record_df = []
        while tmp_time < end_time:
            df = client.klines(self.symbol, interval=self.interval, startTime=tmp_time, endTime=end_time, limit=10000)
            df = pd.DataFrame(df, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "close_time", 
                "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
                "taker_buy_quote_asset_volume", "ignore"
            ])
            record_df.append(df)
            tmp_time = df["timestamp"].max()
            time.sleep(0.5)
        
        df = pd.concat(record_df).set_index("timestamp")
        df.index = pd.to_datetime(df.index, unit="ms")
        df = df.sort_index()
        df["close"] = df["close"].astype(float)
        return df[~df.duplicated()]
    
    
    
crypto_pairs = [
    "BTC/USDT",  # 比特幣
    "ETH/USDT",  # 以太幣
    "BNB/USDT",  # 幣安幣
    "XRP/USDT",  # 瑞波幣
    "ADA/USDT",  # 艾達幣 (Cardano)
    "SOL/USDT",  # Solana
    "DOGE/USDT", # 狗狗幣
    "DOT/USDT",  # Polkadot
    "MATIC/USDT",# Polygon
    "LTC/USDT",  # 萊特幣
    "AVAX/USDT", # Avalanche
    "SHIB/USDT", # Shiba Inu
    "LINK/USDT", # Chainlink
    "ATOM/USDT", # Cosmos
    "FTM/USDT",  # Fantom
    "SAND/USDT", # Sandbox
    "AXS/USDT",  # Axie Infinity
    "UNI/USDT",  # Uniswap
    "NEAR/USDT", # NEAR Protocol
    "ALGO/USDT", # Algorand
    "AAVE/USDT"  # Aave
]

    

def renew_baseline_data():
    for crypto in crypto_pairs:
        print("".join(crypto.split("/")))
        crypto_name = "".join(crypto.split("/"))
        df = BinaceData(crypto_name, start_date="2020-01-01", end_date="2024-05-05")
        df.Get().to_csv(f"/Users/user/Desktop/stock/factor-model/data/base-{crypto_name}.csv")
    
if __name__ == "__main__":
    renew_baseline_data()
