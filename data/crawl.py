import yfinance as yf
import pandas as pd
from datetime import datetime
import requests, json
from abc import ABC
from typing import List
from binance.spot import Spot
import time
import ccxt
from config.symbols import Symbols

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
    
class FundingRate(Crawler):
    def __init__(self, symbol = "BTCUSDT", start_date="2017-01-01", end_date="2024-01-01"):
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol

    def Get(self) -> pd.DataFrame:
        binance = ccxt.binance()
        binance.options = {
            'defaultType': 'future',
            'adjustForTimeDifference': True
        }

        start_time = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
        end_time = int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
        tmp_time = start_time

        record_df = []
        while tmp_time < end_time:
            df = binance.fetch_funding_rate_history(symbol=self.symbol, since=tmp_time, params={"until": end_time})
            df = [ data["info"] for data in df] 
            df = pd.DataFrame(df)
            record_df.append(df)
            tmp_time = int(df["fundingTime"].max())

        df = pd.concat(record_df).set_index("fundingTime")
        df.index = pd.to_datetime(df.index, unit="ms")
        df.sort_index()

        df["fundingRate"] = df["fundingRate"].astype(float)

        df.drop(columns=["markPrice"], inplace=True)

        return df

# only 30 days data
class LongShortRatio(Crawler):
    def __init__(self, symbol="BTC/USDT:USDT", interval="4h", limit=500):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit

    def Get(self) -> pd.DataFrame:
        binance = ccxt.binance()

        df = binance.fetch_long_short_ratio_history(symbol=self.symbol, timeframe=self.interval, limit=self.limit)
        df = [ data["info"] for data in df] 
        df = pd.DataFrame(df)

        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, unit="ms")
        df.sort_index()

        df["longAccount"] = df["longAccount"].astype(float)
        df["shortAccount"] = df["shortAccount"].astype(float)
        df["longShortRatio"] = df["longShortRatio"].astype(float)

        return df

class OpenInterest(Crawler):
    def __init__(self, symbol="BTC/USDT:USDT", interval="4h", limit=500):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
    
    def Get(self) -> pd.DataFrame:
        binance = ccxt.binance()

        df = binance.fetch_open_interest_history(symbol=self.symbol, timeframe=self.interval, limit=self.limit)
        df = [ data["info"] for data in df]
        df = pd.DataFrame(df)

        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, unit="ms")
        df.sort_index()

        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)

        return df

def renew_data():
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    base_path = "./data/storage"
    base = "USDT"

    # Base Data
    print(f'Base Data:')
    for symbol in Symbols:
        pair = symbol.value + base
        print(f'Pair: {symbol.value}/{base}')

        binance_data = BinaceData(symbol=pair, start_date=start_date, end_date=end_date)
        base_df = binance_data.Get()
        base_df.to_csv(f"{base_path}/base/{pair}.csv")

    # Funding Rate Data (BTC, ETH)
    symbols = []
    symbols.append(Symbols.BTC)
    symbols.append(Symbols.ETH)

    print(f'Funding Rate Data:')
    for symbol in symbols:
        pair = symbol.value + base
        print(f'Pair: {symbol.value}/{base}')

        funding_rate = FundingRate(symbol=pair, start_date=start_date, end_date=end_date)
        funding_rate_df = funding_rate.Get()
        funding_rate_df.to_csv(f"{base_path}/funding-rate/{pair}.csv")

    # Long Short Ratio Data (BTC, ETH) 30 days only
    print(f'Long Short Ratio Data:')
    for symbol in symbols:
        pair = f'{symbol.value}/{base}:{base}'
        print(f'Pair: {pair}')

        long_short_ratio = LongShortRatio(symbol=pair)
        long_short_ratio_df = long_short_ratio.Get()

        start_date = long_short_ratio_df.index.min().strftime('%Y-%m-%d %H:%M:%S')

        # old data
        try:
            df = pd.read_csv(f"{base_path}/long-short-ratio/{symbol.value + base}.csv")
            df.set_index('timestamp', inplace=True)

            repeated_df = df[df.index >= start_date]
            df = df.drop(repeated_df.index, axis=0)

        except:
            df = pd.DataFrame()

        df = pd.concat([df, long_short_ratio_df])
        df.to_csv(f"{base_path}/long-short-ratio/{symbol.value + base}.csv")

    # Open Interest Data (BTC, ETH) 30 days only
    print(f'Open Interest Data:')
    for symbol in symbols:
        pair = f'{symbol.value}/{base}:{base}'
        print(f'Pair: {pair}')

        open_interest = OpenInterest(symbol=pair)
        open_interest_df = open_interest.Get()

        start_date = open_interest_df.index.min().strftime('%Y-%m-%d %H:%M:%S')

        # old data
        try:
            df = pd.read_csv(f"{base_path}/open-interest/{symbol.value + base}.csv")
            df.set_index('timestamp', inplace=True)

            repeated_df = df[df.index >= start_date]
            df = df.drop(repeated_df.index, axis=0)

        except:
            df = pd.DataFrame()

        df = pd.concat([df, open_interest_df])
        df.to_csv(f"{base_path}/open-interest/{symbol.value + base}.csv")

if __name__ == "__main__":
    renew_data()
