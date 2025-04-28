import yfinance as yf
import pandas as pd
from datetime import datetime
import requests, json, psycopg2, time
from data.db import *
from abc import ABC
from typing import List
from binance.spot import Spot
from config.symbols import Symbols
from psycopg2.extras import execute_values
from itertools import product
import requests
from bs4 import BeautifulSoup

class Crawler(ABC):
    def __init__(self): ...
    def Get(self) -> pd.DataFrame: ...
    
    def batch_insert(self, df, table_name, batch_size=10000, show=True):
        """ 高效批量插入数据到 PostgreSQL """
        conn = get_connection()
        records = df.values.tolist()
        columns = ", ".join(df.columns)
        insert_query = f"""
        INSERT INTO {table_name} ({columns})
        VALUES %s
        """
        if show:print("Start inserting values.")
        with conn.cursor() as cursor:
            execute_values(cursor, insert_query, records, page_size=batch_size)
        conn.commit()
        if show:print("End inserting values.")
        release_connection(conn)

class SingleStock:
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS stock_prices (
        date DATE NOT NULL,             -- 日期
        open NUMERIC,                   -- 開盤價
        high NUMERIC,                   -- 最高價
        low NUMERIC,                    -- 最低價
        close NUMERIC,                  -- 收盤價
        adj_close NUMERIC,              -- 調整後收盤價
        volume BIGINT,                  -- 成交量
        stock_id VARCHAR NOT NULL,      -- 股票代碼
        PRIMARY KEY (date, stock_id)    -- 主鍵：日期 + 股票代碼，避免重複數據
    );
    '''
    
    def __init__(self, start_date: str = '2005-01-01', end_date: str = '2024-10-04', stock_id: str = "2330.TW"):
        self.start_date = start_date
        self.end_date = end_date
        self.stock_id = stock_id

    def get_stock_data(self) -> pd.DataFrame:
        try:
            data = yf.download(self.stock_id, start=self.start_date, end=self.end_date, interval='15m', auto_adjust=False)
            data.reset_index(inplace=True)
            data['StockID'] = self.stock_id
            return data
        except Exception as e:
            print(f"Error fetching data for {self.stock_id}: {e}")
            return pd.DataFrame()

    def write_to_sql(self, data: pd.DataFrame):
        if data.empty:
            print("No data to write to the database.")
            return

        try:
            # 建立資料庫連接
            conn = get_connection()

            # 將 DataFrame 寫入資料庫
            with conn.cursor() as cursor:
                for _, row in data.iterrows():
                    cursor.execute(f"""
                        INSERT INTO stock_prices (date, open, high, low, close, adj_close, volume, stock_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (date, stock_id) DO NOTHING;
                    """, (
                        row['Date'].item(),
                        row['Open'].item(),
                        row['High'].item(),
                        row['Low'].item(),
                        row['Close'].item(),
                        row['Adj Close'].item(),
                        row['Volume'].item(),
                        row['StockID'].item()
                    ))

            # 提交交易
            conn.commit()
            print(f"Data successfully written to table.")
        except Exception as e:
            print(f"Error writing to the database: {e}")
        finally:
            release_connection(conn)

class SingleUSStock(Crawler):
    def __init__(self,start_date:str = '2005-01-01', end_date:str='2024-10-04', stock_id:str="QQQ"):
        self.start_date = start_date
        self.end_date = end_date
        self.stock_id = stock_id
    
    def Get(self) -> pd.DataFrame:
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
    
class SpotData(Crawler):
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

class FutureData(Crawler):
    def __init__(self, symbol="BTC/USDT:USDT", start_date="2017-01-01", end_date="2024-01-01", interval="4h"):
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol
        self.interval = interval

    def Get(self) -> pd.DataFrame:
        binance = ccxt.binance()

        start_time = int(datetime.strptime(self.start_date, "%Y-%m-%d").timestamp() * 1000)
        end_time = int(datetime.strptime(self.end_date, "%Y-%m-%d").timestamp() * 1000)
        tmp_time = start_time

        record_df = []
        while tmp_time < end_time:
            df = binance.fetch_ohlcv(symbol=self.symbol, timeframe=self.interval, since=tmp_time, params={"until": end_time})
            df = pd.DataFrame(df, columns=["timestamp", "open", "high", "low", "close", "volume"])
            new_tmp_time = int(df["timestamp"].max())
            if tmp_time == new_tmp_time:
                break

            record_df.append(df)
            tmp_time = new_tmp_time

        df = pd.concat(record_df).set_index("timestamp")
        df.index = pd.to_datetime(df.index.astype(float), unit="ms")
        df = df.sort_index()
        df["close"] = df["close"].astype(float)

        return df
    
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
            new_tmp_time = int(df["fundingTime"].max())
            if tmp_time == new_tmp_time:
                break
            
            record_df.append(df)
            tmp_time = new_tmp_time

        df = pd.concat(record_df).set_index("fundingTime")
        df.index = pd.to_datetime(df.index.astype(float), unit="ms")
        df.index = df.index.round('s')
        df = df.sort_index()
        df = df.drop_duplicates(keep='first')
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
        df.index = pd.to_datetime(df.index.astype(float), unit="ms")
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
        df.index = pd.to_datetime(df.index.astype(float), unit="ms")
        df.sort_index()

        df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
        df["sumOpenInterestValue"] = df["sumOpenInterestValue"].astype(float)

        return df

# class StableCoin(Crawler):
#     def __init__(self):
#         ...
    
#     def Get(self) -> pd.DataFrame:
#         url = "https://stablecoins.llama.fi/stablecoincharts/all"
#         df = requests.get(url, params={"stablecoin": 1}).json()

#         df = [(i["date"], i["totalCirculatingUSD"]["peggedUSD"]) for i in df]
#         df = pd.DataFrame(df, columns=["timestamp", "totalCirculatingUSD"])

#         df.set_index("timestamp", inplace=True)
#         df.index = pd.to_datetime(df.index.astype(float), unit="s")
#         df.sort_index()

#         df["totalCirculatingUSD"] = df["totalCirculatingUSD"].astype(int)

#         return df



class FearGreedIndex(Crawler):
    def __init__(self):
        ...
    
    def Get(self) -> pd.DataFrame:
        url = "https://api.alternative.me/fng/"
        df = requests.get(url, params={"limit": 0}).json()
        df = pd.DataFrame(df["data"])
        df.drop(columns=["time_until_update"], inplace=True)
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index.astype(float), unit="s")
        df.sort_index()
        df["value"] = df["value"].astype(int)
        return df

class ShareDistribution(Crawler):
    
        
    def __init__(self):
        self._create_share_distribution_table()
    
    def Get(self, symbols=[]):
        headers = {
            'User-Agent':  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
        }
        rs = requests.session()
        res = rs.get('https://www.tdcc.com.tw/portal/zh/smWeb/qryStock', headers = headers)
        soup = BeautifulSoup(res.text, 'lxml')
        valid_date = [v.get("value") for v in soup.find_all("option")]
        
        conn = get_connection()
        for symbol, date in product(symbols, valid_date):
            res = rs.get('https://www.tdcc.com.tw/portal/zh/smWeb/qryStock', headers = headers)
            soup = BeautifulSoup(res.text, 'lxml')
            payload = dict([ (e.get('name'), e.get('value')) for e in soup.select('input[type=hidden]')])
            
            print(date)
            time.sleep(1)
            payload['scaDate'] = date
            payload['sqlMethod'] = 'StockNo'
            payload['stockNo'] = str(symbol)

            res2 = rs.post('https://www.tdcc.com.tw/portal/zh/smWeb/qryStock', headers = headers, data = payload)

            dfs = pd.read_html(res2.text)
            share_table = dfs[1]
            
            share_table.columns = ["", "holding_range", "holder_count", "share_count", "percentage"]
            share_table = share_table.loc[:14,share_table.columns[1:]]
            print(share_table)
            share_table["holding_range"] = share_table["holding_range"].str.split('-').str[1]
            share_table["holding_range"] = share_table["holding_range"].str.replace(',', '')
            share_table.loc[14, "holding_range"] = "999999999"
            share_table["holding_range"] = share_table["holding_range"].astype(int)
            
            for _, row in share_table.iterrows():
                row_dict = row.to_dict()
                row_dict["symbol"] = symbol
                row_dict["date"] = f"'{date}'"
                insert_query = '''
                    INSERT INTO stock_distribution 
                    (symbol, date, holding_range, holder_count, share_count, percentage)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, date, holding_range) DO NOTHING;
                '''
                
                # 提取欄位值並確保資料型別正確
                values = (
                    row_dict["symbol"],
                    row_dict["date"],
                    row_dict["holding_range"],
                    row_dict["holder_count"],
                    row_dict["share_count"],
                    row_dict["percentage"]
                )

                cursor = conn.cursor()
                cursor.execute(insert_query, values)
                
                conn.commit()
        release_connection(conn)

    def _create_share_distribution_table(self):
        create_table_sql = '''
        CREATE TABLE IF NOT EXISTS stock_distribution (
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            holding_range VARCHAR(50) NOT NULL,
            holder_count INT NOT NULL,
            share_count BIGINT NOT NULL,
            percentage DECIMAL(10, 2) NOT NULL,
            PRIMARY KEY (symbol, date, holding_range) -- If needed, you can set a composite primary key
        );
        '''
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        release_connection(conn)

class StockTickData(Crawler):
    def __init__(self, dtype='raw'):
        self.dtype = dtype
    
    def Get(self):
        if self.dtype == 'raw':
            df = self.readData()
            self.batch_insert(
                df,
                table_name="stock_tick_data", 
                batch_size=10000)
            print('inserted raw data.')
        
    
    def readData(self):
        url = "https://api.kibot.com/?action=history&symbol=IVE&interval=tickbidask&bp=1&user=guest"
        
        print("Read data from url:%s." % (url))
        try:
            df = pd.read_csv(url, header=None, names=[
                "date", "time", "price", "bid", "ask", "volume"
            ])
            # df = pd.read_csv("/Users/user/Desktop/CryptoPredictor/data/storage/SPY_tick.txt", header=None, names=[
            #     "date", "time", "price", "bid", "ask", "volume"
            # ])
        except Exception as e:
            print("Error %s" % e)
            raise e
        
        print("Success reading data from url:%s." % (url))
        df["datetime"] = pd.to_datetime(df["date"] + df["time"],format='%m/%d/%Y%H:%M:%S')
        df["datetime"] = df["datetime"].dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        df["symbol"] = "SPY"
        
        return df[["symbol", "datetime", "price", "bid", "ask", "volume"]]
    
    @classmethod
    def initStockTable(cls):
        sql = """CREATE TABLE IF NOT EXISTS stock_tick_data(
            symbol CHAR(8),       -- 股票代號，固定 8 字元長度
            datetime TIMESTAMP,            -- 時間
            price DECIMAL(18, 4),  -- 開盤價，精確到小數點後 4 位
            bid DECIMAL(18, 4),  -- 最高價，精確到小數點後 4 位
            ask DECIMAL(18, 4),   -- 最低價，精確到小數點後 4 位
            volume INT,            -- 成交量，整數類型
            PRIMARY KEY (datetime, symbol)    -- 主鍵：日期 + 股票代碼，避免重複數據
        );"""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        release_connection(conn)


def create_stock_table():
    try:
        print("creating stock table.")
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(SingleStock.create_table_sql)
        conn.commit()
        print("success creating stock table.")
        release_connection(conn)
    except Exception as e:
        print(f"Error creating table: {e}.")


def renew_stock():
    start_date = "2020-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    stockCrawler = SingleStock(start_date=start_date, end_date=end_date, stock_id="SPY")
    df = stockCrawler.get_stock_data()
    stockCrawler.write_to_sql(df)


def renew_data():
    start_date = "2020-01-01"
    # end_date = "2024-12-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    base_path = "./data/storage"
    base = "USDT"

    # Taiwan Stock data
    stockCrawler = SingleStock(start_date=start_date, end_date=end_date, stock_id="AAPL")
    df = stockCrawler.get_stock_data()
    print(df)
    stockCrawler.write_to_sql(df, )
    crawler = ShareDistribution()
    crawler.Get(["2330",])
    
    # Base Data
    print(f'Spot Data:')
    binance_data = SpotData(symbol=Symbols.BTC.value + base, start_date=start_date, end_date=end_date)
    match_index = binance_data.Get().index
    
    for symbol in Symbols:
        pair = symbol.value + base
        print(f'Pair: {symbol.value}/{base}')

        binance_data = SpotData(symbol=pair, start_date=start_date, end_date=end_date)
        base_df = binance_data.Get()
        if (not base_df.index.equals(match_index)):
            print("%s does not have enough data." % (pair))
            continue
        base_df.to_csv(f"{base_path}/spot/{pair}.csv")
        
    

    symbols = []
    symbols.append(Symbols.BTC)
    symbols.append(Symbols.ETH)

    # Funding Rate Data (BTC, ETH)
    print(f'\nFunding Rate Data:')
    for symbol in symbols:
        pair = symbol.value + base
        print(f'Pair: {symbol.value}/{base}')

        funding_rate = FundingRate(symbol=pair, start_date=start_date, end_date=end_date)
        funding_rate_df = funding_rate.Get()
        funding_rate_df.to_csv(f"{base_path}/funding-rate/{pair}.csv")

    # Future Data (BTC, ETH)
    print(f'\nFuture Data:')
    for symbol in symbols:
        pair = f'{symbol.value}/{base}:{base}'
        print(f'Pair: {pair}')

        future_data = FutureData(symbol=pair, start_date=start_date, end_date=end_date)
        future_df = future_data.Get()
        future_df.to_csv(f"{base_path}/future/{symbol.value + base}.csv")

    # Long Short Ratio Data (BTC, ETH) 30 days only
    print(f'\nLong Short Ratio Data:')
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
    print(f'\nOpen Interest Data:')
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

    # Fear & Greed Index Data
    print(f'\nFear & Greed Index Data')
    fear_greed_index = FearGreedIndex()
    fear_greed_index_df = fear_greed_index.Get()
    fear_greed_index_df.to_csv(f"{base_path}/fear-greed-index/fear-greed-index.csv")
    
    # Stable Coin Data
    print(f'\nStable Coin Data')
    stable_coin = StableCoin()
    stable_coin_df = stable_coin.Get()
    stable_coin_df.to_csv(f"{base_path}/stablecoin/stablecoin.csv")

if __name__ == "__main__":
    StockTickData.initStockTable()
    stockTickDataCrawler = StockTickData(dtype="raw")
    stockTickDataCrawler.Get()

    # create_stock_table()
    # renew_stock()
