from data.db import *
import pandas as pd

def get_spy():
    conn = get_connection()
    sql = """
    select * from stock_prices where stock_id = 'SPY';
    """
    out = pd.read_sql_query(sql, conn)
    # out.reset_index()
    print(out)
    exit()
    out = out[["Adj Close", "Volume"]]
    out.columns = ["price", "volume"]
    return out
    
    
    

def get_volume_bars(df, volume_threshold):
    """
    生成 Volume Bars DataFrame。
    
    Parameters:
        df (pd.DataFrame): 包含交易數據的 DataFrame，需包含以下列：
            - 'timestamp': 時間戳
            - 'price': 成交價格
            - 'volume': 成交量
        volume_threshold (int): 每個 Volume Bar 的累計成交量閾值。
    
    Returns:
        pd.DataFrame: Volume Bars DataFrame，包含以下列：
            - 'timestamp': 該 Volume Bar 結束的時間戳
            - 'price_open': 該 Volume Bar 的開盤價
            - 'price_high': 該 Volume Bar 的最高價
            - 'price_low': 該 Volume Bar 的最低價
            - 'price_close': 該 Volume Bar 的收盤價
            - 'volume': 該 Volume Bar 的總成交量
    """
    # 初始化結果列表
    volume_bars = []
    
    # 初始化累計變數
    cumulative_volume = 0
    price_open, price_high, price_low = None, None, None
    volume_accumulated = 0
    
    # 遍歷原始數據
    for idx, row in df.iterrows():
        price = row['price']
        volume = row['volume']
        timestamp = row['timestamp']
        
        if price_open is None:
            price_open = price
        price_high = max(price_high, price) if price_high is not None else price
        price_low = min(price_low, price) if price_low is not None else price
        
        cumulative_volume += volume
        volume_accumulated += volume
        
        # 如果累計成交量超過閾值，生成一個 Volume Bar
        if cumulative_volume >= volume_threshold:
            volume_bars.append({
                'timestamp': timestamp,
                'price_open': price_open,
                'price_high': price_high,
                'price_low': price_low,
                'price_close': price,
                'volume': volume_accumulated
            })
            
            # 重置累計變數
            cumulative_volume = 0
            price_open, price_high, price_low = None, None, None
            volume_accumulated = 0
    
    # 將結果轉換為 DataFrame
    volume_bars_df = pd.DataFrame(volume_bars)
    return volume_bars_df

# # 範例數據
# data = {
#     'timestamp': pd.date_range('2023-01-01 09:00:00', periods=10, freq='T'),
#     'price': [100, 101, 102, 101, 100, 99, 100, 101, 102, 103],
#     'volume': [300, 500, 700, 200, 600, 100, 400, 300, 800, 500]
# }
# df = pd.DataFrame(data)

# # 使用範例
# volume_threshold = 1000
# volume_bars_df = get_volume_bars(df, volume_threshold)
# print(volume_bars_df)

    
close = get_spy()