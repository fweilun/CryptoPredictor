from factor.factors import Factor
import pandas as pd
import numpy as np

class bang03(Factor):
    need = ["funding_rate", "close"]
    
    def __init__(self, periods: list, dpo_n: int, displacement: int, threshold: float = 1.0):
        """
        参数:
            periods: 用于计算资金费率指标的周期列表
            dpo_n: DPO 计算中的移动平均周期数
            displacement: DPO 计算中的偏移周期数
            threshold: 信号生成的灵敏度阈值
        """
        self.periods = periods
        self.dpo_n = dpo_n
        self.displacement = displacement
        self.threshold = threshold

    def calculate_dpo(self, close: pd.Series, n: int, displacement: int):
        moving_average = close.rolling(window=n).mean()
        displaced_moving_average = moving_average.shift(displacement)
        dpo = close - displaced_moving_average
        return dpo

    def Gen(self, x: pd.DataFrame):
        for col in self.need:
            assert col in x.columns, f"{col} 不存在"
        
        funding_rate = x["funding_rate"]
        close = x["close"]
        
        # 计算资金费率特征
        funding_metrics = []
        for period in self.periods:
            mean_rate = funding_rate.rolling(window=period).mean().iloc[-1]
            std_rate = funding_rate.rolling(window=period).std().iloc[-1]
            recent_rate = funding_rate.iloc[-1]
            normalized_score = (recent_rate - mean_rate) / (std_rate + 1e-5)
            funding_metrics.append(normalized_score)
        
        avg_funding_score = float(np.mean(funding_metrics))
        
        # 计算 DPO 指标
        dpo_series = self.calculate_dpo(close, self.dpo_n, self.displacement)
        dpo_value = dpo_series.iloc[-1]
        # 归一化 DPO 信号（根据需要调整归一化方法）
        dpo_signal = dpo_value / (close.std() + 1e-5)
        
        # 总分数
        combined_score = avg_funding_score + dpo_signal
        
        # 判断信号：涨是 1，跌是 -1
        if combined_score > self.threshold:
            return 1
        elif combined_score < -self.threshold:
            return -1
        return 0

    def GenAll(self, x: pd.DataFrame):
    # 遍历所需的列，确保每个必要的列都存在于数据框中
        for col in self.need:
            assert col in x.columns, f"{col} 不存在"

        # 提取资金费率和收盘价的数据列
        funding_rate = x["funding_rate"]
        close = x["close"]

        # 初始化一个信号序列，索引与输入数据相同，数据类型为整数
        signals = pd.Series(index=x.index, dtype=int)

        # 确定循环的起始点，避免在数据不足时进行计算
        start_index = max(self.periods + [self.dpo_n + self.displacement])

        # 从起始索引开始遍历数据，直到数据的末尾
        for i in range(start_index, len(x)):
        # 获取当前索引之前的所有数据切片
            slice_data = x.iloc[:i]
            slice_funding_rate = slice_data["funding_rate"]
            slice_close = slice_data["close"]

            # 初始化资金费率指标的列表
            funding_metrics = []

            # 遍历每个指定的周期，计算资金费率的标准化得分
            for period in self.periods:
                if i >= period:
                    # 计算过去 'period' 个周期的平均资金费率
                    mean_rate = slice_funding_rate.iloc[-period:].mean()
                    # 计算过去 'period' 个周期的资金费率标准差
                    std_rate = slice_funding_rate.iloc[-period:].std()
                    # 获取最近一个周期的资金费率
                    recent_rate = slice_funding_rate.iloc[-1]
                    # 计算标准化得分，避免除以零
                    normalized_score = (recent_rate - mean_rate) / (std_rate + 1e-5)
                    funding_metrics.append(normalized_score)
                else:
                    # 如果数据不足指定周期，得分设为0
                    funding_metrics.append(0)

            # 计算所有周期的资金费率得分的平均值
            avg_funding_score = float(np.mean(funding_metrics))

            # 计算DPO（Detrended Price Oscillator）指标序列
            dpo_series = self.calculate_dpo(slice_close, self.dpo_n, self.displacement)

            if len(dpo_series.dropna()) > 0:
                # 获取最新的DPO值
                dpo_value = dpo_series.iloc[-1]
                # 标准化DPO值，避免除以零
                dpo_signal = dpo_value / (slice_close.std() + 1e-5)
            else:
                # 如果DPO序列为空，信号设为0
                dpo_signal = 0

            # 组合资金费率得分和DPO信号
            combined_score = avg_funding_score + dpo_signal

            # 根据组合得分判断买卖信号
            if combined_score > self.threshold:
                signals.iloc[i] = 1  # 买入信号
            elif combined_score < -self.threshold:
                signals.iloc[i] = -1  # 卖出信号
            else:
                signals.iloc[i] = 0  # 持有信号

        # 将初始部分的信号设为0，避免因数据不足导致的错误信号
        signals.iloc[:start_index] = 0
        return signals

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{'_'.join(map(str, self.periods))}_{self.threshold}_DPO_{self.dpo_n}_{self.displacement}"