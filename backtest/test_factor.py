from factor.factors import Factor
from factor.signal.Cross import Cross
from data.loader import Loader
from config.load import load_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib, tqdm, factor, argparse, os
import loggings as log
from backtest.utils import FactorTest, test_factor_signal_report
from typing import List
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

_debug_flag = False

log.add_file_handler("logs/test-factor.log")
if _debug_flag:
    log.set_level(log.DEBUG)
else:
    log.set_level(log.INFO)
_logger = log.get_logger(name=__name__)

__all__ = ["FactorRunner"]

config = load_config()
TARGET = config.get("TARGET")
DELAY = config.get("DELAY")
HOURS = config.get("HOURS")

start_date = config.get("test_factor").get("start_date")
end_date = config.get("test_factor").get("end_date")

class FactorRunner:
    def __init__(self ,X=None, y=None, target=None, mode="default"):
        if mode == "default":
            train, test = Loader.make_not_overlap(target=TARGET, delay=DELAY, hours=HOURS, dtype="continuous")
            self.X = train
            self.y = test
        else:
            self.X = X
            self.y = y
            
        self.start_date = self.y.index[0]
        self.end_date = self.y.index[-1]
        self.target = target
        self.__save = True
    
    def run1taclass(self, factor_class:Factor, rerun=True, plot=False):
        signal_results = pd.DataFrame()
        _logger.info("Run on a signal class: %s", str(factor_class))
        for signal in factor_class.load_signals():
            _logger.debug("Current signal: %s, loading target: %s", signal, self.target)
            signal.load_target(self.target)
            
            padding = 30
            signal_output = None
            if rerun or (not signal.output_exist):
                _logger.info("Running signal %s due to %s.", signal.__str__(), "rerun" if rerun else "output not exist")
                signal_output = signal.GenAll(self.X)
                signal.delete_exist_result()
                signal.make_result_dir()
            else:
                _logger.debug("Loading signal output.")
                signal_output = signal.load_signal_output()
                
            _logger.debug("%s %s", "    start date:".ljust(padding), signal_output.index[0])
            _logger.debug("%s %s", "    end date:".ljust(padding), signal_output.index[-1])
            _logger.debug("%s %s", "    cases:".ljust(padding), signal_output.index.__len__())
            signal_output = signal_output.loc[self.start_date: self.end_date]
            _logger.debug("%s %s", "    filtered start date:".ljust(padding), signal_output.index[0])
            _logger.debug("%s %s", "    filtered end date:".ljust(padding), signal_output.index[-1])
            _logger.debug("%s %s", "    filtered cases:".ljust(padding), signal_output.index.__len__())

            signal_results[signal.__str__()] = signal_output
            if plot: self.__save_plots(signal, signal_output)
            
            if rerun or (not signal.output_exist):
                signal.save_signal_output(signal_output)
        
        return signal_results
  
    def __save_plots(self, signal:Factor, signal_output):
        distribution_path = os.path.join(signal.result_path, "distribution.png")
        plt.hist(signal_output, bins=100)
        plt.savefig(distribution_path)
        plt.close()
        
        ts_plot_path = os.path.join(signal.result_path, "ts_plot.png")
        plt.plot(signal_output, label=signal.__str__())
        plt.savefig(ts_plot_path)
        plt.close()



    def run1class(self, fctr:Factor, rerun=True, plot=False, show=False):
        signal_results = self.run1taclass(fctr, rerun=rerun, plot=plot)
        # train by fixed interval, setting.yml
        STEP = HOURS//4
        signal_results = signal_results.loc[start_date:end_date:STEP]
        test = self.y.loc[start_date:end_date:STEP]
        
        scoring = test_factor_signal_report(signal_results=signal_results, test=test)
        result_class_path:str = fctr.result_class_path(target=TARGET)
        scoring.to_csv(os.path.join(result_class_path, "report.csv"))
        signal_results.corr().to_csv(os.path.join(result_class_path, "corr.csv"))
        
        if show:
            print("factor results:")
            print(scoring)
            print("correlation matrix:")
            print(signal_results.corr())
    


def main():
    parser = argparse.ArgumentParser(description="Run factor tests with given parameters.")
    parser.add_argument("factor_name", type=str, help="Name of the factor to test")
    parser.add_argument("-r", "--rerun", action="store_true", help="Flag to rerun the tests")  # 新增 -r 參數
    args = parser.parse_args()
    
    if args.factor_name == "all":
        _logger.info("Running tests for all factors.")
        factors_cls:List[Factor] = [
            factor.weilun01,
            factor.weilun02,
            factor.weilun03,
            factor.weilun04,
            factor.weilunta,
            factor.FundingRate,
            factor.MarketParticipation,
            factor.Momentum,
            factor.Skewness,
            factor.Slope,
            factor.SpotFutureSpread,
            factor.VolatilityCrossMarket,
            factor.VolumeAnomaly,
            # factor.StablecoinFlow,
            
        ]
        runner = FactorRunner(target=TARGET, mode="default")
        all_signals:List[Factor] = []
        
        for fctr_cls in factors_cls:
            if args.rerun: runner.run1class(fctr_cls, rerun=True, plot=True, show=False)
            factors = fctr_cls.load_signals()
            [fctr.load_target(TARGET) for fctr in factors]
            all_signals.extend(factors)
        
        all_output = pd.concat([fctr_cls.load_signal_output() for fctr_cls in all_signals], axis=1)
        all_report = test_factor_signal_report(signal_results=all_output, test=runner.y)
        
        print(all_report)
            
    else:
        _logger.info("Running tests for factor: {}".format(args.factor_name))
        factor_name = args.factor_name
        runner = FactorRunner(target=TARGET, mode="default")
        
        module = importlib.import_module(f'factor.signal.{factor_name}')
        fctr: Factor = getattr(module, factor_name)
        runner.run1class(fctr, rerun=True, plot=True, show=True)
        






if __name__ == "__main__":
    main()
