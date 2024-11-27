from factor.factors import Factor
from factor.signal.Cross import Cross
from data.loader import Loader
from config.load import load_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib, tqdm, factor, argparse, os
import loggings as log
from backtest.utils import FactorTest
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

_debug_flag = False

log.add_file_handler("logs/test-factor.log")
if _debug_flag:
    log.set_level(log.DEBUG)
else:
    log.set_level(log.INFO)
_logger = log.get_logger(name=__name__)

__all__ = ["FactorRunner", "test1factor_by_class", "test1factor_by_name"]

config = load_config()
TARGET = config.get("TARGET")
DELAY = config.get("DELAY")
HOURS = config.get("HOURS")

start_date = config.get("test_factor").get("start_date")
end_date = config.get("test_factor").get("end_date")

def make_signal_report(signal_results, test):
    scoring = pd.DataFrame(index=signal_results.columns)
    keys = signal_results.columns

    scoring["correlation"] = [FactorTest.correlation(signal_results[k], test) for k in keys]
    scoring["correlation_stable"] = [FactorTest.correlation_stable(signal_results[k], test)  for k in keys]
    scoring["accuracy"] = [FactorTest.accuracy(signal_results[index], test)  for index in keys]
    scoring["mean"] = [FactorTest.mean(signal_results[index])  for index in keys]
    scoring["std"] = [FactorTest.std(signal_results[index])  for index in keys]
    scoring["skew"] = [FactorTest.skewness(signal_results[index])  for index in keys]
    scoring["zeros"] = [FactorTest.zero_percent(signal_results[index])  for index in keys]
    
    return scoring

class FactorRunner:
    def __init__(self, X, y, target=None):
        self.X = X
        self.y = y
        self.start_date = y.index[0]
        self.end_date = y.index[-1]
        self.target = target
        self.__save = True
    
    def run1factor(self, factor:Factor):
        df = pd.Series([factor.Gen(row) for row in self.X], index=self.y.index, name=str(factor))
        if factor.exist:
            factor.delete_exist_result()
        factor.make_result_dir()
        if self.__save:
            factor.save_signal_output(df)
        return df
    
    def run1class(self, factor_class:Factor, rerun=True, plot=False):
        signal_results = pd.DataFrame()
        _logger.info("Run on a signal class: %s", str(factor_class))
        for signal in factor_class.load_signals():
            _logger.debug("Current signal: %s, loading target: %s", signal, self.target)
            signal.load_target(self.target)
            
            padding = 30
            signal_output = None
            if rerun or (not signal.output_exist):
                _logger.info("Running signal %s due to %s.", signal.__str__(), "rerun" if rerun else "output not exist")
                signal_output = pd.Series([signal.Gen(row) for row in tqdm.tqdm(self.X, "one factor")], index=self.y.index, name=str(signal))
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

# def test1factor_by_class(factor: Factor, rerun=True, plot=False):
#     train, test = Loader.make_not_overlap(target=TARGET, delay=DELAY, hours=HOURS)
#     factor_runner = FactorRunner(train, test, target=TARGET)
#     index = test.index
    
#     _logger.info("Start time: {}. End time: {}.".format(index[0], index[-1]))
#     _logger.info("Number of cases: {}".format(len(train)))
#     _logger.info("Predict interval: {} hours".format(HOURS))
    
#     signal_results = factor_runner.run1class(factor, rerun=rerun, plot=plot)

#     # train by fixed interval, setting.yml
#     signal_results = signal_results.loc[start_date:end_date]
#     test = test.loc[start_date:end_date]
    
#     _logger.info("Report generate on:")
#     _logger.info("      Start time: %s", signal_results.index[0])
#     _logger.info("      End time: %s", signal_results.index[-1])
#     scoring = make_signal_report(signal_results=signal_results, test=test)
#     result_class_path:str = factor.result_class_path(target=TARGET)
#     scoring.to_csv(os.path.join(result_class_path, "report.csv"))
#     signal_results.corr().to_csv(os.path.join(result_class_path, "corr.csv"))
    
    
#     print("factor results:")
#     print(scoring)
#     print("correlation matrix:")
#     print(signal_results.corr())

# def test1factor_by_name(factor_name, rerun=True, plot=False):
#     module = importlib.import_module(f'factor.signal.{factor_name}')
#     BaseClass: Factor = getattr(module, factor_name)
#     test1factor_by_class(BaseClass, rerun=rerun, plot=plot)
    
def test1tafactor_by_class(factor: Factor, rerun=True, plot=False):
    train, test = Loader.make_not_overlap(target=TARGET, delay=DELAY, hours=HOURS, dtype="continuous")
    factor_runner = FactorRunner(train, test, target=TARGET)
    index = test.index
    
    signal_results = factor_runner.run1taclass(factor, rerun=rerun, plot=plot)
    # train by fixed interval, setting.yml
    signal_results = signal_results.loc[start_date:end_date]
    test = test.loc[start_date:end_date]
    
    scoring = make_signal_report(signal_results=signal_results, test=test)
    result_class_path:str = factor.result_class_path(target=TARGET)
    scoring.to_csv(os.path.join(result_class_path, "report.csv"))
    signal_results.corr().to_csv(os.path.join(result_class_path, "corr.csv"))
    
    
    print("factor results:")
    print(scoring)
    print("correlation matrix:")
    print(signal_results.corr())
    
def test1tafactor_by_name(factor_name, rerun=True, plot=False):
    module = importlib.import_module(f'factor.signal.{factor_name}')
    BaseClass: Factor = getattr(module, factor_name)
    test1tafactor_by_class(BaseClass, rerun=rerun, plot=plot)
    

# def show_all():
#     factor_cls_list = [
#         factor.weilun01,
#         factor.weilun02,
#         factor.weilun03,
#         factor.weilun04,
#         factor.weilun05,
#         factor.weilun06,
#         factor.weilun07,
#         factor.weilun08,
#         factor.weilun09,
#         factor.Cross,
#         factor.CrossRSI,
#         factor.Slope,
#         factor.Skewness,
#     ]
    
#     # train, test = Loader.make_not_overlap(target=TARGET, delay=DELAY, hours=HOURS)
#     train, test = Loader.make_not_overlap(target=TARGET, delay=DELAY, hours=HOURS, dtype="continuous")
    
#     factor_runner = FactorRunner(X=train, y=test, target=TARGET)
#     scoring_df = pd.DataFrame()
#     for factor_cls in factor_cls_list:
#         signal_results = factor_runner.run1class(factor_cls, rerun=False, plot=False)
#         scoring = make_signal_report(signal_results=signal_results, test=test)
#         result_class_path:str = factor_cls.result_class_path(target=TARGET)
#         scoring.to_csv(os.path.join(result_class_path, "report.csv"))
#         signal_results.corr().to_csv(os.path.join(result_class_path, "corr.csv"))
#         scoring_df = pd.concat([scoring_df, scoring])
        
#     pd.set_option('display.max_rows', 100)
#     print(scoring_df)




def main():
    parser = argparse.ArgumentParser(description="Run factor tests with given parameters.")
    parser.add_argument("factor_name", type=str, help="Name of the factor to test")
    args = parser.parse_args()
    
    if args.factor_name == "all":
        _logger.info("Running tests for all factors.")
        # show_all()
        
    else:
        _logger.info("Running tests for factor: {}".format(args.factor_name))
        test1tafactor_by_name(args.factor_name, rerun=True, plot=True)

if __name__ == "__main__":
    main()
