import pandas as pd
from abc import ABC
import os, shutil, json, importlib



class Factor(ABC):
    def __init__(self):
        self.target__ = None
        
    def Gen(self, cls, x:pd.Series) -> float: ...
    
    def GenAll(self, cls, x:pd.Series) -> pd.Series: ...
    
    def __str__(self):
        return "not defined ..."
    
    def load_target(self, target):
        self.target__ = target
        
    @classmethod
    def load_signal_config(cls):
        module_dir = os.path.dirname(__file__)
        path = os.path.join(module_dir, "cfg", cls.__name__ + ".json")
        f = open(path)
        return json.load(f)
    
    @classmethod
    def find_signal_class(cls, factor_name:str):
        module = importlib.import_module(f'factor.signal.{factor_name}')
        signal_class: Factor = getattr(module, factor_name)
        return signal_class
    
    @classmethod
    def load_signals(cls):
        signal_config = cls.load_signal_config()
        return [cls(**config["argument"]) for config in signal_config["signal_list"]]
    
    @classmethod
    def result_class_path(cls, target):
        module_dir = os.path.dirname(__file__)
        return os.path.join(module_dir, "result", target, cls.__name__)
    
    @property
    def result_path(self):
        assert(self.target__ != None), "target is not defined."
        module_dir = os.path.dirname(__file__)
        return os.path.join(module_dir, "result", self.target__, self.__class__.__name__,str(self))
    
    @property
    def signal_output_path(self):
        return os.path.join(self.result_path, "signal_output.csv")
    
    @property
    def output_exist(self):
        return os.path.exists(self.signal_output_path)
    
    @property
    def signal_config_path(self):
        module_dir = os.path.dirname(__file__)
        return os.path.join(module_dir, "cfg", self.__class__.__name__, str(self))
    
    @property
    def exist(self):
        return os.path.exists(self.result_path)
    
    def delete_exist_result(self) -> None:
        try:
            shutil.rmtree(self.result_path)
        except:
            pass
    
    def save_signal_output(self, output:pd.Series):
        output.to_csv(self.signal_output_path)
    
    def save_plot(self, x, y):
        pass
        
    def load_signal_output(self):
        result = pd.read_csv(self.signal_output_path, index_col=0, parse_dates=True).iloc[:,0]
        result.name = str(self)
        return result
    
    def make_result_dir(self):
        os.makedirs(self.result_path)
    
    def __repr__(self):
        return self.__str__()
        
    

