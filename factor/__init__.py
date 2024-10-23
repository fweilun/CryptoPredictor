import os, importlib, json
from .factors import Factor as BaseFactor

cfg = {}

module_dir = os.path.dirname(__file__)
json_dir = os.path.dirname(__file__)
cfg_dir = os.path.join(module_dir, 'cfg')
for filename in os.listdir(cfg_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(cfg_dir, filename)
        module_name = filename[:-5]
        with open(file_path, 'r') as file:
            cfg[module_name] = json.load(file)


signal_dir = os.path.join(module_dir, 'signal')
for filename in os.listdir(signal_dir):
    
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]
        module = importlib.import_module(f'.signal.{module_name}', package=__name__)
        setattr(module, 'a', cfg[module_name])
        globals()[module_name] = getattr(module, module_name)






