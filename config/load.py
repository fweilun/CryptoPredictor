import yaml, os

def load_config():
    file = os.path.dirname(__file__)
    file_path = os.path.join(file, "setting.yml")
    config = None
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if not config:
        raise FileNotFoundError("setting.yml not found")
    
    return config