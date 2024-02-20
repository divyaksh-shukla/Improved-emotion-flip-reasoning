import torch
import os
import sys
import yaml

def load_config():
    return yaml.safe_load(open('config.yaml', 'r'))

CONFIG = load_config()

def save_model(model, model_name=CONFIG['model_name']):
    torch.save({
        'model': model,
        'model_state_dict': model.state_dict(),
    }, os.path.join(CONFIG['model_dir'], f"{model_name}.{CONFIG['model_ext']}"))
    
def load_model(model, model_name=CONFIG['model_name']):
    return torch.load(os.path.join(CONFIG['model_dir'], f"{model_name}.{CONFIG['model_ext']}"))

def save_model_and_exit(model, model_name=CONFIG['model_name']):
    save_model(model, model_name=model_name)
    sys.exit(1)
    
class ContainsNansError(Exception):
    pass