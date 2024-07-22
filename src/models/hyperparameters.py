import yaml
import os

configs_path = os.path.join('configurations', 'config.yaml')

with open(configs_path, "r") as config_file:
    config = yaml.safe_load(config_file)

BERT_MODEL_NAME = config['bert_model_name']
DATA_PATH = config['data_path']
TRAINED_MODEL_PATH = config['trained_model_path']
SAVE_MODEL_PATH = config['model_save_path']
DEVICE = config['device']

MAX_LENGTH = int(config['max_len'])
BATCH_SIZE = int(config['batch_size'])
EPOCHS = int(config['epochs'])
LEARNING_RATE = float(config['learning_rate'])
LORA_R = int(config['lora_r'])
LORA_ALPHA = int(config['lora_alpha'])
LORA_DROPOUT = float(config['lora_dropout'])
