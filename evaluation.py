import torch
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from src.models.model import load_model
from src.data.features import load_data, preprocess_data
from src.models.hyperparameters import config
from transformers import BertTokenizerFast

def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len = config['max_len']

    df = load_data(config['data_path'])
    tokenizer = BertTokenizerFast.from_pretrained(config['bert_model_name'])
    _, _, _, test_seq, test_mask, test_y = preprocess_data(df, tokenizer, max_len)

    model = load_model(config['trained_model_path'], device, config['bert_model_name'])

    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()

    preds = np.argmax(preds, axis=1)

    print(classification_report(test_y, preds))
    print(pd.crosstab(test_y, preds))

if __name__ == "__main__":
    main()
