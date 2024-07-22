import torch
from torch import nn
from transformers import AutoModel
from get_hyperparameters import BERT_MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT

class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: int, dropout: float):
        super(LoRALayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(in_features, r, bias=False)
        self.lora_b = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_b(self.lora_a(x))
        return x + self.dropout(lora_output) * self.scaling

class BERT(nn.Module):
    def __init__(self, bert_model_name: str = BERT_MODEL_NAME) -> None:
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name, return_dict=False)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = LoRALayer(768, 768, LORA_R, LORA_ALPHA, LORA_DROPOUT)
        self.fc2 = nn.Linear(768, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def load_model(model_path: str, device: torch.device, bert_model_name: str = BERT_MODEL_NAME) -> BERT:
    model = BERT(bert_model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model
