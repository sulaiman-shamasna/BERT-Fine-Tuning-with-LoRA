import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

def create_dataloader(seq: torch.Tensor, mask: torch.Tensor, y: torch.Tensor, batch_size: int = 32) -> DataLoader:
    data = TensorDataset(seq, mask, y)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader