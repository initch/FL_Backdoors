import torch
from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader


@dataclass
class FLUser:
    user_id: int = 0
    compromised: bool = False
    n_samples: int = 0
    train_loader: DataLoader = None
