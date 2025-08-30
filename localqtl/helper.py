import torch

def _prepare_tensor(df, dtype=torch.float32, device='cpu'):
    return torch.tensor(df, dtype=dtype).to(device)
