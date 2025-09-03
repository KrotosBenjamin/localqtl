import torch

def _prepare_tensor(x, dtype=torch.float32, device='cpu'):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device)
