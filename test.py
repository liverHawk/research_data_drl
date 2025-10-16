import torch
import os

# Try each GPU individually
print(f'Testing {torch.cuda.device_count()} GPUs...')
for i in range(torch.cuda.device_count()):
    try:
        device = torch.device(f'cuda:{i}')
        print(f'Testing GPU {i}...')
        x = torch.tensor([1.0], device=device)
        print(f'✓ GPU {i} works!')
        x = None
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'✗ GPU {i} failed: {e}')
        continue