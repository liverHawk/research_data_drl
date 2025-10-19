import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# Try each GPU individually
for i in range(4):
    try:
        device = torch.device(f'cuda:{i}')
        print(f'Testing GPU {i}...')
        # torch.cuda.set_device(device)

        # get GPU information (ex: temperature, utilization, memory)
        temperature = torch.cuda.temperature(i)
        utilization = torch.cuda.utilization(i)
        # memory = torch.cuda.memory_stats(i)
        # str_memory = ', '.join([f'{k}: {v}' for k, v in memory.items()])
        print(f'GPU {i} - Temperature: {temperature}°C, Utilization: {utilization}%')

        x = torch.tensor([1.0], device=device)
        print(f'✓ GPU {i} works!')
        x = None
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'✗ GPU {i} failed: {e}')
        continue