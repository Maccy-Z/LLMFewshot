import sys
print("Python version:", sys.version)

import os
print("Current working directory:", os.getcwd())

import torch
print("Pytorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
x = torch.tensor(1, device='cuda')
