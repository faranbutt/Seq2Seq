import torch
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    print("GPU is available")
else:
    print("GPU is not available")