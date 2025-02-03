import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("\nNo CUDA GPU available. Using CPU.")
    print("\nCPU information:")
    print(f"Number of CPU cores: {torch.get_num_threads()}")
