import torch
from pytorch3d.ops import sample_farthest_points

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())


points = torch.rand((1, 1000, 3), device='cuda')
_, idx = sample_farthest_points(points, K=100)
print("Success! Sampled indices:", idx.shape)