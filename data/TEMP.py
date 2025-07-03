import numpy as np
import torch

t = torch.zeros(2,1,3,3)

print(t.shape)

m = t.mean((0), keepdims=True)
print(m)
print(m.shape)

n = t.mean((2,3))
print(n.shape, n)