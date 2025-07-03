import numpy as np
import torch

t1 = torch.zeros(4, 3, 10, 3, 3)
t2 = torch.zeros(4, 3, 4, 3, 3)
u = torch.cat([t1, t2], dim=2)
#print(u.shape)

p = torch.tensor([[3088, 3088, 3088],
        [3088, 3136, 3143],
        [3088, 3088, 3143],
        [3088, 3112, 3143],
        [3080, 3085, 3090],
        [3090, 3135, 3145],
        [3085, 3090, 3145],
        [3090, 3110, 3145]], device='cuda:0')

print(p.shape)
p = p.type(torch.float64)
m = p.mean()
print(m.shape, m)
