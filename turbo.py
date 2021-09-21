import torch
import time

def gpu_turbo():
    a = torch.zeros([1])
    a = a.cuda()
    while True:
        if int(time.time()) % 2 == 0:
            b = torch.abs(a)

gpu_turbo()
