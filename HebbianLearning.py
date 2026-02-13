import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianLearning(nn.Module):
    def __init__(self, in_features, out_features, lr=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.lr = lr

    def forward(self, x):
        y = torch.matmul(x, self.weight)
        return y
    
    def hebbian_update(self, x, y):
        delta_w = torch.einsum('bi,bj->ij', y, x) / x.size(0)
        self.weight += self.lr * delta_w