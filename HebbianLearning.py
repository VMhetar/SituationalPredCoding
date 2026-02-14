import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianLearning(nn.Module):
    def __init__(self, in_features, out_features, lr=0.01):
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(out_features, in_features)*0.1,
            requires_grad=False # no backpropagation
        )
        self.lr = lr
    
    def forward(self, x):
        y = x @ self.W.T

        # Hebbian Update (Oja's rule)
        hebb_term = y.T @ x  / x.size(0)
        y_mean = (y ** 2).mean(dim=0).unsqueeze(1)
        decay_term = y_mean * self.W  
        self.W.data += self.lr * (hebb_term - decay_term)

        return y