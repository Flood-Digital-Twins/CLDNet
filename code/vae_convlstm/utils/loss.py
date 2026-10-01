import torch 
import torch.nn as nn

class RelativeL2Loss(nn.Module):
    def __init__(self, eps=1e-10, reduction='sum'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def l2_norm(self, x):
        if x.dim() == 2:
            return torch.sum(x**2, dim=1)**(1/2)
        elif x.dim() == 3:
            return torch.sum(x**2, dim=(1,2))**(1/2)
        elif x.dim() == 4:
            return torch.sum(x**2, dim=(1,2,3))**(1/2)
        else:
            raise ValueError('Unsupported tensor shape')

    def forward(self, outputs, labels):
        diff = outputs - labels
        loss = self.l2_norm(diff) / (self.l2_norm(labels) + self.eps)
        if self.reduction == 'sum':
            return torch.sum(loss, dim=0)
        else:
            return torch.mean(loss, dim=0)
        
        
        