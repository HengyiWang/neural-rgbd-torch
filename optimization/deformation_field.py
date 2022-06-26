import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformationField(nn.Module):
    def __init__(self, D=6, W=128, input_ch=2, output_ch=2, skips=[3]):
        super(DeformationField, self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.skips = skips        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.output_linear = nn.Linear(W, output_ch)
        

    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        
        outputs = self.output_linear(h)
        return outputs