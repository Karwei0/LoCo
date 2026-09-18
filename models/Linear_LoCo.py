import torch
import torch.nn as nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.N = args.input_dim
        self.L = args.seq_len
        self.NL = self.N * self.L
        self.linear = nn.Linear(self.NL, self.N, bias=False)
        self.topk = args.topk

        if args.mask is not None:
            self.register_buffer('mask', torch.tensor(args.mask, dtype=torch.float))
        else:
            self.mask = None
        
        self.init_weights()
    
    def init_weights(self, init_type='uniform', **kwargs):
        if init_type == 'uniform':
            a = kwargs.get('a', -0.00001)
            b = kwargs.get('b', 0.00001)
            nn.init.uniform_(self.linear.weight, a, b)
        elif init_type == 'normal':
            mean = kwargs.get('mean', 0.)
            std = kwargs.get('std', 0.02)
            nn.init.normal_(self.linear.weight, mean, std)
        else:
            raise NotImplementedError(f'init type {init_type} is not supported')
        
    def forward(self, x, *args):
        x = x.reshape(x.shape[0], -1)
        if self.mask is not None:
            W_masked = self.linear.weight * self.mask
            print(f'[debug] here')
            return F.linear(x, W_masked).unsqueeze(1)
        else:
            return self.linear(x).unsqueeze(1)
        
    @torch.no_grad()
    def topk_gradient_mask(self):
        if self.linear.weight.grad is None:
            return
        
        new_grad = torch.zeros_like(self.linear.weight.grad)

        # TODO: whether is ture or not
        for i in range(self.N):
            abs_grad = torch.abs(self.linear.weight.grad[i])
            _, topk_idx = torch.topk(abs_grad, min(self.topk, len(abs_grad)), stored=False)
            new_grad[i, topk_idx] = self.linear.weight.grad[i, topk_idx]
        
        self.linear.weight.grad = new_grad

    def get_causal_matrix(self):
        W = self.linear.weight.data.clone()
        if self.mask is not None:
               W = W * self.mask
        return torch.abs(W) # TODO: CHECK abs