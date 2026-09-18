import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.RFF import RFF

class Model(nn.Module):
     def __init__(self, args):
          super(Model, self).__init__()
          self.N = args.input_dim
          self.L = args.seq_len
          self.D = args.D
          self.NL = self.N * self.L
          self.NLD = self.NL * self.D
          self.topk =args.topk

          self.rff = RFF(self.N, self.L, self.D, sigma=args.sigma, seed=None, device=args.gpu)
          self.linear = nn.Linear(self.NLD, self.N, bias=False)

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
          x = x.reshape(x.size(0), -1)
          phi = self.rff(x)

          if self.mask is not None:
               # mask (N, NL) ->  (N, NL * D)
               mask_full = self.mask.unsqueeze(-1).repeat(1, 1, self.D).view(self.N, -1)
               W_masked = self.linear.weight * mask_full
               return F.linear(phi, W_masked)
          res = self.linear(phi)
          
          return self.linear(phi).unsqueeze(1)
     
     @torch.no_grad() 
     def topk_gradient_mask(self):
          if self.linear.weight.grad is None:
               return 
          
          new_grad = torch.zeros_like(self.linear.weight.grad)
          grad = self.linear.weight.grad
          grad_reshaped = grad.view(self.N, self.NL, self.D)
          new_grad_reshaped = new_grad.view(self.N, self.NL, self.D)

          for i in range(self.N):
               group_norms = torch.norm(grad_reshaped[i], dim=1) # (NL, )
               _, topk_idx = torch.topk(group_norms, min(self.topk, self.NL), sorted=False)
               new_grad_reshaped[i, topk_idx] = grad_reshaped[i, topk_idx]
          
          self.linear.weight.grad = new_grad

     def get_causal_matrix(self):
          W = self.linear.weight.data # (N, NL*D)
          W_reshaped = W.view(self.N, self.NL, self.D)
          S = torch.norm(W_reshaped, dim=2) # (N, NL)
          if self.mask is not None:
               S = S * self.mask
          return S
