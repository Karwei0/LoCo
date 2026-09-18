import numpy as np
import torch
import torch.nn as nn

class RFF(object):
     def __init__(self, N, L, D, sigma=1.0, seed=None, device='cuda:0'):
          """
          D: 每个滞后变量的(映射的）特征维度
          sigma: 高斯核带宽
          """
          self.N = N
          self.L = L
          self.D = D
          self.NL = N * L
          self.sigma = sigma

          if seed is not None:
               torch.manual_seed(seed)
          
          # ω_p ~ N(0, 1/sigma^2)
          self.omega = torch.randn(self.NL, self.D, device=device) / self.sigma

          # b_p ~ Uniform(0, 2π)
          self.bias = torch.rand(self.NL, self.D, device=device) * 2 * np.pi
     def __call__(self, x):
          """
          phi(x) = (batch, N*L*D)
          """
          B = x.shape[0]

          x_expanded = x.view(B, self.NL, 1) # B, NL, 1

          # omega: (NL, D) -> (1, NL, D)
          omega = self.omega.unsqueeze(0)
          bias = self.bias.unsqueeze(0)

          # ω * z + b: (B, NL, D)
          arg = x_expanded * omega + bias
          phi = torch.cos(arg) * np.sqrt(2. / self.D)

          return phi.view(B, -1)