import torch
import torch.nn as nn

class Model(nn.Module):
     def __init__(self, args):
          super(Model, self).__init__()
          self.N = args.input_dim
          self.L = args.seq_len
          self.NL = self.N * self.L
          self.linear = nn.Linear(self.NL, self.N, bias=False)
          nn.init.xavier_uniform_(self.linear.weight)

     def forward(self, x, *args):
          x = x.reshape(x.shape[0], -1)
          return self.linear(x).unsqueeze(-1)
     
     def set_weights(self, coef):
          device = next(self.linear.parameters()).device
          with torch.no_grad():
               self.linear.weight.data = torch.tensor(coef, dtype=torch.float).to(device)
     def get_causal_matrix(self):
          return torch.abs(self.linear.weight.data)