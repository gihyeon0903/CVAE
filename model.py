import torch
import torch.nn as nn

class gaussian_MLP_encoder(nn.Module):
    def __init__(self, dim_z=2, n_hidden=500, keep_prob=0.5):
      super(gaussian_MLP_encoder, self).__init__()

      self.dim_z = dim_z
      self.softplus = nn.Softplus()

      self.fc1 = nn.Sequential(
          nn.Linear(784 + 10, n_hidden),
          nn.ELU(),
          nn.Dropout(p=keep_prob)
      )

      self.fc2 = nn.Sequential(
          nn.Linear(n_hidden, n_hidden),
          nn.Tanh(),
          nn.Dropout(p=keep_prob)
      )

      self.output_layer = nn.Sequential(
          nn.Linear(n_hidden, dim_z*2),
      )

      for m in self.modules():
        if isinstance(m, nn.Linear):
          torch.nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
      x = self.fc1(x)
      x = self.fc2(x)
      gaussain_params = self.output_layer(x)

      mean   = gaussain_params[:, :self.dim_z]
      stddev = 1e-6 + self.softplus(gaussain_params[:, self.dim_z:])

      return mean, stddev

class bernoulli_MLP_decoder(nn.Module):
    def __init__(self, dim_z=2, n_hidden=500, keep_prob=0.5):
      super(bernoulli_MLP_decoder, self).__init__()

      self.fc1 = nn.Sequential(
          nn.Linear(dim_z + 10, n_hidden),
          nn.Tanh(),
          nn.Dropout(keep_prob)
      )

      self.fc2 = nn.Sequential(
          nn.Linear(n_hidden, n_hidden),
          nn.ELU(),
          nn.Dropout(keep_prob)
      )

      self.output_layer = nn.Sequential(
          nn.Linear(n_hidden, 784),
          nn.Sigmoid()
      )

      for m in self.modules():
        if isinstance(m, nn.Linear):
          torch.nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):

      x = self.fc1(x)
      x = self.fc2(x)
      out = self.output_layer(x)

      return out

class Conditional_Auto_Encoder(nn.Module):
  def __init__(self, dim_z=2, n_hidden=500, keep_prob=0.5, device='cpu'):
      super(Conditional_Auto_Encoder, self).__init__()
      self.device = device
      self.encoder = gaussian_MLP_encoder(dim_z, n_hidden, keep_prob).to(device)
      self.decoder = bernoulli_MLP_decoder(dim_z, n_hidden, keep_prob).to(device)

  def forward(self, x, y):
    xy_concat = torch.concatenate((x, y), dim=1)
    mean, stddev = self.encoder(xy_concat)
    
    z = mean + stddev * torch.normal(0, 1, (mean.size()[0], 1)).to(self.device)

    zy_concat = torch.concatenate((z, y), dim=1)
    y = self.decoder(zy_concat)
    return y, (mean, stddev), z    