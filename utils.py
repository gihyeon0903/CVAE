import torch

def ELBO_loss(y, y_, mean, stddev):
  # y : x_reconst y_ : x(label)
  marginal_likelihood = torch.sum(y_*torch.log(y) + (1-y_)*torch.log(1-y), dim=1)
  KL_divergence = 0.5 * torch.sum(torch.square(mean) + torch.square(stddev) - torch.log(1e-8 + torch.square(stddev)) - 1, dim=1)
  
  marginal_likelihood = torch.mean(marginal_likelihood)
  KL_divergence       = torch.mean(KL_divergence)

  ELBO =  marginal_likelihood - KL_divergence # 최대화

  return -ELBO, (-marginal_likelihood, KL_divergence) # loss