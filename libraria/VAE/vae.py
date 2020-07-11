import torch
from torch import nn, distributions, optim


class VAEBernoulli(nn.Module):

    def __init__(self, dim, input):
        super.__init__()
        self._latent_dim = dim
        self._input_dim = input.shape[1]

        self.fc_encode_mean = nn.Linear(self._input_dim, self._latent_dim)
        self.fc_encode_std = nn.Linear(self._input_dim, self._latent_dim)

        self.fc_decode_mean = nn.Linear(self._latent_dim, self._input_dim)

        self._prior = torch.Tensor([0.5 for _ in range(self._input_dim)])

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def input_dim(self):
        return self._input_dim

    def encode(self, x:torch.Tensor, return_parameters=False) -> torch.Tensor:
        mean = nn.ReLU(self.fc_encode_mean(x))
        std = nn.ReLU(self.fc_encode_std(x))
        if return_parameters:
            return mean, std
        return distributions.Normal(mean, std).sample()

    def decode(self, z: torch.Tensor, return_parameters=False) -> torch.Tensor:
        logits =nn.ReLU(self.fc_decode_mean(z))
        b = distributions.continuous_bernoulli.ContinuousBernoulli(logits=logits).sample()
        if return_parameters:
            return b
        return b.sample()

    @classmethod
    def divergence(cls, mean, sigma):
        return 0.5 * torch.sum((1 + torch.log(torch.square(sigma)) - torch.square(mean) - torch.square(sigma)))

    def loss_func(self, x):
        mean, std = self.encode(x, return_parameters=True)
        latent = distributions.Normal(mean, std).sample()
        b = self.decode(latent, return_parameters=True)

        return torch.mean(b.log_prob(x)) + VAEBernoulli.divergence(mean, std)

    def optimize(self, x, num_epochs=100):
        optimizer = optim.SGD(self.parameters())

        for epoch in range(num_epochs):
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            optimizer.zero_grad()
            loss = self.loss_func(x)
            loss.backward()
            optimizer.step()