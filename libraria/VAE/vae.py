import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data

from typing import Union
import numpy as np


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20, hidden_dim: int = 400):
        super(VAE, self).__init__()
        self._latent_dim = latent_dim
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 784)

    def _encode(self, x: torch.Tensor):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _decode(self, z: torch.Tensor):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def encode(self, x: np.ndarray):
        if x.ndim == 2:
            x = torch.Tensor(x).view(1, 784)
            mu, logvar = self._encode(x)
            np_output = self._sample(mu, logvar).detach().numpy()
            return np_output.squeeze()

        x = torch.Tensor(x).view(-1, 784)
        mu, logvar = self._encode(x)
        return self._sample(mu, logvar).detach().numpy()

    def decode(self, z: Union[float, np.ndarray]):
        z = np.array([z]) if isinstance(z, float) else z

        if z.ndim == 1 or z.ndim == 0:
            z = torch.Tensor(z).view(1, self._latent_dim)
            return self._decode(z).view(28, 28).detach().numpy()

        z = torch.Tensor(z).view(-1, self._latent_dim)
        return self._decode(z).view(-1, 28, 28).detach().numpy()

    def loss_function(self, x):
        mu, logvar = self._encode(x.view(-1, 784))
        z = self._sample(mu, logvar)
        recon_x = self._decode(z)

        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def fit(self, x, epochs=200, batch_size=128, verbose=True):
        train_loader = torch.utils.data.DataLoader(
            x, batch_size=batch_size, shuffle=True
        )
        for epoch in range(1, epochs + 1):
            train_loss = 0
            optimizer = optim.Adam(self.parameters(), lr=1e-2)
            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.loss_function(data)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            if verbose:
                print(
                    "Epoch: {} Average loss: {:.4f}".format(
                        epoch, train_loss / len(train_loader.dataset)
                    )
                )
