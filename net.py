import torch
import torch.nn as nn
from torch.distributions import Normal


def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    final_activation: str
) -> torch.nn.modules.container.Sequential:

    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth -1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())
    elif final_activation =='linear':
        pass

    return nn.Sequential(*layers)


class GaussPolicyMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(
        self, 
        state_dim: int, hidden_dim: int, depth: int, action_dim: int, 
    ) -> None:
        super().__init__()
        self._net = MLP(state_dim, hidden_dim, depth, action_dim, 'linear')
        self._raw_log_std = nn.Parameter(data=torch.Tensor([0.,] * action_dim),
                                         requires_grad=True)

        self._log_std_bound = (-5., 2.)


    def forward(
        self, s: torch.Tensor
    ) -> torch.distributions:

        mu = self._net(s)
        mu = nn.Tanh()(mu)
        log_std = torch.clamp(self._raw_log_std, 
                              self._log_std_bound[0], 
                              self._log_std_bound[1])
        std = log_std.exp()
        dist = Normal(mu, std)
        return dist