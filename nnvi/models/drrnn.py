from typing import Callable, Optional

import torch
import torch.nn as nn

from nnvi.examples import Example


class Block(nn.Module):
    """
    Implementation of the block used in the Deep Ritz paper, taken from https://github.com/xdfeng7370/Deep-Ritz-Method/blob/master/deep_ritz_ls.py

    Arguments:
        in_N: dimension of the input
        width: number of nodes in the interior middle layer
        out_N: dimension of the output
        activation: name of the activation function
    """

    def __init__(self, in_N: int, width: int, out_N: int, activation: str = "ReLU") -> None:
        super(Block, self).__init__()
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        self.phi = getattr(nn, activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi(self.L2(self.phi(self.L1(x)))) + x


class Drrnn(nn.Module):
    """
    Drrnn -- Deep Ritz Residual Neural Network
    Implements a network with the architecture used in the Deep Ritz paper. This implementation generalises that in https://github.com/xdfeng7370/Deep-Ritz-Method/blob/master/deep_ritz_ls.py

    Arguments:
        in_N: input dimension
        out_N: output dimension
        width: width of layers that form blocks
        depth: number of blocks to be stacked
        activation: name of the activation function
    """

    def __init__(
        self,
        in_N: int,
        width: int,
        out_N: int,
        depth: int = 4,
        activation: str = "ReLU",
        force_bc: bool = False,
        example: Optional[Example] = None,
        **ignored,
    ) -> None:
        super().__init__()

        if force_bc:
            assert example is not None, f"`example` cannot be None if `force_bc=True`."
            self.example = example
        else:
            self.example = None

        self.force_bc = force_bc
        self.stack = torch.nn.ModuleList()
        self.stack.append(torch.nn.Linear(in_N, width))

        for _ in range(depth):
            self.stack.append(Block(width, width, width, activation=activation))

        self.stack.append(torch.nn.Linear(width, out_N))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.force_bc:
            boundary_data = self.example.boundary_data_setter_for_testfn(x).to(x.device)
        for layer in self.stack:
            x = layer(x)
        if self.force_bc:
            x = x * boundary_data
        return x


class DrrnnWithZeroBC(Drrnn):
    """
    Drrnn where zero boundary condition is always enforced.
    """

    def __init__(
        self,
        example: Example,
        width: int,
        out_N: int,
        depth: int,
        activation: str = "ReLU",
        **ignored,
    ) -> None:
        super().__init__(example.domain.pdim, width, out_N, depth, activation, force_bc=True, example=example)


class DrrnnWithSpecificBC(nn.Module):
    """
    Model of the form DrrnnWithZeroBC + h_NN, where h_NN satisfies the boundary condition (to a low error) and is larger than or equal to the obstacle
    """

    def __init__(
        self,
        f_nn: DrrnnWithZeroBC,
        interpolation: Drrnn,
    ) -> None:
        super().__init__()
        self.f_nn = f_nn
        self.interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.interpolation(x) + self.f_nn(x)

    def freeze_only_interpolation(self) -> None:
        for param in self.interpolation.parameters():
            param.requires_grad_(False)
        for param in self.f_nn.parameters():
            param.requires_grad_(True)

    def freeze_all(self) -> None:
        for param in self.interpolation.parameters():
            param.requires_grad_(False)
        for param in self.f_nn.parameters():
            param.requires_grad_(False)

    def parameters(self, recurse: bool = True):
        return self.f_nn.parameters(recurse)


class DrrnnWithObstacleEnforcement(nn.Module):
    """
    Model of the form Drrnn^2 + obstacle
    """

    def __init__(
        self,
        f_nn: Drrnn,
        obstacle: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.f_nn = f_nn
        self.obstacle = obstacle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.obstacle(x) + torch.pow(self.f_nn(x), 2)
