from abc import ABC, abstractmethod
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from nnvi.examples import Example
from nnvi.visualise import plot_2d, plot_3d


class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __call__(
        self,
        soln: Tensor,
        exact_soln: Tensor,
        testfn: Tensor,
        xr: Tensor,
        grad_soln: Tensor = None,
        grad_exact_soln: Tensor = None,
    ) -> Any:
        pass


def error_l2(x: Tensor, y: Tensor, relative: bool = True) -> Tensor:
    """
    :param x: predicted value
    :param y: exact value
    :return: L^2 error
    """
    if not relative:
        return torch.norm(x - y)
    return torch.norm(x - y) / torch.norm(y)


class L2Error(Metric):
    def __init__(self, relative: bool = True) -> None:
        super().__init__()
        self.relative = relative
        self.uses_grad = False

    @property
    def name(self) -> str:
        return "error_l2"

    def __call__(
        self,
        soln: Tensor,
        exact_soln: Tensor,
        testfn: Tensor,
        xr: Tensor,
        grad_soln: Optional[Tensor] = None,
        grad_exact_soln: Optional[Tensor] = None,
    ) -> Any:
        return error_l2(soln, exact_soln, self.relative)


class LInftyError(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.uses_grad = False

    @property
    def name(self) -> str:
        return "error_linfty"

    def __call__(
        self,
        soln: Tensor,
        exact_soln: Tensor,
        testfn: Tensor,
        xr: Tensor,
        grad_soln: Optional[Tensor] = None,
        grad_exact_soln: Optional[Tensor] = None,
    ) -> Any:
        return torch.max(torch.abs(soln - exact_soln))


class H_one_Norm(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.uses_grad = True

    @property
    def name(self) -> str:
        return "H_one_norm"

    def __call__(
        self, soln: Tensor, exact_soln: Tensor, testfn: Tensor, xr: Tensor, grad_soln: Tensor, grad_exact_soln: Tensor
    ) -> Any:
        if grad_exact_soln is not None:
            dist = torch.square(torch.norm(soln - exact_soln, p=2))
            dist_grads = torch.square(torch.norm(grad_soln - grad_exact_soln, p=2))

            norm_exact_sol = torch.sqrt(
                torch.square(torch.norm(exact_soln, p=2)) + torch.square(torch.norm(grad_exact_soln, p=2))
            )
            rel_norm = torch.sqrt(dist + dist_grads) / norm_exact_sol
        else:
            rel_norm = torch.tensor([0.0])
        return rel_norm


class Visualise(Metric):
    def __init__(self, example: Example, use_trisurf=False) -> None:
        super().__init__()
        self.uses_grad = False
        self.example = example
        self.use_trisurf = use_trisurf

    @property
    def name(self) -> str:
        return "solution_plot"

    def __call__(
        self,
        soln: Tensor,
        exact_soln: Tensor,
        testfn: Tensor,
        xr: Tensor,
        grad_soln: Optional[Tensor] = None,
        grad_exact_soln: Optional[Tensor] = None,
    ) -> Any:
        X = xr[:, 0].detach().cpu().numpy()

        Z_soln = torch.reshape(soln, (-1,)).detach().cpu().numpy()
        Z_exact_soln = exact_soln.detach().cpu().numpy()
        Z_obstacle = self.example.obstacle(xr).detach().cpu().numpy()
        Z_test = testfn.detach().cpu().numpy()
        error = np.abs(Z_exact_soln - Z_soln.reshape(-1, 1))

        fig = plt.figure()

        if self.example.domain.pdim == 1:
            ax = fig.add_subplot(2, 2, 1)
            plot_2d("Test function", [X, X], [Z_test, Z_obstacle], ax)
            ax = fig.add_subplot(2, 2, 2)
            plot_2d("NN solution", [X, X], [Z_soln, Z_obstacle], ax)
            ax = fig.add_subplot(2, 2, 3)
            plot_2d("True solution", [X, X], [Z_exact_soln, Z_obstacle], ax)
            ax = fig.add_subplot(2, 2, 4)
            plot_2d("Linfty error", [X], [error], ax)
        elif self.example.domain.pdim == 2:
            Y = xr[:, 1].detach().cpu().numpy()

            ax = fig.add_subplot(2, 2, 1, projection="3d")
            plot_3d("Test function", [X, X], [Y, Y], [Z_test, Z_obstacle], ax, use_trisurf=self.use_trisurf)
            ax = fig.add_subplot(2, 2, 2, projection="3d")
            plot_3d("NN solution", [X, X], [Y, Y], [Z_soln, Z_obstacle], ax, use_trisurf=self.use_trisurf)
            ax = fig.add_subplot(2, 2, 3, projection="3d")
            plot_3d("True solution", [X, X], [Y, Y], [Z_exact_soln, Z_obstacle], ax, use_trisurf=self.use_trisurf)
            ax = fig.add_subplot(2, 2, 4, projection="3d")
            plot_3d("Linfty error", [X], [Y], [error], ax, use_trisurf=self.use_trisurf)

        else:
            raise NotImplementedError("Only one or two dimensions is supported.")
        return fig


class VisualiseInterpolation(Metric):
    def __init__(self, example: Example, use_trisurf=False) -> None:
        super().__init__()
        self.example = example
        self.use_trisurf = use_trisurf
        self.uses_grad = False

    @property
    def name(self) -> str:
        return "solution_plot"

    def __call__(self, soln: Tensor, exact_soln: Tensor, testfn: Tensor, xr: Tensor) -> Any:
        X = xr[:, 0].detach().cpu().numpy()

        Z_soln = torch.reshape(soln, (-1,)).detach().cpu().numpy()

        fig = plt.figure()

        if self.example.domain.pdim == 1:
            ax = fig.add_subplot(2, 2, 1)
            plot_2d("Interpolation of BC", [X], [Z_soln], ax)

        elif self.example.domain.pdim == 2:
            Y = xr[:, 1].detach().cpu().numpy()
            ax = fig.add_subplot(2, 2, 1, projection="3d")
            plot_3d("Interpolation of BC", [X], [Y], [Z_soln], ax, use_trisurf=self.use_trisurf)

        else:
            raise NotImplementedError("Only one or two dimensions is supported.")
        return fig
