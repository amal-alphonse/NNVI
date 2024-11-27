from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, autograd

from nnvi.config import Config
from nnvi.examples import Example


def gradients(input: Tensor, output: Tensor) -> Tensor:
    return autograd.grad(
        outputs=output,
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]


def interior_integral(
    example: Example,
    soln_interior: Tensor,
    grad_soln: Tensor,
    testfn_interior: Tensor,
    grad_testfn: Tensor,
    f_value: Tensor,
    points: Optional[Tensor] = None,
    integration_type: str = "mc",
    weight: Optional[float] = None,
) -> Tensor:
    if example.differential_operator == "Laplacian":
        diff_operator_part = torch.sum(grad_soln * (grad_soln - grad_testfn), dim=1).reshape([-1, 1])
    elif example.differential_operator == "LaplacianPlusDerivative":
        diff_operator_part = torch.sum(
            grad_soln * (grad_soln - grad_testfn) + grad_soln * (soln_interior - testfn_interior),
            dim=1,
        ).reshape([-1, 1])
    elif example.differential_operator == "LaplacianPlusDerivative2D":
        diff_operator_part = torch.sum(
            grad_soln * (grad_soln - grad_testfn) + grad_soln[:, 0:1] * (soln_interior - testfn_interior),
            dim=1,
        ).reshape([-1, 1])
    else:
        raise NotImplementedError(f"Differential operator {example.differential_operator} not supported.")

    integrand = diff_operator_part - f_value * (soln_interior - testfn_interior)

    if integration_type == "mc":
        integral = example.domain.get_volume() * torch.mean(integrand)
    elif integration_type == "trapz":
        if points is None:
            raise ValueError("`points` must be not None if `integration_type` is 'trapz'.")
        integral = torch.trapz(integrand.reshape(-1), points.reshape(-1))
    elif integration_type == "weighted_mc":
        if weight is None:
            weight = 1.0
        integral = example.domain.get_volume() * torch.mean(integrand / weight)
    else:
        raise NotImplementedError(f"Integration type {integration_type} not supported.")

    return integral


def obstacle_loss(obstacle: Tensor, fn_interior: Tensor) -> Tensor:
    return torch.mean(
        torch.pow(torch.maximum(torch.zeros(fn_interior.shape).to(fn_interior.device), obstacle - fn_interior), 2)
    )


class BaseLoss(ABC):
    @abstractmethod
    def __call__(
        self,
        xr: Tensor,
        xb: Tensor,
        soln_interior: Tensor,
        soln_boundary: Tensor,
        testfn_interior: Tensor,
        f_value: Tensor,
        testfn_boundary: Tensor,
    ) -> dict[str, Tensor]:
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, example: Example, cfg: Config) -> "BaseLoss":
        pass


class BaseInterpolationLoss(ABC):
    @abstractmethod
    def __call__(
        self,
        xr: Tensor,
        xb: Tensor,
        soln_interior: Tensor,
        soln_boundary: Tensor,
    ) -> dict[str, Tensor]:
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, example: Example, cfg: Config) -> "BaseInterpolationLoss":
        pass


class InterpolationLoss(BaseInterpolationLoss):
    def __init__(
        self,
        example: Example,
        integration_type: str = "mc",
        weight_b: float = 1.0,
        weight_soln_obs: float = 1.0,
    ) -> None:
        self.weight_b = weight_b
        self.weight_soln_obs = weight_soln_obs
        self._example = example
        self._integration_type = integration_type

    def __call__(
        self,
        xr: Tensor,
        xb: Tensor,
        soln_interior: Tensor,
        soln_boundary: Tensor,
    ) -> dict[str, Tensor]:
        device = soln_boundary.device

        obstacle_interior = self._example.obstacle(xr).to(device)
        loss_soln_obs = obstacle_loss(obstacle_interior, soln_interior)
        loss_b = torch.mean(torch.pow(soln_boundary - self._example.boundary_data(xb), 2))
        loss1 = self.weight_b * loss_b + self.weight_soln_obs * loss_soln_obs

        losses = {"loss1": loss1, "loss_b": loss_b, "loss_soln_obs": loss_soln_obs}

        return losses

    @classmethod
    def from_config(cls, example: Example, _: Config, integration_type: str = "mc") -> "InterpolationLoss":
        return cls(example=example, integration_type=integration_type)


class StandardLoss(BaseLoss):
    def __init__(
        self,
        example: Example,
        weight_b: float,
        weight_b_test: float = None,
        weight_soln_obs: float = None,
        weight_testfn_obs: float = None,
        integration_type: str = "mc",
    ) -> None:
        self._example = example
        self._weight_b = weight_b
        if weight_b_test is not None:
            self._weight_b_test = weight_b_test
        else:
            self._weight_b_test = self._weight_b
        self._weight_soln_obs = weight_soln_obs
        self._weight_testfn_obs = weight_testfn_obs
        self._integration_type = integration_type

    def __call__(
        self,
        xr: Tensor,
        xb: Tensor,
        soln_interior: Tensor,
        soln_boundary: Tensor,
        testfn_interior: Tensor,
        f_value: Tensor,
        weight: Tensor = None,
        testfn_boundary: Tensor = None,
    ) -> dict[str, Tensor]:
        device = soln_interior.device
        obstacle_interior = self._example.obstacle(xr).to(device)

        grad_soln = gradients(xr, soln_interior)
        grad_testfn = gradients(xr, testfn_interior)

        loss_r = interior_integral(
            self._example,
            soln_interior,
            grad_soln,
            testfn_interior,
            grad_testfn,
            f_value,
            integration_type=self._integration_type,
            weight=weight,
        )

        loss_b = torch.mean(torch.pow(soln_boundary - self._example.boundary_data(xb), 2))
        loss_test_b = torch.mean(torch.pow(testfn_boundary - self._example.boundary_data(xb), 2))
        loss_soln_obs = obstacle_loss(obstacle_interior, soln_interior)
        loss_testfn_obs = obstacle_loss(obstacle_interior, testfn_interior)

        weighted_loss_b = self._weight_b * loss_b
        weighted_loss_test_b = self._weight_b_test * loss_test_b
        weighted_loss_soln_obs = self._weight_soln_obs * loss_soln_obs
        weighted_loss_tesfn_obs = self._weight_testfn_obs * loss_testfn_obs

        obs_and_bc_loss = weighted_loss_b + weighted_loss_test_b + weighted_loss_soln_obs + weighted_loss_tesfn_obs
        loss1 = obs_and_bc_loss + loss_r
        loss2 = obs_and_bc_loss - loss_r

        losses = {
            "loss1": loss1,
            "loss2": loss2,
            "loss_r": loss_r,
            "loss_b": loss_b,
            "loss_test_b": loss_test_b,
            "loss_soln_obs": loss_soln_obs,
            "loss_testfn_obs": loss_testfn_obs,
            "weighted_loss_b": weighted_loss_b,
            "weighted_loss_test_b": weighted_loss_test_b,
            "weighted_loss_soln_obs": weighted_loss_soln_obs,
            "weighted_loss_testfn_obs": weighted_loss_tesfn_obs,
        }

        return losses

    @classmethod
    def from_config(cls, example: Example, cfg: Config, integration_type: str = "mc") -> "StandardLoss":
        return cls(
            example=example,
            weight_b=cfg.weight_b,
            weight_soln_obs=cfg.weight_soln_obs,
            weight_testfn_obs=cfg.weight_testfn_obs,
            integration_type=integration_type,
            weight_b_test=cfg.weight_b_test,
        )


class RegularisedGapLoss(StandardLoss):
    def __init__(
        self,
        example: Example,
        weight_b: float,
        weight_b_test: float = None,
        weight_soln_obs: float = None,
        weight_testfn_obs: float = None,
        weight_gap_term: float = None,
        integration_type: str = "mc",
        use_H1_norm_for_gap_term: bool = None,
    ) -> None:
        super().__init__(example, weight_b, weight_b_test, weight_soln_obs, weight_testfn_obs, integration_type)
        self.weight_gap_term = weight_gap_term
        self.p = 2
        self.use_H1_norm_for_gap_term = use_H1_norm_for_gap_term

    def set_weight_gap_term(self, force=0.0):
        self.weight_gap_term = force

    def __call__(
        self,
        xr: Tensor,
        xb: Tensor,
        soln_interior: Tensor,
        soln_boundary: Tensor,
        testfn_interior: Tensor,
        f_value: Tensor,
        weight: Tensor = None,
        testfn_boundary: Tensor = None,
    ) -> dict[str, Tensor]:
        losses = super().__call__(
            xr, xb, soln_interior, soln_boundary, testfn_interior, f_value, weight, testfn_boundary
        )
        if self.use_H1_norm_for_gap_term:
            grad_soln = gradients(xr, soln_interior)
            grad_testfn = gradients(xr, testfn_interior)
            distance_to_diagonal = torch.square(torch.norm(soln_interior - testfn_interior, p=2)) + torch.square(
                torch.norm(grad_soln - grad_testfn, p=2)
            )
        else:
            distance_to_diagonal = torch.square(torch.norm(soln_interior - testfn_interior, p=self.p))
        losses["loss2"] += self.weight_gap_term * distance_to_diagonal
        losses["loss1"] -= self.weight_gap_term * distance_to_diagonal
        losses["distance_to_diagonal"] = distance_to_diagonal

        return losses

    @classmethod
    def from_config(cls, example: Example, cfg: Config, integration_type: str = "mc") -> "RegularisedGapLoss":
        return cls(
            example=example,
            weight_b=cfg.weight_b,
            weight_soln_obs=cfg.weight_soln_obs,
            weight_testfn_obs=cfg.weight_testfn_obs,
            weight_gap_term=cfg.weight_gap_term,
            integration_type=integration_type,
            weight_b_test=cfg.weight_b_test,
            use_H1_norm_for_gap_term=cfg.use_H1_norm_for_gap_term,
        )


LOSS_MAP = {
    "StandardLoss": StandardLoss,
    "InterpolationLoss": InterpolationLoss,
    "RegularisedGapLoss": RegularisedGapLoss,
}


def build_loss(example: Example, cfg: Config, integration_type: str = "mc") -> BaseLoss:
    return LOSS_MAP[cfg.loss_name].from_config(example, cfg, integration_type)
