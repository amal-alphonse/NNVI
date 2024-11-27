import copy
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor, optim

import nnvi
from nnvi.callbacks import Callback, TensorboardLogger
from nnvi.config import Config
from nnvi.examples import EXAMPLES_MAP, Example
from nnvi.grids.grid import Grid, SimpleGrid
from nnvi.loss import BaseInterpolationLoss, build_loss
from nnvi.metrics import Metric, VisualiseInterpolation
from nnvi.models import build_model
from nnvi.train import seed_everything


class InterpolationTrainer:
    def __init__(
        self,
        cfg: Config,
        example: Example,
        grid: Grid,
        soln_model: nn.Module,
        loss: BaseInterpolationLoss,
        metrics: list[Metric],
        callbacks: list[Callback],
    ) -> None:
        self.cfg = cfg
        self.example = example
        self.grid = grid
        self.soln_model = soln_model.to(cfg.device)
        self.loss = loss
        self.metrics = metrics

        self.optim_soln = getattr(optim, cfg.optim_soln)(soln_model.parameters(), lr=cfg.lr_soln, amsgrad=True)
        self.lr_sched_soln = getattr(optim.lr_scheduler, cfg.lr_scheduler_soln)(
            self.optim_soln,
            **cfg.lr_scheduler_soln_args,
        )

        self.optim_soln_init_state = self.optim_soln.state_dict()

        for callback in callbacks:
            callback.soln_model = self.soln_model

        self.callbacks = callbacks

    def print_stats(
        self, epoch: int, losses: dict[str, Tensor], metrics: Optional[dict[str, Any]], verbal=True
    ) -> None:
        args = [f"epoch: {epoch}"]
        for name, loss in losses.items():
            args.append(f"{name}: {loss.item(): 0.4}")

        if metrics is not None:
            for name, metric in metrics.items():
                if isinstance(metric, Tensor):
                    args.append(f"{name}: {metric.item(): 0.4}")
        print(*args)

    def on_epoch_end(self, epoch: int, losses: dict[str, Tensor], metrics: Optional[dict[str, Any]]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, losses, metrics)

    def on_train_end(self) -> None:
        for callback in self.callbacks:
            callback.on_train_end()

    def train(self) -> dict[str, Any]:
        def soln_step(loss: Tensor) -> None:
            self.optim_soln.zero_grad()
            loss.backward()
            self.optim_soln.step()
            self.lr_sched_soln.step()

        for epoch in range(self.cfg.epochs):
            losses = self.train_step()

            soln_step(losses["loss1"])
            metrics = None

            if epoch % self.cfg.eval_freq == 0:
                metrics = self.eval_step()

            if epoch % self.cfg.print_freq == 0:
                self.print_stats(epoch, losses, metrics)

            self.on_epoch_end(epoch, losses, metrics)

        return self.eval_step()

    def train_step(self) -> dict[str, Tensor]:
        device = self.cfg.device
        self.soln_model.train()
        xr = self.grid.get_interior_points().to(device)
        xb = self.grid.get_boundary_points().to(device)

        soln_interior = self.soln_model(xr)
        soln_boundary = self.soln_model(xb)

        losses = self.loss(xr, xb, soln_interior, soln_boundary)

        return losses

    def eval_step(self) -> dict[str, Any]:
        device = self.cfg.device
        self.soln_model.eval()

        if self.cfg.use_uniform_grid_for_eval:
            uniform_grid = SimpleGrid(self.example.domain, self.cfg.n_interior, self.cfg.n_boundary, False)
            xr = uniform_grid.get_interior_points().to(device)
        else:
            xr = self.grid.get_interior_points().to(device)

        with torch.no_grad():
            soln_interior = self.soln_model(xr)

        return {metric.name: metric(soln_interior, None, None, xr) for metric in self.metrics}


def train_interpolation(cfg: Config) -> nn.Module:
    cfg = copy.deepcopy(cfg)
    seed_everything(cfg.seed)

    if isinstance(cfg.soln_model_args["width"], tuple):
        cfg.soln_model_args["width"] = cfg.soln_model_args["width"][0]
        cfg.soln_model_args["depth"] = cfg.soln_model_args["depth"][0]

    example = EXAMPLES_MAP[cfg.example]

    soln_model = build_model(
        cfg.soln_model_name,
        in_N=example.domain.pdim,
        out_N=1,
        force_bc=cfg.force_bc,
        example=example,
        **cfg.soln_model_args,
    )

    loss = build_loss(example, cfg, integration_type="weighted_mc")
    grid = getattr(nnvi.grids.grid, cfg.grid_cls)(example.domain, cfg.n_interior, cfg.n_boundary)

    callbacks = [TensorboardLogger(cfg.chpt_folder)]
    metrics = [VisualiseInterpolation(example, use_trisurf=True)]
    trainer = InterpolationTrainer(cfg, example, grid, soln_model, loss, metrics, callbacks)

    trainer.train()

    return soln_model
