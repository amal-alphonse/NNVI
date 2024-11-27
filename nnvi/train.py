import copy
import json
import logging
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim

import nnvi
from nnvi.callbacks import Callback, CheckpointLogger, TensorboardLogger, Tunes
from nnvi.config import Config
from nnvi.examples import EXAMPLES_MAP, Example
from nnvi.grids.grid import Grid, SimpleGrid
from nnvi.loss import BaseLoss, build_loss, gradients
from nnvi.metrics import H_one_Norm, L2Error, LInftyError, Metric, Visualise
from nnvi.models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(
        self,
        cfg: Config,
        example: Example,
        grid: Grid,
        soln_model: nn.Module,
        testfn_model: nn.Module,
        loss: BaseLoss,
        metrics: list[Metric],
        callbacks: list[Callback],
    ) -> None:
        self.cfg = cfg
        self.example = example
        self.grid = grid
        self.soln_model = soln_model.to(cfg.device)
        self.testfn_model = testfn_model.to(cfg.device)
        self.loss = loss
        self.metrics = metrics
        self.testfn_model_original_state = testfn_model.state_dict()

        self.best_l2 = float("inf")
        self.loss2_corr_to_best = float("inf")
        self.epoch_corr_to_best = 0

        self.optim_soln = getattr(optim, cfg.optim_soln)(self.soln_model.parameters(), lr=cfg.lr_soln, amsgrad=True)
        self.optim_testfn = getattr(optim, cfg.optim_testfn)(
            self.testfn_model.parameters(), lr=cfg.lr_testfn, amsgrad=True
        )
        self.lr_sched_soln = self.get_lr_scheduler(cfg.lr_scheduler_soln, cfg.lr_scheduler_soln_args, self.optim_soln)
        self.lr_sched_testfn = self.get_lr_scheduler(
            cfg.lr_scheduler_testfn, cfg.lr_scheduler_testfn_args, self.optim_testfn
        )

        for callback in callbacks:
            callback.soln_model = self.soln_model
            callback.testfn_model = self.testfn_model

        self.callbacks = callbacks

    @staticmethod
    def get_lr_scheduler(
        scheduler_name: str, scheduler_args: Dict, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler._LRScheduler:
        if scheduler_name == "DecayedCosineAnnealingWarmRestarts":
            scheduler1 = getattr(optim.lr_scheduler, "CosineAnnealingWarmRestarts")(
                optimizer,
                **scheduler_args,
            )

            T_0 = scheduler_args["T_0"]
            T_mult = scheduler_args["T_mult"]
            milestones = [T_0 * T_mult**2, T_0 * T_mult**3, T_0 * T_mult**4, T_0 * T_mult**5, T_0 * T_mult**6]

            scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.00001)
            lr_scheduler = optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)
        else:
            lr_scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_args)

        return lr_scheduler

    def print_stats(
        self, epoch: int, losses: dict[str, Tensor], metrics: Optional[dict[str, Any]], verbose: bool = True
    ) -> None:
        args = [f"epoch: {epoch}"]
        for name, loss in losses.items():
            if verbose or name in ["loss1", "loss2", "loss_r", "distance_to_diagonal"]:
                args.append(f"{name}: {loss.item(): 0.4}")

        if metrics is not None:
            for name, metric in metrics.items():
                if isinstance(metric, Tensor):
                    args.append(f"{name}: {metric.item(): 0.4}")
                if name == "error_l2":
                    if metric.item() < self.best_l2:
                        self.loss2_corr_to_best = losses["loss2"]
                        self.epoch_corr_to_best = epoch
                        self.best_l2 = metric.item()

        args.append(
            f" | E: {self.cfg.example} L: {self.lr_sched_soln.get_last_lr()[0]:0.4} |  E: {self.epoch_corr_to_best} B2: {self.loss2_corr_to_best : 0.4} B: {self.best_l2 : 0.4}"
        )

        print(*args)

    def on_epoch_end(self, epoch: int, losses: dict[str, Tensor], metrics: Optional[dict[str, Any]]) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, losses, metrics)

    def on_train_end(self) -> None:
        for callback in self.callbacks:
            callback.on_train_end()

    def train(self) -> float:
        def soln_step(loss: Tensor) -> None:
            self.optim_soln.zero_grad()
            loss.backward()
            last_loss1_grad = torch.mean(torch.tensor([torch.norm(p.grad) for p in self.soln_model.parameters()]))

            if not torch.isnan(last_loss1_grad):
                self.optim_soln.step()
            else:
                logging.warning("NaN in gradient of solution")

            self.lr_sched_soln.step()

        def testfn_step(loss: Tensor) -> None:
            self.optim_testfn.zero_grad()
            loss.backward()
            last_loss2_grad = torch.mean(torch.tensor([torch.norm(p.grad) for p in self.testfn_model.parameters()]))

            if not torch.isnan(last_loss2_grad):
                self.optim_testfn.step()
            else:
                logging.warning("NaN in gradient of test function")

            self.lr_sched_testfn.step()

        for epoch in range(self.cfg.epochs):
            losses = self.train_step()

            if self.cfg.max_step_period > 1:
                if epoch % self.cfg.max_step_period == 1:
                    testfn_step(losses["loss2"])
                else:
                    soln_step(losses["loss1"])
            else:
                min_step_period = int(1 / self.cfg.max_step_period)
                if epoch % min_step_period == 1:
                    soln_step(losses["loss1"])
                else:
                    testfn_step(losses["loss2"])

            metrics = None

            if epoch % self.cfg.eval_freq == 0 or epoch == self.cfg.epochs - 1:
                metrics = self.eval_step()

            if epoch % self.cfg.print_freq == 0 or epoch == self.cfg.epochs - 1:
                self.print_stats(epoch, losses, metrics)

            self.on_epoch_end(epoch, losses, metrics)

        self.on_train_end()

        return metrics["error_l2"].item()

    def train_step(self) -> dict[str, Tensor]:
        device = self.cfg.device
        self.soln_model.train()
        self.testfn_model.train()

        xr = self.grid.get_interior_points().to(device)
        xb = self.grid.get_boundary_points().to(device)
        xr.requires_grad_()

        f_value = self.example.source_term(xr)
        soln_interior = self.soln_model(xr)
        soln_boundary = self.soln_model(xb)
        testfn_interior = self.testfn_model(xr)
        testfn_boundary = self.testfn_model(xb)

        losses = self.loss(
            xr,
            xb,
            soln_interior,
            soln_boundary,
            testfn_interior,
            f_value,
            testfn_boundary=testfn_boundary,
        )

        return losses

    def eval_step(self) -> dict[str, Any]:
        device = self.cfg.device
        self.soln_model.eval()
        self.testfn_model.eval()

        if self.cfg.use_uniform_grid_for_eval:
            uniform_grid = SimpleGrid(self.example.domain, self.cfg.n_interior, self.cfg.n_boundary, False)
            xr = uniform_grid.get_interior_points().to(device)
        else:
            xr = self.grid.get_interior_points().to(device)

        exact_soln = self.example.exact_solution(xr)
        if [m.uses_grad for m in self.metrics]:
            xr.requires_grad = True
            soln_interior = self.soln_model(xr)
            grad_soln = gradients(xr, soln_interior)
            if self.example.exact_solution_grad is not None:
                grad_exact_soln = self.example.exact_solution_grad(xr)
            else:
                grad_exact_soln = None
            xr.requires_grad = False

        with torch.no_grad():
            soln_interior = self.soln_model(xr)
            testfn_interior = self.testfn_model(xr)
        return {
            metric.name: metric(soln_interior, exact_soln, testfn_interior, xr, grad_soln, grad_exact_soln)
            for metric in self.metrics
        }


def train_cfg(cfg: Config) -> float:
    cfg = copy.deepcopy(cfg)
    if isinstance(cfg.soln_model_args["width"], tuple):
        cfg.soln_model_args["width"] = cfg.soln_model_args["width"][0]
        cfg.soln_model_args["depth"] = cfg.soln_model_args["depth"][0]

    seed_everything(cfg.seed)

    save_cfg_to = Path(cfg.chpt_folder) / "cfg.json"
    save_cfg_to.parent.mkdir(parents=True, exist_ok=True)
    with save_cfg_to.open("w", encoding="UTF-8") as target:
        json.dump(asdict(cfg), target)

    example = EXAMPLES_MAP[cfg.example]

    soln_model = build_model(
        cfg.soln_model_name,
        in_N=example.domain.pdim,
        out_N=1,
        force_bc=cfg.force_bc,
        example=example,
        **cfg.soln_model_args,
    )
    testfn_model = build_model(
        cfg.testfn_model_name,
        in_N=example.domain.pdim,
        out_N=1,
        example=example,
        force_bc=cfg.force_bc,
        **cfg.testfn_model_args,
    )
    loss = build_loss(example, cfg, integration_type="weighted_mc")
    grid = getattr(nnvi.grids.grid, cfg.grid_cls)(example.domain, cfg.n_interior, cfg.n_boundary)

    tensorboard_dir = os.path.join(cfg.chpt_folder, "board")
    chpt_dir = os.path.join(cfg.chpt_folder, "check")

    callbacks = [
        TensorboardLogger(tensorboard_dir, add_time_stamp=False, tensor_freq=100),
        CheckpointLogger(chpt_dir, epochs=cfg.epochs, nbr_of_intermediate_chepts=None),
    ]

    if cfg.use_ray:
        callbacks = [Tunes(best_l2=10.81)] + callbacks

    if cfg.save_plots:
        metrics = [L2Error(), LInftyError(), Visualise(example, use_trisurf=True), H_one_Norm()]
    else:
        metrics = [L2Error(), LInftyError(), H_one_Norm()]

    trainer = Trainer(cfg, example, grid, soln_model, testfn_model, loss, metrics, callbacks)
    v = trainer.train()

    return v


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = Config(**cfg_dict)
    train_cfg(cfg)
