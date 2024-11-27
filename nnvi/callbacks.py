import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib
import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class Callback(ABC):
    def __init__(self) -> None:
        self.soln_model = None
        self.testfn_model = None
        self.i_sampler_for_solution = None
        self.i_sampler_for_test = None

    @abstractmethod
    def on_epoch_end(self, epoch: int, losses: dict[str, Tensor], metrics: Optional[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def on_train_end(self) -> None:
        pass


class TensorboardLogger(Callback):
    def __init__(self, log_dir: Optional[Union[str, Path]] = None, add_time_stamp=True, tensor_freq=1) -> None:
        super().__init__()
        if (log_dir is not None) and add_time_stamp:
            log_dir = Path(log_dir) / datetime.datetime.now().isoformat().replace(":", "-").replace(".", "-")
            log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir)
        self.tensor_freq = tensor_freq

    def on_epoch_end(self, epoch: int, losses: dict[str, Tensor], metrics: Optional[dict[str, Any]]) -> None:
        if epoch % self.tensor_freq == 0:
            for name, loss in losses.items():
                self.writer.add_scalar(name, loss, epoch)
            if metrics is not None:
                for name, metric in metrics.items():
                    if isinstance(metric, matplotlib.figure.Figure):
                        try:
                            self.writer.add_figure(name, metric, epoch)
                        except RuntimeWarning:
                            pass
                    elif isinstance(metric, Tensor):
                        self.writer.add_scalar(name, metric, epoch)
                    else:
                        raise NotImplementedError(f"Unsupported metric type: {type(metric)}.")

    def on_train_end(self) -> None:
        self.writer.flush()
        self.writer.close()


class CheckpointLogger(Callback):
    def __init__(self, log_dir: Union[str, Path], epochs=None, nbr_of_intermediate_chepts=2) -> None:
        super().__init__()
        self.log_dir = Path(log_dir) / datetime.datetime.now().isoformat().replace(":", "-").replace(".", "-")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        if self.epochs is not None and nbr_of_intermediate_chepts is not None:
            self.save_freq = self.epochs // nbr_of_intermediate_chepts
        else:
            self.save_freq = None

    def on_epoch_end(self, epoch: int, losses: dict[str, Tensor], metrics: Optional[dict[str, Any]]) -> None:
        if self.save_freq is not None:
            if epoch % self.save_freq == 0 and not (epoch == self.epochs):
                save_path = Path(self.log_dir) / str(epoch)
                save_path.mkdir(exist_ok=True, parents=True)
                for m, n in [[self.soln_model, "soln.pt"], [self.testfn_model, "test.pt"]]:
                    torch.save(m.state_dict(), save_path / n)

    def on_train_end(self) -> None:
        for m, n in [[self.soln_model, "soln.pt"], [self.testfn_model, "test.pt"]]:
            torch.save(m.state_dict(), Path(self.log_dir) / n)


class SolutionGradient(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_epoch_end(self, epoch: int, losses: dict[str, Tensor], metrics: Optional[dict[str, Any]]) -> None:
        grad_of_solution = np.sum([torch.norm(para.grad).item() for para in self.soln_model.parameters()])
        losses["grad_of_solution"] = torch.tensor(grad_of_solution)

    def on_train_end(self) -> None:
        pass


class Tunes(Callback):
    def __init__(self, best_l2=1.0) -> None:
        from ray import train

        super().__init__()
        if best_l2:
            self.best_l2 = best_l2
        else:
            self.best_l2 = 5.0
        self.epoch_of_last_update = 0

    def on_epoch_end(self, epoch: int, losses: dict[str, Tensor], metrics: Optional[dict[str, Any]]) -> None:
        if metrics != None and epoch % 100 == 0:
            if "error_l2" in metrics.keys():
                current_metric = metrics["error_l2"].cpu().detach().numpy().item()
                self.last_error = current_metric
                if current_metric < self.best_l2:
                    self.epoch_of_last_update = epoch
                self.best_l2 = np.min([self.best_l2, current_metric])
            if "H_one_norm" in metrics.keys():
                self.h_one_err = metrics["H_one_norm"].cpu().detach().numpy().item()
            else:
                self.h_one_err = 1.0
            if "error_linfty" in metrics.keys():
                self.e_infty = metrics["error_linfty"].cpu().detach().numpy().item()
            else:
                self.e_infty = 1.0
            stats = {"epoch": epoch}
            as_np = {}
            for name, value in losses.items():
                as_np[name] = value.cpu().detach().numpy()
            for name, value in metrics.items():
                if torch.is_tensor(value):
                    as_np[name] = value.cpu().detach().numpy().item()
            as_np["error_l2"] = self.last_error
            as_np["best_l2"] = self.best_l2
            as_np["error_l2"] = self.last_error
            if "H_one_norm" in metrics.keys():
                as_np["H_one_norm"] = self.h_one_err
            if "error_linfty" in metrics.keys():
                as_np["error_linfty"] = self.e_infty
            stats.update(as_np)
            train.report(stats)

    def on_train_end(self) -> None:
        stats = {"epoch": 150001}
        as_np = {}
        as_np["best_l2"] = self.last_error  # self.best_l2
        as_np["error_l2"] = self.last_error
        as_np["H_one_norm"] = self.h_one_err
        as_np["error_linfty"] = self.e_infty
        stats.update(as_np)
        train.report(stats)
