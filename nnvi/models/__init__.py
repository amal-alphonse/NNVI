from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nnvi.config import Config
from nnvi.models.drrnn import (
    Drrnn,
    DrrnnWithObstacleEnforcement,
    DrrnnWithSpecificBC,
    DrrnnWithZeroBC,
)

MODEL_MAP = {
    "Drrnn": Drrnn,
    "DrrnnWithZeroBC": DrrnnWithZeroBC,
    "DrrnnWithSpecificBC": DrrnnWithSpecificBC,
    "DrrnnWithObstacleEnforcement": DrrnnWithObstacleEnforcement,
}


def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def build_model(
    model_name: str, in_N: int, out_N: int, force_bc: bool = True, pretrained: bool = False, **kwargs
) -> nn.Module:
    if model_name == "DrrnnWithObstacleEnforcement":
        example = kwargs.get("example")
        assert example is not None, f"`example` can not None for model {model_name}."

        f_nn = Drrnn(in_N=in_N, out_N=out_N, **kwargs)
        model = DrrnnWithObstacleEnforcement(f_nn, example.obstacle)
    elif model_name == "DrrnnWithSpecificBC":
        soln_model_args = {"width": 40, "depth": 4, "activation": "Tanh"}

        if pretrained:
            g_nn = Drrnn(in_N=in_N, out_N=out_N, **soln_model_args)
        else:
            example = kwargs.get("example")
            assert example is not None, f"`example` can not None for model {model_name}."

            interpolant_save_path = Path("interpolants") / "saved_models" / f"{example.name}_interpolant.pt"

            if interpolant_save_path.exists():
                print("Loading previously saved interpolant from ", str(interpolant_save_path))
                g_nn = torch.load(interpolant_save_path)
            else:
                print("Creating interpolation function:")
                from nnvi.train_interpolation import train_interpolation

                interpolant_chpt_folder = Path("plots") / "interpolated_functions"
                interpolant_chpt_folder.mkdir(exist_ok=True, parents=True)
                seed = np.random.randint(low=2, high=9999)
                interpolant_cfg = Config(
                    example=example.name,
                    n_interior=5000,
                    n_boundary=1000,
                    epochs=2000,
                    soln_model_name="Drrnn",
                    soln_model_args=soln_model_args,
                    lr_soln=0.006659396851225986,
                    lr_testfn=0.012417555451614158,
                    lr_scheduler_soln_args={"gamma": 0.6, "step_size": 1000},
                    optim_soln="AdamW",
                    loss_name="InterpolationLoss",
                    chpt_folder=str(interpolant_chpt_folder),
                    use_uniform_grid_for_eval=True,
                    print_freq=100,
                    eval_freq=100,
                    seed=seed,
                )

                g_nn = train_interpolation(interpolant_cfg)

                interpolant_save_path.parent.mkdir(exist_ok=True, parents=True)
                torch.save(g_nn, interpolant_save_path)

            g_nn.eval()
            for param in g_nn.parameters():
                param.requires_grad = False

        f_nn = DrrnnWithZeroBC(in_N=in_N, out_N=out_N, **kwargs)
        model = DrrnnWithSpecificBC(f_nn, g_nn)
    else:
        model = MODEL_MAP[model_name](in_N=in_N, out_N=out_N, force_bc=force_bc, **kwargs)
        model.apply(weights_init)

    return model
