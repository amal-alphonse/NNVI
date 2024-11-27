import inspect
from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    example: str
    n_interior: int = 1024
    n_boundary: int = 256
    epochs: int = 6000
    grid_cls: str = "UniformRandomCube"
    soln_model_name: str = "Drrnn"
    soln_model_args: dict = field(default_factory=lambda: {"width": 40, "depth": 4})
    testfn_model_name: str = "Drrnn"
    testfn_model_args: dict = field(default_factory=lambda: {"width": 40, "depth": 4})
    loss_name: str = "RegularisedGapLoss"
    weight_b: float = 100.0
    weight_b_test: float = 100.0
    weight_soln_obs: float = 10.0
    weight_testfn_obs: float = 10.0
    weight_gap_term: float = 10.0
    optim_soln: str = "AdamW"
    optim_testfn: str = "AdamW"
    lr_soln: float = 1e-2
    lr_testfn: float = 1e-3
    lr_scheduler_soln: str = "StepLR"
    lr_scheduler_soln_args: dict = field(default_factory=lambda: {"gamma": 0.2, "step_size": 1000})
    lr_scheduler_testfn: str = "StepLR"
    lr_scheduler_testfn_args: dict = field(default_factory=lambda: {"gamma": 0.2, "step_size": 1000})
    max_step_period: int = 3
    print_freq: int = 100
    eval_freq: int = 25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 123
    use_H1_norm_for_gap_term: bool = True
    force_bc: bool = False
    chpt_folder: str = "./plots/"
    use_uniform_grid_for_eval: bool = False
    use_ray: bool = False
    save_plots: bool = False

    @classmethod
    def from_dict(cls, env):
        """
        To avoid typ error if env contains more fields then the one defined above
        """
        return cls(**{k: v for k, v in env.items() if k in inspect.signature(cls).parameters})
