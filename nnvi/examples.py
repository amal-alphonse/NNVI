import numpy as np
import torch

from nnvi.grids.domain import Domain

__all__ = ["Example", "EXAMPLES_MAP"]


################################################################################################################
## example on [-1,1]^2 from Keith & Surowiec
def KS_exact_solution_vi(x):
    value = torch.where(x[:, 0:1] < 0, 0, torch.pow(x[:, 0:1], 4))
    return value


def KS_gradient_exact_solution_vi(x):
    value_x = torch.where(x[:, 0:1] < 0, 0, 4 * torch.pow(x[:, 0:1], 3))
    value_y = torch.where(x[:, 0:1] < 0, 0, 0)
    return torch.concatenate([value_x, value_y], dim=1)


def KS_obstacle(x):
    return (0 * x[:, 0]).reshape(-1, 1)


def KS_function_l_exact(x):
    return (1 + x[:, 0:1]) * (1 + x[:, 1:2]) * (1 - x[:, 0:1]) * (1 - x[:, 1:2])


def KS_vi_source_f(x):
    value = torch.where(x[:, 0:1] < 0, 0, -12 * torch.pow(x[:, 0:1], 2))
    return value


################################################################################################################
################################################################################################################
## nonsmooth example on [-1,1]^2 from Keith & Surowiec
def KS_ns_exact_solution_vi(x):
    value = torch.where(
        x[:, 0:1] ** 2 + x[:, 1:2] ** 2 < 0.25, torch.pow(1 - 4 * x[:, 0:1] ** 2 - 4 * x[:, 1:2] ** 2, 4), 0
    )
    return value


def KS_ns_gradient_exact_solution_vi(x):
    value_x = torch.where(
        x[:, 0:1] ** 2 + x[:, 1:2] ** 2 < 0.25,
        -32 * x[:, 0:1] * torch.pow(1 - 4 * x[:, 0:1] ** 2 - 4 * x[:, 1:2] ** 2, 3),
        0,
    )
    value_y = torch.where(
        x[:, 0:1] ** 2 + x[:, 1:2] ** 2 < 0.25,
        -32 * x[:, 1:2] * torch.pow(1 - 4 * x[:, 0:1] ** 2 - 4 * x[:, 1:2] ** 2, 3),
        0,
    )
    return torch.concatenate([value_x, value_y], dim=1)


def KS_ns_obstacle(x):
    return (0 * x[:, 0]).reshape(-1, 1)


def KS_ns_function_l_exact(x):
    return (1 + x[:, 0:1]) * (1 + x[:, 1:2]) * (1 - x[:, 0:1]) * (1 - x[:, 1:2])


def KS_ns_laplacian(x):
    value = torch.where(
        x[:, 0:1] ** 2 + x[:, 1:2] ** 2 < 0.25,
        -64 * (1 - 4 * x[:, 0:1] ** 2 - 4 * x[:, 1:2] ** 2) ** 3
        + 768 * (x[:, 0:1] ** 2 + x[:, 1:2] ** 2) * (1 - 4 * x[:, 0:1] ** 2 - 4 * x[:, 1:2] ** 2) ** 2,
        0,
    )
    return value


def KS_ns_vi_source_f(x):
    value = -1 * KS_ns_laplacian(x) - torch.where(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 > 0.75, 1, 0)
    return value


################################################################################################################


################################################################################################################
## new example on [-1,1]^2 from NSV
def r_for_NSV(is_cuda):
    if is_cuda:
        r = torch.tensor([0.5]).to("cuda")
    else:
        r = torch.tensor([0.5]).to("cpu")
    return r


def NSV_two_dim_exact_solution_vi(x):
    r = r_for_NSV(x.is_cuda)
    value = torch.square(
        torch.maximum(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 - r**2, torch.zeros(x[:, 0:1].shape, device=x.device))
    )
    return value


def NSV_two_dim_obstacle(x):
    return (0 * x[:, 0]).reshape(-1, 1)


def NSV_two_dim_function_l_exact(x):
    return (1 + x[:, 0:1]) * (1 + x[:, 1:2]) * (1 - x[:, 0:1]) * (1 - x[:, 1:2])


def NSV_two_dim_vi_source_f(x):
    r = r_for_NSV(x.is_cuda)
    value = torch.where(
        x[:, 0:1] ** 2 + x[:, 1:2] ** 2 > r**2,
        -4 * (2 * (x[:, 0:1] ** 2 + x[:, 1:2] ** 2) + 2 * (x[:, 0:1] ** 2 + x[:, 1:2] ** 2 - r**2)),
        -8 * r**2 * (1 - x[:, 0:1] ** 2 - x[:, 1:2] ** 2 + r**2),
    )
    return value


################################################################################################################


################################################################################################################
## example on [-2,2]^2
def two_dim_exact_solution_vi(x):
    r = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
    rz = 0.6979651482

    value = torch.where(
        r <= rz,
        torch.sqrt(1 - torch.where(r <= rz, r**2, 0)),
        -1 * rz**2 * torch.log(r / 2.0) / (np.sqrt(1 - rz**2)),
    )
    return value


def two_dim_exact_solution_for_plot(x, y):
    r = np.sqrt(x**2 + y**2)
    rz = 0.6979651482

    value = np.where(r <= rz, np.sqrt(1 - r**2), -1 * rz**2 * np.log(r / 2.0) / (np.sqrt(1 - rz**2)))
    return value


def two_dim_obstacle(x):
    # Alex's solution
    i2 = x[:, 0:1] ** 2 + x[:, 1:2] ** 2
    r2 = torch.sqrt(torch.relu(i2))
    alternative = torch.sqrt(torch.relu(1.0 - r2**2))
    value2 = torch.where(r2 <= 1, alternative, -1)

    # Solution from https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
    r = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
    value_new = torch.where(r <= 1, torch.sqrt(1.0 - torch.where(r <= 1, r, 0) ** 2), -1)

    if torch.equal(value2, value_new) == False:
        assert False, "problem with two_dim_obstacle!"
    return value_new


def two_dim_function_l_exact(x):
    return (2 + x[:, 0:1]) * (2 + x[:, 1:2]) * (2 - x[:, 0:1]) * (2 - x[:, 1:2])


def two_dim_vi_source_f(x):
    return 0


################################################################################################################


################################################################################################################
## non-symmetric example on [-2,2]^2
def ns_two_dim_exact_solution_vi(x):
    r = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
    rz = 0.6979651482
    value = torch.where(
        r <= rz,
        torch.sqrt(1 - torch.where(r <= rz, r**2, 0)),
        -1 * rz**2 * torch.log(r / 2.0) / (np.sqrt(1 - rz**2)),
    )
    return value


def ns_two_dim_obstacle(x):
    r = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
    value_new = torch.where(r <= 1, torch.sqrt(1.0 - torch.where(r <= 1, r, 0) ** 2), -1)
    return value_new


def ns_two_dim_function_l_exact(x):
    return (2 + x[:, 0:1]) * (2 + x[:, 1:2]) * (2 - x[:, 0:1]) * (2 - x[:, 1:2])


def ns_two_dim_vi_source_f(x):
    r = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
    rz = 0.6979651482
    f0 = (2 - r**2) / torch.sqrt((1 - r**2) ** 3) - x[:, 0:1] / (torch.sqrt(1 - r**2))
    f1 = -rz / (np.sqrt(1 - rz**2)) * x[:, 0:1] / (r**2)
    value = torch.where(r <= rz, f0, f1)
    return value


def ns_two_dim_exact_solution_for_plot(x, y):
    r = np.sqrt(x**2 + y**2)
    rz = 0.6979651482

    value = np.where(r <= rz, np.sqrt(1 - r**2), -1 * rz**2 * np.log(r / 2.0) / (np.sqrt(1 - rz**2)))
    return value


################################################################################################################


################################################################################################################
## example on (0,1)
def one_dim_exact_solution_vi(x):
    ref_point = 1 / (2 * np.sqrt(2))
    v1 = torch.where((x >= ref_point) & (x <= 0.5), 100 * x * (1 - x) - 12.5, 0)
    v2 = torch.where((x >= 0) & (x < ref_point), (100 - 50 * np.sqrt(2)) * x, v1)
    v3 = torch.where((x > 0.5) & (x <= 1 - ref_point), 100 * x * (1 - x) - 12.5, v2)
    v4 = torch.where((x > 1 - ref_point) & (x <= 1), (50 * np.sqrt(2) - 100) * (x - 1), v3)  # change ths!

    return v4


def one_dim_boundary_condition(x):
    return torch.zeros_like(x)


def one_dim_obstacle(x):
    v1 = torch.where((x >= 0.25) & (x <= 0.5), 100 * x * (1 - x) - 12.5, 0)
    v2 = torch.where((x >= 0) & (x < 0.25), 100 * x**2, v1)
    v3 = torch.where((x > 0.5) & (x <= 0.75), 100 * x * (1 - x) - 12.5, v2)
    v4 = torch.where((x > 0.75) & (x <= 1), 100 * (1 - x) ** 2, v3)
    return v4


def one_dim_function_l_exact(x):
    return x[:, 0:1] * (1 - x[:, 0:1])


def one_dim_sigmoid_1(x):
    return torch.sigmoid((one_dim_function_l_exact(x) - 0.0005) * 5 / 0.0005)


def one_dim_vi_source_f(x):
    return torch.zeros_like(x)


def one_dim_exact_solution_vi_gradient(x):
    ref_point = 1 / (2 * np.sqrt(2))
    v1 = torch.where((x >= ref_point) & (x <= 0.5), 100 * (1 - x) + 100 * x * -1, 0)
    v2 = torch.where((x >= 0) & (x < ref_point), 100 - 50 * np.sqrt(2), v1)
    v3 = torch.where((x > 0.5) & (x <= 1 - ref_point), 100 * (1 - x) + 100 * x * -1, v2)
    v4 = torch.where((x > 1 - ref_point) & (x <= 1), 50 * np.sqrt(2) - 100, v3)
    return v4


def one_dim_exact_solution_vi_laplacian(x):
    ref_point = 1 / (2 * np.sqrt(2))
    v1 = torch.where((x >= ref_point) & (x <= 0.5), 100 * (-1) + 100 * -1, 0)
    v2 = torch.where((x >= 0) & (x < ref_point), 0, v1)
    v3 = torch.where((x > 0.5) & (x <= 1 - ref_point), 100 * (-1) + 100 * -1, v2)
    v4 = torch.where((x > 1 - ref_point) & (x <= 1), 0, v3)
    return v4


def one_dim_obstacle_gradient(x):
    v1 = torch.where((x >= 0.25) & (x <= 0.5), 100 * (1 - x) + 100 * x * -1, 0)
    v2 = torch.where((x >= 0) & (x < 0.25), 2 * 100 * x, v1)
    v3 = torch.where((x > 0.5) & (x <= 0.75), 100 * (1 - x) + 100 * x * -1, v2)
    v4 = torch.where((x > 0.75) & (x <= 1), 2 * 100 * (1 - x) * -1, v3)
    return v4


################################################################################################################


################################################################################################################
def ns_one_dim_exact_solution_vi(x):
    A = 4 - 2 * np.sqrt(3)
    v1 = torch.where((x >= -2) & (x <= -2 + np.sqrt(3)), A * (x + 2), 0)
    v2 = torch.where((x > -2 + np.sqrt(3)) & (x <= 2 - np.sqrt(3)), 1 - x**2, v1)
    v3 = torch.where((x > 2 - np.sqrt(3)) & (x <= 2), A * (2 - x), v2)
    return v3


def ns_one_dim_gradient_exact_solution_vi(x):
    A = 4 - 2 * np.sqrt(3)
    v1 = torch.where((x >= -2) & (x <= -2 + np.sqrt(3)), A, 0)
    v2 = torch.where((x > -2 + np.sqrt(3)) & (x <= 2 - np.sqrt(3)), -2 * x, v1)
    v3 = torch.where((x > 2 - np.sqrt(3)) & (x <= 2), -1 * A, v2)
    return v3


def ns_one_dim_obstacle(x):
    return 1 - x**2


def ns_one_dim_boundary_condition(x):
    return torch.zeros_like(x)


def ns_one_dim_function_l_exact(x):
    return (2 + x[:, 0:1]) * (2 - x[:, 0:1])


def ns_one_dim_function_sigmoid_l(x):
    return torch.sigmoid((ns_one_dim_function_l_exact(x) - 0.005) * 5 / 0.005)


def ns_one_dim_vi_source_f(x):
    A = 4 - 2 * np.sqrt(3)
    v1 = torch.where((x >= -2) & (x <= -2 + np.sqrt(3)), A, 0)
    v2 = torch.where((x > -2 + np.sqrt(3)) & (x <= 2 - np.sqrt(3)), 2 - 2 * np.sqrt(3), v1)
    v3 = torch.where((x > 2 - np.sqrt(3)) & (x <= 2), -1 * A, v2)
    return v3


################################################################################################################


################################################################################################################
## example on [0,1]^2 from "A priori finite element error analysis for optimal control of the obstacle problem" by C. Meyer and O. Thoma
def z1(x):
    return -4096 * x**6 + 6144 * x**5 - 3072 * x**4 + 512 * x**3


def z2(x):
    return -244.140625 * x**6 + 585.9375 * x**5 - 468.75 * x**4 + 125 * x**3


def z1_grad(x):
    return -6 * 4096 * x**5 + 5 * 6144 * x**4 - 4 * 3072 * x**3 + 3 * 512 * x**2


def z2_grad(x):
    return -6 * 244.140625 * x**5 + 5 * 585.9375 * x**4 - 4 * 468.75 * x**3 + 3 * 125 * x**2


def z1_secondderiv(x):
    return -6 * 5 * 4096 * x**4 + 5 * 4 * 6144 * x**3 - 4 * 3 * 3072 * x**2 + 3 * 2 * 512 * x


def z2_secondderiv(x):
    return -6 * 5 * 244.140625 * x**4 + 5 * 4 * 585.9375 * x**3 - 4 * 3 * 468.75 * x**2 + 3 * 2 * 125 * x


def MT_two_dim_exact_solution_vi(x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    value = torch.where((x1 < 0.5) & (x2 < 0.8), z1(x1) * z2(x2), 0)

    return value


def MT_two_dim_exact_gardient(x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    value_x1 = torch.where((x1 < 0.5) & (x2 < 0.8), z1_grad(x1) * z2(x2), 0)
    value_x2 = torch.where((x1 < 0.5) & (x2 < 0.8), z1(x1) * z2_grad(x2), 0)

    return torch.concatenate([value_x1, value_x2], dim=1)


def MT_zero_solution(x):
    return torch.ones_like(x)[:, 0].reshape((-1, 1)) * 0.0000000001


def MT_two_dim_boundary_condition(x):
    return torch.zeros_like(x)[:, 0].reshape((-1, 1))


def MT_two_dim_exact_solution_for_plot(x, y):
    return np.where((x < 0.5) & (y < 0.8), z1(x) * z2(y), 0)


def MT_two_dim_obstacle(x):
    return (x * 0)[:, 0:1]


def MT_two_dim_function_l_exact(x):
    return (x[:, 0:1]) * (x[:, 1:2]) * (1 - x[:, 0:1]) * (1 - x[:, 1:2])


def MT_laplacian(x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    Laplacian = torch.where((x1 < 0.5) & (x2 < 0.8), z2(x2) * z1_secondderiv(x1) + z1(x1) * z2_secondderiv(x2), 0)
    value = Laplacian
    return value


def MT_two_dim_vi_source_f(x):
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    zeta_term = torch.where((x1 > 0.5) & (x1 < 1) & (x2 < 0.8) & (x2 > 0), z1(x1 - 0.5) * z2(x2), 0)
    Laplacian = torch.where((x1 < 0.5) & (x2 < 0.8), z2(x2) * z1_secondderiv(x1) + z1(x1) * z2_secondderiv(x2), 0)
    value = -1 * Laplacian - zeta_term
    return value


################################################################################################################


################################################################################################################
# Example from https://www.uni-due.de/imperia/md/images/mathematik/ruhr_pad/ruhrpad-2016-10.pdf


def exp_func(x):
    value_old = torch.where(x > 0, torch.exp(-1 / x), 0)  # the original

    safe_x = torch.where(x > 0, x, 1)
    value_new = torch.where(x > 0, torch.exp(-1 / safe_x), 0)
    if torch.equal(value_old, value_new) == False:
        assert False, "problem with exp_func!"
    return value_new


def exp_func_der(x):
    return torch.where(x > 0, torch.exp(-1 / x) / (x**2), 0)


def ccinf_function(x):
    return exp_func(0.4 - torch.abs(x)) / (exp_func(torch.abs(x) - 0.3) + exp_func(0.4 - torch.abs(x)))


def CM_obstacle(x):
    alpha = 0.4

    value = torch.where(
        x <= 0,
        ccinf_function(x + 0.5) * (1.5 - 12 * torch.abs(x + 0.5) ** (2 - alpha)) - 0.5,
        ccinf_function(x - 0.5) * (1.5 - 12 * torch.abs(x - 0.5) ** (2 - alpha)) - 0.5,
    )
    return value


def CM_exact_solution(x):
    if x.is_cuda:
        eta_alpha = torch.tensor([0.02376]).to("cuda")
    else:
        eta_alpha = torch.tensor([0.02376]).to("cpu")

    v1 = torch.where(
        (x > -1) & (x < -1 * eta_alpha - 0.5), CM_obstacle(-1 * eta_alpha - 0.5) * (x + 1) / (0.5 - eta_alpha), 0
    )
    v2 = torch.where((x >= -0.5 - 1 * eta_alpha) & (x < -0.5), CM_obstacle(x), v1)
    v3 = torch.where((x >= -0.5) & (x < 0.5), 1, v2)
    v4 = torch.where((x < 0.5 + eta_alpha) & (x >= 0.5), CM_obstacle(x), v3)
    v5 = torch.where((x >= eta_alpha + 0.5) & (x < 1), CM_obstacle(eta_alpha + 0.5) * (x - 1) / (-0.5 + eta_alpha), v4)
    return v5


def CM_boundary_condition(x):
    return torch.zeros_like(x)


def CM_function_l_exact(x):
    return (1 + x[:, 0:1]) * (1 - x[:, 0:1])


def CM_function_sigmoid(x):
    return torch.sigmoid((CM_function_l_exact(x) - 0.005) * 5 / 0.005)


def CM_source(x):
    return 0


################################################################################################################


################################################################################################################
## example on (0,1) from the above paper
def sin_exact_solution_vi(x):
    v1 = torch.where((x >= 0) & (x <= 0.25), 10 * torch.sin(2 * torch.pi * x), 0)
    v2 = torch.where((x > 0.25) & (x <= 0.5), 10, v1)
    v3 = torch.where((x > 0.5) & (x <= 0.75), 10, v2)
    v4 = torch.where((x > 0.75) & (x <= 1), 10 * torch.sin(2 * torch.pi * (1 - x)), v3)

    return v4


def sin_boundary_condition(x):
    return torch.zeros_like(x)


def sin_obstacle(x):
    v1 = torch.where((x >= 0) & (x <= 0.25), 10 * torch.sin(2 * torch.pi * x), 0)
    v2 = torch.where((x > 0.25) & (x <= 0.5), 5 * torch.cos(torch.pi * (4 * x - 1)) + 5, v1)
    v3 = torch.where((x > 0.5) & (x <= 0.75), 5 * torch.cos(torch.pi * (4 * (1 - x) - 1)) + 5, v2)
    v4 = torch.where((x > 0.75) & (x <= 1), 10 * torch.sin(2 * torch.pi * (1 - x)), v3)
    return v4


def sin_function_l_exact(x):
    return x[:, 0:1] * (1 - x[:, 0:1])


def sin_function_sigmoid(x):
    return torch.sigmoid((sin_function_l_exact(x) - 0.005) * 5 / 0.005)


def sin_source_f(x):
    return torch.zeros_like(x)


################################################################################################################


class Example:
    def __init__(
        self,
        domain,
        source_term,
        obstacle,
        exact_solution,
        boundary_data,
        boundary_data_setter_for_testfn,
        differential_operator="Laplacian",
        exact_solution_for_plot=None,
        name=None,
        exact_solution_grad=None,
    ):
        self.domain = domain
        self.shift_obstacle = 0.0
        self.source_term = source_term
        self.original_obstacle = obstacle
        self.obstacle = self.obstacle_shift_function(obstacle, self.shift_obstacle)
        self.exact_solution = exact_solution
        self.boundary_data = boundary_data
        self.boundary_data_setter_for_testfn = boundary_data_setter_for_testfn
        self.differential_operator = differential_operator
        self.exact_solution_for_plot = exact_solution_for_plot
        self.name = name
        self.exact_solution_grad = exact_solution_grad

    def obstacle_shift_function(self, obstacle, shift):
        def fn(x):
            return obstacle(x) + shift

        return fn

    def set_shift_obstacle(self, shift):
        if shift is None:
            self.shift_obstacle = 0
        else:
            self.shift_obstacle = shift
        self.obstacle = self.obstacle_shift_function(self.original_obstacle, self.shift_obstacle)


### domains
unit_interval = Domain(pdim=1, left_coord=0, right_coord=1)
unit_square = Domain(pdim=2, left_coord=0, right_coord=1)
interval_minus2_to_2 = Domain(pdim=1, left_coord=-2, right_coord=2)
square_minus2_to_2 = Domain(pdim=2, left_coord=-2, right_coord=2)
square_minus1_to_1 = Domain(pdim=2, left_coord=-1, right_coord=1)
interval_minus1_to_1 = Domain(pdim=1, left_coord=-1, right_coord=1)


EXAMPLES_MAP = {
    "one_dim": Example(
        domain=unit_interval,
        source_term=one_dim_vi_source_f,
        obstacle=one_dim_obstacle,
        exact_solution=one_dim_exact_solution_vi,
        boundary_data=one_dim_boundary_condition,
        boundary_data_setter_for_testfn=one_dim_function_l_exact,
        name="one_dim",
    ),
    "ns_one_dim": Example(
        domain=interval_minus2_to_2,
        source_term=ns_one_dim_vi_source_f,
        obstacle=ns_one_dim_obstacle,
        exact_solution=ns_one_dim_exact_solution_vi,
        boundary_data=ns_one_dim_boundary_condition,
        boundary_data_setter_for_testfn=ns_one_dim_function_l_exact,
        differential_operator="LaplacianPlusDerivative",
        exact_solution_for_plot=ns_one_dim_exact_solution_vi,
        exact_solution_grad=ns_one_dim_gradient_exact_solution_vi,
        name="ns_one_dim",
    ),
    "two_dim": Example(
        domain=square_minus2_to_2,
        source_term=two_dim_vi_source_f,
        obstacle=two_dim_obstacle,
        exact_solution=two_dim_exact_solution_vi,
        boundary_data=two_dim_exact_solution_vi,
        boundary_data_setter_for_testfn=two_dim_function_l_exact,
        exact_solution_for_plot=two_dim_exact_solution_for_plot,
        name="two_dim",
    ),
    "ns_two_dim": Example(
        domain=square_minus2_to_2,
        source_term=ns_two_dim_vi_source_f,
        obstacle=ns_two_dim_obstacle,
        exact_solution=ns_two_dim_exact_solution_vi,
        boundary_data=ns_two_dim_exact_solution_vi,
        boundary_data_setter_for_testfn=ns_two_dim_function_l_exact,
        differential_operator="LaplacianPlusDerivative2D",
        exact_solution_for_plot=ns_two_dim_exact_solution_for_plot,
        name="ns_two_dim",
    ),
    "NSV_two_dim": Example(
        domain=square_minus1_to_1,
        source_term=NSV_two_dim_vi_source_f,
        obstacle=NSV_two_dim_obstacle,
        exact_solution=NSV_two_dim_exact_solution_vi,
        boundary_data=NSV_two_dim_exact_solution_vi,
        boundary_data_setter_for_testfn=NSV_two_dim_function_l_exact,
        name="NSV_two_dim",
    ),
    "KS": Example(
        domain=square_minus1_to_1,
        source_term=KS_vi_source_f,
        obstacle=KS_obstacle,
        exact_solution=KS_exact_solution_vi,
        boundary_data=KS_exact_solution_vi,
        boundary_data_setter_for_testfn=KS_function_l_exact,
        name="KS",
        exact_solution_grad=KS_gradient_exact_solution_vi,
    ),
    "KS_ns": Example(
        domain=square_minus1_to_1,
        source_term=KS_ns_vi_source_f,
        obstacle=KS_ns_obstacle,
        exact_solution=KS_ns_exact_solution_vi,
        boundary_data=KS_ns_exact_solution_vi,
        boundary_data_setter_for_testfn=KS_ns_function_l_exact,
        name="KS_ns",
        exact_solution_grad=KS_ns_gradient_exact_solution_vi,
    ),
    "MT": Example(
        domain=unit_square,
        source_term=MT_two_dim_vi_source_f,
        obstacle=MT_two_dim_obstacle,
        exact_solution=MT_two_dim_exact_solution_vi,
        boundary_data=MT_two_dim_boundary_condition,
        boundary_data_setter_for_testfn=MT_two_dim_function_l_exact,
        exact_solution_for_plot=MT_two_dim_exact_solution_for_plot,
        name="MT",
        exact_solution_grad=MT_two_dim_exact_gardient,
    ),
    "CM": Example(
        domain=interval_minus1_to_1,
        source_term=CM_source,
        obstacle=CM_obstacle,
        exact_solution=CM_exact_solution,
        boundary_data=CM_boundary_condition,
        boundary_data_setter_for_testfn=CM_function_l_exact,
        name="CM",
    ),
    "sine": Example(
        domain=unit_interval,
        source_term=sin_source_f,
        obstacle=sin_obstacle,
        exact_solution=sin_exact_solution_vi,
        boundary_data=sin_boundary_condition,
        boundary_data_setter_for_testfn=sin_function_l_exact,
        name="sine",
    ),
    "MT_zero": Example(
        domain=unit_square,
        source_term=MT_two_dim_vi_source_f,
        obstacle=MT_two_dim_obstacle,
        exact_solution=MT_zero_solution,
        boundary_data=MT_two_dim_boundary_condition,
        boundary_data_setter_for_testfn=MT_two_dim_function_l_exact,
        exact_solution_for_plot=MT_zero_solution,
        name="MT_zero",
    ),
}
