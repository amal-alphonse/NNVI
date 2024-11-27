from abc import ABC, abstractmethod
import math

import numpy as np
import torch
from scipy.stats import qmc

from nnvi.grids.domain import Domain
from nnvi.loss import gradients

__all__ = ["Grid", "SimpleGrid", "UniformRandomCube", "SobolGrid"]


class Grid(ABC):
    @abstractmethod
    def get_interior_points(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_boundary_points(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def interior_points_count(self) -> int:
        pass


class SimpleGrid(Grid):
    def __init__(self, domain: Domain, N_interior: int, N_boundary: int, randomise: bool = False) -> None:
        self.domain = domain
        self.pdim = domain.get_pdim()
        self.square_left_coord = domain.get_left_coord()
        self.square_right_coord = domain.get_right_coord()
        self.N_interior = N_interior
        self.N_boundary = N_boundary
        self.randomise = randomise

        if self.pdim == 1:
            self.n_points_per_axis = self.N_interior
        else:  # self.pdim == 2
            self.n_points_per_axis = int(np.floor(np.sqrt(self.N_interior)))

        if not randomise:
            self.linspace_or_uniform = np.linspace
        else:
            self.linspace_or_uniform = np.random.uniform

    def get_interior_points(self) -> torch.Tensor:
        if self.linspace_or_uniform == np.linspace:
            # +1 because we remove the starting point (we only want interior points). linspace includes low and high
            # (endpoint=False excludes high)
            X_axis = self.linspace_or_uniform(
                self.square_left_coord, self.square_right_coord, self.n_points_per_axis + 1, endpoint=False
            )[1:]
        else:
            # uniform excludes high but includes low.
            X_axis = np.sort(
                self.linspace_or_uniform(self.square_left_coord, self.square_right_coord, self.n_points_per_axis + 1)
            )[1:]

        if self.pdim == 1:
            return torch.from_numpy(np.reshape(X_axis, [-1, 1])).float()
        else:  # self.pdim == 2
            if self.linspace_or_uniform == np.linspace:
                Y_axis = self.linspace_or_uniform(
                    self.square_left_coord, self.square_right_coord, self.n_points_per_axis + 1, endpoint=False
                )[1:]
            else:
                Y_axis = np.sort(
                    self.linspace_or_uniform(
                        self.square_left_coord, self.square_right_coord, self.n_points_per_axis + 1
                    )
                )[1:]

            X, Y = np.meshgrid(np.sort(X_axis), np.sort(Y_axis))
            X_flat = np.reshape(X, [-1, 1])
            Y_flat = np.reshape(Y, [-1, 1])
            Input_all_np = np.stack([X_flat, Y_flat], 1).squeeze()
            xr = torch.from_numpy(Input_all_np).float()

            return xr

    def get_mesh_points(self) -> torch.Tensor:
        if self.linspace_or_uniform == np.linspace:
            # +1 because we remove the starting point (we only want interior points). linspace includes low and high
            # (endpoint=False excludes high)
            X_axis = self.linspace_or_uniform(
                self.square_left_coord, self.square_right_coord, self.n_points_per_axis + 1, endpoint=True
            )
        else:
            # uniform excludes high but includes low.
            X_axis = np.sort(
                self.linspace_or_uniform(self.square_left_coord, self.square_right_coord, self.n_points_per_axis + 1)
            )

        if self.pdim == 1:
            X=None
            Y=None
            return X,Y,torch.from_numpy(np.reshape(X_axis, [-1, 1])).float()
        else:  # self.pdim == 2
            if self.linspace_or_uniform == np.linspace:
                Y_axis = self.linspace_or_uniform(
                    self.square_left_coord, self.square_right_coord, self.n_points_per_axis + 1, endpoint=False
                )
            else:
                Y_axis = np.sort(
                    self.linspace_or_uniform(
                        self.square_left_coord, self.square_right_coord, self.n_points_per_axis + 1
                    )
                )

            X, Y = np.meshgrid(np.sort(X_axis), np.sort(Y_axis))
            X_flat = np.reshape(X, [-1, 1])
            Y_flat = np.reshape(Y, [-1, 1])
            Input_all_np = np.stack([X_flat, Y_flat], 1).squeeze()
            xr = torch.from_numpy(Input_all_np).float()

            return X,Y,xr

    @property
    def interior_points_count(self) -> int:
        return self.n_points_per_axis**self.pdim

    def get_boundary_points(self) -> torch.Tensor:
        if self.pdim == 1:
            return torch.from_numpy(
                np.reshape(np.array([self.square_left_coord, self.square_right_coord]), [-1, 1])
            ).float()
        else:
            n_points = int(np.floor(np.sqrt(self.N_boundary)))
            X_axis = self.linspace_or_uniform(self.square_left_coord, self.square_right_coord, n_points)
            Y_axis = self.linspace_or_uniform(self.square_left_coord, self.square_right_coord, n_points)
            if self.randomise == True:
                X_axis = np.sort(X_axis)
                Y_axis = np.sort(Y_axis)
                X_axis[0] = self.square_left_coord
                X_axis[-1] = self.square_right_coord
                Y_axis[0] = self.square_left_coord
                Y_axis[-1] = self.square_right_coord

            X, Y = np.meshgrid(np.sort(X_axis), np.sort(Y_axis))

            Input_boundary_x = np.concatenate([X[0, :], X[-1, :], X[:, 0], X[:, -1]])
            Input_boundary_y = np.concatenate([Y[0, :], Y[-1, :], Y[:, 0], Y[:, -1]])
            Input_boundary = np.stack([Input_boundary_x, Input_boundary_y], 1)
            xb = torch.from_numpy(Input_boundary).float()

            for point in xb:
                assert (
                    point[0] == self.square_left_coord
                    or point[0] == self.square_right_coord
                    or point[1] == self.square_left_coord
                    or point[1] == self.square_right_coord
                ), "get_boundary_points() did not retrieve a boundary point"

            return xb



class UniformRandomCube(Grid):
    def __init__(self, domain: Domain, N_interior: int, N_boundary: int,randomise: bool = True) -> None:
        assert domain.pdim in {1, 2}, "pdim must be 1 or 2"
        self.domain = domain
        self.N_interior = N_interior
        self.N_boundary = math.ceil(N_boundary / 4) * 4 if self.domain.pdim > 1 else 2
        self.square_left_coord = domain.get_left_coord()
        self.square_right_coord = domain.get_right_coord()

    @property
    def interior_points_count(self) -> int:
        return self.N_interior

    def get_interior_points(self) -> torch.Tensor:
        interior = torch.rand(self.N_interior, self.domain.pdim)
        size = self.domain.get_right_coord() - self.domain.get_left_coord()
        offset = self.domain.get_left_coord()

        return (interior * size) + offset

    def get_boundary_points(self) -> torch.Tensor:
        size = self.domain.get_right_coord() - self.domain.get_left_coord()
        offset = self.domain.get_left_coord()

        if self.domain.pdim == 1:
            boundary = (torch.tensor([0.0, 1.0]).unsqueeze(-1) * size) + offset
        elif self.domain.pdim == 2:
            bdy = torch.rand(self.N_boundary // 4)
            xb1 = torch.stack((torch.zeros_like(bdy), bdy), dim=1)
            bdy = torch.rand(self.N_boundary // 4)
            xb2 = torch.stack((torch.ones_like(bdy), bdy), dim=1)
            bdy = torch.rand(self.N_boundary // 4)
            xb3 = torch.stack((bdy, torch.zeros_like(bdy)), dim=1)
            bdy = torch.rand(self.N_boundary // 4)
            xb4 = torch.stack((bdy, torch.ones_like(bdy)), dim=1)

            boundary = (torch.concat((xb1, xb2, xb3, xb4)) * size) + offset
            for point in boundary:
                assert (
                    point[0] == self.square_left_coord
                    or point[0] == self.square_right_coord
                    or point[1] == self.square_left_coord
                    or point[1] == self.square_right_coord
                ), "get_boundary_points() did not retrieve a boundary point"
        else:
            raise NotImplementedError()

        return boundary


class SobolGrid(Grid):
    """
    Scramble=True destroys reproduceability (every run of the program gives different points). We can set seed,
    but then we get the same set of points every epoch. Scramble=False doesn't seem to fill the grid so well.
    """

    def __init__(self, domain: Domain, N_interior: int, N_boundary: int,randomise: bool = False) -> None:
        assert domain.pdim in {1, 2}, "pdim must be 1 or 2"
        self.domain = domain
        self.N_interior = N_interior
        self.N_boundary = math.ceil(N_boundary / 4) * 4 if self.domain.pdim > 1 else 2
        self.square_left_coord = domain.get_left_coord()
        self.square_right_coord = domain.get_right_coord()

    @property
    def interior_points_count(self) -> int:
        return self.N_interior

    def get_interior_points(self) -> torch.Tensor:
        if self.domain.pdim == 1:
            l_bounds = [self.square_left_coord]
            u_bounds = [self.square_right_coord]
        elif self.domain.pdim == 2:
            l_bounds = [self.square_left_coord, self.square_left_coord]
            u_bounds = [self.square_right_coord, self.square_right_coord]

        sampler = qmc.Sobol(d=self.domain.pdim, scramble=True, seed=123)
        m = np.log2(self.N_interior)
        sample = sampler.random_base2(np.int32(m))
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        points = torch.tensor(sample_scaled).to(dtype=torch.float32)

        return points

    def get_boundary_points(self) -> torch.Tensor:
        size = self.domain.get_right_coord() - self.domain.get_left_coord()
        offset = self.domain.get_left_coord()

        if self.domain.pdim == 1:
            boundary = (torch.tensor([0.0, 1.0]).unsqueeze(-1) * size) + offset
        elif self.domain.pdim == 2:
            sampler = qmc.Sobol(d=1, scramble=True, seed=123)
            m = np.log2(self.N_boundary // 4)
            sample = sampler.random_base2(np.int32(m))
            sample_scaled = qmc.scale(sample, l_bounds=[self.square_left_coord], u_bounds=[self.square_right_coord])

            bdy = torch.tensor(sample_scaled).reshape(-1).to(dtype=torch.float32)
            xb1 = torch.stack((torch.ones_like(bdy) * self.square_left_coord, bdy), dim=1)
            xb2 = torch.stack((torch.ones_like(bdy) * self.square_right_coord, bdy), dim=1)
            xb3 = torch.stack((bdy, torch.ones_like(bdy) * self.square_left_coord), dim=1)
            xb4 = torch.stack((bdy, torch.ones_like(bdy) * self.square_right_coord), dim=1)

            boundary = torch.concat((xb1, xb2, xb3, xb4))
            for point in boundary:
                assert (
                    point[0] == self.square_left_coord
                    or point[0] == self.square_right_coord
                    or point[1] == self.square_left_coord
                    or point[1] == self.square_right_coord
                ), "get_boundary_points() did not retrieve a boundary point"
        else:
            raise NotImplementedError()

        return boundary.to(dtype=torch.float32)

class ImportanceSampled(Grid):
    def __init__(self,grid,prob_of_replace_by_new=0.6,randomise: bool = False):
        self.grid=grid
        self.prob_of_replace_by_new=prob_of_replace_by_new
        
        self.N_interior = grid.interior_points_count
        self.N_boundary = grid.N_boundary
        self.interior_points = grid.get_interior_points()
        self.interior_weights = torch.ones(self.N_interior)*(1/self.N_interior)
        self.weights_for_integration = self.interior_weights
        self.updated=True
        
    @property
    def interior_points_count(self) -> int:
        return self.N_interior

    def get_boundary_points(self):
        return self.grid.get_boundary_points()
    
    def get_interior_points(self):
        self.updated=False
        return self.interior_points 
        
    def update_points(self):
        self.updated=True
        raise NotImplementedError()
    def update_only_pts_int_weigs(self):
        self.updated= True
        if self.prob_of_replace_by_new<0.9:
            nb_of_new_points = np.max((np.random.binomial(n=self.N_interior,p=self.prob_of_replace_by_new),10))
            self.new_points = self.grid.get_interior_points()[:nb_of_new_points]
            
            unique_pts, inverse_indices = torch.unique(self.interior_points,return_inverse=True,dim=0)
            nb_of_unique =  unique_pts.size(dim=0)
            idx_of_unique = [torch.min(torch.arange(start=0,end=self.N_interior)[inverse_indices == i_i]) for i_i in range(nb_of_unique)]
            weights = np.squeeze((self.interior_weights[idx_of_unique]).numpy(force=True))
            norm = np.sum(weights)
            self.size_old=self.N_interior-nb_of_new_points
            if norm > 0:
                self.idx_for_sample_from_old = np.random.choice(idx_of_unique,size=self.size_old,p=weights/norm)
            else: 
                self.idx_for_sample_from_old = np.random.choice(idx_of_unique,size=self.size_old)
                #Update of the new points
            self.interior_points = torch.cat((self.new_points,self.interior_points[self.idx_for_sample_from_old]),dim=0 )
        
            w_new_points = self.prob_of_replace_by_new*torch.ones(nb_of_new_points)*(1/nb_of_new_points)
            pre_weights=torch.tensor(self.interior_weights[self.idx_for_sample_from_old]).to('cpu')
            w_old = (1-self.prob_of_replace_by_new)*pre_weights/torch.sum(pre_weights)
            self.weights_for_integration = (torch.cat((w_new_points,w_old),dim=0)*self.N_interior).detach()
        else:
            self.new_points = self.grid.get_interior_points()
            self.size_old = 0
            self.idx_for_sample_from_old = []
            self.interior_points = self.new_points
            self.weights_for_integration = self.N_interior * torch.ones_like(self.new_points)
            
import copy 
class ImportanceForSolution(ImportanceSampled):
    def __init__(self,grid,test_fn,prob_of_replace_by_new=0.6,randomise: bool = False):
        super(ImportanceForSolution,self).__init__(grid,prob_of_replace_by_new)
        self.prior_test_fn=copy.deepcopy(test_fn)
        
        
    def update_points(self,updated_test_fn,device):
        self.update_only_pts_int_weigs()
        
        
        pts=self.interior_points.to(device)
        old_fn_val = self.prior_test_fn(pts)
        new_fn_val = updated_test_fn(pts)
        self.interior_weights = torch.abs(old_fn_val-new_fn_val)
        self.prior_test_fn = copy.deepcopy(updated_test_fn)

class ImportanceForTest(ImportanceSampled):
    def __init__(self,grid,prob_of_replace_by_new=0.6):
        super(ImportanceForTest,self).__init__(grid,prob_of_replace_by_new)

    def update_points(self,updated_sol_fn,device):
        self.update_only_pts_int_weigs()
        self.new_points = self.new_points.to(device)

        if self.prob_of_replace_by_new<0.9:
            self.new_points.requires_grad_()
            vals=updated_sol_fn(self.new_points)
            self.new_sol_val = torch.abs(vals)
            grad_at_new = gradients(input=self.new_points, output=vals)
            self.new_sol_grad = torch.abs(grad_at_new)
            self.new_sol_grad=torch.sum(self.new_sol_grad,1)*0.5

            
            mask = torch.bernoulli(torch.ones_like(self.new_sol_val)*0.5)
            w_new_points = self.prob_of_replace_by_new*(torch.squeeze(self.new_sol_val*mask)
                                                        +
                                                        (self.new_sol_grad*torch.squeeze(1-mask)))
            
            weights = np.squeeze(self.interior_weights.numpy(force=True))
            
            w_old = (1-self.prob_of_replace_by_new)*torch.tensor(weights[self.idx_for_sample_from_old])
            w_old=w_old.to(device)
            
            self.interior_weights = torch.cat((w_new_points,w_old),dim=0).detach()
        else:
            self.interior_weights = torch.ones_like(self.new_points)
