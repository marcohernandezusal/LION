import random
from typing import Callable
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.optim.optimizer import Optimizer
from tomosipo.torch_support import to_autograd
from tqdm import tqdm
from LION.CTtools.ct_geometry import Geometry
from LION.classical_algorithms.fdk import fdk
from LION.exceptions.exceptions import LIONSolverException
from LION.models.LIONmodel import LIONmodel, ModelInputType
from LION.optimizers.LIONsolver import LIONsolver, SolverParams


def get_rotation_matrix(angle: float):
    theta = torch.tensor(angle)
    s = torch.sin(theta)
    c = torch.cos(theta)
    return torch.tensor([[c, -s], [s, c]])


class EquivariantSolverParams(SolverParams):
    def __init__(
        self, transformation_group: list[Callable], equivariance_strength: float
    ):
        super().__init__()
        self.transformation_group = transformation_group
        self.equivariance_strength = equivariance_strength


class EquivariantSolver(LIONsolver):
    def __init__(
        self,
        model: LIONmodel,
        optimizer: Optimizer,
        loss_fn: Callable,
        geometry: Geometry,
        verbose: bool = True,
        device: torch.device = torch.device(f"cuda:{torch.cuda.current_device()}"),
        solver_params: SolverParams | None = None,
    ) -> None:
        super().__init__(
            model, optimizer, loss_fn, geometry, verbose, device, solver_params
        )
        self.transformation_group = self.solver_params.transformation_group
        self.alpha = self.solver_params.equivariance_strength
        self.A = to_autograd(self.op, num_extra_dims=1)
        self.AT = to_autograd(self.op.T, num_extra_dims=1)

    @staticmethod
    def rotation_group(cardinality: int):
        assert 360 % cardinality == 0
        angle_increment = 360 / cardinality

        return [lambda x: TF.rotate(x, i * angle_increment) for i in range(cardinality)]

    @staticmethod
    def default_parameters() -> EquivariantSolverParams:
        return EquivariantSolverParams(EquivariantSolver.rotation_group(360), 100)

    def mini_batch_step(self, sino_batch, target_batch) -> torch.Tensor:
        random_transform = random.choice(self.transformation_group)

        if needs_image := (self.model.get_input_type() == ModelInputType.IMAGE):
            recon1 = self.model(fdk(sino_batch, self.op))
        else:
            recon1 = self.model(sino_batch)

        transformed_recon1 = random_transform(recon1)

        if needs_image:
            recon2 = self.model(fdk(self.A(transformed_recon1), self.op))
        else:
            recon2 = self.model(self.A(transformed_recon1))

        return self.loss_fn(self.A(recon1), sino_batch) + self.alpha * self.loss_fn(
            recon2, transformed_recon1
        )
        # data consistency + equivariance
