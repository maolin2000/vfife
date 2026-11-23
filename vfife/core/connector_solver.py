"""Standalone entry point for six-component connector Line2 meshes."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .problem import Problem
from . import member_axes
from .mixed_structural_solver import MixedStructuralSolver
from ..elements import connector


class ConnectorSolver(MixedStructuralSolver):
    """A homogeneous connector mesh using the common six-DOF member solver."""

    def __init__(self, problem, mesh, material, log=print):
        problem = Problem.of(problem)        # typed, and its dict
        props = deepcopy(dict(problem["element"]))
        props["region"] = "all"
        super().__init__(problem.derive(members=[props]), mesh, material, log)

    def summary(self):
        props = self.member_props[0]
        K = connector.effective_stiffness(props)
        C = connector.matrix(props.get("connector_damping"))
        return (f"  mesh          {self.nn} nodes, {self.nel} Line2 elements\n"
                f"  element       connector2, 6 DOF/node, {props.get('coordinate_system','local')} axes\n"
                f"  stiffness     diag={np.diag(K).tolist()}\n"
                f"  damping       diag={np.diag(C).tolist()}\n"
                f"  material      {self.material.name}; backend numpy\n"
                f"  mode          {self.mode}; dt={self.dt:.4e}, {self.n_steps} steps"
                + member_axes.summary_lines(self))        # v1.0
