"""Neural ODE implementations and solvers."""

import numpy as np
from typing import Callable, Optional, Tuple
from scipy.integrate import solve_ivp
from core import (
    Estimator,
    check_array,
    get_logger,
    ValidationError
)

class NeuralODE(Estimator):
    """Neural Ordinary Differential Equation model."""
    
    def __init__(self,
                 net: Callable,
                 t_span: Tuple[float, float],
                 solver: str = 'dopri5'):
        self.net = net
        self.t_span = t_span
        self.solver = solver
        
    def forward(self, x0: np.ndarray, t_eval: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve ODE system forward in time."""
        solution = solve_ivp(
            fun=self.net,
            t_span=self.t_span,
            y0=x0,
            method=self.solver,
            t_eval=t_eval
        )
        return solution.y