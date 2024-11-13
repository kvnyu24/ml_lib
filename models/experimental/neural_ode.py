"""Neural ODE implementations and solvers."""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Union
from scipy.integrate import solve_ivp
from core import (
    Estimator,
    check_array,
    get_logger,
    ValidationError,
    EPSILON
)

logger = get_logger(__name__)

class NeuralODE(Estimator):
    """Neural Ordinary Differential Equation model with robust solvers and adjoint methods."""
    
    SUPPORTED_SOLVERS = ['dopri5', 'rk45', 'rk23', 'dop853', 'radau', 'bdf']
    
    def __init__(
        self,
        net: Callable,
        t_span: Tuple[float, float],
        solver: str = 'dopri5',
        rtol: float = 1e-7,
        atol: float = 1e-9,
        method: str = 'adjoint',
        regularization: Optional[float] = None
    ):
        """Initialize Neural ODE.
        
        Args:
            net: Neural network defining the ODE dynamics
            t_span: Time span to solve over (t0, t1)
            solver: ODE solver method
            rtol: Relative tolerance for solver
            atol: Absolute tolerance for solver
            method: Integration method ('adjoint' or 'direct')
            regularization: L2 regularization coefficient
        """
        if not callable(net):
            raise ValidationError("net must be a callable function")
            
        if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
            raise ValidationError("t_span must be tuple of (t0, t1)")
            
        if t_span[1] <= t_span[0]:
            raise ValidationError("t_span[1] must be greater than t_span[0]")
            
        if solver not in self.SUPPORTED_SOLVERS:
            raise ValidationError(f"solver must be one of {self.SUPPORTED_SOLVERS}")
            
        self.net = net
        self.t_span = t_span
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.regularization = regularization
        
    def _regularize(self, x: np.ndarray) -> float:
        """Apply L2 regularization."""
        if self.regularization:
            return self.regularization * np.sum(x ** 2)
        return 0.0
        
    def forward(self, x0: np.ndarray, t_eval: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve ODE system forward in time with error handling and regularization.
        
        Args:
            x0: Initial conditions
            t_eval: Times at which to evaluate solution
            
        Returns:
            Solution trajectories
            
        Raises:
            ValidationError: If solver fails or inputs invalid
        """
        x0 = check_array(x0)
        
        if t_eval is not None:
            t_eval = check_array(t_eval)
            if not np.all(np.diff(t_eval) > 0):
                raise ValidationError("t_eval must be strictly increasing")
        
        try:
            solution = solve_ivp(
                fun=lambda t, x: self.net(t, x) + self._regularize(x),
                t_span=self.t_span,
                y0=x0,
                method=self.solver,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
                dense_output=True
            )
            
            if not solution.success:
                raise ValidationError(f"Solver failed: {solution.message}")
                
            return solution.y
            
        except Exception as e:
            logger.error(f"ODE solver failed: {str(e)}")
            raise ValidationError(f"ODE solver failed: {str(e)}")