"""Global optimization algorithms."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.optimize import fsolve

from core import (
    Optimizer,
    Function,
    EPSILON,
    get_logger
)

# Configure logging
logger = get_logger(__name__)

class ParticleSwarmOptimizer(Optimizer):
    """Particle Swarm Optimization (PSO) algorithm."""
    
    def __init__(self, n_particles: int = 30, omega: float = 0.7,
                 phi_p: float = 2.0, phi_g: float = 2.0):
        super().__init__()
        self.n_particles = n_particles
        self.omega = omega  # Inertia weight
        self.phi_p = phi_p  # Personal best weight
        self.phi_g = phi_g  # Global best weight
        
    def minimize(self, func: Function, bounds: np.ndarray, 
                max_iter: int = 100) -> Tuple[np.ndarray, float]:
        dim = len(bounds)
        # Initialize particles and velocities
        particles = np.random.uniform(bounds[:,0], bounds[:,1], 
                                    size=(self.n_particles, dim))
        velocities = np.zeros((self.n_particles, dim))
        
        # Initialize personal and global bests
        p_best_pos = particles.copy()
        p_best_scores = np.array([func(p) for p in particles])
        g_best_idx = np.argmin(p_best_scores)
        g_best_pos = p_best_pos[g_best_idx].copy()
        
        for _ in range(max_iter):
            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            velocities = (self.omega * velocities + 
                        self.phi_p * r1 * (p_best_pos - particles) +
                        self.phi_g * r2 * (g_best_pos - particles))
            
            particles += velocities
            
            # Clip to bounds
            np.clip(particles, bounds[:,0], bounds[:,1], out=particles)
            
            # Update bests
            scores = np.array([func(p) for p in particles])
            better_idxs = scores < p_best_scores
            p_best_scores[better_idxs] = scores[better_idxs]
            p_best_pos[better_idxs] = particles[better_idxs]
            
            g_best_idx = np.argmin(p_best_scores)
            g_best_pos = p_best_pos[g_best_idx].copy()
            
        return g_best_pos, p_best_scores[g_best_idx]

class NelderMead(Optimizer):
    """Nelder-Mead simplex optimization algorithm."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5,
                 gamma: float = 2.0, delta: float = 0.5):
        super().__init__()
        self.alpha = alpha  # Reflection coefficient
        self.beta = beta   # Contraction coefficient 
        self.gamma = gamma # Expansion coefficient
        self.delta = delta # Shrink coefficient
        
    def minimize(self, func: Function, x0: np.ndarray, 
                max_iter: int = 1000, tol: float = 1e-8) -> Tuple[np.ndarray, float]:
        n = len(x0)
        # Initialize simplex
        simplex = np.zeros((n+1, n))
        simplex[0] = x0
        for i in range(n):
            point = x0.copy()
            point[i] = point[i] + 0.05
            simplex[i+1] = point
            
        # Evaluate function at all points
        values = np.array([func(point) for point in simplex])
        
        for _ in range(max_iter):
            # Order points
            order = np.argsort(values)
            simplex = simplex[order]
            values = values[order]
            
            # Check convergence
            if np.max(np.abs(values[1:] - values[0])) < tol:
                break
                
            # Calculate centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            reflected = centroid + self.alpha*(centroid - simplex[-1])
            reflected_val = func(reflected)
            
            if values[0] <= reflected_val < values[-2]:
                simplex[-1] = reflected
                values[-1] = reflected_val
                continue
                
            # Expansion
            if reflected_val < values[0]:
                expanded = centroid + self.gamma*(reflected - centroid)
                expanded_val = func(expanded)
                if expanded_val < reflected_val:
                    simplex[-1] = expanded
                    values[-1] = expanded_val
                else:
                    simplex[-1] = reflected
                    values[-1] = reflected_val
                continue
                
            # Contraction
            contracted = centroid + self.beta*(simplex[-1] - centroid)
            contracted_val = func(contracted)
            if contracted_val < values[-1]:
                simplex[-1] = contracted
                values[-1] = contracted_val
                continue
                
            # Shrink
            for i in range(1, n+1):
                simplex[i] = simplex[0] + self.delta*(simplex[i] - simplex[0])
                values[i] = func(simplex[i])
                
        return simplex[0], values[0]

class TrustRegionOptimizer(Optimizer):
    """Trust Region optimization with dogleg method."""
    
    def __init__(self, initial_radius: float = 1.0,
                 max_radius: float = 10.0, eta: float = 0.15):
        super().__init__()
        self.radius = initial_radius
        self.max_radius = max_radius
        self.eta = eta
        
    def minimize(self, func: Function, x0: np.ndarray,
                max_iter: int = 100, tol: float = 1e-8) -> Tuple[np.ndarray, float]:
        x = x0.copy()
        n = len(x)
        I = np.eye(n)
        
        for _ in range(max_iter):
            f = func(x)
            g = func.gradient(x)
            H = func.hessian(x)
            
            # Solve trust region subproblem using dogleg method
            try:
                p_u = -np.dot(g.T, g) / np.dot(g.T, np.dot(H, g)) * g
                p_b = -np.linalg.solve(H, g)
                
                if np.linalg.norm(p_b) <= self.radius:
                    p = p_b
                elif np.linalg.norm(p_u) >= self.radius:
                    p = self.radius * p_u / np.linalg.norm(p_u)
                else:
                    # Find tau for dogleg curve
                    def tau_equation(tau):
                        if tau <= 1:
                            p_tau = tau * p_u
                        else:
                            p_tau = p_u + (tau-1)*(p_b - p_u)
                        return np.linalg.norm(p_tau) - self.radius
                    
                    tau = fsolve(tau_equation, 1.0)[0]
                    
                    if tau <= 1:
                        p = tau * p_u
                    else:
                        p = p_u + (tau-1)*(p_b - p_u)
            except:
                # Fallback to steepest descent if numerical issues
                p = -self.radius * g / np.linalg.norm(g)
            
            # Compute actual vs predicted reduction
            actual_red = f - func(x + p)
            pred_red = -(np.dot(g, p) + 0.5*np.dot(p, np.dot(H, p)))
            rho = actual_red / pred_red if pred_red != 0 else 0
            
            # Update trust region radius
            if rho < 0.25:
                self.radius *= 0.25
            elif rho > 0.75 and np.linalg.norm(p) == self.radius:
                self.radius = min(2*self.radius, self.max_radius)
                
            # Update point if reduction is sufficient
            if rho > self.eta:
                x = x + p
                
            if np.linalg.norm(g) < tol:
                break
                
        return x, func(x)