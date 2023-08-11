from src.physics_system.physics_system import PhysicsSystem
import sympy as sp
import numpy as np


class Rocket(PhysicsSystem):
    def _init_(self, m=1, L=1, g=9.81):
        constants = dict(zip(sp.symbols('m, L, g'), [m, L, g]))
        super().__init__(constants)

    def create_x_and_u(self):
        x = sp.Function('x')(self.t)
        y = sp.Function('y')(self.t)
        # define theta = 0 at positive y-axis, increasing clockwise
        theta = sp.Function('theta')(self.t)
        # define alpha = 0 for aligned with rocket, increasing clockwise
        alpha = sp.Function('alpha')(self.t)
        f = sp.Function('f')(self.t)
        return np.array([x, y, theta]), np.array([f, alpha])

    def create_lagrangian(self):
        m, L, g, _ = self.constants.keys()
        x, y, theta = self.x
        I = m * L ** 2 / 12
        KE = (m / 2) * (sp.diff(x, self.t) * 2 + sp.diff(y, self.t) * 2) + (I / 2) * sp.diff(theta, self.t) ** 2
        PE = m * g * y
        L = KE - PE
        return L

    def create_f_external(self):
        _, L, _ = self.constants.keys()
        _, _, theta = self.x
        f, alpha = self.u
        f_x = f * sp.cos(theta + alpha)
        f_y = f * sp.sin(theta + alpha)
        T = -f * sp.sin(alpha) * (L / 2)
        return np.array([f_x, f_y, T])
