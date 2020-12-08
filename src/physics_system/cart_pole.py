from src.physics_system.physics_system import PhysicsSystem
import sympy as sp
import numpy as np


class CartPole(PhysicsSystem):
    def __init__(self, M=1.0, m=0.1, L=1.0, b=1.0, g=9.81):
        constants = dict(zip(sp.symbols('M, m, L, b, g'), [M, m, L, b, g]))
        super().__init__(constants)

    def create_x_and_u(self):
        x = sp.Function('x')(self.t)
        theta = sp.Function('theta')(self.t)
        f = sp.Function('f')(self.t)
        return np.array([x, theta]), np.array([f])

    def create_lagrangian(self):
        M, m, L, _, g = self.constants.keys()
        x, theta = self.x

        x_rel = -L * sp.sin(theta)
        v_x_rel = sp.diff(x_rel, self.t)
        v_x_pole = sp.diff(x, self.t) + v_x_rel

        y_rel = L * sp.cos(theta)
        v_y_pole = sp.diff(y_rel, self.t)  # relative and absolute velocities are the same

        KE_cart = (M * sp.diff(x, self.t) ** 2) / 2
        KE_pole = (m * v_x_pole ** 2) / 2 + (m * v_y_pole ** 2) / 2
        KE = KE_cart + KE_pole
        PE = m * g * L * sp.cos(theta)
        L = sp.simplify(KE - PE)
        return L

    def create_f_external(self):
        _, _, _, b, _ = self.constants.keys()
        x = self.x[0]
        f = self.u[0]
        f_x_external = f - b * sp.diff(x, self.t)  # dissipative and external forces for the x direction
        f_theta_external = 0
        return np.array([f_x_external, f_theta_external])