from src.physics_system.physics_system import PhysicsSystem
import sympy as sp
import numpy as np


class Pendulum(PhysicsSystem):
    def __init__(self, m=1, L=1, b=1, g=9.81):
        constants = dict(zip(sp.symbols('m, L, b, g'), [m, L, b, g]))
        super().__init__(constants)

    def create_x_and_u(self):
        theta = sp.Function('theta')(self.t)  # positive upwards, increasing clockwise
        torque = sp.symbols('torque')(self.t)
        return np.array([theta]), np.array([torque])

    def create_lagrangian(self):
        theta = self.x[0]
        m, L, g = self.constants.keys()
        I = m * L ** 2
        KE = (I / 2) * sp.diff(theta, self.t) ** 2
        PE = m * g * L * sp.cos(theta)
        L = KE - PE
        return L

    def create_f_external(self):
        _, _, b, _ = self.constants.keys()
        theta = self.x[0]
        torque = self.u[0]
        return torque - b * sp.diff(theta, self.t)
