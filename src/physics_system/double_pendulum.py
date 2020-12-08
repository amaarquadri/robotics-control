from src.physics_system.physics_system import PhysicsSystem
import sympy as sp
import numpy as np


class DoublePendulum(PhysicsSystem):
    def __init__(self, m_1=1, m_2=1, L_1=1, L_2=1, b_1=1, b_2=1, g=9.81):
        constants = dict(zip(sp.symbols('m_1, m_2, L_1, L_2, b_1, b_2, g'), [m_1, m_2, L_1, L_2, b_1, b_2, g]))
        super().__init__(constants)

    def create_x_and_u(self):
        theta_1 = sp.Function('theta_1')(self.t)  # angle clockwise from straight up
        theta_2 = sp.Function('theta_2')(self.t)  # angle clockwise from theta_1
        torque = sp.Function('T')(self.t)  # applied at the middle joint
        return np.array([theta_1, theta_2]), np.array([torque])

    def create_lagrangian(self):
        # TODO: verify
        m_1, m_2, L_1, L_2, b_1, b_2, g = self.constants.keys()
        theta_1, theta_2 = self.x

        v_1_squared = (L_1 * sp.diff(theta_1, self.t)) ** 2
        v_2_squared = (L_2 * sp.diff(theta_2, self.t)) ** 2
        cross_velocity = 2 * L_1 * L_2 * sp.diff(theta_1, self.t) * sp.diff(theta_2, self.t) * sp.cos(theta_2)
        KE = (m_1 / 2) * v_1_squared + (m_2 / 2) * (v_1_squared + v_2_squared + cross_velocity)

        y_1 = L_1 * sp.cos(theta_1)
        y_2 = L_2 * sp.cos(theta_1 + theta_2)
        PE = -m_1 * g * y_1 - m_2 * g * (y_1 + y_2)

        L = KE - PE
        return L

    def create_f_external(self):
        pass
