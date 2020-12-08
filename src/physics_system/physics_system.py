from abc import ABC, abstractmethod
import numpy as np
import sympy as sp
from src.linear_control.utils import interweave


class PhysicsSystem(ABC):
    def __init__(self, constants):
        self.constants = constants
        self.t = sp.symbols('t')

        self.x, self.u = self.create_x_and_u()
        self.x_dim, self.u_dim = len(self.x), len(self.u)
        self.lagrangian = self.create_lagrangian()
        self.f_external = self.create_f_external()

        self.equations_of_motion = self.create_equations_of_motion()
        self.equation_of_motion_funcs = self.create_equation_of_motion_funcs()

    def apply_substitutions(self, expression):
        if isinstance(expression, np.ndarray):
            result = np.array([self.apply_substitutions(v) for v in expression])
            try:
                return result.astype(float)
            except TypeError:
                pass
            return result
        elif isinstance(expression, sp.core.Expr):
            return expression.subs(self.constants)
        else:
            return expression

    @abstractmethod
    def create_x_and_u(self):
        pass

    @abstractmethod
    def create_lagrangian(self):
        pass

    @abstractmethod
    def create_f_external(self):
        pass

    def create_equations_of_motion(self):
        lagrange_equations = [sp.Eq(sp.diff(sp.diff(self.lagrangian, sp.diff(x, self.t)), self.t),
                                    sp.diff(self.lagrangian, x) + F_external)
                              for x, F_external in zip(self.x, self.f_external)]
        result = sp.solve(lagrange_equations, [sp.diff(x, (self.t, 2)) for x in self.x])

        equations_of_motion = np.array([sp.simplify(result[sp.diff(x, (self.t, 2))])
                                        for x in self.x])
        return equations_of_motion

    def create_equation_of_motion_funcs(self):
        x_params = interweave(self.x, [sp.diff(x, self.t) for x in self.x])
        return np.array([sp.lambdify([x_params, self.u], self.apply_substitutions(eq))
                         for eq in self.equations_of_motion])

    def equations_of_motion_func(self, x, u):
        return interweave(x[1::2], np.array([eq(x, u) for eq in self.equation_of_motion_funcs]))

    def get_linearized_equations_of_motion(self, operating_point=None):
        """
        The operating points are not included in the linearization which means the resulting linear system
        (x_dot = A * x + B * u) is in reference to the primed coordinates.
        Thus the K gains for the controller will also correspond to the primed coordinates.
        However, K * (x_r' - x') = K * (x_r - x) so the controller works the same regardless.
        """
        # set unspecified required values to 0
        operating_point = operating_point if operating_point is not None else {}
        for x in self.x:
            if x not in operating_point:
                operating_point[x] = 0
            if sp.diff(x, self.t) not in operating_point:
                operating_point[sp.diff(x, self.t)] = 0
        for u in self.u:
            if u not in operating_point:
                operating_point[u] = 0

        A_velocity_rows = interweave(np.zeros((self.x_dim, self.x_dim)), np.identity(self.x_dim), rows=False)
        A_x_dot = np.array([[sp.diff(eq, x).subs(operating_point) for x in self.x]
                            for eq in self.equations_of_motion])
        A_x_ddot = np.array([[sp.diff(eq, sp.diff(x, self.t)).subs(operating_point) for x in self.x]
                             for eq in self.equations_of_motion])
        A_acceleration_rows = interweave(A_x_dot, A_x_ddot, rows=False)
        A = interweave(A_velocity_rows, A_acceleration_rows, rows=True)

        B_x_ddot = np.array([[sp.diff(eq, u).subs(operating_point) for u in self.u]
                             for eq in self.equations_of_motion])
        B_x_dot = np.zeros_like(B_x_ddot)
        B = interweave(B_x_dot, B_x_ddot)
        return A, B
