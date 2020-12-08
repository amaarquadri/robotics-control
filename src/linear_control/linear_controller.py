import numpy as np
from matplotlib import use
from src.linear_control.utils import linear_quadratic_regulator
use('TkAgg')


class LinearController:
    def __init__(self, physics_system, operating_point=None, C=None, Q=None, R=None, V=None, W=None):
        self.phys = physics_system
        self.A, self.B = physics_system.get_linearized_equations_of_motion(operating_point)
        if C is not None:
            self.C = C
        else:
            # measure just first coordinate by default
            self.C = np.zeros((1, 2 * self.phys.x_dim))
            self.C[0, 0] = 1
        self.y_dim = self.C.shape[0]

        self.Q = Q if Q is not None else np.identity(2 * self.phys.x_dim)
        self.R = R if R is not None else np.identity(self.phys.u_dim)
        self.V = V if V is not None else np.identity(2 * self.phys.x_dim)  # model noise
        self.W = W if W is not None else np.identity(self.y_dim)  # measurement noise

        self.K = linear_quadratic_regulator(self.phys.apply_substitutions(self.A),
                                            self.phys.apply_substitutions(self.B),
                                            self.Q, self.R)
        self.L = linear_quadratic_regulator(self.phys.apply_substitutions(self.A.T),
                                            self.phys.apply_substitutions(self.C.T),
                                            self.V, self.W).T

        controllability_matrix = self.controllability_matrix(self.A, self.B)
        if np.linalg.matrix_rank(self.phys.apply_substitutions(controllability_matrix)) < 2 * self.phys.x_dim:
            raise Exception('Not Controllable!')

        observability_matrix = self.observability_matrix(self.A, self.C)
        if np.linalg.matrix_rank(self.phys.apply_substitutions(observability_matrix)) < 2 * self.phys.x_dim:
            raise Exception('Not Observable!')

        overall_dynamics = np.block([[self.A - np.dot(self.B, self.K), np.dot(self.B, self.K)],
                                     [np.zeros_like(self.A), self.A - np.dot(self.L, self.C)]])
        eigenvalues, _ = np.linalg.eig(self.phys.apply_substitutions(overall_dynamics))
        if np.any([eigenvalue.real > 0 for eigenvalue in eigenvalues]):
            raise Exception('Overall Dynamics are unstable!')
        print(np.max([eigenvalue.real for eigenvalue in eigenvalues]))

    @staticmethod
    def controllability_matrix(A, B):
        return np.column_stack([(np.linalg.multi_dot(i * [A] + [B]) if i > 0 else B) for i in range(A.shape[0])])

    @staticmethod
    def observability_matrix(A, C):
        return np.row_stack([(np.linalg.multi_dot([C] + i * [A]) if i > 0 else C) for i in range(A.shape[0])])

    def get_controller_function(self, x_r_func=None, u_max=None):
        if x_r_func is None:
            # use step input for first variable in x_r
            target = np.zeros(self.phys.x_dim)
            target[0] = 1

            def x_r_func(_):
                return target

        if u_max is None:
            u_max = np.inf * np.ones(self.phys.u_dim)

        A_num = self.phys.apply_substitutions(self.A)
        B_num = self.phys.apply_substitutions(self.B)

        def controller(t, x_hat, y):
            u = np.dot(self.K, x_r_func(t) - x_hat)
            u = np.sign(u) * np.minimum(np.abs(u), u_max)
            x_hat_dot = np.dot(A_num, x_hat) + np.dot(B_num, u) + np.dot(self.L, y - np.dot(self.C, x_hat))
            return u, x_hat_dot

        return controller
