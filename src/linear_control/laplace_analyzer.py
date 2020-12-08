import sympy as sp
import numpy as np
from tbcontrol.symbolic import routh
from src.linear_control.utils import interweave, get_gain, to_string


class LaplaceAnalyzer:
    def __init__(self, physics_system, A, B):
        self.phys = physics_system
        self.A = A
        self.B = B

        self.s = sp.symbols('s')
        self.X = np.array([sp.Function(x.name.capitalize())(self.s) for x in self.phys.x])
        self.U = np.array([sp.Function(u.name.capitalize())(self.s) for u in self.phys.u])
        self.X_r = np.array([sp.Function(X.name + '_r')(self.s) for X in self.X])

        self.transfer_functions = None
        self.k_pd_mat = None
        self.control_tfs = None

    def get_transfer_functions(self):
        if self.transfer_functions is not None:
            return self.transfer_functions

        laplace_tf = {}
        for x, X in zip(self.phys.x, self.X):
            laplace_tf[x] = X
            laplace_tf[sp.diff(x)] = self.phys.s * X
            laplace_tf[sp.diff(x, (self.phys.t, 2))] = self.phys.s ** 2 * X
        for u, U in zip(self.phys.u, self.U):
            laplace_tf[u] = U

        x_vec = interweave(self.phys.x, [sp.diff(x, self.phys.t) for x in self.phys.x])
        linear_eq = np.dot(self.A, x_vec) + np.dot(self.B, self.phys.u)
        laplace_equations = [sp.Eq(sp.diff(x, (self.phys.t, 2)), eq).subs(laplace_tf)
                             for x, eq in zip(self.phys.x, linear_eq[1::2])]
        result = sp.solve(laplace_equations, list(self.X))  # convert to list because sympy doesn't support numpy arrays

        self.transfer_functions = np.array([sp.simplify(result[X]) for X in self.X])
        return self.transfer_functions

    def get_controller_transfer_functions(self):
        if self.control_tfs is not None:
            return self.control_tfs

        if self.transfer_functions is None:
            self.get_transfer_functions()

        k_p_mat = np.array([[sp.symbols(f'k_p_{U.name.lower()}_{X.name.lower()}') for X in self.X]
                            for U in self.U])
        k_d_mat = np.array([[sp.symbols(f'k_d_{U.name.lower()}_{X.name.lower()}') for X in self.X]
                            for U in self.U])

        self.k_pd_mat = np.zeros((self.phys.u_dim, 2 * self.phys.x_dim)).astype(object)
        self.k_pd_mat[:, 0::2] = k_p_mat
        self.k_pd_mat[:, 1::2] = k_d_mat

        U_control = np.dot(k_p_mat, (self.X_r - self.X)) - np.dot(k_d_mat, self.s * self.X)

        eqs = [sp.Eq(X, tf).subs(zip(self.U, U_control))
               for X, tf in zip(self.X, self.transfer_functions)]
        result = sp.solve(eqs, list(self.X))  # convert to list because sympy doesn't support numpy arrays
        self.control_tfs = np.array([result[X] for X in self.X])

        return self.control_tfs

    def analyze_controller(self, K):
        if self.control_tfs is None:
            self.get_controller_transfer_functions()

        for X_r in self.X_r:
            # set all others in X_r to zero
            tfs = [sp.simplify(tf.subs([(X_r_, 0) for X_r_ in self.X_r if X_r_ is not X_r]) / X_r)
                   for tf in self.control_tfs]
            for X, tf in zip(self.X, tfs):
                numerator, denominator = tf.as_numer_denom()

                routh_table = np.array(routh(sp.Poly(denominator, self.s)))
                routh_conditions = routh_table[:, 0]

                tf = self.phys.apply_substitutions(tf)
                numerator = self.phys.apply_substitutions(numerator)
                denominator = self.phys.apply_substitutions(denominator)

                zeros = sp.roots(numerator.subs(zip(self.k_pd_mat.flatten(), K.flatten())), self.s)
                poles = sp.roots(denominator.subs(zip(self.k_pd_mat.flatten(), K.flatten())), self.s)
                gain = get_gain(self.s, tf.subs(zip(self.k_pd_mat.flatten(), K.flatten())), zeros, poles)

                zeros_description = ', '.join(
                    [str(zero.evalf(6)) if multiplicity == 1 else f'{zero.evalf(6)} (x{multiplicity})'
                     for zero, multiplicity in zeros.items()])
                poles_description = ', '.join(
                    [str(pole.evalf(6)) if multiplicity == 1 else f'{pole.evalf(6)} (x{multiplicity})'
                     for pole, multiplicity in poles.items()])
                routh_conditions_description = '\n'.join(
                    [to_string(condition.evalf(3)) for condition in routh_conditions])
                print(f'{X}/{X_r}: Routh Conditions:\n{routh_conditions_description}\nGain: {gain.evalf(6)}\n'
                      f'Poles: {poles_description}\nZeros: {zeros_description}\n\n')
