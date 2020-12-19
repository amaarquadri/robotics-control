import numpy as np
import sympy as sp
from sympy.physics.vector.printing import vlatex
from scipy.linalg import solve_continuous_are


def linear_quadratic_regulator(A, B, Q, R):
    """
    https://youtu.be/bMiiC94FJ5E?t=3276
    https://github.com/markwmuller/controlpy/blob/master/controlpy/synthesis.py
    System is defined by dx/dt = Ax + Bu
    Minimizing integral (x.T*Q*x + u.T*R*u) dt from 0 to infinity
    Returns K such that optimal control is u = -Kx
    """
    # first, try to solve the Ricatti equation
    P = solve_continuous_are(A, B, Q, R)

    # compute the LQR gain
    K = np.linalg.multi_dot([np.linalg.inv(R), B.T, P])
    return K


def linear_quadratic_estimator(A, C, V, W):
    return linear_quadratic_regulator(A.T, C.T, V, W).T


def interweave(x, x_dot, rows=True):
    if len(x.shape) == 1:
        return np.column_stack((x, x_dot)).flatten()
    elif len(x.shape) == 2:
        dtype = float if isinstance(x, float) and isinstance(x_dot, float) else object
        if rows:
            result = np.zeros((2 * x.shape[0], x.shape[1]), dtype)
            result[0::2, :] = x
            result[1::2, :] = x_dot
        else:
            result = np.zeros((x.shape[0], 2 * x.shape[1]), dtype)
            result[:, 0::2] = x
            result[:, 1::2] = x_dot
        return result


def get_gain(s, ratio, zeros, poles, epsilon=0.1):
    # find a value sufficiently far from all poles and zeros
    i = 0
    while any([abs(i - zero) < epsilon for zero, _ in zeros.items()]) or \
            any([abs(i - pole) < epsilon for pole, _ in poles.items()]):
        i += 1

    gain = ratio.subs(s, i)
    gain *= np.product([(i - pole) ** multiplicity for pole, multiplicity in poles.items()])
    gain /= np.product([(i - zeros) ** multiplicity for zeros, multiplicity in zeros.items()])
    return sp.re(gain.evalf())


def to_string(expression, to_word=True):
    text = vlatex(expression).replace(r'\operatorname{Theta}', r'\Theta')
    if to_word:
        text = text.replace(r'p f x', r'p,x') \
            .replace(r'd f x', r'd,x') \
            .replace(r'p f \theta', r'p,\theta') \
            .replace(r'd f \theta', r'd,\theta')
    else:
        text = text.replace(r'p f x', r'px') \
            .replace(r'd f x', r'dx') \
            .replace(r'p f \theta', r'pt') \
            .replace(r'd f \theta', r'dt')
    return text


def process_observation_matrix(C=None, physics_system=None):
    if C is None:
        if physics_system is None:
            raise ValueError('Must provide either C or physics system!')
        # measure just first coordinate by default
        C = np.zeros((1, 2 * physics_system.x_dim))
        C[0, 0] = 1
    y_dim = C.shape[0]
    return C, y_dim
