import numpy as np
from scipy.integrate import solve_ivp


def simulate(physics_system, C, controller, state_0=None, t_f=10, rtol=1e-3):
    """
    :param physics_system:
    :param C:
    :param controller: The controller function which accepts (t, x_hat, y) and returns (u, x_hat_dot).
    :param state_0: A numpy array of the starting x and x_hat values.
    :param t_f:
    :param rtol:
    """
    if state_0 is None:
        state_0 = np.zeros(2 * physics_system.x_dim)

    def state_derivative(t, state):
        x = state[:physics_system.x_dim]
        x_hat = state[physics_system.x_dim:]

        y = np.dot(C, x)
        u, x_hat_dot = controller(t, x_hat, y)
        x_dot = physics_system.equations_of_motion_func(x, u)

        return np.concatenate((x_dot, x_hat_dot))

    result = solve_ivp(state_derivative, (0, t_f), state_0, method='RK45', rtol=rtol)
    # noinspection PyUnresolvedReferences
    return result.t, result.y
