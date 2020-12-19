import numpy as np
from scipy.integrate import solve_ivp
from src.simulation.noise import smooth_gaussian_noise_function, gaussian_noise_function


def simulate(physics_system, C, controller, V=None, W=None, state_0=None, t_f=10, rtol=1e-3):
    """
    :param physics_system:
    :param C:
    :param controller: The controller function which accepts (t, x_hat, y) and returns (u, x_hat_dot).
    :param V:
    :param W:
    :param state_0: A numpy array of the starting x and x_hat values.
    :param t_f:
    :param rtol:
    """
    if state_0 is None:
        state_0 = np.zeros(2 * physics_system.x_dim)

    model_noise = gaussian_noise_function(variances=V)
    control_noise = gaussian_noise_function(variances=W)
    A, B = physics_system.get_linearized_equations_of_motion()
    A = physics_system.apply_substitutions(A)
    B = physics_system.apply_substitutions(B)

    def state_derivative(t, state):
        print(t)
        midpoint = len(state) // 2
        x = state[:midpoint]
        x_hat = state[midpoint:]

        y = np.dot(C, x)  # + control_noise(t)
        u, x_hat_dot = controller(t, x_hat, y)
        x_dot = np.dot(A, x) + np.dot(B, u)  # physics_system.equations_of_motion_func(x, u)# + model_noise(t)

        return np.concatenate((x_dot, x_hat_dot))

    result = solve_ivp(state_derivative, (0, t_f), state_0, method='RK45', rtol=1e99, atol=1e99, max_step=0.01)
    # noinspection PyUnresolvedReferences
    return result.t, result.y


def simulate_direct(controller, state_0=None, t_f=10, rtol=1e-3):
    """
    :param controller: The controller function which accepts (t, x_hat, y) and returns (u, x_hat_dot).
    :param state_0: A numpy array of the starting x and x_hat values.
    :param t_f:
    :param rtol:
    """
    if state_0 is None:
        state_0 = np.zeros(2 * controller.phys.x_dim)

    M = controller.phys.apply_substitutions(controller.overall_dynamics)

    def state_derivative(_, state):
        return np.dot(M, state)

    result = solve_ivp(state_derivative, (0, t_f), state_0, method='RK45')  # , rtol=1e99, atol=1e99, max_step=0.01)
    # noinspection PyUnresolvedReferences
    return result.t, result.y
