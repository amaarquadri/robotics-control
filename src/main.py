import numpy as np
import sympy as sp
from src.physics_system.cart_pole import CartPole
from src.linear_control.linear_controller import LinearController
from src.linear_control.laplace_analyzer import LaplaceAnalyzer
from src.simulation.simulator import simulate
from src.linear_control.utils import to_string
import matplotlib.pyplot as plt
from matplotlib import use
use('TkAgg')


def test_system(controller, x_r_func=None, state_0=None, t_f=10):
    """
    Generates

    :param constants: A dictionary that maps sympy variables to floating point values.
                      The sympy variables must be in the order that physics_func expects.
    :param physics_func:
    :param operating_point:
    :param Q:
    :param R:
    :param V:
    :param R:
    :param W:
    :param x_r_func:
    :param x_0:
    :param t_f:
    """
    physics_system = controller.phys
    print('For displaying LaTeX:', 'https://latex.codecogs.com/eqneditor/editor.php')

    print('Lagrangian:', to_string(physics_system.lagrangian))
    print('\nExternal Forces:', to_string(physics_system.f_external))

    print('\nEquations of Motion:')
    for x, eq in zip(physics_system.x, physics_system.equations_of_motion):
        print(to_string(sp.Eq(sp.diff(x, physics_system.t), eq)))

    print('\nA (in Primed Variables):\n', to_string(controller.A),
          '\nB (in Primed Variables):\n', to_string(controller.B))

    laplace_analyzer = LaplaceAnalyzer(physics_system, controller.A, controller.B)

    transfer_functions = [sp.expand(tf) for tf in laplace_analyzer.get_transfer_functions()]
    print('\nTransfer Functions (in Primed Variables):')
    for X, tf in zip(laplace_analyzer.X, transfer_functions):
        print(to_string(sp.Eq(X, tf)))

    print('\nControl Transfer Functions (in Primed Variables):')
    for X, tf in zip(laplace_analyzer.X, laplace_analyzer.get_controller_transfer_functions()):
        print(to_string(sp.Eq(X, tf)))

    print('K:\n', controller.K)
    print('L:\n', controller.L)

    laplace_analyzer.analyze_controller(controller.K)

    t_vals, state_vals = simulate(physics_system, controller.C, controller.get_controller_function(x_r_func),
                                  controller.V, controller.W,
                                  state_0, t_f)
    forces = [np.dot(controller.K, x_r_func(t_) - state_vals[2 * physics_system.x_dim:, i])[0]
              for i, t_ in enumerate(t_vals)]
    powers = [force * velocity for force, velocity in zip(forces, state_vals[1, :])]
    print('\nMax Force:', np.max(np.abs(forces)))
    print('Max Power Draw:', np.max(np.abs(powers)))
    print('Mean Power Draw:', np.mean(np.abs(powers)))

    return t_vals, state_vals


def main():
    cart_pole = CartPole(M=1.994376, m=0.105425, L=0.110996, b=1.6359, g=9.81)
    controller = LinearController(cart_pole, operating_point=None, C=np.array([[1, 0, 0, 0],
                                                                               [0, 0, 1, 0]]),
                                  Q=np.diag([10, 20, 100, 50]), R=np.array([[1]]),
                                  V=np.diag([0, 0.001, 0, 0.001]), W=0.0001*np.identity(2))

    t_vals, state_vals = test_system(controller, x_r_func=lambda t: [(t // 7) % 2, 0, 0, 0],
                                     state_0=np.array([0, 0, 0, 0,
                                                       0, 0, 0, 0]),
                                     t_f=30)
    for i, x in enumerate(cart_pole.x):
        plt.plot(t_vals, state_vals[2 * i, :], label=f'${to_string(x)}$')
    for i, x in enumerate(cart_pole.x):
        plt.plot(t_vals, state_vals[2 * (cart_pole.x_dim + i), :], label=f'$\\hat{{{to_string(x)}}}$')

    # plt.plot(t_vals, x_r_func(t_vals)[0], label=r'$x_{r}$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Position (m), Velocity (m/s), $\theta$ (rad), $\omega$ (rad/s)')
    plt.title(r'Recovery from x Perturbation of 15 Meters')
    plt.legend()
    # plt.savefig('theta_perturbation.png')
    plt.show()


if __name__ == '__main__':
    main()
