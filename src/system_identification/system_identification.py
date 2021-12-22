import numpy as np
import rosbag
from src.physics_system.trexo_leg import TrexoLeg


def fit_physics_system_constants(physics_system, t_data, x_data, u_data):
    """
    Determines the optimal constants for the physics system to match the given input and output data.
    :@param x_data:
    """
    def error(constants):
        predicted_x_data = np.copy(x_data)
        for i in range(len(t_data) - 1):
            x_dot = physics_system.equations_of_motion


def main():
    dataset = trexo_status_bag_to_np('dataset.bag')
    t_data, x_data, u_data = extract_trexo_leg_x_data(dataset, right_leg=True)
    trexo_leg = TrexoLeg()
    constants = fit_physics_system_constants(trexo_leg, t_data, x_data, u_data)
    print(constants)


if __name__ == '__main__':
    main()
