from keras.models import Sequential, Input
from keras.layers import Dense
from src.linear_control.utils import process_observation_matrix


class NetworkTrainer:
    """
    Trains a neural network that maps states (x) to control inputs (u).
    For now, a central thread will maintain the current network, and dispatch a copy of it to each worker thread.
    The worker threads will then run a fixed number of simulations each with random initial conditions.
    """

    def __init__(self, physics_system, reward_func, hidden_layers=2, hidden_neurons=10):
        """
        :param reward_func: Maps state to a reward. x
                            Each episode's reward is determined by applying this to the final state.
        """
        self.phys = physics_system
        self.reward_func = reward_func
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons

        self.model = None

    def create_model(self):
        model = Sequential()
        model.add(Input(shape=(self.phys.x_dim,)))
        for _ in range(self.hidden_layers):
            model.add(Dense(self.hidden_neurons, activation='relu'))
        model.add(Dense(self.phys.u_dim, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
