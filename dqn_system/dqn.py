import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class DQNSystem:
    def __init__(self, state_size: int, action_list: list,
                 learning_rate: float = 0.001, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995):
        """
        Initialize the DQN system.
        :param state_size: Size of the state vector.
        :param action_list: List of possible actions.
        """
        self.state_size = state_size
        self.action_list = action_list
        self.action_size = len(action_list)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()

    def build_model(self) -> Sequential:
        """
        Build the DQN model.
        """
        model = Sequential(name="DQN")
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        logger.debug("DQN model built and compiled.")
        return model

    def choose_action(self, state: np.ndarray) -> str:
        """
        Choose an action based on the current state.
        :param state: Current state vector.
        """
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_list)
            logger.info(f'DQN: Exploratory action "{action}"')
            return action
        q_values = self.model.predict(state, verbose=0)
        action_idx = np.argmax(q_values[0])
        action = self.action_list[action_idx]
        logger.info(f'DQN: Exploitative action "{action}"')
        return action

    def train(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Train the DQN model with experience replay.
        :param state: Current state vector.
        :param action: Chosen action.
        :param reward: Reward received.
        :param next_state: Next state vector.
        :param done: Whether the episode is done.
        """
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
        target_f = self.model.predict(state, verbose=0)
        action_idx = self.action_list.index(action)
        target_f[0][action_idx] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        logger.info(f'DQN: Epsilon updated to {self.epsilon}')
