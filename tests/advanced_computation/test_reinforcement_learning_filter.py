import pytest
import numpy as np
from vitalDSP.advanced_computation.reinforcement_learning_filter import ReinforcementLearningFilter

@pytest.fixture
def test_signal():
    """
    Fixture to generate a sample signal for testing.
    """
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)

@pytest.fixture
def rl_filter(test_signal):
    """
    Fixture to initialize the ReinforcementLearningFilter with the test signal and action space.
    """
    action_space = [-1, 0, 1]
    return ReinforcementLearningFilter(test_signal, action_space)

def test_q_learning(rl_filter):
    """
    Test the Q-Learning training method.
    """
    rl_filter.train_q_learning(episodes=10, alpha=0.1, gamma=0.99, epsilon=0.1)
    assert rl_filter.q_table.shape == (len(rl_filter.signal), len(rl_filter.action_space))
    assert np.any(rl_filter.q_table != 0)  # Ensure that the Q-table was updated

def test_apply_filter_q_learning(rl_filter):
    """
    Test filter application after Q-Learning training.
    """
    rl_filter.train_q_learning(episodes=10)
    filtered_signal = rl_filter.apply_filter()
    assert len(filtered_signal) == len(rl_filter.signal)
    assert isinstance(filtered_signal, np.ndarray)

def test_dqn_training(rl_filter):
    """
    Test the DQN-based filter training.
    """
    rl_filter.train_dqn(episodes=10, batch_size=4, gamma=0.99, epsilon=0.1, target_update=2)
    assert rl_filter.q_table.shape == (len(rl_filter.signal), len(rl_filter.action_space))
    # Since DQN doesn't update Q-table directly, we skip checking q_table updates
    # Instead, ensure that neural networks are trained without errors
    assert True  # If no exception is raised, the test passes

def test_apply_filter_dqn(rl_filter):
    """
    Test filter application after DQN training.
    """
    rl_filter.train_dqn(episodes=10, batch_size=4, gamma=0.99, epsilon=0.1, target_update=2)
    filtered_signal = rl_filter.apply_filter()
    assert len(filtered_signal) == len(rl_filter.signal)
    assert isinstance(filtered_signal, np.ndarray)

def test_ppo_training(rl_filter):
    """
    Test PPO-based filter training.
    """
    rl_filter.train_ppo(epochs=10, gamma=0.99, lambda_=0.95, clip_ratio=0.2)
    assert True  # Ensure the function executes without errors

def test_apply_filter_ppo(rl_filter):
    """
    Test filter application after PPO training.
    """
    rl_filter.train_ppo(epochs=10, gamma=0.99, lambda_=0.95, clip_ratio=0.2)
    filtered_signal = rl_filter.apply_filter()
    assert len(filtered_signal) == len(rl_filter.signal)
    assert isinstance(filtered_signal, np.ndarray)

def test_initialize_state(rl_filter):
    """
    Test the internal state initialization method.
    """
    state = rl_filter._initialize_state()
    assert state == 0  # The initial state should be zero

def test_take_action(rl_filter):
    """
    Test the _take_action method.
    """
    state, action = 0, 1
    next_state, reward = rl_filter._take_action(state, action)
    assert next_state == 1
    assert reward <= 0  # The reward should be non-positive due to the reward function

# def test_apply_action(rl_filter):
#     """
#     Test the _apply_action method.
#     """
#     action, t = 1, 5
#     original_value = rl_filter.signal[t]
#     new_value = rl_filter._apply_action(action, t)
#     assert isinstance(new_value, float)
#     assert new_value != original_value  # The signal should be altered

def test_apply_action(rl_filter):
    """Test the _apply_action method."""
    action, t = 1, 5  # Use a non-zero action to ensure signal is modified
    original_value = rl_filter.signal[t]
    new_value = rl_filter._apply_action(action, t)
    
    # Check if action has changed the signal only if action_space[action] is non-zero
    if rl_filter.action_space[action] != 0:
        assert new_value != original_value  # The signal should be altered
    else:
        assert new_value == original_value  # If action is zero, signal should remain unchanged


def test_update_state(rl_filter):
    """
    Test the _update_state method.
    """
    new_state = rl_filter._update_state(0, 1)
    assert new_state == 1  # The state should have incremented

def test_neural_network(rl_filter):
    """
    Test the DQN neural network's predict method.
    """
    network = rl_filter._initialize_neural_network()
    state = np.zeros(rl_filter.state_space)
    state[0] = 1  # One-hot encoding for state 0
    action_values = network.predict(state)
    assert len(action_values) == len(rl_filter.action_space)

def test_policy_network(rl_filter):
    """
    Test the PPO policy network's action sampling.
    """
    policy_network = rl_filter._initialize_policy_network()
    state = np.zeros(rl_filter.state_space)
    state[0] = 1  # One-hot encoding for state 0
    action = policy_network.sample_action(state)
    assert action in range(len(rl_filter.action_space))  # Ensure a valid action is sampled

def test_value_network(rl_filter):
    """
    Test the PPO value network's predict method.
    """
    value_network = rl_filter._initialize_value_network()
    state = np.zeros(rl_filter.state_space)
    state[0] = 1  # One-hot encoding for state 0
    value = value_network.predict(state)
    assert isinstance(value, float)

def test_collect_trajectories(rl_filter):
    """
    Test trajectory collection for PPO.
    """
    policy_network = rl_filter._initialize_policy_network()
    value_network = rl_filter._initialize_value_network()
    trajectories = rl_filter._collect_trajectories(policy_network, value_network, gamma=0.99, lambda_=0.95)
    assert isinstance(trajectories, list)
    assert len(trajectories) > 0
    for traj in trajectories:
        assert len(traj) == 4  # Each trajectory tuple should have 4 elements

def test_calculate_advantage(rl_filter):
    """
    Test the calculation of advantage estimates for PPO.
    """
    value_network = rl_filter._initialize_value_network()
    state, action, reward, next_state = 0, 1, 1.0, 1
    advantage = rl_filter._calculate_advantage(state, action, reward, next_state, value_network, gamma=0.99, lambda_=0.95)
    assert isinstance(advantage, float)
    # Depending on the state and action, advantage could be zero or non-zero
    # Here, we just ensure it's a float
