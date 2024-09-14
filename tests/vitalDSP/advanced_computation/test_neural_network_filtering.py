import pytest
import numpy as np
from unittest.mock import MagicMock
from vitalDSP.advanced_computation.neural_network_filtering import NeuralNetworkFiltering
from vitalDSP.advanced_computation.neural_network_filtering import FeedforwardNetwork
from vitalDSP.advanced_computation.neural_network_filtering import ConvolutionalNetwork
from vitalDSP.advanced_computation.neural_network_filtering import RecurrentNetwork

# @pytest.fixture
# def test_signal():
#     np.random.seed(42)
#     return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
@pytest.fixture
def test_signal():
    np.random.seed(42)
    # Adjust the shape to (100, 1) for feedforward and recurrent networks, and (10, 5, 5) for convolutional network
    return np.random.randn(100, 1)

@pytest.fixture
def sample_convolutional_data():
    np.random.seed(42)
    # Assuming input shape for convolutional network (batch_size, height, width)
    X = np.random.randn(5, 5, 5)  # Batch of 5 images, 5x5 pixels
    y = np.random.randn(5)  # Target values
    return X, y

@pytest.fixture
def sample_recurrent_data():
    np.random.seed(42)
    # Assuming input shape for recurrent network (batch_size, sequence_length, features)
    X = np.random.randn(5, 10)  # 5 sequences of length 10
    y = np.random.randn(5)  # Target values
    return X, y

@pytest.fixture
def nn_feedforward(test_signal):
    # Patch the FeedforwardNetwork to mock its methods
    network = NeuralNetworkFiltering(
        signal=test_signal, network_type="feedforward", hidden_layers=[64, 64]
    )
    network.network.train = MagicMock()
    network.network.predict = MagicMock(return_value=np.random.randn(len(test_signal) - 1))
    return network


@pytest.fixture
def nn_convolutional(test_signal):
    # Patch the ConvolutionalNetwork to mock its methods
    network = NeuralNetworkFiltering(signal=test_signal, network_type="convolutional")
    network.network.train = MagicMock()
    network.network.predict = MagicMock(return_value=np.random.randn(len(test_signal) - 1))
    return network


@pytest.fixture
def nn_recurrent(test_signal):
    # Patch the RecurrentNetwork to mock its methods
    network = NeuralNetworkFiltering(signal=test_signal, network_type="recurrent")
    network.network.train = MagicMock()
    network.network.predict = MagicMock(return_value=np.random.randn(len(test_signal) - 1))
    return network


def test_feedforward_train(nn_feedforward):
    nn_feedforward.train()
    # Ensure the network's train method is called
    nn_feedforward.network.train.assert_called_once()


def test_feedforward_apply_filter(nn_feedforward):
    filtered_signal = nn_feedforward.apply_filter()
    # Ensure the network's predict method is called and filtered signal is returned
    nn_feedforward.network.predict.assert_called_once()
    assert isinstance(filtered_signal, np.ndarray)


def test_feedforward_evaluate(nn_feedforward, test_signal):
    mse = nn_feedforward.evaluate(test_signal)
    # Ensure predict is called and the MSE is calculated correctly
    nn_feedforward.network.predict.assert_called_once()
    assert isinstance(mse, float)


def test_convolutional_train(nn_convolutional):
    nn_convolutional.train()
    # Ensure the network's train method is called
    nn_convolutional.network.train.assert_called_once()


def test_convolutional_apply_filter(nn_convolutional):
    filtered_signal = nn_convolutional.apply_filter()
    # Ensure the network's predict method is called and filtered signal is returned
    nn_convolutional.network.predict.assert_called_once()
    assert isinstance(filtered_signal, np.ndarray)


def test_convolutional_evaluate(nn_convolutional, test_signal):
    mse = nn_convolutional.evaluate(test_signal)
    # Ensure predict is called and the MSE is calculated correctly
    nn_convolutional.network.predict.assert_called_once()
    assert isinstance(mse, float)


def test_recurrent_train(nn_recurrent):
    nn_recurrent.train()
    # Ensure the network's train method is called
    nn_recurrent.network.train.assert_called_once()


def test_recurrent_apply_filter(nn_recurrent):
    filtered_signal = nn_recurrent.apply_filter()
    # Ensure the network's predict method is called and filtered signal is returned
    nn_recurrent.network.predict.assert_called_once()
    assert isinstance(filtered_signal, np.ndarray)


def test_recurrent_evaluate(nn_recurrent, test_signal):
    mse = nn_recurrent.evaluate(test_signal)
    # Ensure predict is called and the MSE is calculated correctly
    nn_recurrent.network.predict.assert_called_once()
    assert isinstance(mse, float)


def test_invalid_network_type(test_signal):
    # Test for invalid network type
    with pytest.raises(ValueError):
        NeuralNetworkFiltering(signal=test_signal, network_type="invalid")

@pytest.fixture
def feedforward_network():
    hidden_layers = [64, 64]
    dropout_rate = 0.1
    batch_norm = True
    return FeedforwardNetwork(hidden_layers, dropout_rate, batch_norm)

@pytest.fixture
def sample_data():
    np.random.seed(0)
    X = np.random.randn(100, 1)
    y = np.random.randn(100)
    return X, y

def test_feedforward_initialization(feedforward_network):
    assert feedforward_network.hidden_layers == [64, 64]
    assert feedforward_network.dropout_rate == 0.1
    assert feedforward_network.batch_norm is True
    assert len(feedforward_network.weights) == 3  # 3 weight matrices: input->hidden1, hidden1->hidden2, hidden2->output

def test_feedforward_train(feedforward_network, sample_data):
    X, y = sample_data
    y = y.reshape(-1, 1)  # Ensure y has the correct shape
    # Mock learning parameters
    learning_rate = 0.01
    epochs = 2
    batch_size = 10
    # Ensure training runs without errors
    try:
        feedforward_network.train(X, y, learning_rate, epochs, batch_size)
    except Exception as e:
        pytest.fail(f"Training failed with exception: {e}")

def test_feedforward_predict(feedforward_network, sample_data):
    X, _ = sample_data
    predictions = feedforward_network.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X.shape[0], 1)

def test_feedforward_relu(feedforward_network):
    Z = np.array([-1, 0, 1, 2])
    A = feedforward_network._relu(Z)
    expected = np.array([0, 0, 1, 2])
    np.testing.assert_array_equal(A, expected)

def test_feedforward_batch_normalization(feedforward_network):
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    normalized_Z = feedforward_network._batch_normalization(Z)
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    expected = (Z - mean) / np.sqrt(variance + 1e-8)
    np.testing.assert_array_almost_equal(normalized_Z, expected)

@pytest.fixture
def convolutional_network():
    dropout_rate = 0.2
    batch_norm = False
    return ConvolutionalNetwork(dropout_rate, batch_norm)

@pytest.fixture
def sample_convolutional_data():
    # Assuming the convolution expects 3D input (e.g., channels, height, width)
    np.random.seed(0)
    X = np.random.randn(10, 5, 5)  # 10 samples, 5x5 each
    y = np.random.randn(10)
    return X, y

def test_convolutional_initialization(convolutional_network):
    assert convolutional_network.dropout_rate == 0.2
    assert convolutional_network.batch_norm is False
    assert len(convolutional_network.filters) == 16
    assert convolutional_network.weights == [convolutional_network.weights[0], convolutional_network.weights[1]]
    assert len(convolutional_network.biases) == 2

def test_convolutional_train(convolutional_network, sample_convolutional_data):
    X, y = sample_convolutional_data
    y = y.reshape(-1, 1)  # Ensure y has the correct shape
    learning_rate = 0.001
    epochs = 1
    batch_size = 5
    try:
        convolutional_network.train(X, y, learning_rate, epochs, batch_size)
    except Exception as e:
        pytest.fail(f"Training failed with exception: {e}")

def test_convolutional_predict(convolutional_network, sample_convolutional_data):
    X, _ = sample_convolutional_data
    predictions = convolutional_network.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X.shape[0], 1)  # Assuming output layer has 1 neuron

def test_convolution_operation(convolutional_network):
    X = np.random.randn(1, 5, 5)  # Single sample
    output = convolutional_network._convolution(X)
    # Expected output shape: (1, 3, 3, 16) because (5-2, 5-2, 16)
    assert output.shape == (1, 3, 3, 16)
    
@pytest.fixture
def recurrent_network():
    recurrent_type = "lstm"
    dropout_rate = 0.3
    batch_norm = True
    return RecurrentNetwork(recurrent_type, dropout_rate, batch_norm)

@pytest.fixture
def recurrent_network_gru():
    recurrent_type = "gru"
    dropout_rate = 0.3
    batch_norm = True
    return RecurrentNetwork(recurrent_type, dropout_rate, batch_norm)

@pytest.fixture
def sample_recurrent_data():
    # Assuming input shape: (batch_size, sequence_length, features)
    np.random.seed(0)
    X = np.random.randn(5, 10, 10)  # 5 samples, 10 time steps, 10 features
    y = np.random.randn(5)
    return X, y

def test_recurrent_initialization(recurrent_network):
    assert recurrent_network.recurrent_type == "lstm"
    assert recurrent_network.dropout_rate == 0.3
    assert recurrent_network.batch_norm is True
    assert recurrent_network.Wx.shape == (10, 64)
    assert recurrent_network.Wh.shape == (64, 64)
    assert recurrent_network.Wy.shape == (64, 1)
    assert recurrent_network.b.shape == (64,)
    assert recurrent_network.by.shape == (1,)

def test_recurrent_train(recurrent_network, sample_recurrent_data):
    X, y = sample_recurrent_data
    learning_rate = 0.005
    epochs = 1
    batch_size = 2
    # Ensure training runs without errors
    try:
        recurrent_network.train(X, y, learning_rate, epochs, batch_size)
    except Exception as e:
        pytest.fail(f"Training failed with exception: {e}")

def test_recurrent_predict(recurrent_network, sample_recurrent_data):
    X, _ = sample_recurrent_data
    predictions = recurrent_network.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X.shape[0], 1)  # Assuming output layer has 1 neuron

def test_recurrent_lstm_forward(recurrent_network, sample_recurrent_data):
    X, _ = sample_recurrent_data
    h, c = recurrent_network._lstm(X, np.zeros((X.shape[0], 64)))
    assert h.shape == (X.shape[0], 64)
    assert c.shape == (X.shape[0], 64)

def test_recurrent_gru_forward(recurrent_network, sample_recurrent_data):
    # Switch to GRU
    recurrent_network.recurrent_type = "gru"
    X, _ = sample_recurrent_data
    h = recurrent_network._gru(X, np.zeros((X.shape[0], 64)))
    assert h.shape == (X.shape[0], 64)
