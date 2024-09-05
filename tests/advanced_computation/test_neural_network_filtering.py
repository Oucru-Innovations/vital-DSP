import pytest
import numpy as np
from unittest.mock import MagicMock
from vitalDSP.advanced_computation.neural_network_filtering import NeuralNetworkFiltering


@pytest.fixture
def test_signal():
    np.random.seed(42)
    return np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)


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
