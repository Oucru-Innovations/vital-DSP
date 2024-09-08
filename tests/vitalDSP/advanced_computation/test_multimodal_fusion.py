import pytest
import numpy as np
from vitalDSP.advanced_computation.multimodal_fusion import MultimodalFusion

# Sample test signals
@pytest.fixture
def signal1():
    return np.sin(np.linspace(0, 10, 100))

@pytest.fixture
def signal2():
    return np.cos(np.linspace(0, 10, 100))

@pytest.fixture
def fusion_instance(signal1, signal2):
    return MultimodalFusion([signal1, signal2])

def test_init_with_valid_signals(signal1, signal2):
    # Testing correct initialization
    fusion = MultimodalFusion([signal1, signal2])
    assert fusion.signals[0] is signal1
    assert fusion.signals[1] is signal2

def test_init_with_invalid_signals():
    # Testing invalid initialization (non-list input)
    with pytest.raises(ValueError):
        MultimodalFusion(np.sin(np.linspace(0, 10, 100)))

    # Testing invalid initialization (signals of different lengths)
    with pytest.raises(ValueError):
        MultimodalFusion([np.array([1, 2, 3]), np.array([4, 5])])

def test_weighted_sum_fusion(fusion_instance, signal1, signal2):
    # Testing weighted sum fusion
    weights = [0.6, 0.4]
    fused_signal = fusion_instance.fuse(strategy="weighted_sum", weights=weights)
    expected_signal = 0.6 * signal1 + 0.4 * signal2
    np.testing.assert_array_almost_equal(fused_signal, expected_signal)

    # Testing invalid weights length
    with pytest.raises(ValueError):
        fusion_instance.fuse(strategy="weighted_sum", weights=[0.6])

def test_concatenation_fusion(fusion_instance, signal1, signal2):
    # Testing concatenation fusion
    fused_signal = fusion_instance.fuse(strategy="concatenation")
    expected_signal = np.concatenate([signal1, signal2])
    np.testing.assert_array_equal(fused_signal, expected_signal)

def test_pca_fusion(fusion_instance):
    # Testing PCA fusion with 1 component
    fused_signal = fusion_instance.fuse(strategy="pca", n_components=1)
    assert fused_signal.shape == (100,)

    # Testing PCA fusion with multiple components
    fused_signal = fusion_instance.fuse(strategy="pca", n_components=2)
    assert fused_signal.shape == (100, 2)

def test_maximum_fusion(fusion_instance):
    # Testing maximum fusion
    fused_signal = fusion_instance.fuse(strategy="maximum")
    expected_signal = np.maximum(fusion_instance.signals[0], fusion_instance.signals[1])
    np.testing.assert_array_equal(fused_signal, expected_signal)

def test_minimum_fusion(fusion_instance):
    # Testing minimum fusion
    fused_signal = fusion_instance.fuse(strategy="minimum")
    expected_signal = np.minimum(fusion_instance.signals[0], fusion_instance.signals[1])
    np.testing.assert_array_equal(fused_signal, expected_signal)

def test_invalid_fusion_strategy(fusion_instance):
    # Testing invalid fusion strategy
    with pytest.raises(ValueError):
        fusion_instance.fuse(strategy="invalid_strategy")
