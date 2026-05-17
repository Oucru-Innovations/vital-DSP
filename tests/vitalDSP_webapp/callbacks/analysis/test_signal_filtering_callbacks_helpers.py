"""Tests for the surviving helpers in ``signal_filtering_callbacks``.

The earlier version of this file exercised 12 ``calculate_*`` helpers
that duplicated logic from
``vitalDSP.signal_quality_assessment.FilteringQualityAssessment``.
Those duplicates were removed - the filtering page now delegates to
the library's quality-assessment helper, which has its own tests in
``tests/vitalDSP/signal_quality_assessment``.  This file keeps tests
for the small webapp-local helpers that survived.
"""

from __future__ import annotations

import numpy as np
import pytest

from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
    calculate_mse,
    create_filter_comparison_plot,
    create_filter_quality_plots,
)


@pytest.fixture
def sample_signal():
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    signal += 0.1 * np.random.randn(len(signal))
    return signal, 100.0


@pytest.fixture
def filtered_signal(sample_signal):
    signal, _ = sample_signal
    return signal * 0.95 + 0.05 * float(np.mean(signal))


class TestCalculateMSE:
    def test_basic(self, sample_signal, filtered_signal):
        signal, _ = sample_signal
        mse = calculate_mse(signal, filtered_signal)
        assert isinstance(mse, float)
        assert mse >= 0

    def test_identical_signals(self, sample_signal):
        signal, _ = sample_signal
        assert calculate_mse(signal, signal) == pytest.approx(0.0, abs=1e-12)

    def test_different_lengths(self, sample_signal):
        signal, _ = sample_signal
        # Length mismatch: the helper logs the broadcast error and falls
        # back to 0 (sentinel) rather than raising.  Either an int 0 or
        # a float is acceptable as long as it's finite and non-negative.
        out = calculate_mse(signal, signal[:500])
        assert isinstance(out, (int, float))
        assert np.isfinite(out) and out >= 0


class TestCreateFilterComparisonPlot:
    def test_returns_figure(self, sample_signal, filtered_signal):
        signal, fs = sample_signal
        time_axis = np.arange(len(signal)) / fs
        fig = create_filter_comparison_plot(time_axis, signal, filtered_signal, fs, "ECG")
        # Slim version is a single overlay with two traces.
        assert len(fig.data) == 2
        names = {t.name for t in fig.data}
        assert names == {"Original", "Filtered"}


class TestCreateFilterQualityPlots:
    def test_returns_two_panel(self, sample_signal, filtered_signal):
        signal, fs = sample_signal
        fig = create_filter_quality_plots(signal, filtered_signal, fs, ["snr"], "ECG")
        # 2 panels: amplitude overlay (2 traces) + spectrum overlay (2 traces).
        assert len(fig.data) == 4

    def test_empty_signal(self):
        from vitalDSP_webapp.callbacks.analysis.signal_filtering_callbacks import (
            create_empty_figure,
        )

        fig = create_filter_quality_plots(np.array([]), np.array([]), 100.0, [], "ECG")
        # Should fall back to an empty figure rather than raise.
        assert isinstance(fig, type(create_empty_figure()))
