import math

import numpy as np
import pytest

from src.utils.toolkit.auroc import calculate_dynamic_auroc


@pytest.mark.unit
class TestToolkitAuroc:
    def test_returns_nan_for_empty_inputs(self) -> None:
        targets = np.array([])
        probs = np.array([])

        result = calculate_dynamic_auroc(targets, probs)

        assert math.isnan(result)

    def test_returns_nan_for_non_2d_probabilities(self) -> None:
        targets = np.array([0, 1, 0, 1])
        probs = np.array([0.1, 0.9, 0.2, 0.8])

        result = calculate_dynamic_auroc(targets, probs)

        assert math.isnan(result)

    def test_returns_nan_when_only_one_class_present(self) -> None:
        targets = np.array([1, 1, 1, 1])
        probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.15, 0.85], [0.05, 0.95]])

        result = calculate_dynamic_auroc(targets, probs)

        assert math.isnan(result)

    def test_binary_case_returns_expected_auroc(self) -> None:
        targets = np.array([0, 0, 1, 1])
        probs = np.array(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        )

        result = calculate_dynamic_auroc(targets, probs)

        assert result == pytest.approx(1.0)

    def test_multiclass_case_returns_expected_auroc(self) -> None:
        targets = np.array([0, 1, 2])
        probs = np.array(
            [
                [0.99, 0.005, 0.005],
                [0.005, 0.99, 0.005],
                [0.005, 0.005, 0.99],
            ]
        )

        result = calculate_dynamic_auroc(targets, probs)

        assert result == pytest.approx(1.0)

    def test_returns_nan_on_metric_value_error(self) -> None:
        targets = np.array([0, 1, 0, 1])
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]])

        result = calculate_dynamic_auroc(targets, probs)

        assert math.isnan(result)

