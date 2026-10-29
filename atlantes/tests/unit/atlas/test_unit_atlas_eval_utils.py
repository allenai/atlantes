"""Unit tests for the atlas_eval_utils module."""

import numpy as np
from atlantes.atlas.atlas_eval_utils import (
    binarize_by_fishing, convert_output_probs_to_binary_fishing_probs)


def test_binarize_by_fishing() -> None:
    """Test the binarize_by_fishing function."""
    label_array_1 = np.array([1, 2, 3, 0, 1, 1, 0])
    label_array_2 = np.array([1, 0, 0, 0, 0, 0, 0])
    label_array_3 = np.array([0, 0, 0, 0, 0, 0, 0])
    label_array_4 = np.array([1, 1, 1, 2, 1, 3, 1, 1])

    expected_output_1 = np.array([0, 0, 0, 1, 0, 0, 1])
    expected_output_2 = np.array([0, 1, 1, 1, 1, 1, 1])
    expected_output_3 = np.array([1, 1, 1, 1, 1, 1, 1])
    expected_output_4 = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    output_1 = binarize_by_fishing(label_array_1)
    output_2 = binarize_by_fishing(label_array_2)
    output_3 = binarize_by_fishing(label_array_3)
    output_4 = binarize_by_fishing(label_array_4)

    assert np.array_equal(output_1, expected_output_1)
    assert np.array_equal(output_2, expected_output_2)
    assert np.array_equal(output_3, expected_output_3)
    assert np.array_equal(output_4, expected_output_4)


def test_convert_output_probs_to_binary_fishing_probs() -> None:
    """Test the convert_output_probs_to_binary_fishing_probs function."""
    # 4 classes all probs add to 1
    probs_1 = np.array([[0.5, 0.3, 0.1, 0.1]])
    probs_2 = np.array(
        [[0.1, 0.1, 0.7, 0.1], [0.3, 0.1, 0.1, 0.5], [0.25, 0.25, 0.25, 0.25]]
    )

    expected_output_1 = np.array([[0.5, 0.5]])
    expected_output_2 = np.array([[0.9, 0.1], [0.7, 0.3], [0.75, 0.25]])

    output_1 = convert_output_probs_to_binary_fishing_probs(probs_1)
    output_2 = convert_output_probs_to_binary_fishing_probs(probs_2)

    assert np.array_equal(output_1, expected_output_1)
    assert np.array_equal(output_2, expected_output_2)
