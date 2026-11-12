import numpy as np
import pytest

from core.moo import BaselineSeededSampling
from core.objectives import DesignPoint, validate_design_points


class FakeProblem:
    xl = np.full(18, -1.0)
    xu = np.full(18, 1.0)


def test_baseline_sampling_is_reproducible_with_seed():
    baseline = np.zeros(18)
    first = BaselineSeededSampling(baseline, seed=123)._do(FakeProblem(), 12)
    second = BaselineSeededSampling(baseline, seed=123)._do(FakeProblem(), 12)

    np.testing.assert_array_equal(first, second)


def test_duplicate_design_point_labels_are_rejected():
    points = [
        DesignPoint(label="cruise", re=1e6, cl_target=0.3),
        DesignPoint(label="cruise", re=6e5, cl_target=0.9),
    ]

    with pytest.raises(ValueError, match="unique"):
        validate_design_points(points)
