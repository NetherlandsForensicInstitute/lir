import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from scipy.stats import cramervonmises_2samp

from lir.algorithms.gof_test import (
    WithinSourceGofResult,
    _comparison_matrix,
    _cvm_statistic,
    _cvm_statistic_many,
    _reml_sums_of_squares,
    within_source_gof_test,
)
from lir.data.models import FeatureData


def euclidean(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y))


class TestComparisonMatrix:
    def test_structure(self):
        for n in (3, 4, 7, 9):
            matrix = _comparison_matrix(n)
            row_indices, col_indices = np.triu_indices(n, 1)
            assert matrix.shape == (len(row_indices), n)
            for t in range(len(row_indices)):
                assert matrix[t, row_indices[t]] == 1.0
                assert matrix[t, col_indices[t]] == 1.0
                assert matrix[t].sum() == 2.0

    def test_maps_effects_to_pair_scores(self):
        n = 5
        matrix = _comparison_matrix(n)
        row_indices, col_indices = np.triu_indices(n, 1)
        effects = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(matrix @ effects, effects[row_indices] + effects[col_indices])


class TestCvmStatistic:
    def test_known_value_with_ties(self):
        # pooled: 1, 2, 2, 3, 3, 4
        # contributions: (1/3 - 0)^2 * 1/6 + (2/3 - 1/3)^2 * 2/6 + (1 - 2/3)^2 * 2/6 + 0
        assert_almost_equal(_cvm_statistic(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])), 5 / 54)

    def test_zero_for_identical_distributions(self):
        assert _cvm_statistic(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0
        assert _cvm_statistic(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0

    def test_matches_scipy_for_continuous_data(self):
        rng = np.random.default_rng(42)
        scores_1 = rng.normal(0.0, 1.0, 10)
        scores_2 = rng.normal(0.5, 1.0, 12)
        n1, n2 = scores_1.shape[0], scores_2.shape[0]
        # scipy returns n1*n2/(n1+n2) * omega^2, so the scaling must match
        assert_almost_equal(
            cramervonmises_2samp(scores_1, scores_2).statistic,
            _cvm_statistic(scores_1, scores_2) * n1 * n2 / (n1 + n2),
        )

    def test_invariance_to_monotonic_transformations(self):
        rng = np.random.default_rng(42)
        scores_1 = rng.normal(0.0, 1.0, 10) + 5.0
        scores_2 = rng.normal(0.5, 1.0, 12) + 5.0
        statistic = _cvm_statistic(scores_1, scores_2)
        for transform in (lambda s: np.log(s), lambda s: -s, lambda s: s**3, lambda s: np.exp(s)):
            assert_almost_equal(_cvm_statistic(transform(scores_1), transform(scores_2)), statistic, decimal=10)

    def test_many_matches_single(self):
        rng = np.random.default_rng(42)
        batch_1 = rng.normal(0.0, 1.0, (50, 6))
        batch_2 = rng.normal(0.5, 1.0, (50, 8))
        expected = np.array([_cvm_statistic(s1, s2) for s1, s2 in zip(batch_1, batch_2, strict=True)])
        assert_allclose(_cvm_statistic_many(batch_1, batch_2), expected, rtol=0.0, atol=1e-14)


class TestRemlSumsOfSquares:
    def test_known_values(self):
        # score matrix of a source with 4 objects
        score_matrix = np.array(
            [
                [0.0, 2.0, 4.0, 6.0],
                [2.0, 0.0, 8.0, 10.0],
                [4.0, 8.0, 0.0, 13.0],
                [6.0, 10.0, 13.0, 0.0],
            ]
        )
        theta, ss_a, ss_t, ss_e = _reml_sums_of_squares(score_matrix)
        assert_almost_equal(theta, 43 / 6)
        assert_almost_equal(ss_a, 161 / 2)
        assert_almost_equal(ss_t, 485 / 6)
        assert_almost_equal(ss_e, 1 / 3)

    def test_constant_scores(self):
        # constant off-diagonal scores (the diagonal is zero by definition of the score matrix)
        score_matrix = 3.0 * (np.ones((5, 5)) - np.eye(5))
        theta, ss_a, ss_t, ss_e = _reml_sums_of_squares(score_matrix)
        assert theta == 3.0
        assert ss_a == 0.0
        assert ss_t == 0.0
        assert ss_e == 0.0


class TestWithinSourceGofTest:
    def _score_function_from_matrix(self, matrix_1: np.ndarray, matrix_2: np.ndarray):
        """Return a score function that reproduces the given score matrices for two sources."""
        tables = {}
        for source_id, matrix in ((0, matrix_1), (1, matrix_2)):
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[0]):
                    tables[(source_id, i, j)] = matrix[i, j]

        def score_function(x: np.ndarray, y: np.ndarray) -> float:
            return tables[(int(x[0]), int(x[1]), int(y[1]))]

        objects_1 = np.array([[0, i] for i in range(matrix_1.shape[0])], dtype=float)
        objects_2 = np.array([[1, i] for i in range(matrix_2.shape[0])], dtype=float)
        return objects_1, objects_2, score_function

    def test_pooled_reml_estimates(self):
        matrix_1 = np.array(
            [
                [0.0, 2.0, 4.0, 6.0],
                [2.0, 0.0, 8.0, 10.0],
                [4.0, 8.0, 0.0, 13.0],
                [6.0, 10.0, 13.0, 0.0],
            ]
        )
        # per source: theta = 43/6, ss_a = 161/2, ss_e = 1/3
        objects_1, objects_2, score_function = self._score_function_from_matrix(matrix_1, matrix_1)
        result = within_source_gof_test(objects_1, objects_2, score_function, n_bootstrap=100, random_state=42)
        assert_almost_equal(result.theta_p, 43 / 6)
        # sigma_e2_p = (1/3 + 1/3) / (6 + 6 - 4 - 4) = 1/6
        assert_almost_equal(result.sigma_e2_p, 1 / 6)
        # sigma_a2_p = (161/2 + 161/2 - 1/6 * 6) / (3 * 2 + 3 * 2) = (161 - 1) / 12 = 40/3
        assert_almost_equal(result.sigma_a2_p, 40 / 3)
        # identical score distributions: the observed statistic is zero
        assert result.statistic == 0.0
        assert 0.0 <= result.p_value <= 1.0

    def test_result_fields(self):
        rng = np.random.default_rng(0)
        source_1 = rng.normal(0.0, 1.0, (8, 3))
        source_2 = rng.normal(0.0, 1.0, (8, 3))
        result = within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=100, random_state=0)
        assert isinstance(result, WithinSourceGofResult)
        assert result.statistic >= 0.0
        assert 0.0 <= result.p_value <= 1.0
        assert result.theta_p > 0.0
        assert result.sigma_a2_p >= 0.0
        assert result.sigma_e2_p >= 0.0

    def test_deterministic_for_fixed_random_state(self):
        rng = np.random.default_rng(0)
        source_1 = rng.normal(0.0, 1.0, (8, 3))
        source_2 = rng.normal(0.0, 1.0, (8, 3))
        result_a = within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=100, random_state=0)
        result_b = within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=100, random_state=0)
        assert result_a == result_b
        # a different seed can give a different (discretized) bootstrap p-value
        result_c = within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=100, random_state=1)
        assert result_c.statistic == result_a.statistic
        assert result_c.theta_p == result_a.theta_p

    def test_feature_data_input(self):
        rng = np.random.default_rng(0)
        features_1 = rng.normal(0.0, 1.0, (8, 3))
        features_2 = rng.normal(0.0, 1.0, (8, 3))
        result_array = within_source_gof_test(features_1, features_2, euclidean, n_bootstrap=100, random_state=0)
        result_feature_data = within_source_gof_test(
            FeatureData(features=features_1),
            FeatureData(features=features_2),
            euclidean,
            n_bootstrap=100,
            random_state=0,
        )
        assert result_feature_data == result_array

    def test_statistic_invariant_to_monotonic_score_transform(self):
        rng = np.random.default_rng(0)
        source_1 = rng.normal(0.0, 1.0, (8, 3)) + 10.0
        source_2 = rng.normal(0.0, 1.0, (8, 3)) + 10.0

        def log_euclidean(x: np.ndarray, y: np.ndarray) -> float:
            return float(np.log(np.linalg.norm(x - y)))

        result_euclidean = within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=100, random_state=0)
        result_log = within_source_gof_test(source_1, source_2, log_euclidean, n_bootstrap=100, random_state=0)
        assert_almost_equal(result_euclidean.statistic, result_log.statistic)

    def test_error_too_few_objects(self):
        with pytest.raises(ValueError, match='at least 3 objects'):
            within_source_gof_test(np.ones((2, 2)), np.ones((5, 2)), euclidean)

    def test_error_both_sources_have_three_objects(self):
        with pytest.raises(ValueError, match='at least 4 objects'):
            within_source_gof_test(np.ones((3, 2)), np.ones((3, 2)), euclidean)

    def test_error_one_source_has_four_objects(self):
        # the error variance is estimable as long as one source has 4 or more objects
        result = within_source_gof_test(np.ones((3, 2)), 2.0 * np.ones((4, 2)), euclidean, random_state=0)
        assert isinstance(result, WithinSourceGofResult)

    def test_error_non_numeric_source(self):
        with pytest.raises(ValueError, match='numeric'):
            within_source_gof_test(np.array([['a'], ['b'], ['c']]), np.ones((5, 2)), euclidean)

    def test_error_non_finite_source(self):
        with pytest.raises(ValueError, match='non-finite'):
            within_source_gof_test(np.array([[np.nan], [1.0], [2.0]]), np.ones((5, 2)), euclidean)

    def test_error_one_dimensional_source(self):
        with pytest.raises(ValueError, match='two-dimensional'):
            within_source_gof_test(np.ones(5), np.ones((5, 2)), euclidean)

    def test_error_non_callable_score_function(self):
        with pytest.raises(ValueError, match='callable'):
            within_source_gof_test(np.ones((5, 2)), np.ones((5, 2)), 3.0)  # type: ignore[arg-type]

    def test_error_non_finite_score(self):
        def non_finite_score(x: np.ndarray, y: np.ndarray) -> float:
            return float('nan')

        with pytest.raises(ValueError, match='non-finite score'):
            within_source_gof_test(np.ones((5, 2)), np.ones((5, 2)), non_finite_score)

    def test_error_invalid_n_bootstrap(self):
        source_1 = np.ones((5, 2))
        source_2 = np.ones((5, 2))
        with pytest.raises(ValueError, match='n_bootstrap'):
            within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=0)
        with pytest.raises(ValueError, match='n_bootstrap'):
            within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=-1)
        with pytest.raises(ValueError, match='n_bootstrap'):
            within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=1.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match='n_bootstrap'):
            within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=True)  # type: ignore[arg-type]


class TestWithinSourceGofTestSimulation:
    """
    Sanity checks of the Type I error and power of the test, based on the simulation scenarios of
    Hanka et al. (2026): 9 objects per source, the Euclidean distance as score function, and
    B = 1000 bootstrap samples. The scenarios and the expected behavior (approximately valid Type I
    error control and non-trivial power) are reproduced with a fixed random seed; the bounds are
    deliberately wide because the test has limited power for small sample sizes.
    """

    def test_type_i_error(self):
        rejections = 0
        p_values = []
        for rep in range(60):
            rng = np.random.default_rng(1000 + rep)
            source_1 = rng.normal(0.0, 1.0, (9, 6))
            source_2 = rng.normal(0.0, 1.0, (9, 6))
            result = within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=1000, random_state=2000 + rep)
            p_values.append(result.p_value)
            if result.p_value < 0.05:
                rejections += 1
        # nominal Type I error is 0.05; allow for the wide bounds of small-sample tests
        assert rejections / 60 <= 0.20
        assert 0.25 <= float(np.mean(p_values)) <= 0.75

    def test_power_different_random_effect_variance(self):
        rejections = 0
        for rep in range(60):
            rng = np.random.default_rng(1000 + rep)
            source_1 = rng.standard_t(3, 9)[:, None]
            source_2 = 3.0 * rng.standard_t(3, 9)[:, None]
            result = within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=1000, random_state=2000 + rep)
            if result.p_value < 0.05:
                rejections += 1
        assert rejections / 60 >= 0.30

    def test_power_different_scale(self):
        rejections = 0
        for rep in range(60):
            rng = np.random.default_rng(1000 + rep)
            source_1 = rng.normal(0.0, 1.0, (9, 6))
            source_2 = 3.0 * rng.normal(0.0, 1.0, (9, 6))
            result = within_source_gof_test(source_1, source_2, euclidean, n_bootstrap=1000, random_state=2000 + rep)
            if result.p_value < 0.05:
                rejections += 1
        assert rejections / 60 >= 0.80
