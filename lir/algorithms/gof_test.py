"""
Goodness-of-fit test for comparing within-source distributions of scores.

This module implements a goodness-of-fit test for the null hypothesis that two sources share a common
marginal distribution of within-source scores. The test is based on the Cramer-von Mises (CvM) statistic,
whose invariance to strictly monotonic transformations is exploited to approximate the null distribution
with a parametric bootstrap. The bootstrap resamples scores from a random-effects model with pooled REML
estimates of the variance components, which yields proper Type I error control for the small sample sizes
(per source) that are typical in forensic science.

The method is described in Hanka, Ommen, Hietpas, Buscaglia & Saunders (2026), "Development and evaluation
of a goodness-of-fit test for comparing within-source distributions of scores suitable for small sample
sizes", Law, Probability and Risk, 25(1), mgag008. The REML estimates of the variance components follow
the random-effects model of Miller, Gantz & Saunders (2009).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lir.data.models import FeatureData


@dataclass(frozen=True)
class WithinSourceGofResult:
    """
    Result of a within-source goodness-of-fit test.

    Attributes
    ----------
    - statistic: the observed value of the Cramer-von Mises test statistic, i.e. the integral of the
      squared difference between the two within-source empirical score distributions, weighted by the
      pooled empirical distribution.
    - p_value: the p-value of the test with respect to the bootstrapped null distribution, i.e. the
      proportion of bootstrapped test statistics that are strictly larger than the observed statistic.
      A small p-value indicates that the within-source score distributions of the two sources are not
      statistically similar, and that the scores of the two sources should not be pooled.
    - theta_p: the pooled REML estimate of the common mean of the within-source scores.
    - sigma_a2_p: the pooled REML estimate of the variance component of the per-object random effects.
    - sigma_e2_p: the pooled REML estimate of the variance of the residual (lack-of-fit) term.
    """

    statistic: float
    p_value: float
    theta_p: float
    sigma_a2_p: float
    sigma_e2_p: float


def _as_objects(source: np.ndarray | FeatureData, name: str) -> np.ndarray:
    """
    Convert a source of objects to a two-dimensional numpy array.

    Parameters
    ----------
    source : np.ndarray | FeatureData
        The objects of a single source, either as a `FeatureData` object or as a two-dimensional array
        with one row per object.
    name : str
        The name of the parameter, used in error messages.

    Returns
    -------
    np.ndarray
        Two-dimensional array with one row per object.
    """
    objects = source.features if isinstance(source, FeatureData) else np.asarray(source)

    if objects.ndim != 2:
        raise ValueError(f'{name} must be two-dimensional with one row per object; found shape {objects.shape}')
    if not np.issubdtype(objects.dtype, np.number):
        raise ValueError(f'{name} must contain numeric values; found dtype {objects.dtype}')
    if np.any(~np.isfinite(objects)):
        raise ValueError(f'{name} contains non-finite values')

    return objects


def _within_source_scores(objects: np.ndarray, score_function: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """
    Calculate the within-source scores of all pairs of objects of a single source.

    Parameters
    ----------
    objects : np.ndarray
        Two-dimensional array with one row per object.
    score_function : Callable[[np.ndarray, np.ndarray], float]
        Function that maps a pair of objects (one-dimensional feature vectors) to a real-valued score.

    Returns
    -------
    np.ndarray
        Symmetric score matrix with one row and column per object. The score of object pair `(i, j)` is
        stored at `(i, j)` and `(j, i)`, and the diagonal is zero.
    """
    n = objects.shape[0]
    scores = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            try:
                value = float(score_function(objects[i], objects[j]))
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f'score_function must map a pair of objects to a real number; error for objects {i} and {j}: {err}'
                ) from err
            if not np.isfinite(value):
                raise ValueError(f'score_function returned a non-finite score for objects {i} and {j}: {value}')
            scores[i, j] = value
            scores[j, i] = value

    return scores


def _reml_sums_of_squares(score_matrix: np.ndarray) -> tuple[float, float, float, float]:
    r"""
    Calculate the REML sums of squares of the random-effects model for a single source.

    The random-effects model for the within-source score `s_ij` of objects `i` and `j` is

    .. math::

        s_{ij} = \theta + a_i + a_j + \epsilon_{ij}

    where `a_i` are the per-object random effects with variance `sigma_a2`, and `epsilon_ij` are the
    residual (lack-of-fit) terms with variance `sigma_e2`.

    Parameters
    ----------
    score_matrix : np.ndarray
        Symmetric score matrix with one row and column per object, with zero diagonal.

    Returns
    -------
    tuple[float, float, float, float]
        The REML estimate `theta` of the mean, and the sums of squares `ss_a`, `ss_t` and `ss_e` for
        the random effects, the total and the error, respectively.
    """
    n = score_matrix.shape[0]
    scores = score_matrix[np.triu_indices(n, 1)]

    theta = float(scores.mean())

    # the average of all scores that contain the m-th object
    row_means = score_matrix.sum(axis=1) / (n - 1)

    ss_a = (n - 1) ** 2 / (n - 2) * float(np.sum((row_means - theta) ** 2))
    ss_t = float(np.sum((scores - theta) ** 2))
    ss_e = ss_t - ss_a

    return theta, ss_a, ss_t, ss_e


def _comparison_matrix(n: int) -> np.ndarray:
    """
    Build the comparison matrix of the within-source score pairs.

    Each row corresponds to one pair of objects `(i, j)` with `i < j`, and has a one in the i-th and
    j-th column.

    Parameters
    ----------
    n : int
        Number of objects.

    Returns
    -------
    np.ndarray
        Comparison matrix of shape `(n * (n - 1) // 2, n)`.
    """
    indices = np.triu_indices(n, 1)
    matrix = np.zeros((len(indices[0]), n))
    row_indices = np.arange(matrix.shape[0])
    matrix[row_indices, indices[0]] = 1.0
    matrix[row_indices, indices[1]] = 1.0

    return matrix


def _cvm_statistic(scores_1: np.ndarray, scores_2: np.ndarray) -> float:
    r"""
    Calculate the two-sample Cramer-von Mises statistic for two vectors of within-source scores.

    The statistic is the integral of the squared difference between the two empirical score
    distributions, weighted by the pooled empirical distribution:

    .. math::

        \omega^2(s_1, s_2) = \int (F_{n_1}(\tau) - G_{n_2}(\tau))^2 dH_{n_1 n_2}(\tau)

    where `F_{n_1}` and `G_{n_2}` are the empirical distributions of the scores of the first and
    second source, and `H_{n_1 n_2}` is the pooled empirical distribution. The statistic is invariant
    to strictly monotonic transformations of the scores.

    Parameters
    ----------
    scores_1 : np.ndarray
        Vector of within-source scores of the first source.
    scores_2 : np.ndarray
        Vector of within-source scores of the second source.

    Returns
    -------
    float
        The two-sample Cramer-von Mises statistic.
    """
    n1, n2 = scores_1.shape[0], scores_2.shape[0]
    pooled = np.concatenate([scores_1, scores_2])
    order = np.argsort(pooled, kind='stable')
    is_first = order < n1
    cumulative_first = np.cumsum(is_first)
    cumulative_second = np.cumsum(~is_first)

    # group tied values; the empirical distributions are evaluated at the end of each tie group
    _, start_indices, counts = np.unique(pooled[order], return_index=True, return_counts=True)
    end_indices = start_indices + counts - 1

    first_cdf = cumulative_first[end_indices] / n1
    second_cdf = cumulative_second[end_indices] / n2
    pooled_cdf_increments = counts / (n1 + n2)

    return float(np.sum((first_cdf - second_cdf) ** 2 * pooled_cdf_increments))


def _cvm_statistic_many(scores_1: np.ndarray, scores_2: np.ndarray) -> np.ndarray:
    """
    Calculate the two-sample Cramer-von Mises statistic for multiple pairs of score vectors.

    This is a vectorized version of `_cvm_statistic()` for bootstrapping. It assumes that the scores
    are continuous, such that tied values have probability zero.

    Parameters
    ----------
    scores_1 : np.ndarray
        Array of score vectors of the first source, with shape `(b, n_1)`.
    scores_2 : np.ndarray
        Array of score vectors of the second source, with shape `(b, n_2)`.

    Returns
    -------
    np.ndarray
        Array of test statistics with shape `(b,)`.
    """
    n1, n2 = scores_1.shape[1], scores_2.shape[1]
    pooled = np.concatenate([scores_1, scores_2], axis=1)
    order = np.argsort(pooled, axis=1, kind='stable')
    is_first = order < n1
    cumulative_first = np.cumsum(is_first, axis=1)
    cumulative_second = np.cumsum(~is_first, axis=1)
    cdf_difference = cumulative_first / n1 - cumulative_second / n2

    return np.sum(cdf_difference**2, axis=1) / (n1 + n2)


def within_source_gof_test(
    source_1: np.ndarray | FeatureData,
    source_2: np.ndarray | FeatureData,
    score_function: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> WithinSourceGofResult:
    r"""
    Test whether the within-source score distributions of two sources are statistically similar.

    This function tests the null hypothesis that the two sources share a common marginal distribution
    of within-source scores, against the alternative that the distributions differ. It is a
    goodness-of-fit test that is specifically designed for the small sample sizes (number of objects
    per source) that are typical in forensic science, where a common practice is to pool the
    within-source scores of all sources to estimate a common within-source score distribution. This
    test can be used to check whether that pooling assumption is reasonable for a pair of sources.

    The test uses the two-sample Cramer-von Mises statistic, calculated on the vectors of all pairwise
    within-source scores of the two sources. The Cramer-von Mises statistic is invariant to strictly
    monotonic transformations of the scores. This invariance is exploited to approximate the null
    distribution of the statistic with a parametric bootstrap: the scores are resampled from the
    random-effects model

    .. math::

        s_{ij} = \theta + a_i + a_j + \epsilon_{ij}

    where the per-object random effects `a_i` are normal with variance `sigma_a2`, and the residual
    terms `epsilon_ij` are normal with variance `sigma_e2`. The parameters are the pooled REML
    estimates of the two sources. The p-value is the proportion of bootstrapped test statistics that
    are strictly larger than the observed test statistic.

    Parameters
    ----------
    source_1 : np.ndarray | FeatureData
        The objects of the first source, either as a `FeatureData` object or as a two-dimensional
        array with one row per object. The number of objects must be at least 3, and at least one of
        the two sources must contain at least 4 objects.
    source_2 : np.ndarray | FeatureData
        The objects of the second source, in the same format as `source_1`.
    score_function : Callable[[np.ndarray, np.ndarray], float]
        The within-source score function. This function maps a pair of objects (one-dimensional
        feature vectors) to a real-valued score. For example, a Euclidean distance can be used as
        score function: `lambda x, y: np.linalg.norm(x - y)`.
    n_bootstrap : int, optional
        The number of bootstrap samples used to approximate the null distribution. The default is
        1000. Must be a positive integer.
    random_state : int | np.random.Generator | None, optional
        Seed or generator used to initialize the random number generator of the bootstrap. The
        default is `None`, in which case a seed is derived from system entropy.

    Returns
    -------
    WithinSourceGofResult
        The result of the test, containing the observed test statistic, the p-value and the pooled
        REML estimates of the mean and the variance components.

    Raises
    ------
    ValueError
        If the sources contain fewer than 3 objects, if both sources contain exactly 3 objects, if the
        objects contain non-finite values, if `score_function` does not return finite real numbers, or
        if `n_bootstrap` is not a positive integer.

    References
    ----------
    Hanka, J. L., Ommen, D. M., Hietpas, J., Buscaglia, J., & Saunders, C. P. (2026). Development and evaluation
    of a goodness-of-fit test for comparing within-source distributions of scores suitable for small sample sizes.
    *Law, Probability and Risk*, 25(1), mgag008.

    Miller, J. J., Gantz, D. T., & Saunders, C. P. (2009). Quantifying the effects of database size and sample
    quality on measures of individualization validity and accuracy in forensics. NIJ, Phase I, Part B, pp. 51-67.
    """
    if not callable(score_function):
        raise ValueError(f'score_function must be callable; found: {type(score_function)}')
    if not isinstance(n_bootstrap, (int, np.integer)) or isinstance(n_bootstrap, bool) or n_bootstrap < 1:
        raise ValueError(f'n_bootstrap must be a positive integer; found: {n_bootstrap!r}')

    objects_1 = _as_objects(source_1, 'source_1')
    objects_2 = _as_objects(source_2, 'source_2')

    n1, n2 = objects_1.shape[0], objects_2.shape[0]
    if n1 < 3 or n2 < 3:
        raise ValueError(
            f'each source must contain at least 3 objects; found {n1} objects in source_1 and {n2} objects in source_2'
        )
    n_pairs_1 = n1 * (n1 - 1) // 2
    n_pairs_2 = n2 * (n2 - 1) // 2
    if n_pairs_1 + n_pairs_2 - n1 - n2 <= 0:
        raise ValueError(
            'unable to estimate the error variance of the random-effects model: at least one of the '
            'sources must contain at least 4 objects; found '
            f'{n1} objects in source_1 and {n2} objects in source_2'
        )

    # calculate the within-source score vectors of both sources
    score_matrix_1 = _within_source_scores(objects_1, score_function)
    score_matrix_2 = _within_source_scores(objects_2, score_function)
    scores_1 = score_matrix_1[np.triu_indices(n1, 1)]
    scores_2 = score_matrix_2[np.triu_indices(n2, 1)]

    # calculate the observed test statistic
    statistic = _cvm_statistic(scores_1, scores_2)

    # calculate the REML estimates of the mean and the variance components of both sources
    theta_1, ss_a_1, _, ss_e_1 = _reml_sums_of_squares(score_matrix_1)
    theta_2, ss_a_2, _, ss_e_2 = _reml_sums_of_squares(score_matrix_2)

    # pool the REML estimates across both sources
    theta_p = (n_pairs_1 * theta_1 + n_pairs_2 * theta_2) / (n_pairs_1 + n_pairs_2)
    sigma_e2_p = (ss_e_1 + ss_e_2) / (n_pairs_1 + n_pairs_2 - n1 - n2)

    # a negative error variance is a small-sample artifact; constrain it to zero, similar to the
    # random-effects variance component
    if sigma_e2_p < 0:
        sigma_e2_p = 0.0

    sigma_a2_p = (ss_a_1 + ss_a_2) - sigma_e2_p * (n1 + n2 - 2)
    sigma_a2_p /= (n1 - 1) * (n1 - 2) + (n2 - 1) * (n2 - 2)
    sigma_a2_p = max(0.0, sigma_a2_p)

    # parametric bootstrap: resample the scores from the random-effects model with the pooled REML
    # estimates, and calculate the test statistic of each resample
    rng = np.random.default_rng(random_state)
    comparison_matrix_1 = _comparison_matrix(n1)
    comparison_matrix_2 = _comparison_matrix(n2)

    if sigma_a2_p > 0.0:
        random_effects_1 = rng.normal(0.0, np.sqrt(sigma_a2_p), (n_bootstrap, n1))
        random_effects_2 = rng.normal(0.0, np.sqrt(sigma_a2_p), (n_bootstrap, n2))
    else:
        random_effects_1 = np.zeros((n_bootstrap, n1))
        random_effects_2 = np.zeros((n_bootstrap, n2))

    if sigma_e2_p > 0.0:
        residuals_1 = rng.normal(0.0, np.sqrt(sigma_e2_p), (n_bootstrap, n_pairs_1))
        residuals_2 = rng.normal(0.0, np.sqrt(sigma_e2_p), (n_bootstrap, n_pairs_2))
    else:
        residuals_1 = np.zeros((n_bootstrap, n_pairs_1))
        residuals_2 = np.zeros((n_bootstrap, n_pairs_2))

    pseudo_scores_1 = theta_p + random_effects_1 @ comparison_matrix_1.T + residuals_1
    pseudo_scores_2 = theta_p + random_effects_2 @ comparison_matrix_2.T + residuals_2

    bootstrap_statistics = _cvm_statistic_many(pseudo_scores_1, pseudo_scores_2)

    p_value = float(np.mean(bootstrap_statistics > statistic))

    return WithinSourceGofResult(
        statistic=statistic,
        p_value=p_value,
        theta_p=theta_p,
        sigma_a2_p=sigma_a2_p,
        sigma_e2_p=sigma_e2_p,
    )
