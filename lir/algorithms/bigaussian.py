"""
Bi-Gaussianized calibration of likelihood ratios.

Implementation of the method described in:

    Morrison G.S. (2024). Bi-Gaussianized calibration of likelihood ratios.
    Law, Probability & Risk, 23, mgae004. https://doi.org/10.1093/lpr/mgae004

The code follows the reference implementation available from
https://forensic-data-science.net/calibration-and-validation/#biGauss
(demo_biGaussianized_calibration - Python - 2025-07-10a).

Like the other calibrators in LiR, :class:`BiGaussianCalibrator` maps input scores (uncalibrated LLRs) to calibrated
LLRs. The mapping is constructed by matching the empirical cdf of the (equally weighted) same-source and
different-source training scores to the cdf of a perfectly calibrated bi-Gaussian system with variance
``sigma2``. The value of ``sigma2`` is either derived from a user-provided ``target_cllr``, or it is derived from the
Cllr that the training data obtains after regularized logistic-regression calibration.
"""

import logging
from collections.abc import Callable
from typing import Self

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import expit
from scipy.stats import norm

from lir import Transformer
from lir.data.models import FeatureData, InstanceData, LLRData
from lir.metrics import cllr
from lir.util import Xn_to_Xy, Xy_to_Xn, check_type, ln_to_log10


LOG = logging.getLogger(__name__)

# Empirical regression coefficients from Morrison (2024) mapping Cllr to sigma2 of a perfectly calibrated
# bi-Gaussian system with the same Cllr.
DEFAULT_CLLR_TO_SIGMA2_COEFS: tuple[float, float] = (17.665396790464737, 0.009333834837656)

# Default parameters for the regularized logistic regression that is used to estimate the Cllr of the training data
# when no target Cllr is provided: (prior, kappa, df, max_iter). See
# :func:`_train_logreg_fusion_regularized`.
DEFAULT_LOGREG_REGULARIZATION_COEFS: tuple[float, float, int | None, int] = (0.5, 0.01, None, 50000)


def cllr_to_sigma2(cllr_value: float, coefs: tuple[float, float] = DEFAULT_CLLR_TO_SIGMA2_COEFS) -> float:
    """
    Map a Cllr value to the variance of a perfectly calibrated bi-Gaussian system.

    A perfectly calibrated bi-Gaussian system has different-source log-likelihood-ratios distributed as
    ``N(-sigma2/2, sigma2)`` and same-source log-likelihood-ratios distributed as ``N(sigma2/2, sigma2)``. Such a
    system has a specific Cllr value for each ``sigma2``. This function inverts that relationship using the empirical
    fit described in Morrison (2024).

    Parameters
    ----------
    cllr_value : float
        Cllr value to map; must be in the range (0, 1).
    coefs : tuple[float, float], optional
        Regression coefficients ``(b, c)`` of the empirical fit, by default
        :data:`DEFAULT_CLLR_TO_SIGMA2_COEFS`.

    Returns
    -------
    float
        Variance (``sigma2``) of the perfectly calibrated bi-Gaussian system with the given Cllr.
    """
    b, c = coefs
    return float(-np.log(np.log(cllr_value) / b + 1) / c)


def np_unique_last(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the unique elements in an array with the indices of their last occurrences.

    Equivalent to the MATLAB function ``unique(A, 'last')``.

    Parameters
    ----------
    A : np.ndarray
        1-dimensional array with values in non-decreasing order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Unique values and the indices of their last occurrences in ``A``.
    """
    _, indices = np.unique(A[::-1], return_index=True)
    indices = len(A) - 1 - indices
    return A[indices], indices


def _as_score_data(instances: InstanceData) -> LLRData:
    """
    Return the instances as LLRData, requiring a single score column.

    Parameters
    ----------
    instances : InstanceData
        Input instances.

    Returns
    -------
    LLRData
        The input instances as LLRData.
    """
    instances = check_type(FeatureData, instances)
    if isinstance(instances, LLRData):
        return instances
    if instances.features.ndim > 1 and instances.features.shape[-1] != 1:
        raise ValueError(
            f'bi-Gaussianized calibration maps a single score to an LLR, but the input has '
            f'{instances.features.shape[-1]} features per instance; use it as a calibration step on uncalibrated scores'
        )
    return instances.replace_as(LLRData)


def _train_logreg_fusion_regularized(
    tar: np.ndarray,
    non: np.ndarray,
    prior: float = 0.5,
    kappa: float = 0.0,
    df: int | None = None,
    max_iter: int = 50000,
) -> np.ndarray:
    """
    Train a (optionally regularized) prior-weighted logistic regression.

    Trains a linear fusion of the scores with a prior-weighted logistic-regression objective, such that the output is a
    well-calibrated natural-log likelihood ratio: ``lnLR = weights[0] * score + weights[1]``. The regularization adds
    pseudo data points that shrink the log-likelihood-ratio output towards 0; see Morrison G.S., Poh N. (2018), Science
    & Justice 58, 200-218. This implementation is a port of the reference implementation in
    https://geoff-morrison.net/#shrunk_LRs (FoCal/Matlab origin by N. Brümmer, T. Minka); the pseudo data is realized
    by mirroring the training data with additional sample weights and solving the objective with conjugate gradient
    descent.

    Parameters
    ----------
    tar : np.ndarray
        1-dimensional array of same-source (target) scores.
    non : np.ndarray
        1-dimensional array of different-source (non-target) scores.
    prior : float, optional
        Prior probability of same-source, by default 0.5.
    kappa : float, optional
        Strength of the pseudo data points in units of pseudodata points; 0 means no regularization, by default 0.0.
    df : int | None, optional
        Pseudo degrees of freedom of the regularization; if None, the number of same-source scores is used, by default
        None. Irrelevant if ``kappa`` is 0.
    max_iter : int, optional
        Maximum number of conjugate-gradient iterations, by default 50000.

    Returns
    -------
    np.ndarray
        Weight array ``[slope, intercept]``.
    """
    nt, nn = tar.shape[0], non.shape[0]
    prop = nt / (nt + nn)

    # each column of x is one observation: [score; 1], negated for the different-source class; the ±1 in the second
    # row encodes the prior as an offset in the linear predictor
    x = np.vstack((np.concatenate((tar, -non)), np.concatenate((np.ones(nt), -np.ones(nn)))))
    weights = np.concatenate((np.full(nt, prior / prop), np.full(nn, (1 - prior) / (1 - prop))))
    offset = float(np.log(prior / (1 - prior))) * x[1]

    if kappa != 0:
        if df is None:
            df = nt
        # add the maximally uninformative uniform distribution as pseudo data, with total weight kappa
        x = np.concatenate((x, x, -x), axis=1)
        weights = np.concatenate((weights, np.full(2 * weights.size, kappa / (2 * df))))
        offset = np.concatenate((offset, offset, -offset))

    w = np.zeros(x.shape[0])
    g = u = np.zeros(x.shape[0])
    for iteration in range(max_iter):
        old_w, old_g = w, g
        s1 = expit(-(w @ x + offset))
        g = x @ (s1 * weights)
        if iteration == 0:
            u = g
        else:
            delta = g - old_g
            den = u @ delta
            u = np.zeros_like(g) if den == 0 else g - ((g @ delta) / den) * u

        step = (u @ g) / ((u @ x) ** 2 @ (weights * s1 * (1 - s1)))
        w = w + step * u
        if np.max(np.abs(w - old_w)) < 1e-5:
            break
    return w


class BiGaussianCalibrator(Transformer):
    """
    Calibrate scores using bi-Gaussianized calibration.

    Maps scores to log-likelihood-ratios by transforming the empirical cdf of the training data (with equal weight
    for the same-source and the different-source class) onto the cdf of a perfectly calibrated bi-Gaussian system with
    variance ``sigma2``. See the module documentation of :mod:`lir.algorithms.bigaussian` for a description of the
    method and references.

    Infinite values in the input are ignored, except if they are misleading, which is an error. Different-source
    scores of ``-inf`` are treated as a point mass at the lowest finite different-source score; same-source scores of
    ``inf`` are treated as a point mass at the highest finite same-source score. Test scores outside the range of the
    training scores are mapped to the boundary values.

    Parameters
    ----------
    target_cllr : float | None, optional
        If provided, the target Cllr (in the range (0, 1)) of the bi-Gaussian system to calibrate to. If not provided,
        the Cllr of the training data after logistic-regression calibration is used, by default None.
    cllr_to_sigma2_coefs : tuple[float, float], optional
        Regression coefficients ``(b, c)`` for :func:`cllr_to_sigma2`, by default
        :data:`DEFAULT_CLLR_TO_SIGMA2_COEFS`.
    logreg_regularization_coefs : tuple[float, float, int | None, int], optional
        Parameters ``(prior, kappa, df, max_iter)`` for the regularized logistic regression that estimates the Cllr of
        the training data; if ``df`` is None, the number of same-source training scores is used, by default
        :data:`DEFAULT_LOGREG_REGULARIZATION_COEFS`. Only used when ``target_cllr`` is not provided.
    """

    def __init__(
        self,
        target_cllr: float | None = None,
        cllr_to_sigma2_coefs: tuple[float, float] = DEFAULT_CLLR_TO_SIGMA2_COEFS,
        logreg_regularization_coefs: tuple[float, float, int | None, int] = DEFAULT_LOGREG_REGULARIZATION_COEFS,
    ):
        if target_cllr is not None and not 0 < target_cllr < 1:
            raise ValueError(f'target_cllr must be a numeric value in the range (0, 1); found: {target_cllr}')

        self.target_cllr = target_cllr
        self.cllr_to_sigma2_coefs = cllr_to_sigma2_coefs
        self.logreg_regularization_coefs = logreg_regularization_coefs
        self.sigma2: float | None = None
        self._score_to_cdf: Callable | None = None
        self._cdf_to_lnllr: Callable | None = None
        self._min_score: float | None = None
        self._max_score: float | None = None

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the bi-Gaussian calibration model on training data.

        Parameters
        ----------
        instances : InstanceData
            Training instances.

        Returns
        -------
        Self
            Fitted calibrator.
        """
        instances = _as_score_data(instances)
        instances.check_misleading_finite()

        scores_d, scores_s = (
            scores.flatten() for scores in Xy_to_Xn(instances.llrs.reshape(-1, 1), instances.require_labels)
        )
        num_d_train = len(scores_d)
        num_s_train = len(scores_s)
        num_train = num_d_train + num_s_train
        if num_d_train == 0 or num_s_train == 0:
            raise ValueError('bi-Gaussianized calibration needs both same-source and different-source training scores')

        # Treat -inf different-source scores as a point mass at the lowest finite different-source score, and +inf
        # same-source scores as a point mass at the highest finite same-source score.
        if np.any(np.isneginf(scores_d)):
            scores_d = np.where(np.isneginf(scores_d), np.min(scores_d[np.isfinite(scores_d)]), scores_d)
        if np.any(np.isposinf(scores_s)):
            scores_s = np.where(np.isposinf(scores_s), np.max(scores_s[np.isfinite(scores_s)]), scores_s)

        # Empirical cdf for training data, giving equal weight to the same-source and different-source classes. The 1
        # added to the denominators ensures the cdf does not reach 0 or 1 at the extremes of the training data.
        props = np.concatenate(
            (np.full(num_d_train, 0.5 / (num_d_train + 1)), np.full(num_s_train, 0.5 / (num_s_train + 1)))
        )
        scores_train = np.concatenate((scores_d, scores_s))
        ID_sorted = scores_train.argsort()
        scores_train_sorted = scores_train[ID_sorted]
        cdf_empirical = np.cumsum(props[ID_sorted])

        # In case of repeated score values, use the cdf of the last (highest) occurrence of each unique score value
        unique_scores, ID_unique = np_unique_last(scores_train_sorted)
        self._score_to_cdf = interp1d(unique_scores, cdf_empirical[ID_unique], kind='linear', fill_value='extrapolate')
        self._min_score = float(unique_scores[0])
        self._max_score = float(unique_scores[-1])

        # Determine sigma2, the variance of the target bi-Gaussian system, and build the inverse cdf of that system
        sigma2_target = self._target_sigma2(scores_s, scores_d, num_s_train)
        self.sigma2 = sigma2_target
        LOG.info(f'biGaussianized calibration with sigma2 = {sigma2_target}')

        self._cdf_to_lnllr = self._build_target_cdf_inverter(sigma2_target, num_train)
        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """
        Apply the fitted bi-Gaussian calibration model to new data.

        Parameters
        ----------
        instances : InstanceData
            Instances to calibrate.

        Returns
        -------
        LLRData
            Calibrated log-likelihood-ratio data.
        """
        if self._score_to_cdf is None or self._cdf_to_lnllr is None or self._min_score is None:
            raise ValueError('trying to use a model before fitting')

        instances = _as_score_data(instances)

        # Map scores to cdf values, then to bi-Gaussianized-calibrated lnLRs. Scores outside the training range are
        # clamped to the extremes of the training range, so -inf and +inf input obtain finite calibrated LLRs.
        scores = np.clip(instances.llrs, self._min_score, self._max_score)
        lnllrs = self._cdf_to_lnllr(self._score_to_cdf(scores))
        llrs = ln_to_log10(lnllrs)

        return instances.replace(features=llrs.reshape(-1, 1), llr_lower_bound=None, llr_upper_bound=None)

    def _target_sigma2(self, scores_s: np.ndarray, scores_d: np.ndarray, num_s_train: int) -> float:
        """
        Determine the variance of the target bi-Gaussian system.

        If a target Cllr was provided, it is used directly. Otherwise the Cllr of the training data after regularized
        logistic-regression calibration is calculated and mapped to a sigma2 value.

        Parameters
        ----------
        scores_s : np.ndarray
            1-dimensional array of same-source training scores.
        scores_d : np.ndarray
            1-dimensional array of different-source training scores.
        num_s_train : int
            Number of same-source training scores.

        Returns
        -------
        float
            Variance (sigma2) of the target bi-Gaussian system.
        """
        if self.target_cllr is not None:
            return cllr_to_sigma2(self.target_cllr, self.cllr_to_sigma2_coefs)

        prior, kappa, df, max_iter = self.logreg_regularization_coefs
        weights = _train_logreg_fusion_regularized(
            scores_s, scores_d, prior=prior, kappa=kappa, df=num_s_train if df is None else df, max_iter=max_iter
        )

        # Cllr of the training data after logistic-regression calibration
        scores, labels = Xn_to_Xy(scores_d, scores_s)
        lnlrs = weights[0] * scores + weights[1]
        llr_data = LLRData(features=ln_to_log10(lnlrs).reshape(-1, 1), hypothesis=labels)
        return cllr_to_sigma2(cllr(llr_data), self.cllr_to_sigma2_coefs)

    @staticmethod
    def _build_target_cdf_inverter(sigma2_target: float, num_train: int) -> Callable:
        """
        Build a function mapping cdf values to the lnLRs of a perfectly calibrated bi-Gaussian system.

        The target system has different-source lnLRs distributed as ``N(-sigma2/2, sigma2)`` and same-source lnLRs
        distributed as ``N(sigma2/2, sigma2)``, with equal class priors. The range of the grid is chosen such that the
        full range of lnLRs with appreciable probability mass is covered, assuming all values are within ``mu_d - 4
        sigma`` to ``mu_s + 4 sigma``.

        Parameters
        ----------
        sigma2_target : float
            Variance of the perfectly calibrated bi-Gaussian system.
        num_train : int
            Total number of training scores; determines the grid resolution.

        Returns
        -------
        Callable
            Interpolation function mapping cdf values to lnLRs.
        """
        half_sigma2_target = sigma2_target / 2
        sigma_target = np.sqrt(sigma2_target)

        lnlr_max = half_sigma2_target + 4 * sigma_target
        lnlr_step = lnlr_max / (4 * (num_train + 1))
        lnlr_grid = np.arange(-lnlr_max, lnlr_max, lnlr_step)

        # GMM cdf of the perfectly calibrated bi-Gaussian system
        cdf_grid = 0.5 * norm.cdf(lnlr_grid, loc=-half_sigma2_target, scale=sigma_target) + 0.5 * norm.cdf(
            lnlr_grid, loc=half_sigma2_target, scale=sigma_target
        )

        # Convert the grid to unique cdf values to avoid problems due to numerical constraints on the calculation of
        # the target cdf values
        cdf_unique, ID_unique = np.unique(cdf_grid, return_index=True)
        return interp1d(cdf_unique, lnlr_grid[ID_unique], kind='linear', fill_value='extrapolate')
