from typing import Self

import numpy as np
from scipy.special import logsumexp

from lir import Transformer
from lir.config.base import check_not_none
from lir.data.models import FeatureData, InstanceData, check_type
from lir.lrsystems.lrsystems import LLRData, LRSystem
from lir.transform.pairing import PairingMethod
from lir.transform.pipeline import Pipeline


class TwoLevelModelNormalKDE:
    """Implement two-level model as outlined by Bolck et al.

    An implementation of the two-level model as outlined in FSI191(2009)42 by Bolck et al. "Different likelihood
       ratio approaches to evaluate the strength of evidence of MDMA tablet comparisons".

       Model description:

       Definitions
       X_ij = vector, measurement of reference j, ith repetition, with i=1..n
       Y_kl = vector, measurement of trace l, kth repetition, with k=1..m

       Model:

       First level of variance:
       X_ij ~ N(theta_j, sigma_within)
       Y_kl ~ N(theta_k, sigma_within)
       where theta_j is the true but unknown mean of the reference and theta_k the true but unknown mean of the trace.
       sigma_within is assumed equal for trace and reference (and for repeated measurements of some background data)

       Second level of variance:
       theta_j ~ theta_k ~ KDE(means background database, h)
       with h the kernel bandwidth.

       H1: theta_j = theta_k
       H2: theta_j independent of theta_k

       Numerator LR = Integral_theta N(X_Mean|theta, sigma_within, n) * N(Y_mean|theta, sigma_within, m) * \
            KDE(theta|means background database, h)
       Denominator LR = Integral_theta N(X_Mean|theta, sigma_within, n) * KDE(theta|means background database, h) * \
            Integral_theta N(Y_Mean|theta, sigma_within, m) * KDE(theta|means background database, h)

       In Bolck et al. in the appendix one finds a closed-form solution for the evaluation of these integrals.

       sigma_within and h (and other parameters) are estimated from repeated measurements of background data.
    """

    def __init__(self) -> None:
        self.model_fitted = False
        self.n_features_train: int | None = None
        self.n_sources: int | None = None
        self.mean_within_covars: np.ndarray | None = None
        self.means_per_source: np.ndarray | None = None
        self.kernel_bandwidth_sq: float | None = None
        self.between_covars: np.ndarray | None = None

    def fit_on_unpaired_instances(self, X: np.ndarray, y: np.ndarray) -> 'TwoLevelModelNormalKDE':
        """Fit the model on unpaired instances.

        X np.ndarray of measurements, rows are sources/repetitions, columns are features
        y np 1d-array of labels. For each source a unique identifier (label). Repetitions get the same label.

        Construct the necessary matrices/scores/etc based on test data (X) so that we can predict a score later on.
        Store any calculated parameters in `self`.
        """
        assert len(X.shape) == 2, f'fit(X, y) requires X to be 2-dimensional; found dimensions {X.shape}'
        self.n_sources = self._get_n_sources(y)
        self.n_features_train = X.shape[1]
        self.mean_within_covars = self._get_mean_covariance_within(X, y)
        self.means_per_source = self._get_means_per_source(X, y)
        self.kernel_bandwidth_sq = self._get_kernel_bandwidth_squared(self.n_sources, self.n_features_train)
        self.between_covars = self._get_between_covariance(X, y, self.mean_within_covars)
        self.model_fitted = True

        return self

    def transform(self, X_trace: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
        """Transform the input data using the fitted model.

        Predict odds scores, making use of the parameters constructed during `self.fit()` (which should
        now be stored in `self`).

        X_trace measurements of trace object. np.ndarray of shape (instances, repetitions_trace, features)
        X_ref measurements of reference object. np.ndarray of shape (instances, repetitions_ref, features)

        returns: odds of same source / different source: one-dimensional np.ndarray with one element per instance
        """
        assert self.model_fitted, 'fit() must be called before transform()'
        log10_lr_score = self._predict_log10_lr_score(X_trace, X_ref)
        return log10_lr_score

    def predict_proba(self, X_trace: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
        """Predict probability scores, using the fitted model.

        Predict probability scores, making use of the parameters constructed during `self.fit()` (which should
        now be stored in `self`).

        X_trace measurements of trace object. np.ndarray of shape (instances, repetitions_trace, features)
        X_ref measurements of reference object. np.ndarray of shape (instances, repetitions_ref, features)

        returns: probabilities for same source and different source: np.ndarray with shape (instances, 2)
        """
        logodds_score = self.transform(X_trace, X_ref)
        p0 = 1 / (1 + 10**logodds_score)
        p1 = 1 - p0
        return np.stack([p0, p1], axis=1)

    def _predict_log10_lr_score(self, X_trace: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
        """Predict natural log LR scores (ln_LR scores) using the fitted model.

        Predict ln_LR scores, making use of the parameters constructed during `self.fit()` (which should
                now be stored in `self`).

        X_trace measurements of trace object. np.ndarray of shape (instances, repetitions_trace, features)
        X_ref measurements of reference object. np.ndarray of shape (instances, repetitions_ref, features)
        returns: log10_LR_scores according to the two_level_model in Bolck et al. np.ndarray of shape (instances,)
        """
        if not self.model_fitted:
            raise ValueError('The model is not fitted; fit it before you use it for predicting')
        elif self._get_n_features(X_trace) != self.n_features_train:
            raise ValueError(
                'The number of features in the training data is different from the number of features in the trace'
            )
        elif self._get_n_features(X_ref) != self.n_features_train:
            raise ValueError(
                'The number of features in the training data is different from the number of features in the reference'
            )

        log10lr = []

        # TODO use matrix multiplication instead of for-loop (means that all functions called here should be adjusted)
        for x_trace, x_ref in zip(X_trace, X_ref, strict=True):
            x_trace = x_trace[~np.isnan(x_trace).any(axis=1)]
            x_ref = x_ref[~np.isnan(x_ref).any(axis=1)]

            (
                covars_trace,
                covars_trace_update,
                covars_ref,
                covars_trace_inv,
                covars_trace_update_inv,
                covars_ref_inv,
            ) = self._predict_covariances_trace_ref(x_trace, x_ref)
            updated_ref_mean = self._predict_updated_ref_mean(x_ref, covars_ref_inv)
            ln_num = self._predict_ln_num(
                x_trace,
                x_ref,
                covars_ref_inv,
                covars_trace_update_inv,
                updated_ref_mean,
            )
            ln_den_left = self._predict_ln_den_term(x_ref, covars_ref_inv)
            ln_den_right = self._predict_ln_den_term(x_trace, covars_trace_inv)

            log10lr.append(
                self._predict_log10_LR_from_formula_Bolck(
                    covars_trace, covars_trace_update, ln_num, ln_den_left, ln_den_right
                )
            )

        return np.array(log10lr)

    @staticmethod
    def _get_n_features(X: np.ndarray, feature_ix: int = 2) -> int:
        """Calculate the number of features in `X`."""
        return X.shape[feature_ix]

    @staticmethod
    def _get_n_sources(y: np.ndarray) -> int:
        """Calculate the number of sources in `y`.

        Y np 1d-array of labels. labels from {1, ..., n} with n the number of sources. Repetitions get the same label.
        returns: number of sources in y (int).
        """
        return len(np.unique(y))

    @staticmethod
    def _get_mean_covariance_within(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate a matrix of mean covariances within each of the sources.

        X np.array of measurements, rows are sources/repetitions, columns are features
        y np 1d-array of labels. labels from {1, ..., n} with n the number of sources. Repetitions get the same label.
        returns: mean within covariance matrix, np.array.

        This function calculates a matrix of mean covariances within each of the sources, it does so by grouping the
        data per source, calculating the covariance matrices per source and then taking the mean per feature.
        """
        # Get unique sources
        unique_sources = np.unique(y)

        # Collect covariance matrices for sources with multiple repetitions
        covariance_matrices = []

        for source in unique_sources:
            # Get measurements for this source
            source_measurements = X[y == source]

            # Only include sources with more than one repetition
            if len(source_measurements) > 1:
                # Calculate covariance matrix for this source
                cov_matrix = np.cov(source_measurements, rowvar=False, ddof=1)
                # Ensure cov_matrix is always 2D (handle single feature case)
                cov_matrix = np.atleast_2d(cov_matrix)
                covariance_matrices.append(cov_matrix)

        # Calculate mean covariance matrix across all sources
        if len(covariance_matrices) == 0:
            # If no sources have multiple repetitions, return zero matrix
            return np.zeros((X.shape[1], X.shape[1]))

        return np.mean(covariance_matrices, axis=0)

    @staticmethod
    def _get_means_per_source(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Provide numpy array of means per source.

        X np.array of measurements, rows are sources/repetitions, columns are features
        y np 1d-array of labels. For each source a unique identifier (label). Repetitions get the same label.
        returns: means per source in a np.array matrix of size: number of sources * number of features.
        """
        # Get unique sources and calculate means for each
        unique_sources = np.unique(y)
        return np.array([np.mean(X[y == source], axis=0) for source in unique_sources])

    @staticmethod
    def _get_kernel_bandwidth_squared(n_sources: int, n_features_train: int) -> float:
        """Calculate kernel bandwidth and return it.

        Reference: 'Density estimation for statistics and data analysis', B.W. Silverman,
            page 86 formula 4.14 with A(K) the second row in the table on page 87.
        """
        # calculate kernel bandwidth and square it, using Silverman's rule for multivariate data
        kernel_bandwidth = (4 / ((n_features_train + 2) * n_sources)) ** (1 / (n_features_train + 4))
        return kernel_bandwidth**2

    @staticmethod
    def _get_between_covariance(X: np.ndarray, y: np.ndarray, mean_within_covars: np.ndarray) -> np.ndarray:
        """Calculate and return the between covariance.

        X np.array of measurements, rows are objects, columns are variables
        y np 1d-array of labels. labels from {1, ..., n} with n the number of objects. Repetitions get the same label.
        returns: estimated covariance of true mean of the features between sources in the population in a np.array
            square matrix with number of features^2 as dimension.
        """
        # Get unique sources and their repetition counts
        unique_sources, counts = np.unique(y, return_counts=True)

        # calculate kappa; kappa represents the "average" number of repetitions per source
        # get the repetitions per source as a column vector
        reps = counts.reshape((-1, 1))
        sum_reps_sq = np.sum(reps**2)
        kappa = ((reps.sum() - sum_reps_sq / reps.sum()) / (len(reps) - 1)).item()

        # calculate sum_of_squares between
        # substitute rows with their corresponding group means
        group_means = np.zeros_like(X)
        for source in unique_sources:
            source_mask = y == source
            source_mean = np.mean(X[source_mask], axis=0)
            group_means[source_mask] = source_mean

        # calculate covariance of measurements
        cov_between_measurement = np.cov(group_means, rowvar=False, ddof=0)
        # Ensure cov_between_measurement is always 2D (handle single feature case)
        cov_between_measurement = np.atleast_2d(cov_between_measurement)
        sum_squares_between = cov_between_measurement * len(group_means)

        # calculate between covariance matrix
        # Kappa converts within variance at measurement level to within variance at mean of source level and
        #   scales the SSQ_between to a mean between variance

        return (sum_squares_between / (len(reps) - 1) - mean_within_covars) / kappa

    def _predict_covariances_trace_ref(
        self, X_trace: np.ndarray, X_ref: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate and return covariances of trace references.

        X_tr np.array of measurements of trace object, rows are repetitions, columns are features
        X_ref np.array of measurements of reference object, rows are repetitions, columns features
        returns: covariance matrices of the trace and reference data and their respective inverses needed for
        LR calculation;
            covars_trace is the covariance matrix for the trace data given a KDE background mean (U_h0),
            covars_trace_update is the covariance matrix for the trace mean with a bayesian update of reference mean
            given a KDE background mean (U_hn),
            covars_ref is the covariance matrix for the reference data given a KDE background mean (U_hx),
            covars_trace_inv is the inverse of covars_trace,
            covars_trace_update_inv is the inverse of covars_trace_update,
            covars_ref_inv is the inverse of covars_ref.
        """
        kernel_bandwidth_sq = check_not_none(self.kernel_bandwidth_sq)
        between_covars = check_not_none(self.between_covars)
        mean_within_covars = check_not_none(self.mean_within_covars)

        # Number of trace and reference measurements
        n_trace = len(X_trace)
        n_reference = len(X_ref)
        # Calculate covariance matrix for the trace data, given the training data (U_h0)
        covars_trace = kernel_bandwidth_sq * between_covars + mean_within_covars / n_trace
        # Calculate covariance matrix for the reference data, given the training data (U_hx)
        covars_ref = kernel_bandwidth_sq * between_covars + mean_within_covars / n_reference
        # take the inverses
        covars_trace_inv = np.linalg.inv(covars_trace)
        covars_ref_inv = np.linalg.inv(covars_ref)
        # Calculate T_hn
        T_hn = kernel_bandwidth_sq * between_covars - np.matmul(
            np.matmul((kernel_bandwidth_sq * between_covars), covars_ref_inv),
            (kernel_bandwidth_sq * between_covars),
        )
        # Calculate covariance matrix for the trace data, given the training data and with a Bayesian update with
        #   the reference data under Hp (U_hn)
        covars_trace_update = T_hn + mean_within_covars / n_trace
        covars_trace_update_inv = np.linalg.inv(covars_trace_update)

        # TODO covars_trace redundant to return?
        return (
            covars_trace,
            covars_trace_update,
            covars_ref,
            covars_trace_inv,
            covars_trace_update_inv,
            covars_ref_inv,
        )

    def _predict_updated_ref_mean(self, X_ref: np.ndarray, covars_ref_inv: np.ndarray) -> np.ndarray:
        """Calculate and return bayesian update of reference mean given KDE background means.

        X_ref np.array of measurements of reference object, rows are repetitions, columns features
        returns: updated_ref_mean, bayesian update of reference mean given KDE background means.
        """
        kernel_bandwidth_sq = check_not_none(self.kernel_bandwidth_sq)
        between_covars = check_not_none(self.between_covars)
        mean_within_covars = check_not_none(self.mean_within_covars)
        means_per_source = check_not_none(self.means_per_source)

        n_reference = len(X_ref)
        mean_X_reference = np.mean(X_ref, axis=0)

        # calculate the two terms for mu_h and add, see Bolck et al
        mu_h_1 = np.matmul(
            np.matmul(kernel_bandwidth_sq * between_covars, covars_ref_inv),
            mean_X_reference,
        ).reshape(-1, 1)
        mu_h_2 = np.matmul(
            np.matmul(mean_within_covars / n_reference, covars_ref_inv),
            means_per_source.transpose(),
        )

        return (mu_h_1 + mu_h_2).transpose()

    def _predict_ln_num(
        self,
        X_trace: np.ndarray,
        X_ref: np.ndarray,
        covars_ref_inv: np.ndarray,
        covars_trace_update_inv: np.ndarray,
        updated_ref_mean: np.ndarray,
    ) -> np.floating:
        """Perform calculation to predict natural log of numerator.

        See Bolck et al. formula in appendix. The formula consists of three sum_terms (and some other terms).
        The numerator sum term is calculated here.
        The numerator is based on the product of two Gaussian PDFs.
        The first PDF: ref_mean ~ N(background_mean, U_hx).
        The second PDF: trace_mean ~ N(updated_ref_mean, U_hn).
        In this function log of the PDF is taken (so the exponentiation is left out and the product becomes a sum).

        X_trace np.array of measurements of trace object, rows are repetitions, columns are variables
        X_ref np.array of measurements of reference object, rows are repetitions, columns are variables
        covars_ref_inv, covars_trace_update_inv, np.arrays as calculated by _predict_covariances_trace_ref
        updated_ref_mean np.array with same dimensions as X, calculated by _predict_updated_ref_mean
        returns: ln_num1, natural log of numerator of the LR-formula in Bolck et al.
        """
        means_per_source = check_not_none(self.means_per_source)
        mean_X_trace = np.mean(X_trace, axis=0).reshape(1, -1)
        mean_X_reference = np.mean(X_ref, axis=0).reshape(1, -1)

        # calculate difference vectors (in matrix form)
        dif_trace = mean_X_trace - updated_ref_mean
        dif_ref = mean_X_reference - means_per_source

        # calculate matrix products and sums
        ln_num_terms = -0.5 * np.sum(np.matmul(dif_trace, covars_trace_update_inv) * dif_trace, axis=1) + -0.5 * np.sum(
            np.matmul(dif_ref, covars_ref_inv) * dif_ref, axis=1
        )

        # exponentiate, sum and take log again
        return logsumexp(ln_num_terms)

    def _predict_ln_den_term(self, X_ref_or_trace: np.ndarray, covars_inv: np.ndarray) -> np.floating:
        """Perform calculation and return natural log of a denominator term of the LR-formula.

        See Bolck et al. formula in appendix. The formula consists of three sum_terms (and some other terms).
        A denominator sum term is calculated here.

        X_ref_or_trace np.array of measurements of reference or trace object, rows are repetitions, columns are features
        U_inv, np.array with respective covariance matrix as calculated by _predict_covariances_trace_ref
        returns: ln_den, natural log of a denominator term of the LR-formula in Bolck et al.
        """
        means_per_source = check_not_none(self.means_per_source)
        # calculate mean of reference or trace measurements and difference vectors (in matrix form)
        mean_X_ref_or_trace = np.mean(X_ref_or_trace, axis=0).reshape(1, -1)
        dif_ref = mean_X_ref_or_trace - means_per_source

        # calculate matrix products and sums
        ln_den_terms = -0.5 * np.sum(np.matmul(dif_ref, covars_inv) * dif_ref, axis=1)

        # exponentiate, sum and take log again
        return logsumexp(ln_den_terms)

    def _predict_log10_LR_from_formula_Bolck(
        self,
        covars_trace: np.ndarray,
        covars_trace_update: np.ndarray,
        ln_num: np.floating,
        ln_den_left: np.floating,
        ln_den_right: np.floating,
    ) -> np.floating:
        """Predict 10-base logarithm LR's from the Bolck formula.

        X_trace np.array of measurements of trace object, rows are repetitions, columns are variables
        covars_trace, covars_trace_update, np.arrays as calculated by _predict_covariances_trace_ref
        ln_num, ln_den_left, ln_den_right: terms in big fraction in Bolck et al, as calculated by _predict_ln_num
            and _predict_ln_den_term
        returns: log10_LR_score, 10log of LR according to the LR-formula in Bolck et al.
        """
        assert self.n_sources is not None
        # calculate ln LR_score and change base to 10log
        ln_LR_score = (
            np.log(self.n_sources)
            - 0.5 * np.log(np.linalg.det(covars_trace_update))
            + 0.5 * np.log(np.linalg.det(covars_trace))
            + ln_num
            - ln_den_left
            - ln_den_right
        )

        return ln_LR_score / np.log(10)


def _split_pairs(pairs: np.ndarray, n_trace: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the input array along the second dimension at position `n_trace`.

    :param pairs: a feature array of dimension: (pairs, instances, features)
    :param n_trace: the number of trace instances
    :return: a tuple of two arrays of trace features and reference features, both of dimension:
        (pairs, instances, features)
    """
    trace_features = pairs[:, :n_trace, :]
    ref_features = pairs[:, n_trace:, :]
    return trace_features, ref_features


class TwoLevelSystem(LRSystem):
    """Implement two level model, common-source feature-based LR system architecture.

    During the training phase, the system calculates statistics on the unpaired instances. On application, it
    calculates LRs for same-source and different-source pairs. Each side of the pair may consist of multiple instances.

    See also: `TwoLevelModelNormalKDE`
    """

    def __init__(
        self,
        preprocessing_pipeline: Transformer | None,
        pairing_function: PairingMethod,
        postprocessing_pipeline: Transformer | None,
        n_trace_instances: int,
        n_ref_instances: int,
    ):
        """Initialize a new TwoLevelSystem instance.

        :param preprocessing_pipeline: a preprocessing pipeline that is applied on unpaired instances
        :param pairing_function: a function to generate same-source and different-source pairs
        :param postprocessing_pipeline: a postprocessing pipeline that is applied *after* applying the two level model;
            it takes LLRs as input.
        """
        self.preprocessing_pipeline = preprocessing_pipeline or Pipeline([])
        self.pairing_function = pairing_function
        self.postprocessing_pipeline = postprocessing_pipeline or Pipeline([])
        self.model = TwoLevelModelNormalKDE()
        self.n_trace_instances = n_trace_instances
        self.n_ref_instances = n_ref_instances

    def fit(self, instances: InstanceData) -> Self:
        """Fit the model based on the instance data."""
        instances = self.preprocessing_pipeline.fit_apply(instances)
        instances = check_type(FeatureData, instances, 'preprocessing pipeline should return FeatureData')
        self.model.fit_on_unpaired_instances(instances.features, instances.source_ids_1d)

        pairs = self.pairing_function.pair(instances, self.n_trace_instances, self.n_ref_instances)
        pair_llrs = pairs.replace_as(LLRData, features=self.model.transform(pairs.features_trace, pairs.features_ref))
        self.postprocessing_pipeline.fit(pair_llrs)

        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """Apply this LR system on a set of instances and return LLR data.

        Applies the two level LR system on a set of instances,
        and returns a representation of the calculated LLR data through the `LLRData` tuple.
        """
        instances = self.preprocessing_pipeline.apply(instances)

        pairs = self.pairing_function.pair(instances, self.n_trace_instances, self.n_ref_instances)
        pair_llrs = pairs.replace_as(LLRData, features=self.model.transform(pairs.features_trace, pairs.features_ref))
        pair_llrs = self.postprocessing_pipeline.apply(pair_llrs)

        return pair_llrs.replace_as(LLRData)
