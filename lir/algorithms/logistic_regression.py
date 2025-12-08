import logging
from collections.abc import Callable
from functools import partial

import numpy as np
import sklearn
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin

from lir.util import (
    Bind,
    check_misleading_finite,
    ln_to_log10,
    probability_to_logodds,
)


LOG = logging.getLogger(__name__)


class LogitCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses logistic regression for interpolation.

    Infinite values in the input are ignored, except if they are misleading, which is an error.
    """

    def __init__(self, **kwargs: dict):
        self._logit = sklearn.linear_model.LogisticRegression(class_weight='balanced', **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogitCalibrator':
        # sanity check
        check_misleading_finite(X, y)

        # if data is sane, remove Inf under H1 and minInf under H2 from the data if present (if present, these prevent
        # logistic regression to train while the loss is zero, so they can be safely removed)
        el = np.isfinite(X)
        y = y[el]
        X = X[el]

        # train logistic regression
        X = X.reshape(-1, 1)
        self._logit.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # initiate llrs_output
        llrs_output = np.empty(np.shape(X))
        self.p0 = np.empty(np.shape(X))
        self.p1 = np.empty(np.shape(X))

        # get boundary log_odds values
        zero_elements = np.where(-1 * np.inf == X)
        ones_elements = np.where(np.inf == X)

        # assign desired output for these boundary values to llrs_output
        llrs_output[zero_elements] = np.multiply(-1, np.inf)
        llrs_output[ones_elements] = np.inf

        # get elements with values between negInf and Inf (the boundary values)
        between_elements = np.all(np.array([np.inf != X, -1 * np.inf != X]), axis=0)

        # get LLRs for X[between_elements]
        lnlrs = self._logit.intercept_ + self._logit.coef_ * X[between_elements]
        llrs = ln_to_log10(lnlrs)
        llrs = np.reshape(llrs, np.sum(between_elements))
        llrs_output[between_elements] = llrs

        # calculation of self.p1 and self.p0 is redundant?
        self.p1[zero_elements] = 0
        self.p1[ones_elements] = 1
        self.p1[between_elements] = self._logit.predict_proba(X[between_elements].reshape(-1, 1))[:, 1]
        self.p0 = 1 - self.p1
        return llrs_output


def _negative_log_likelihood_balanced(X: np.ndarray, y: np.ndarray, model: Callable, params: list) -> np.ndarray:
    """
    calculates neg_llh of probabilistic binary classifier.
    The llh is balanced in the sense that the total weight of '1'-labels is equal to the total weight of '0'-labels.

    inputs:
        X: n * 1 np.array of scores
        y: n * 1 np.array of labels (Booleans). H1 --> 1, H2 --> 0.
        model: model that links score to posterior probabilities
        params: mapping of parameter names to values of the model.
    output:
        neg_llh_balanced: float, balanced negative log likelihood (base = exp)
    """

    probs = model(X, *params)
    neg_llh_balanced = -np.sum(np.log(probs**y * (1 - probs) ** (1 - y)) / (y * np.sum(y) + (1 - y) * np.sum(1 - y)))
    return neg_llh_balanced


class FourParameterLogisticCalibrator:
    """
    Calculates a likelihood ratio of a score value, provided it is from one of two distributions.
    Depending on the training data, a 2-, 3- or 4-parameter logistic model is used.
    """

    def __int__(self) -> None:
        self.coef_ = None
        self.model: Callable

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FourParameterLogisticCalibrator':
        # check for negative inf for '1'-labels or inf for '0'-labels
        estimate_c = np.any(np.isneginf(X[y == 1]))
        estimate_d = np.any(np.isposinf(X[y == 0]))

        # define bounds for a and b
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf)]

        if estimate_c and estimate_d:
            # then define 4PL-logistic model
            self.model = self._four_pl_model
            bounds.extend([(10**-10, 1 - 10**-10), (10**-10, np.inf)])
            LOG.debug(
                'There were -Inf lrs for the same source samples and Inf lrs for the different source samples '
                ', therefore a 4pl calibrator was fitted.'
            )
        elif estimate_c:
            # then define 3-PL logistic model. Set 'd' to 0
            self.model = partial(self._four_pl_model, d=0)
            # use very small values since limits result in -inf llh
            bounds.append((10**-10, 1 - 10**-10))
            LOG.debug('There were -Inf lrs for the same source samples, therefore a 3pl calibrator was fitted.')
        elif estimate_d:
            # then define 3-PL logistic model. Set 'c' to 0
            # use bind since 'c' is intermediate variable. In that case partial does not work.
            self.model = Bind(self._four_pl_model, ..., ..., ..., 0, ...)
            # use very small value since limits result in -inf llh
            bounds.append((10**-10, np.inf))
            LOG.debug('There were Inf lrs for the different source samples, therefore a 3pl calibrator was fitted.')
        else:
            # define ordinary logistic model (no regularization, so maximum likelihood estimates)
            self.model = partial(self._four_pl_model, c=0, d=0)
        # define function to minimize
        objective_function = partial(
            _negative_log_likelihood_balanced,
            X,
            y,
            self.model,  # type: ignore
        )

        result = minimize(
            objective_function,
            np.array([0.1] * (2 + estimate_d + estimate_c)),
            bounds=bounds,
        )
        if not result.success:
            raise Exception('The optimizer did not converge for the calibrator, please check your data.')
        assert result.success
        self.coef_ = result.x

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the odds ratio.
        """
        return probability_to_logodds(self.model(X, *self.coef_))  # type: ignore

    @staticmethod
    def _four_pl_model(s: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        """
        inputs:
                s: n * 1 np.array of scores
                a,b,c,d,: floats defining 4PL model.
                    a and b are the familiar logistic parameters.
                    c and d respectively floor and ceil the posterior probability
                        the flour probability is c and the ceiling probability is c + (1 - c)/(1 + d)
        output:
                p: n * 1 np.array. Posterior probabilities of succes given each s (and a,b,c,d)
        """
        p = c + ((1 - c) / (1 + d)) * 1 / (1 + np.exp(-a * s - b))
        return p
