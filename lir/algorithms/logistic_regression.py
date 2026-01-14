import logging
from collections.abc import Callable
from functools import partial
from typing import Self

import numpy as np
import sklearn
from scipy.optimize import minimize

from lir import Transformer
from lir.data.models import FeatureData, InstanceData, LLRData
from lir.util import (
    Bind,
    check_misleading_finite,
    check_type,
    ln_to_log10,
    probability_to_logodds,
)


LOG = logging.getLogger(__name__)


class LogitCalibrator(Transformer):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses logistic regression for interpolation.

    Infinite values in the input are ignored, except if they are misleading, which is an error.
    """

    def __init__(self, **kwargs: dict):
        self._logit = sklearn.linear_model.LogisticRegression(class_weight='balanced', **kwargs)

    def fit(self, instances: InstanceData) -> Self:
        instances = check_type(FeatureData, instances)
        if not isinstance(FeatureData, LLRData):
            instances = instances.replace_as(LLRData)

        check_misleading_finite(instances.llrs, instances.require_labels)

        # if data is sane, remove Inf under H1 and minInf under H2 from the data if present (if present, these prevent
        # logistic regression to train while the loss is zero, so they can be safely removed)
        el = np.isfinite(instances.llrs)
        finite_instances = instances[el]

        # train logistic regression
        self._logit.fit(finite_instances.llrs.reshape(-1, 1), finite_instances.require_labels)
        return self

    def apply(self, instances: InstanceData) -> LLRData:
        instances = check_type(FeatureData, instances)
        if not isinstance(FeatureData, LLRData):
            instances = instances.replace_as(LLRData)

        # initialize llrs_output
        llrs_output = np.empty(len(instances))

        # copy infinite LLRs directly
        llrs_output[np.isposinf(instances.llrs)] = np.inf
        llrs_output[np.isneginf(instances.llrs)] = -np.inf

        # calibrate finite LLRs
        finite_el = np.isfinite(instances.llrs)
        lnlrs = self._logit.intercept_[0] + self._logit.coef_[0][0] * instances.llrs[finite_el]
        llrs_output[finite_el] = ln_to_log10(lnlrs)

        # build and return calibrated LLRData, reset bounds
        return instances.replace(features=llrs_output.reshape(-1, 1), llr_lower_bound=None, llr_upper_bound=None)


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


class FourParameterLogisticCalibrator(Transformer):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of two distributions.
    Depending on the training data, a 2-, 3- or 4-parameter logistic model is used.
    """

    def __int__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.model: Callable

    def fit(self, instances: InstanceData) -> Self:
        instances = check_type(FeatureData, instances)
        if not isinstance(FeatureData, LLRData):
            instances = instances.replace_as(LLRData)

        # check for negative inf for '1'-labels or inf for '0'-labels
        estimate_c = np.any(np.isneginf(instances.llrs[instances.require_labels == 1]))
        estimate_d = np.any(np.isposinf(instances.llrs[instances.require_labels == 0]))

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
            instances.llrs,
            instances.require_labels,
            self.model,
        )

        result = minimize(objective_function, np.array([0.1] * (2 + estimate_d + estimate_c)), bounds=bounds)  # type: ignore
        if not result.success:
            raise Exception('The optimizer did not converge for the calibrator, please check your data.')
        self.coef_ = result.x

        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """
        Returns the odds ratio.
        """
        if self.coef_ is None:
            raise ValueError('trying to use a model before fitting')

        instances = check_type(FeatureData, instances)
        if not isinstance(FeatureData, LLRData):
            instances = instances.replace_as(LLRData)

        # build and return calibrated LLRData, reset bounds
        llrs = probability_to_logodds(self.model(instances.llrs, *self.coef_))
        return instances.replace(features=llrs.reshape(-1, 1), llr_lower_bound=None, llr_upper_bound=None)

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
