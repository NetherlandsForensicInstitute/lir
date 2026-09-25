from math import log
from typing import Self, Any

import numpy as np

import lir
from lir import Transformer, InstanceData, metrics, LLRData
from lir.util import check_type


class BigaussianCalibrator(Transformer):
    """
    Calibrate data using bi-gaussianized calibration.

    This calibrator uses bi-gaussianized calibration as defined in: Morrison G.S. (2024).
    `Bi-Gaussianized calibration of likelihood ratios`_. Law, Probability & Risk, 23, mgae004.

    Abstract from the article:

        For a perfectly calibrated forensic evaluation system, the likelihood ratio of the likelihood ratio is the
        likelihood ratio. Conversion of uncalibrated log-likelihood ratios (scores) to calibrated log-likelihood ratios
        is often performed using logistic regression. The results, however, may be far from perfectly calibrated. We
        propose and demonstrate a new calibration method, "bi-Gaussianized calibration," that warps scores toward
        perfectly calibrated log-likelihood-ratio distributions. Using both synthetic and real data, we demonstrate that
        bi-Gaussianized calibration leads to better calibration than does logistic regression, that it is robust to
        score distributions that violate the assumption of two Gaussians with the same variance, and that it is
        competitive with logistic-regression calibration in terms of performance measured using log-likelihood-ratio
        cost (Cllr). We also demonstrate advantages of bi-Gaussianized calibration over calibration using pool-adjacent
        violators (PAV). Based on bi-Gaussianized calibration, we also propose a graphical representation that may help
        explain the meaning of likelihood ratios to triers of fact.

    .. Bi-Gaussianized calibration of likelihood ratios: https://doi.org/10.1093/lpr/mgae004
    """
    def __init__(self, calibrator: Transformer):
        self.calibrator = calibrator
        self.target_var: float
        self.h1_mean: np.floating
        self.h1_var: np.floating
        self.h2_mean: np.floating
        self.h2_var: np.floating

    @staticmethod
    def _cllr_to_variance(cllr: float) -> float:
        b = 17.7
        c = 0.00933
        return - log(log(cllr) / b + 1) / c

    def fit(self, instances: InstanceData) -> Self:
        training_llrs = check_type(LLRData, self.calibrator.fit_apply(instances))

        llrs = instances.replace_as(LLRData)
        self.target_var = self._cllr_to_variance(metrics.cllr(training_llrs))
        self.h1_mean = np.mean(llrs.llrs[llrs.hypothesis == 1])
        self.h1_var = np.var(llrs.llrs[llrs.hypothesis == 1])
        self.h2_mean = np.mean(llrs.llrs[llrs.hypothesis == 0])
        self.h2_var = np.var(llrs.llrs[llrs.hypothesis == 0])
        return self

    def _transform(self, llrs: np.ndarray, orig_mean: float, orig_var: float, polarity: int) -> np.ndarray:
        target_mean = polarity * self.target_var / 2
        return (llrs - orig_mean) * self.target_var / orig_var + target_mean

    def apply(self, instances: InstanceData) -> InstanceData:
        llrs = instances.replace_as(LLRData)
        h1_mean = np.mean(llrs.llrs[llrs.hypothesis == 1])
        h1_var = np.var(llrs.llrs[llrs.hypothesis == 1])
        h2_mean = np.mean(llrs.llrs[llrs.hypothesis == 0])
        h2_var = np.var(llrs.llrs[llrs.hypothesis == 0])
        transformed_llrs = llrs.llrs
        transformed_llrs[llrs.hypothesis == 1] = self._transform(transformed_llrs[llrs.hypothesis == 1], h1_mean, h1_var, 1)
        transformed_llrs[llrs.hypothesis == 0] = self._transform(transformed_llrs[llrs.hypothesis == 0], h2_mean, h2_var, -1)
        return llrs.replace(features=transformed_llrs.reshape(-1, 1))
