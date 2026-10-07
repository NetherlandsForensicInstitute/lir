import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import softplus
from scipy.stats import norm

from lir import registry
from lir.algorithms.bigaussian import BiGaussianCalibrator, cllr_to_sigma2
from lir.config.base import GenericConfigParser
from lir.data.models import FeatureData, LLRData
from lir.metrics import cllr
from lir.util import Xn_to_Xy, ln_to_log10


# Training scores and expected outputs from the reference implementation by Morrison/Weber:
# https://forensic-data-science.net/calibration-and-validation/#biGauss
# (demo_biGaussianized_calibration - Python - 2025-07-10a). Scores are natural-log likelihood ratios; the
# BiGaussianCalibrator in lir works with log10-likelihood-ratios, so these scores are converted before use.
LNS = np.array(
    [
        -0.9285818039163802,
        3.157012081865504,
        4.814805629767377,
        3.3738391491909074,
        5.1157318021881295,
        10.135606665233128,
        -0.6223626533355994,
        5.315926353473733,
        -1.0811817127877128,
        3.8419988264056353,
        1.7447743903529762,
        6.242533986326018,
        0.8927524898939588,
        4.272493262441093,
        -0.09895655928413191,
        -2.2887888614134035,
        4.064876351344343,
        7.245674410483116,
        4.27406328057009,
        0.5452051676303371,
        3.6573096487121455,
        4.879186559748752,
        5.918008610478444,
        0.16493409933972802,
        4.705529236789725,
        3.8607107341291687,
        2.9205995866524592,
        3.032288142845038,
        1.336184166446037,
        1.480013163504828,
        6.2607955689302885,
        3.6340222820616717,
        1.8206100837004562,
        -3.103613111195176,
        0.8528660265202579,
        1.7617605773376606,
        -0.1430117404599367,
        -0.2586258315198957,
        5.0232603630516115,
        2.39444935708336,
    ]
)
LND = np.array(
    [
        -7.16341269720547,
        -6.2704432505710015,
        -3.7339623865495533,
        -0.2692919369115927,
        -6.689933018121249,
        0.8949632187679339,
        -4.1936980216994675,
        -7.191327256129153,
        -1.743190782338439,
        0.5201419130398303,
        -3.5433068904083886,
        -1.4107306343228576,
        -3.778671220507038,
        -3.026891123462282,
        1.0794206436470335,
        -0.8057803175621733,
        -5.945759275482066,
        3.8576579010768794,
        -5.501409732235571,
        -0.9225706105596722,
        -1.7799517678263146,
        -3.206838928004752,
        -2.503994675328378,
        -3.4012405727532804,
        -0.9496308271801519,
        -4.745113651568756,
        -5.876068060629845,
        -1.8358360312553386,
        1.2557307583130828,
        -3.334707249990113,
        1.172256864590598,
        -3.2160899333066593,
        0.8144486809810283,
        -0.6401373690995745,
        -1.7195876407142172,
        -0.704459734081945,
        -5.05000075615864,
        -2.7597832219696743,
        -6.847054410082569,
        -7.359792628018654,
    ]
)
TEST_LNLRS = np.array([-6.0, -3.0, -1.5, 0.0, 1.5, 3.0, 6.0, 9.0, -12.0, 20.0])
GOLDEN_LNLRs = np.array(
    [
        -3.974295849574966,
        -2.0526365911133495,
        -1.2525988415975648,
        0.03815412031732535,
        1.230665166294378,
        1.817453294302931,
        4.196910503791019,
        5.173822290250453,
        -5.979559987743823,
        5.350416863939876,
    ]
)
GOLDEN_SIGMA2 = 4.037554321673164


def make_training_data() -> LLRData:
    scores, labels = Xn_to_Xy(LND, LNS)
    return LLRData(features=ln_to_log10(scores).reshape(-1, 1), hypothesis=labels)


def make_test_data(lnlrs: np.ndarray) -> LLRData:
    return LLRData(features=ln_to_log10(lnlrs).reshape(-1, 1), hypothesis=np.zeros(len(lnlrs)))


def cllr_of_sigma2(sigma2: float) -> float:
    """Cllr of a perfectly calibrated bi-Gaussian system with the given variance, by numerical integration."""
    half, sd = sigma2 / 2, np.sqrt(sigma2)
    # -0.5 * [E_H1 log2(expit(x)) + E_H0 log2(1 - expit(x))] == E_{x~N(half, sd)}[softplus(-x)] / ln(2)
    integrand = lambda x: softplus(-x) * norm.pdf(x, half, sd)  # noqa: E731
    value, _ = quad(integrand, -np.inf, np.inf, limit=400)
    return float(value / np.log(2))


def test_bigaussian_matches_reference():
    cal = BiGaussianCalibrator()
    cal.fit(make_training_data())
    result = cal.apply(make_test_data(TEST_LNLRS))

    assert cal.sigma2 == pytest.approx(GOLDEN_SIGMA2, rel=1e-5)
    np.testing.assert_allclose(result.llrs, ln_to_log10(GOLDEN_LNLRs), rtol=1e-5, atol=1e-6)


def test_bigaussian_perfectly_calibrated_data_stays_calibrated():
    rng = np.random.default_rng(42)
    sigma2, n = 6.0, 1500
    lns = rng.normal(sigma2 / 2, np.sqrt(sigma2), n)
    lnd = rng.normal(-sigma2 / 2, np.sqrt(sigma2), n)
    ln_test = np.concatenate(
        (rng.normal(sigma2 / 2, np.sqrt(sigma2), 500), rng.normal(-sigma2 / 2, np.sqrt(sigma2), 500))
    )
    labels_test = np.concatenate((np.ones(500), np.zeros(500)))

    training_scores, training_labels = Xn_to_Xy(lnd, lns)
    cal = BiGaussianCalibrator()
    cal.fit(LLRData(features=ln_to_log10(training_scores).reshape(-1, 1), hypothesis=training_labels))
    test_data = LLRData(features=ln_to_log10(ln_test).reshape(-1, 1), hypothesis=labels_test)
    result = cal.apply(test_data)

    # a perfectly calibrated system should be (approximately) calibrated to itself: the calibrated LLRs should
    # closely resemble the input LLRs and the Cllr should barely change
    np.testing.assert_allclose(result.llrs, ln_to_log10(ln_test), atol=0.25)
    assert cllr(result) == pytest.approx(cllr(test_data), abs=0.05)


def test_cllr_to_sigma2_consistent_with_definition():
    for cllr_value in (0.1, 0.3, 0.5, 0.7, 0.9):
        sigma2 = cllr_to_sigma2(cllr_value)
        # the empirical fit in Morrison (2024) approximates the true relationship; check it is inverted consistently
        assert cllr_of_sigma2(sigma2) == pytest.approx(cllr_value, rel=0.03)


def test_cllr_to_sigma2_monotonic():
    sigma2s = [cllr_to_sigma2(c) for c in (0.9, 0.5, 0.1)]
    assert sigma2s[0] < sigma2s[1] < sigma2s[2]


def test_target_cllr_is_used_directly():
    cal = BiGaussianCalibrator(target_cllr=0.3)
    cal.fit(make_training_data())
    assert cal.sigma2 == pytest.approx(cllr_to_sigma2(0.3))


def test_monotonic_mapping():
    cal = BiGaussianCalibrator()
    cal.fit(make_training_data())
    grid = np.linspace(-7.0, 10.0, 200)  # inside the training range; outside it, scores are clamped to the extremes
    result = cal.apply(make_test_data(grid))
    assert np.all(np.diff(result.llrs) > 0)


def test_infinite_training_scores_are_point_masses():
    scores = np.concatenate(([-np.inf] * 5, LND, LNS, [np.inf] * 5))
    labels = np.concatenate((np.zeros(len(LND) + 5), np.ones(len(LNS) + 5)))
    cal = BiGaussianCalibrator()
    cal.fit(LLRData(features=ln_to_log10(scores).reshape(-1, 1), hypothesis=labels))

    test_values = np.array([-np.inf, -5.0, 0.0, 5.0, np.inf])
    result = cal.apply(make_test_data(test_values))
    assert np.all(np.isfinite(result.llrs))
    assert np.all(np.diff(result.llrs) > 0)


def test_misleading_infinite_training_scores_raise():
    scores = np.concatenate((LND, LNS, [-np.inf]))
    labels = np.concatenate((np.zeros(len(LND)), np.ones(len(LNS) + 1)))
    cal = BiGaussianCalibrator()
    with pytest.raises(ValueError):
        cal.fit(LLRData(features=ln_to_log10(scores).reshape(-1, 1), hypothesis=labels))


@pytest.mark.parametrize('target', [0, 1, -0.5, 1.5])
def test_invalid_target_cllr_raises(target: float):
    with pytest.raises(ValueError):
        BiGaussianCalibrator(target_cllr=target)


def test_multidimensional_input_raises_clear_error():
    features = np.random.default_rng(0).normal(size=(20, 3))
    labels = np.tile(np.array([0, 1]), 10)
    cal = BiGaussianCalibrator()
    with pytest.raises(ValueError, match='single score'):
        cal.fit(FeatureData(features=features, hypothesis=labels))


def test_apply_before_fit_raises():
    cal = BiGaussianCalibrator()
    with pytest.raises(ValueError):
        cal.apply(make_test_data(TEST_LNLRS))


def test_registry_resolves_bigaussian_calibrator():
    parser = registry.get('bigaussian_calibrator', default_config_parser=GenericConfigParser, search_path=['modules'])
    assert isinstance(parser, GenericConfigParser)
    assert parser.component_class is BiGaussianCalibrator
