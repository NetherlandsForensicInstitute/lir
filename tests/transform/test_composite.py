import numpy as np
import pytest

from lir.algorithms.logistic_regression import LogitCalibrator
from lir.data.models import FeatureData, LLRData
from lir.metrics import cllr
from lir.transform.composite import CategoricalCompositeTransformer
from lir.util import check_type


def test_composite():
    features = np.concatenate([np.random.default_rng(42).normal(loc=i, scale=1, size=(100, 1)) for i in range(6)])
    labels = np.repeat([0, 1, 0, 1, 0, 1], 100)
    categories = np.repeat([0, 1, 2], 200)
    instances = FeatureData(features=features, labels=labels, categories=categories)
    simple_llrs = LogitCalibrator().fit(instances).apply(instances)

    lrsystem = CategoricalCompositeTransformer(factory=LogitCalibrator, category_field='categories')

    # apply before fit should fail
    with pytest.raises(ValueError):
        lrsystem.apply(instances)

    composite_llrs = check_type(LLRData, lrsystem.fit_apply(instances))

    assert cllr(composite_llrs) < cllr(simple_llrs), 'Cllr from a composite system should be better'
    assert cllr(composite_llrs[categories == 0]) == pytest.approx(cllr(composite_llrs[categories == 1]))
    assert cllr(composite_llrs[categories == 0]) == pytest.approx(cllr(composite_llrs[categories == 2]))

    # apply lrsystem with missing category field
    with pytest.raises(ValueError):
        lrsystem.apply(instances.replace(categories=None))

    # apply lrsystem with bad category shape
    with pytest.raises(ValueError):
        lrsystem.apply(instances.replace(categories=np.array([0, 1, 2])))

    # apply lrsystem with bad category shape
    with pytest.raises(ValueError):
        lrsystem.apply(instances.replace(categories=categories.reshape(-1, 1)))

    # apply lrsystem with bad category field
    with pytest.raises(ValueError):
        lrsystem.apply(instances.replace(categories='bad value'))

    # apply lrsystem with bad category values
    with pytest.raises(ValueError):
        lrsystem.apply(instances.replace(categories=categories + 1))

    # fit lrsystem with missing category field
    with pytest.raises(ValueError):
        lrsystem.fit(instances.replace(categories=None))

    composite_llrs_alt = check_type(
        LLRData, lrsystem.fit_apply(instances.replace(categories=categories.reshape(-1, 1)))
    )
    assert cllr(composite_llrs) == cllr(composite_llrs_alt), 'reshape should have no effect on cllr'

    composite_llrs_alt = check_type(
        LLRData,
        lrsystem.fit_apply(instances.replace(categories=np.stack([categories, np.ones(categories.size)], axis=1))),
    )
    assert cllr(composite_llrs) == cllr(composite_llrs_alt), 'reshape should have no effect on cllr'

    composite_llrs_alt = check_type(
        LLRData, lrsystem.fit_apply(instances.replace(categories=np.vectorize(str)(categories)))
    )
    assert cllr(composite_llrs) == cllr(composite_llrs_alt), 'transformation to str should have no effect on cllr'
