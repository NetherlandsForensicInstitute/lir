from docs import GetRegistryLink


def test_registry_link():
    assert (
        GetRegistryLink()('lir.config.substitution.parse_categorical')
        == ':class:`lir.config.substitution.parse_categorical <lir.config.substitution.CategoricalHyperparameter>`'
    )
    assert (
        GetRegistryLink()('hyperparameter_types.categorical')
        == ':class:`hyperparameter_types.categorical <lir.config.substitution.CategoricalHyperparameter>`'
    )
