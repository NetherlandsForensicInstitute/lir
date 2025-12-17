import pytest

from lir.config.base import GenericConfigParser, YamlParseError
from lir.config.experiment_strategies import parse_experiments_setup
from lir import registry


def test_parse_external_directly():
    """Test that an external module can be used directly in the configuration.

    In this case, the external module is `tests.resources.external_modules.ExampleExternalData`,
    which points to the `ExampleExternalData` class defined in tests/resources/external_modules.py.

    Two tests are performed: one on a DataStrategy and one on a DataSet.
    """
    try:
        registry.get('tests.resources.external_modules.ExampleExternalData',            
                    search_path=['data_strategies'],
                    default_config_parser=GenericConfigParser)
    except registry.ComponentNotFoundError:
        pytest.fail("Failed to parse external DataStrategy module.")
    
    try:
        registry.get('tests.resources.external_modules.ExampleExternalData',            
                     search_path=['data_sets'],
                    default_config_parser=GenericConfigParser)
    except registry.ComponentNotFoundError:
        pytest.fail("Failed to parse external DataSet module.")