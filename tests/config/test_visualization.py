import yaml
import pytest

from lir.config.base import ContextAwareDict, ContextAwareList, pop_field
from lir.config.visualization import parse_visualizations


def _yaml_to_context_aware_config(config_dict: dict) -> ContextAwareDict:
    """Helper function to produce a ContextAwareDict from a regular dict."""
    config_key, config_values = next(iter(config_dict.items()))

    context_dict = ContextAwareDict(context=['experiments', 'model_selection', 'visualization'])
    context_list = ContextAwareList(context=['experiments', 'model_selection', 'visualization'])

    context_list.extend(config_values)

    context_dict.update({
        config_key: context_list
    })

    return context_dict


@pytest.mark.parametrize('yaml_config', [
"""
visualization:
    - pav:
        h1_color: green
        h2_color: blue
    - ece
""",
"""
visualization:
    - pav:
        - h1_color: green
        - h2_color: blue
    - ece
"""
])
def test_parse_extra_visualization_options(yaml_config: str):
    """Check that extra provided visualization options are handled correctly."""
    config = yaml.safe_load(yaml_config)
    output_path = 'tests/output/config'  # not used in actual test

    # Given that we provide the appropriate configuration
    context_aware_config = _yaml_to_context_aware_config(config)

    # When we collect the parse functions for our visualizations
    visualization_functions = parse_visualizations(
        pop_field(context_aware_config, 'visualization'),
        output_path=output_path,
    )

    assert len(visualization_functions) == 2

    pav_plot_function = visualization_functions[0]
    assert pav_plot_function.func.__name__ == 'pav'

    ece_function = visualization_functions[1]
    assert ece_function.func.__name__ == 'ece'


    # We expected both keyword arguments available
    assert pav_plot_function.keywords['h1_color'] == 'green'
    assert pav_plot_function.keywords['h2_color'] == 'blue'

    # No keywords have been specified for the 'ece' function
    assert len(ece_function.keywords) == 0