from lir.config.base import ConfigValue, pop_field


def pop_experiment_name(config: ConfigValue) -> str:
    """
    Extract and remove experiment name from configuration.

    Reads the experiment name from the ``name`` attribute. If that attribute is missing, a default name is generated
    from the section context.

    Parameters
    ----------
    config : ConfigValue
        Configuration section describing the experiment.

    Returns
    -------
    str
        Experiment name.
    """
    if 'name' in config.check_type(dict):
        return pop_field(config, 'name', validate=str)
    else:
        return f'unnamed_experiment{config.context[-1] if len(config.context) > 0 else ""}'
