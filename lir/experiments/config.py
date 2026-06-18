from lir.config.base import ContextAwareDict, pop_field


def pop_experiment_name(config: ContextAwareDict) -> str:
    """
    Extract and remove experiment name from configuration.

    Reads the experiment name from the ``name`` attribute. If that attribute is missing, a default name is generated
    from the section context.

    Parameters
    ----------
    config : ContextAwareDict
        Configuration section describing the experiment.

    Returns
    -------
    str
        Experiment name.
    """
    if 'name' in config:
        return pop_field(config, 'name')
    else:
        return f'unnamed_experiment{config.context[-1] if len(config.context) > 0 else ""}'
