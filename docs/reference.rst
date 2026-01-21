Registry reference
==================

Experiment components
---------------------

The preferred way to set up an experiment is to use a YAML configuration.
Use the [LR system selection helper](lrsystem_yaml.md) to learn how to use the YAML interface.

This page lists the components that may be needed to set up an experiment.


.. jinja::

    {% for registry_section, friendly_name in [
        ('experiment_strategies', 'Experiment strategies'),
        ('data_strategies', 'Data strategies'),
        ('data_providers', 'Data providers'),
        ('metric', 'Metrics'),
        ('output', 'Output'),
        ('hyperparameter_types', 'Hyperparameters'),
    ] %}

    {{friendly_name}}
    ^^^^^^^^^^^^^^^^^
    Registry section: ``{{ registry_section }}``

    {% for name in registry %}
        {% if name.startswith(registry_section) %}
    - `{{name}} <{{ registry.get(name) | apidocs_uri }}>`_ {{ registry.get(name) | docstr_short }}
        {% endif %}
    {% endfor %}
    {% endfor %}


LR system components
--------------------

You may choose to use either the Python API or a YAML configuration to set up an LR system.

- Use the `Pracitioner's Guide <https://github.com/NetherlandsForensicInstitute/lir/tree/practitioner_guide>`_ to learn how to use the Python API.
- Use the [LR system selection helper](lrsystem_yaml.md) to learn how to use the YAML interface.

This page lists the components that may be needed to define an LR system.

.. jinja::

    {% for registry_section, friendly_name in [
        ('lrsystem_architecture', 'LR system architecture'),
        ('modules', 'LR system modules'),
        ('pairing', 'Pairing methods'),
    ] %}

    {{friendly_name}}
    ^^^^^^^^^^^^^^^^^

    Registry section: `{{ registry_section }}`

    {% for name in registry %}
        {% if name.startswith(registry_section) %}
    - `{{ name }} <{{ registry.get(name) | apidocs_uri }}>`_ {{ registry.get(name) | docstr_short }}
        {% endif %}
    {% endfor %}
    {% endfor %}
