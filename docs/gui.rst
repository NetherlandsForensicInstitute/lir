Graphical user interface (GUI)
==============================

.. warning::

   The GUI is experimental: it is provided as-is, and its interface and behaviour may change or be removed in
   future releases.

The GUI is a web-based form for creating and exploring LiR experiment configurations. The form is generated from
the JSON schema that describes the experiment setup (the same schema that is used to validate YAML
configurations), so all registry options are available as form fields, including the descriptions that explain
their use.

The GUI is a development aid: it does not run experiments.

The form
--------

The form starts with a standard minimal configuration, so the data and LR system sections are visible from the
start. Changing the discriminators (for example, the experiment ``strategy``, the data ``setup``, the provider
``method``, or the LR system ``architecture``) shows the fields that apply to the selected option. Submitting the
form displays the resulting configuration below the form; the output is JSON, which is also valid YAML and can be
saved as a ``.yaml`` file.

.. raw:: html

   <div id="lir-gui"></div>
   <script type="module" src="assets/gui.js"></script>

The source code of the GUI is available in the ``gui/`` folder of the repository.
