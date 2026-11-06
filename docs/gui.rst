Graphical user interface (GUI)
==============================

.. warning::

   The GUI is experimental: it is provided as-is, and its interface and behaviour may change or be removed in
   future releases.

The GUI is a web-based form for creating and exploring LiR experiment configurations. The form is generated from
the JSON schema that describes the experiment setup (the same schema that is used to validate YAML
configurations), so all registry options are available as form fields, including the descriptions that explain
their use.

The GUI is a development aid: it does not run experiments. Submitted form values are written to the browser
console, where they can be inspected and converted into a YAML configuration file.

The form
--------

.. raw:: html

   <div id="lir-gui"></div>
   <script type="module" src="assets/gui.js"></script>

The source code of the GUI is available in the ``gui/`` folder of the repository.
