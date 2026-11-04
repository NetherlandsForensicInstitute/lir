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

Running the GUI
---------------

The GUI lives in the ``gui/`` folder of the repository. Running it requires a development installation of LiR
(see `CONTRIBUTING.md`_) and Node.js with npm.

1. Generate the JSON schema that the form is based on:

   .. code-block:: bash

      pdm lir --generate-schema > gui/lir.schema.json

2. Install the dependencies of the GUI and start its development server:

   .. code-block:: bash

      cd gui
      npm install
      npm run dev

3. Open the URL shown in the terminal (by default ``http://localhost:5173``) in a browser.

.. _CONTRIBUTING.md: https://github.com/NetherlandsForensicInstitute/lir/blob/main/CONTRIBUTING.md
