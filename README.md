Practitioner's Guide - LIR Python Likelihood Ratio Toolkit
===================================

- This branch is dedicated to the Practitioner's Guide Jupyter Notebook, accompanying the paper ["From data to a validated score-based LR system: A practitioner’s guide" - Leegwater et al.](https://doi.org/10.1016/j.forsciint.2024.111994)
- The notebook is also available on [Google Colab](https://colab.research.google.com/github/NetherlandsForensicInstitute/lir/blob/practitioner_guide/practitioners_guide_glass.ipynb)
- The source code of the LiR package can be found on the [main branch](https://github.com/NetherlandsForensicInstitute/lir) of this GitHub repository


Quickstart
----------

This project uses [pdm](https://pdm-project.org/en/latest/) as a dependency manager. For installation of PDM, please consult the
[PDM project website](https://pdm-project.org/en/latest/#installation). It is encouraged to use PDM when contributing to this package.

Get the code and install dependencies:

```shell
git clone -b practitioner_guide git@github.com:NetherlandsForensicInstitute/lir.git
pdm sync
```

Commands available:
- `pdm run notebook`: launch the notebook in a browser
- `pdm run clean`: reset the notebook
- `pdm run test`: run unit tests
