from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_notebook():
    notebook_path = Path(__file__).parent.parent / "practitioners_guide_glass.ipynb"
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(notebook)
