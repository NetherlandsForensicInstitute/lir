"""Build the GUI assets that are embedded in the documentation.

The GUI is a React application that renders the JSON schema of the experiment setup as a form. This
script generates the schema, builds the GUI to static assets, and copies the assets to the
documentation source directory, where Sphinx copies them to the documentation output.
"""

import json
import shutil
import subprocess
from pathlib import Path

from lir.config.schema import generate_schema


GUI_DIR = Path(__file__).resolve().parent.parent / 'gui'
OUTPUT_DIR = Path(__file__).resolve().parent / 'gui_static'


def main() -> None:
    npm = shutil.which('npm')
    if npm is None:
        raise RuntimeError('npm is required to build the GUI; install Node.js and npm')

    # generate the JSON schema that the form is based on; strict mode is required, because the lenient schema
    # expresses optional fields as an any-of-null alternative, which form renderers display as a choice menu
    (GUI_DIR / 'lir.schema.json').write_text(json.dumps(generate_schema(strict=True), indent=2) + '\n')

    # install the dependencies if needed, then build the GUI to static assets
    if not (GUI_DIR / 'node_modules').is_dir():
        subprocess.run([npm, 'ci', '--no-audit', '--no-fund'], cwd=GUI_DIR, check=True)  # noqa: S603
    subprocess.run([npm, 'run', 'build'], cwd=GUI_DIR, check=True)  # noqa: S603

    # copy the built assets to the documentation source directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    shutil.copytree(GUI_DIR / 'dist', OUTPUT_DIR, ignore=shutil.ignore_patterns('index.html'))


if __name__ == '__main__':
    main()
