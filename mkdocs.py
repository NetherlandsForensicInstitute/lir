import inspect
from pathlib import Path
from typing import Any

import lir.registry


def define_env(env):
    """
    This is the hook for defining variables, macros and filters

    - variables: the dictionary that contains the environment variables
    - macro: a decorator function, to declare a macro.
    - filter: a function with one of more arguments,
        used to perform a transformation

    See also: https://mkdocs-macros-plugin.readthedocs.io/en/latest/macros/
    """

    env.variables['registry'] = lir.registry

    @env.macro
    def get_sourcefile(obj: Any) -> Path:
        root = Path(__file__).parent
        sourcefile = Path(inspect.getfile(obj.__class__))
        return sourcefile.relative_to(root)
