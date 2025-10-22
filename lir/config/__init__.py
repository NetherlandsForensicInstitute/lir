# This import is used to be able to write a short-hand version in `registry.yaml`, i.e.
# `lir.config.numpy_csv_writer`. Ignored by linting (F401; unused import).
from .transform import (  # noqa: F401
    NumpyCsvWriterWrappingConfigParser as numpy_csv_writer,
)
