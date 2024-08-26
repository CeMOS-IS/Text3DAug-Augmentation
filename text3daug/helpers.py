import contextlib
import os

import numpy as np


def expand_array_2(x):
    if x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x


@contextlib.contextmanager
def suppress_print():
    """Supresses the print return of a function."""
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        yield
