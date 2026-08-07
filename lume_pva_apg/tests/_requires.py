"""Import guards for tests that need an optional dependency.

The project spans a pure-Python core and four optional dependency sets, so no
single install runs the whole suite. A module-scope ``import`` of an absent
dependency fails *collection*, and a collection error takes down the entire
run -- including the tests that had everything they needed. Guarding the import
turns that into a skip of the one module that could not run.

Only "not installed" is worth skipping for. A dependency that is installed but
fails to load is a broken environment, and skipping past it reports a green run
on a host that cannot serve. That distinction is the package's own, drawn in
:mod:`lume_pva_apg._optional`, so a guarded test module and the module it tests
report an unusable transport the same way.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import NoReturn

import pytest

from lume_pva_apg._optional import missing_extra, module_is_absent

# Which extra provides each optional dependency the suite imports, so a skip
# reason names the extra to install rather than the package that was missing.
EXTRA_FOR_MODULE = {
    "p4p": "pva",
    "pcaspy": "ca",
    "epics": "dev",
    "torch": "torch",
    "lume_torch": "torch",
}


def skip_if_absent(cause: ImportError) -> NoReturn:
    """Skip the calling test module when an optional dependency is not installed.

    Call from the ``except ImportError`` of a module-scope import block::

        try:
            import pcaspy
        except ImportError as exc:
            skip_if_absent(exc)

    Parameters
    ----------
    cause : ImportError
        The failure raised by the guarded import block.

    Raises
    ------
    ImportError :
        Re-raised unchanged if the failure is not an optional dependency of
        this project, and re-raised naming the extra if the dependency is
        installed but fails to load. Neither is a skip: the first is a real
        error, and the second is a broken host rather than an install the suite
        supports.
    """
    module = (cause.name or "").partition(".")[0]
    extra = EXTRA_FOR_MODULE.get(module)
    if extra is None:
        raise cause
    if module_is_absent(module, cause):
        pytest.skip(str(missing_extra(module, extra)), allow_module_level=True)
    raise missing_extra(module, extra, cause) from cause


def optional_module(module: str, extra: str) -> ModuleType | None:
    """Import ``module``, or return ``None`` when it is not installed.

    For a dependency whose absence should thin out a parametrisation rather
    than skip the file -- the caller keeps collecting the cases that do not
    need it.

    Parameters
    ----------
    module : str
        Top-level module to import, e.g. ``"torch"``.
    extra : str
        The extra that provides it, named in the error if it is broken.

    Returns
    -------
    ModuleType | None :
        The module, or None if it is not installed.

    Raises
    ------
    ImportError :
        If the module is installed but fails to load.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if module_is_absent(module, exc):
            return None
        raise missing_extra(module, extra, exc) from exc
