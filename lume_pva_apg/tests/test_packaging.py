"""Tests for what an install of this distribution guarantees.

The core install is pure-Python and carries no EPICS transport, so a consumer
can provision a host without an EPICS toolchain and still import the package,
read its metadata and generate configuration. That guarantee is a property of
what the modules import, not only of what the wheel declares: a convenience
re-export added to ``__init__`` would pull a transport into the core install
while the packaging metadata still said the core was pure.

CI checks the metadata half -- the built wheel is ``py3-none-any`` and a
core-only environment carries no transport. These tests check the import half,
by running a fresh interpreter with the optional dependencies made unimportable
and asserting what the package does with them missing. They also cover the two
answers the package gives for an unusable transport, which differ: an absent
one names the extra to install, a broken one says the install is broken. Told
the wrong one, an operator on a host with a transport that cannot load spends
their time reinstalling a package they already have.

A subprocess per case, because import state is process-global: a module that
has already imported p4p cannot be re-imported without it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path

import pytest

from lume_pva_apg._optional import DISTRIBUTION


def _installed(module: str) -> bool:
    """Whether ``module`` is installed, without importing it.

    Every question here is about what an install contains, so it is answered
    from the module's spec. Importing to find out would make this file
    uncollectable on a host carrying a transport that does not load -- which is
    a condition these tests exist to describe, not one they should die of.
    """
    return importlib.util.find_spec(module) is not None


HAS_P4P = _installed("p4p")
HAS_PCASPY = _installed("pcaspy")

# Installed as a meta-path finder ahead of every other, so a named module
# cannot be imported however it would otherwise have been found. Absent and
# broken are raised as the import system raises them: ModuleNotFoundError with
# the module's name for a package that is not there, and a plain ImportError
# for one that is there but whose extension module will not load.
_BLOCKER = """
import sys


class _Blocker:
    def __init__(self, absent=(), broken=()):
        self.absent = set(absent)
        self.broken = set(broken)

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.partition(".")[0]
        if top in self.absent:
            raise ModuleNotFoundError(f"No module named {top!r}", name=top)
        if top in self.broken:
            raise ImportError(f"dlopen({top}) failed: simulated broken install", name=top)
        return None


sys.meta_path.insert(0, _Blocker(absent=%(absent)r, broken=%(broken)r))
"""


def _run(
    body: str,
    cwd: Path,
    absent: tuple[str, ...] = (),
    broken: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a fresh interpreter with the named modules unimportable.

    Parameters
    ----------
    body : str
        Source to run after the blocker is installed.
    cwd : Path
        Working directory. A directory outside the checkout, so ``-c`` puts no
        source tree on ``sys.path`` ahead of the installed distribution.
    absent : tuple[str, ...]
        Modules to make look uninstalled.
    broken : tuple[str, ...]
        Modules to make look installed but unloadable.

    Returns
    -------
    subprocess.CompletedProcess[str] :
        The completed run. Not checked -- several cases expect a failure.
    """
    source = _BLOCKER % {"absent": tuple(absent), "broken": tuple(broken)} + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_the_package_imports_with_no_transport_installed(tmp_path: Path) -> None:
    """The pure-Python core guarantee, at the level of what gets imported."""
    result = _run(
        """
        import lume_pva_apg

        print(lume_pva_apg.__name__)
        """,
        cwd=tmp_path,
        absent=("pcaspy", "p4p", "torch", "lume_torch"),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lume_pva_apg"


@pytest.mark.skipif(
    not (HAS_P4P and HAS_PCASPY),
    reason="both transports must be installed for their absence to prove anything",
)
def test_importing_the_package_pulls_in_no_transport(tmp_path: Path) -> None:
    """Importable-without and not-imported-with are different properties.

    The transports are installed here, so nothing stops the package reaching
    for one. Anything re-exported from ``__init__`` would load p4p or pcaspy on
    ``import lume_pva_apg`` and cost every consumer the transport they were
    promised they could do without -- while the test above still passed,
    because a re-export of a module that is absent can be made to degrade.
    """
    result = _run(
        """
        import sys

        import lume_pva_apg

        loaded = sorted(m for m in ("pcaspy", "p4p", "torch") if m in sys.modules)
        print(",".join(loaded))
        """,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_the_value_layer_without_pva_names_the_extra(tmp_path: Path) -> None:
    result = _run(
        """
        try:
            import lume_pva_apg.variables
        except ImportError as exc:
            print(exc)
        """,
        cwd=tmp_path,
        absent=("p4p",),
    )

    assert result.returncode == 0, result.stderr
    assert f"pip install '{DISTRIBUTION}[pva]'" in result.stdout


@pytest.mark.skipif(HAS_P4P is False, reason="the runner reaches pcaspy only through p4p")
def test_the_runner_without_ca_names_the_extra(tmp_path: Path) -> None:
    result = _run(
        """
        try:
            import lume_pva_apg.runner
        except ImportError as exc:
            print(type(exc).__name__, exc.name)
            print(exc)
        """,
        cwd=tmp_path,
        absent=("pcaspy",),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "ModuleNotFoundError pcaspy"
    assert f"pip install '{DISTRIBUTION}[ca]'" in result.stdout


@pytest.mark.skipif(HAS_P4P is False, reason="the runner reaches pcaspy only through p4p")
def test_a_transport_that_will_not_load_is_not_reported_as_missing(tmp_path: Path) -> None:
    """The distinction the whole of ``_optional`` exists to draw.

    A transport whose extension module fails to load raises ``ImportError``
    exactly as an absent one does. Told to install the extra, an operator
    reinstalls a package that is already there and learns nothing.

    Carried in the exception type as well as the message, because a caller
    reraising it -- the suite's own import guards do -- has to tell the two
    apart without parsing prose.
    """
    result = _run(
        """
        try:
            import lume_pva_apg.runner
        except ImportError as exc:
            print(type(exc).__name__, exc.name)
            print(exc)
        """,
        cwd=tmp_path,
        broken=("pcaspy",),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "ImportError pcaspy"
    assert "installed but failed to load" in result.stdout
    assert "pip install" not in result.stdout


@pytest.mark.skipif(HAS_P4P is False, reason="the value layer needs p4p whatever torch does")
def test_the_value_layer_serves_the_core_types_without_torch(tmp_path: Path) -> None:
    """torch is optional, and the variable types that do not need it still work.

    The torch extra is the one optional dependency whose absence must *not*
    fail an import: the Torch* handlers drop out and every other handler stays.
    Nothing in a full-install test run exercises that branch.
    """
    result = _run(
        """
        from lume.variables import NDVariable, ScalarVariable

        from lume_pva_apg.variables import TORCH_AVAILABLE, find_variable_handler

        assert TORCH_AVAILABLE is False
        assert find_variable_handler(ScalarVariable) is not None
        assert find_variable_handler(NDVariable) is not None
        print("ok")
        """,
        cwd=tmp_path,
        absent=("torch", "lume_torch"),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_documented_extras_are_declared() -> None:
    """The names a consumer pins against.

    ``pip install 'lume-pva-apg[ca,pva]'`` is what the README tells a consumer
    to write, and pip does not fail on an extra that does not exist -- it warns
    and installs the base distribution, so a renamed extra reaches the consumer
    as a missing transport at run time.
    """
    try:
        declared = set(metadata(DISTRIBUTION).get_all("Provides-Extra") or ())
    except PackageNotFoundError:
        pytest.skip(f"{DISTRIBUTION} is not installed; run against the installed distribution")

    assert {"ca", "pva", "torch", "dev", "docs"} <= declared
