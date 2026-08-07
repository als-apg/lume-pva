"""Helpers for reporting a missing optional EPICS transport.

The core distribution is pure-Python. The transports live behind extras --
``[ca]`` for Channel Access (pcaspy) and ``[pva]`` for PVAccess (p4p) -- so a
module that needs one can be imported by a caller who never installed it. A
bare ``ModuleNotFoundError`` names a third-party package the caller did not ask
for and does not say how to fix it; these helpers name the extra instead.
"""

from __future__ import annotations

DISTRIBUTION = "lume-pva-apg"


def module_is_absent(module: str, cause: ImportError | None) -> bool:
    """Whether ``cause`` means ``module`` is not installed at all.

    An optional dependency that is installed but fails to load -- a wheel
    linked against a library the host does not have -- raises ``ImportError``
    just as an absent one does, and the two want opposite responses: install
    the extra, or repair the install. This is the one place that distinction is
    drawn, so a caller acting on it cannot drift from the message
    :func:`missing_extra` produces.

    ``cause`` of ``None`` reports absent: a caller with no failure in hand is
    asking about a dependency it never got far enough to load.

    Parameters
    ----------
    module : str
        The top-level import that failed, e.g. ``"pcaspy"``. A submodule import
        raises with the *missing* module's name, so pass the top-level name.
    cause : ImportError | None
        The original failure.

    Returns
    -------
    bool :
        True when the module is not installed.
    """
    if cause is None:
        return True
    return isinstance(cause, ModuleNotFoundError) and cause.name == module


def missing_extra(module: str, extra: str, cause: ImportError | None = None) -> ImportError:
    """Build the ImportError to raise when an optional transport is unusable.

    Distinguishes absent from broken. An installed transport that fails to
    load its extension module -- a wheel linked against a library that is not
    on the host, say -- also raises ImportError, and telling that caller to
    install the extra they already have sends them the wrong way.

    Parameters
    ----------
    module : str
        The import that failed, e.g. ``"pcaspy"``.
    extra : str
        The extra that provides it, e.g. ``"ca"``.
    cause : ImportError | None
        The original failure. ``ModuleNotFoundError`` for this module means
        absent; anything else means present but not loadable.

    Returns
    -------
    ImportError :
        An error naming either the extra to install or the broken install.
        Raise it ``from`` the original so the underlying failure stays visible.

        ``ModuleNotFoundError`` for an absent module and a plain
        ``ImportError`` for a broken one, each carrying ``name``. The message
        already says which, but a caller reraising this -- a test module
        deciding whether to skip, an installer deciding whether to install --
        needs it in the exception rather than in prose, and would otherwise see
        every failure as the broken one.
    """
    if module_is_absent(module, cause):
        return ModuleNotFoundError(
            f"'{module}' is required here but is not installed. "
            f"It ships with the '{extra}' extra: pip install '{DISTRIBUTION}[{extra}]'",
            name=module,
        )
    return ImportError(
        f"'{module}' is installed but failed to load; the '{extra}' extra is present "
        f"but not usable on this host. See the underlying error above.",
        name=module,
    )
