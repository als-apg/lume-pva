"""Helpers for reporting a missing optional EPICS transport.

The core distribution is pure-Python. The transports live behind extras --
``[ca]`` for Channel Access (pcaspy) and ``[pva]`` for PVAccess (p4p) -- so a
module that needs one can be imported by a caller who never installed it. A
bare ``ModuleNotFoundError`` names a third-party package the caller did not ask
for and does not say how to fix it; these helpers name the extra instead.
"""

from __future__ import annotations

DISTRIBUTION = "lume-pva-apg"


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
    """
    absent = isinstance(cause, ModuleNotFoundError) and cause.name == module
    if absent or cause is None:
        return ImportError(
            f"'{module}' is required here but is not installed. "
            f"It ships with the '{extra}' extra: pip install '{DISTRIBUTION}[{extra}]'"
        )
    return ImportError(
        f"'{module}' is installed but failed to load; the '{extra}' extra is present "
        f"but not usable on this host. See the underlying error above."
    )
