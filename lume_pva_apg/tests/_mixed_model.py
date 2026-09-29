"""A model serving one variable of every scalar type the runner supports.

The over-the-wire suites (:mod:`test_seams`, :mod:`test_write_path`) each built
their servers around a model of float variables only, so nothing that serves a
value exercised the integer, boolean, string or enum paths through EPICS.
:class:`MixedModel` is the shared fixture for those paths: one float input and
one float output, both carrying a ``value_range``, plus an int, a bool, a str
and an enum input.

It lives in a module of its own, rather than in either test module, so both
harnesses import the same class and a ``spawn``-started server child can locate
it by module path.

Every variable name takes an optional ``tag`` suffix, so a harness that keeps
its servers apart by variable name rather than by prefix can serve several of
these side by side.
"""

from __future__ import annotations

from multiprocessing.synchronize import Event as mpEvent
from typing import Any

from lume.model import LUMEModel
from lume.variables import (
    BoolVariable,
    EnumVariable,
    IntVariable,
    ScalarVariable,
    StrVariable,
    Variable,
)

# Base variable names, before any tag is appended.
MIXED_FLOAT_IN = "mixed_float_in"
MIXED_FLOAT_OUT = "mixed_float_out"
MIXED_INT = "mixed_int"
MIXED_BOOL = "mixed_bool"
MIXED_STR = "mixed_str"
MIXED_ENUM = "mixed_enum"

MIXED_NAMES = (MIXED_FLOAT_IN, MIXED_FLOAT_OUT, MIXED_INT, MIXED_BOOL, MIXED_STR, MIXED_ENUM)

# The operating range of both float variables.
MIXED_RANGE = (-10.0, 10.0)
MIXED_UNIT = "mm"
# What the model does to the float input to produce the float output.
MIXED_GAIN = 2.0
MIXED_ENUM_OPTIONS = ["OFF", "ON", "STANDBY"]

# Each variable's default. None is the zero value of its type, so a value read
# back over the wire can only have come from the model, never from a zeroed record.
MIXED_DEFAULTS: dict[str, Any] = {
    MIXED_FLOAT_IN: 1.5,
    MIXED_FLOAT_OUT: 1.5 * MIXED_GAIN,
    MIXED_INT: 3,
    MIXED_BOOL: True,
    MIXED_STR: "hello",
    MIXED_ENUM: "ON",
}


class MixedModel(LUMEModel):
    """One variable of each scalar type: float in/out, int, bool, str and enum.

    Parameters
    ----------
    started : Event | None
        Set whenever ``set`` or ``reset`` completes, so a server harness can
        wait for the runner's startup cycle. Optional for in-process use.
    tag : str
        Suffix appended to every variable name.
    """

    def __init__(self, started: mpEvent | None = None, tag: str = "") -> None:
        self.tag = tag
        self.started = started
        self.float_in = f"{MIXED_FLOAT_IN}{tag}"
        self.float_out = f"{MIXED_FLOAT_OUT}{tag}"
        self._vars: dict[str, Variable] = {
            self.float_in: ScalarVariable(
                name=self.float_in,
                default_value=MIXED_DEFAULTS[MIXED_FLOAT_IN],
                value_range=MIXED_RANGE,
                unit=MIXED_UNIT,
                read_only=False,
            ),
            self.float_out: ScalarVariable(
                name=self.float_out,
                default_value=MIXED_DEFAULTS[MIXED_FLOAT_OUT],
                value_range=MIXED_RANGE,
                unit=MIXED_UNIT,
                read_only=True,
            ),
            f"{MIXED_INT}{tag}": IntVariable(
                name=f"{MIXED_INT}{tag}",
                default_value=MIXED_DEFAULTS[MIXED_INT],
                read_only=False,
            ),
            f"{MIXED_BOOL}{tag}": BoolVariable(
                name=f"{MIXED_BOOL}{tag}",
                default_value=MIXED_DEFAULTS[MIXED_BOOL],
                read_only=False,
            ),
            f"{MIXED_STR}{tag}": StrVariable(
                name=f"{MIXED_STR}{tag}",
                default_value=MIXED_DEFAULTS[MIXED_STR],
                read_only=False,
            ),
            f"{MIXED_ENUM}{tag}": EnumVariable(
                name=f"{MIXED_ENUM}{tag}",
                options=list(MIXED_ENUM_OPTIONS),
                default_value=MIXED_DEFAULTS[MIXED_ENUM],
                read_only=False,
            ),
        }
        self._state: dict[str, Any] = {}
        self._restore_defaults()

    def _restore_defaults(self) -> None:
        self._state = {f"{base}{self.tag}": value for base, value in MIXED_DEFAULTS.items()}

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._vars

    def _get(self, names) -> dict[str, Any]:
        return {n: self._state[n] for n in names}

    def _set(self, values: dict[str, Any]) -> None:
        self._state.update({k: v for k, v in values.items() if k in self._state})
        self._state[self.float_out] = float(self._state[self.float_in]) * MIXED_GAIN
        if self.started is not None:
            self.started.set()

    def reset(self) -> None:
        self._restore_defaults()
        if self.started is not None:
            self.started.set()
