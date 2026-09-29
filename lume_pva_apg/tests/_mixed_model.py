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

Two knobs exercise the runner's ``output_severity`` handling. ``fail_mode``
gives the model an ``output_severity`` hook -- absent when it is unset -- that
misbehaves while the float input holds :data:`MIXED_SEVERITY_TRIGGER` and
reports nothing otherwise, so a test can post a clean cycle, trigger, and then
recover. ``extra_key`` makes ``_get`` return one key nobody requested, carrying
a value no variable would accept.
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

# While the float input holds this value, an output_severity hook acts on its
# fail_mode. Inside MIXED_RANGE, but it doubles to a float output outside it.
MIXED_SEVERITY_TRIGGER = -7.5
# What output_severity may be asked to do while triggered: report the chosen
# names undefined, raise, or reply with something that is not a valid reply.
FAIL_UDF = "udf"
FAIL_RAISE = "raise"
FAIL_MALFORMED = "malformed"
FAIL_MODES = (FAIL_UDF, FAIL_RAISE, FAIL_MALFORMED)
# The names reported undefined in FAIL_UDF mode when a test chooses none.
MIXED_DEFAULT_UDF = (MIXED_FLOAT_OUT, MIXED_INT)
# The key extra_key adds to every _get reply, and the wrongly typed value it
# carries. No variable is named this, so LUMEModel.get never validates it.
MIXED_EXTRA_KEY = "mixed_extra"
MIXED_EXTRA_VALUE = object()

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
    fail_mode : str | None
        One of :data:`FAIL_MODES`: the model then defines ``output_severity``,
        which while triggered reports ``udf_names`` undefined, raises, or
        returns a malformed reply. ``None`` (the default) defines no hook.
    udf_names : tuple[str, ...]
        Base names (before ``tag``) reported undefined in ``FAIL_UDF`` mode.
    extra_key : bool
        Add :data:`MIXED_EXTRA_KEY` to every ``_get`` reply.
    """

    def __init__(
        self,
        started: mpEvent | None = None,
        tag: str = "",
        *,
        fail_mode: str | None = None,
        udf_names: tuple[str, ...] = MIXED_DEFAULT_UDF,
        extra_key: bool = False,
    ) -> None:
        if fail_mode is not None and fail_mode not in FAIL_MODES:
            raise ValueError(f"fail_mode must be one of {FAIL_MODES} or None, got {fail_mode!r}")
        self.tag = tag
        self.started = started
        self.fail_mode = fail_mode
        self.udf_names = tuple(f"{base}{tag}" for base in udf_names)
        self.extra_key = extra_key
        # Every names list output_severity was called with, in order.
        self.severity_calls: list[list[str]] = []
        # Defined per instance, so a model built without fail_mode has no
        # output_severity attribute at all -- the runner's hook-less path.
        if fail_mode is not None:
            self.output_severity = self._output_severity
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
        values = {n: self._state[n] for n in names}
        if self.extra_key:
            values[MIXED_EXTRA_KEY] = MIXED_EXTRA_VALUE
        return values

    @property
    def triggered(self) -> bool:
        """Whether the float input holds :data:`MIXED_SEVERITY_TRIGGER`."""
        return float(self._state[self.float_in]) == MIXED_SEVERITY_TRIGGER

    def _output_severity(self, names: list[str]) -> Any:
        self.severity_calls.append(list(names))
        if not self.triggered:
            return {}
        if self.fail_mode == FAIL_RAISE:
            raise RuntimeError("output_severity cannot tell")
        if self.fail_mode == FAIL_MALFORMED:
            return {n: {"condition": "no-such-condition"} for n in self.udf_names}
        return {n: {"condition": "udf"} for n in self.udf_names}

    def _set(self, values: dict[str, Any]) -> None:
        self._state.update({k: v for k, v in values.items() if k in self._state})
        self._state[self.float_out] = float(self._state[self.float_in]) * MIXED_GAIN
        if self.started is not None:
            self.started.set()

    def reset(self) -> None:
        self._restore_defaults()
        if self.started is not None:
            self.started.set()
