"""Tests for the names, the metadata and the extension points a
:class:`lume_pva_apg.runner.Runner` puts on the wire.

The CA database is keyed by base PV name and prefixed exactly once, by
``SimpleServer.createPV``, and that same base name is what names a PV in every
driver callback -- ``write``'s ``reason``, ``setParam``, ``updatePV``. Both
halves have to agree: a database keyed by already-prefixed names serves the
prefix twice over *and* hands the driver a reason the runner's own lookup
tables cannot resolve.

Every server here therefore runs under a non-empty ``prefix``. A suite run
entirely at ``prefix=""`` cannot tell a singly-applied prefix from a doubly
applied one, nor an output pass that resolves its names from one that does not.

The metadata tests cover what a variable's ``value_range`` becomes on the CA
side: display limits, and not an alarm threshold.

The remaining tests cover the four seams a subclass builds on -- ``_extend_pvdb``,
``ca_driver_cls``, ``_post_outputs`` and the ``control_pvs`` configuration key.
Each is asserted through EPICS rather than through the object: a hook that is
called but whose result never reaches a client is not a seam. Where the base
class already does something similar -- serving a PV, publishing an output --
a runner *without* the override is served alongside as a negative control, so
the assertions distinguish the seam from what the base class does anyway.

Runners are started in independent subprocesses so each test gets a fresh
server and a fresh configuration. Each server takes a prefix of its own, so no
two tests share a PV name and the client's channel cache stays valid for the
whole session. Clearing that cache between tests is the obvious alternative and
is not used here: it detaches and recreates the CA context underneath PV
objects that are still alive, which pyepics documents as a route to a random
SIGSEGV from inside the EPICS libraries.
"""

import functools
import inspect
import itertools
import multiprocessing
import os
import threading
from collections.abc import Callable, Generator
from multiprocessing.synchronize import Event as mpEvent
from typing import Any

import pytest

from lume_pva_apg.tests._requires import skip_if_absent
from lume_pva_apg.tests._spawn import wait_until_ready

# Keep all EPICS traffic on the loopback interface. Must be set before p4p,
# pyepics, or the pcaspy server (created in Runner.__init__) initialise.
os.environ.setdefault("EPICS_CA_ADDR_LIST", "127.0.0.1")
os.environ.setdefault("EPICS_CA_AUTO_ADDR_LIST", "NO")
os.environ.setdefault("EPICS_PVA_ADDR_LIST", "127.0.0.1")
os.environ.setdefault("EPICS_PVA_AUTO_ADDR_LIST", "NO")

# Both transports serve here and the CA client is pyepics, so this module needs
# the whole dev set. Guarded so an install missing one of them skips this file
# rather than failing collection, which would take the whole suite with it.
try:
    import epics
    import pcaspy
    from lume.model import LUMEModel
    from lume.variables import ScalarVariable
    from p4p.client.thread import Context
    from p4p.client.thread import TimeoutError as PvaTimeoutError

    from lume_pva_apg.runner import RESET_CONTROL_PV, Runner
    from lume_pva_apg.tests._mixed_model import (
        FAIL_MALFORMED,
        FAIL_RAISE,
        FAIL_UDF,
        MIXED_BOOL,
        MIXED_DEFAULTS,
        MIXED_ENUM,
        MIXED_ENUM_OPTIONS,
        MIXED_FLOAT_IN,
        MIXED_FLOAT_OUT,
        MIXED_INT,
        MIXED_SEVERITY_TRIGGER,
        MIXED_STR,
        MixedModel,
    )
except ImportError as exc:
    skip_if_absent(exc)

# Generous upper bound for any single operation to complete.
OP_TIMEOUT = 10.0
# Bound for an operation expected to time out. Long enough that a slow-but-live
# server still answers within it, short enough that a handful of absence checks
# do not dominate the run.
ABSENT_TIMEOUT = 3.0

# The model's variables. Served unprefixed in the pvdb and under the server's
# prefix on the wire.
IN_A = "in_a"
OUT_DOUBLE = "out_double"
# Contributed by _extend_pvdb, and known to no model.
EXTRA_PV = "EXTRA"
# Also contributed by _extend_pvdb, and written by the _post_outputs override.
MIRROR_PV = "MIRROR"
# What the seam driver does to a value written to EXTRA_PV. The stock driver
# resolves a reason through the model's variables, so it refuses EXTRA_PV
# outright; any non-zero readback can only have come from the override.
EXTRA_GAIN = 2.0
# What the _post_outputs override adds to the output it mirrors, so a mirrored
# value cannot be confused with the output itself.
MIRROR_OFFSET = 1.0

RANGE = (-10.0, 10.0)
UNIT = "mm"

_MP = multiprocessing.get_context("spawn")
_TAGS = itertools.count()


class SeamModel(LUMEModel):
    """One writable input and one read-only output derived from it."""

    def __init__(self, started: mpEvent) -> None:
        self._state: dict[str, float] = {IN_A: 0.0, OUT_DOUBLE: 0.0}
        self._vars: dict[str, ScalarVariable] = {
            IN_A: ScalarVariable(
                name=IN_A,
                default_value=0.0,
                value_range=RANGE,
                read_only=False,
                unit=UNIT,
            ),
            OUT_DOUBLE: ScalarVariable(name=OUT_DOUBLE, default_value=0.0, read_only=True),
        }
        self.started = started

    @property
    def supported_variables(self) -> dict[str, ScalarVariable]:
        return self._vars

    def _get(self, names) -> dict[str, float]:
        return {n: self._state.get(n, 0.0) for n in names}

    def _set(self, values: dict[str, Any]) -> None:
        self._state.update({k: float(v) for k, v in values.items() if k in self._state})
        self._state[OUT_DOUBLE] = self._state[IN_A] * 2.0
        self.started.set()

    def reset(self) -> None:
        for key in self._state:
            self._state[key] = 0.0
        self.started.set()


def _extra_entries() -> dict[str, dict[str, Any]]:
    """The pvdb entries a subclass contributes, keyed by base PV name."""
    return {
        EXTRA_PV: {"type": "float", "value": 0.0},
        MIRROR_PV: {"type": "float", "value": 0.0},
    }


class ExtendOnlyRunner(Runner):
    """Contributes PVs and nothing else.

    The negative control for the driver and output seams: it serves the same
    two extra PVs, so a difference in how they behave is attributable to the
    seam under test rather than to their presence.
    """

    def _extend_pvdb(self) -> dict[str, dict[str, Any]]:
        return _extra_entries()


class SeamRunner(ExtendOnlyRunner):
    """A subclass built out of every seam, the way a consumer would build one."""

    class SeamDriver(Runner.CaDriver):
        """Serves EXTRA_PV, a reason the stock driver would refuse."""

        def write(self, reason: str, value: Any) -> bool:
            if reason == EXTRA_PV:
                self.setParam(reason, value * EXTRA_GAIN)
                self.updatePV(reason)
                return True
            return super().write(reason, value)

    ca_driver_cls = SeamDriver

    def _post_outputs(self, out_values: dict[str, Any], ts: float) -> None:
        # Published before delegating, so the base implementation's updatePVs
        # flushes this alongside the model's own outputs.
        if self.ca_driver is not None:
            self.ca_driver.setParam(MIRROR_PV, float(out_values[OUT_DOUBLE]) + MIRROR_OFFSET)
        super()._post_outputs(out_values, ts)


# The variable PvaOnlyVariableRunner keeps off CA.
PVA_ONLY = OUT_DOUBLE


class PvaOnlyVariableRunner(Runner):
    """Serves one variable over PVA only, by clearing ``supports_ca`` around it.

    A consumer that must leave one name to another CA server keeps that name
    off its own CA database this way, without giving up CA for the rest. The
    flag is restored in a ``finally``, so everything read after ``_add_pv`` --
    the other variables, the pvdb merge, the control PVs -- still sees CA on.
    """

    def _add_pv(self, pv: str, var: Any, ro: bool, prefix: str, handler: Any) -> None:
        if var.name != PVA_ONLY:
            super()._add_pv(pv, var, ro, prefix, handler)
            return
        supports_ca = self.supports_ca
        self.supports_ca = False
        try:
            super()._add_pv(pv, var, ro, prefix, handler)
        finally:
            self.supports_ca = supports_ca


_RUNNERS: dict[str, type[Runner]] = {
    "stock": Runner,
    "extend_only": ExtendOnlyRunner,
    "seam": SeamRunner,
    "pva_only_variable": PvaOnlyVariableRunner,
    # The stock runner, serving a MixedModel instead of a SeamModel.
    "mixed": Runner,
    # The stock runner, serving a MixedModel with output_severity / extra-key knobs.
    "mixed_udf": Runner,
    "mixed_udf_int": Runner,
    "mixed_raise": Runner,
    "mixed_malformed": Runner,
    "mixed_extra": Runner,
    "mixed_udf_extra": Runner,
}

# The model each runner key serves, where it is not SeamModel.
_MODELS: dict[str, Callable[[mpEvent], LUMEModel]] = {
    "mixed": MixedModel,
    "mixed_udf": functools.partial(MixedModel, fail_mode=FAIL_UDF),
    "mixed_udf_int": functools.partial(MixedModel, fail_mode=FAIL_UDF, udf_names=(MIXED_INT,)),
    "mixed_raise": functools.partial(MixedModel, fail_mode=FAIL_RAISE),
    "mixed_malformed": functools.partial(MixedModel, fail_mode=FAIL_MALFORMED),
    "mixed_extra": functools.partial(MixedModel, extra_key=True),
    "mixed_udf_extra": functools.partial(MixedModel, fail_mode=FAIL_UDF, extra_key=True),
}


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Merge `overrides` into a generated config.

    Every key replaces the config's own, except ``variables``: that one maps a
    variable name to the keys to merge into *that variable's* generated entry,
    so a test can change one variable's ``pv`` or ``mode`` without restating
    the rest of the config. A name the model does not serve is an error rather
    than a new entry, so a typo cannot pass as an override that did nothing.
    """
    overrides = dict(overrides)
    for name, entry in overrides.pop("variables", {}).items():
        if name not in config["variables"]:
            raise KeyError(f"override names {name!r}, which the model does not serve")
        config["variables"][name].update(entry)
    config.update(overrides)


def _serve(
    prefix: str,
    runner_key: str,
    overrides: dict[str, Any],
    started: mpEvent,
    ready: mpEvent,
) -> None:
    """Child-process entry point: serve the key's model under `prefix`.

    The model is SeamModel unless ``_MODELS`` names another for `runner_key`.
    Must be importable at module top level so the ``spawn`` start method can
    locate it. Blocks forever once ready; the parent terminates the process.
    """
    model = _MODELS.get(runner_key, SeamModel)(started)
    config = Runner.generate_config(model, prefix=prefix)
    config["update_rate"] = 0.0
    _apply_overrides(config, overrides)

    runner = _RUNNERS[runner_key](model=model, config=config)
    threading.Thread(target=runner._run, daemon=True).start()

    # Runner.__init__ enqueues an empty update; wait for that cycle to land so
    # the parent never races the server's first publish.
    if not started.wait(timeout=OP_TIMEOUT):
        raise RuntimeError("startup cycle never ran in child")
    ready.set()
    threading.Event().wait()


@pytest.fixture(scope="function")
def serve() -> Generator[Callable[..., str], None, None]:
    """Yield a factory that starts a configured Runner and returns its prefix.

    ``serve(runner_key, **overrides)``: keyword overrides replace config keys,
    and ``variables={name: {...}}`` is merged into each named variable's
    generated entry (see :func:`_apply_overrides`).
    """
    procs: list[Any] = []

    def _start(runner_key: str = "stock", **overrides: Any) -> str:
        prefix = f"SEAM{next(_TAGS)}:"
        started = _MP.Event()
        ready = _MP.Event()
        proc = _MP.Process(
            target=_serve, args=(prefix, runner_key, overrides, started, ready), daemon=True
        )
        proc.start()
        procs.append(proc)
        wait_until_ready(proc, ready)
        return prefix

    try:
        yield _start
    finally:
        for proc in procs:
            proc.terminate()
            proc.join(timeout=OP_TIMEOUT)


def _read(name: str) -> Any:
    """Read a CA PV from the server, never from the client's monitor cache."""
    value = epics.caget(name, use_monitor=False, timeout=OP_TIMEOUT)
    assert value is not None, f"{name} did not respond"
    return value


def _absent(name: str) -> None:
    """Assert no server answers to `name`."""
    value = epics.caget(name, use_monitor=False, timeout=ABSENT_TIMEOUT)
    assert value is None, f"{name} answered with {value!r}; nothing should serve it"


def _put(name: str, value: Any) -> None:
    """Issue a completion-aware caput and assert the client was not left hanging."""
    rc = epics.caput(name, value, wait=True, timeout=OP_TIMEOUT)
    assert rc == 1, f"caput on {name} did not complete (rc={rc})"


def _ctrlvars(name: str) -> dict[str, Any]:
    pv = epics.get_pv(name, timeout=OP_TIMEOUT)
    assert pv.wait_for_connection(timeout=OP_TIMEOUT), f"{name} never connected"
    return pv.get_ctrlvars(timeout=OP_TIMEOUT)


def _severity(name: str) -> tuple[int, int]:
    """Return (severity, status) as the server currently reports them."""
    pv = epics.get_pv(name, timeout=OP_TIMEOUT)
    pv.get(use_monitor=False, timeout=OP_TIMEOUT)
    return pv.severity, pv.status


# --------------------------------------------------------------------------
# (a) the prefix reaches the wire exactly once, on both halves of the CA path
# --------------------------------------------------------------------------


def test_ca_names_are_prefixed_exactly_once(serve) -> None:
    """Every served CA name carries the prefix once, and nothing carries it twice.

    ``createPV`` builds the served name by prepending the prefix to each pvdb
    key, so a pvdb keyed by an already-prefixed name serves the prefix twice
    over and the singly-prefixed name a client asks for does not exist.
    """
    prefix = serve()

    for pv in (IN_A, OUT_DOUBLE, RESET_CONTROL_PV):
        _read(f"{prefix}{pv}")
        _absent(f"{prefix}{prefix}{pv}")


def test_prefixed_ca_write_reaches_the_model(serve) -> None:
    """A CA write under a prefix resolves to its variable.

    The driver names a PV by its pvdb key, so a database keyed by prefixed
    names hands ``write`` a reason that the runner's own base-name lookup
    tables cannot resolve, and the write is refused.
    """
    prefix = serve(echo_unconfirmed_writes=False)

    _put(f"{prefix}{IN_A}", 3.0)

    assert _read(f"{prefix}{IN_A}") == pytest.approx(3.0)
    assert _read(f"{prefix}{OUT_DOUBLE}") == pytest.approx(6.0)


def test_prefixed_output_pass_completes_the_cycle(serve) -> None:
    """The cycle's output pass resolves its CA names under a prefix.

    Driven over PVA so the CA write path is not involved: what is under test is
    the output pass, which addresses the CA driver by the name it holds in
    ``ca_pvs``. A key that is not in the pvdb raises there, and the exception
    fails the whole cycle -- rolling the model back and completing the put with
    an error -- rather than surfacing as a missing PV.
    """
    prefix = serve(echo_unconfirmed_writes=False)

    with Context("pva") as ctx:
        # Raises on a failed cycle; the runner completes the put with the
        # error the cycle raised.
        ctx.put(f"{prefix}{IN_A}", 4.0, timeout=OP_TIMEOUT)
        assert ctx.get(f"{prefix}{OUT_DOUBLE}", timeout=OP_TIMEOUT).raw.value == pytest.approx(8.0)

    # The same value reached the CA transport, from the same pass.
    assert _read(f"{prefix}{OUT_DOUBLE}") == pytest.approx(8.0)
    # The echo was withheld on a failed cycle, so its presence is independent
    # evidence that the cycle succeeded.
    assert _read(f"{prefix}{IN_A}") == pytest.approx(4.0)


# --------------------------------------------------------------------------
# (b) ca_pvspec publishes value_range as display limits and nothing else
# --------------------------------------------------------------------------


def test_value_range_is_published_as_display_limits_only(serve) -> None:
    """``value_range`` describes the operating range, not an alarm threshold.

    pcaspy evaluates a numeric alarm only where ``lolo < hihi`` and
    ``low < high`` hold, so leaving all four unset -- rather than setting them
    to zero, which would still leave them defined -- is what takes the range
    out of the alarm calculation.
    """
    prefix = serve()

    ctrl = _ctrlvars(f"{prefix}{IN_A}")

    assert ctrl["lower_disp_limit"] == pytest.approx(RANGE[0])
    assert ctrl["upper_disp_limit"] == pytest.approx(RANGE[1])
    assert ctrl["units"] == UNIT
    for key in (
        "lower_alarm_limit",
        "upper_alarm_limit",
        "lower_warning_limit",
        "upper_warning_limit",
    ):
        assert ctrl[key] == pytest.approx(0.0), f"{key} is still an alarm threshold"


def test_a_value_at_its_own_limit_is_not_in_alarm(serve) -> None:
    """pcaspy compares with ``<=``/``>=``, so an alarm threshold taken from the
    variable's own range puts a value driven to either end of its legal span
    into MAJOR alarm."""
    prefix = serve()

    _put(f"{prefix}{IN_A}", RANGE[1])
    assert _read(f"{prefix}{IN_A}") == pytest.approx(RANGE[1])
    assert _severity(f"{prefix}{IN_A}") == (0, 0)

    _put(f"{prefix}{IN_A}", RANGE[0])
    assert _read(f"{prefix}{IN_A}") == pytest.approx(RANGE[0])
    assert _severity(f"{prefix}{IN_A}") == (0, 0)


# --------------------------------------------------------------------------
# (c) _extend_pvdb: a subclass's entries are served
# --------------------------------------------------------------------------


def test_extend_pvdb_entries_are_served(serve) -> None:
    prefix = serve("extend_only")

    assert _read(f"{prefix}{EXTRA_PV}") == pytest.approx(0.0)
    assert _read(f"{prefix}{MIRROR_PV}") == pytest.approx(0.0)
    # Contributed by base name, and prefixed by the server like any other.
    _absent(f"{prefix}{prefix}{EXTRA_PV}")


def test_stock_runner_contributes_nothing(serve) -> None:
    """The base implementation adds no PVs, so the entries above are the hook's."""
    prefix = serve("stock")

    _read(f"{prefix}{IN_A}")
    _absent(f"{prefix}{EXTRA_PV}")
    _absent(f"{prefix}{MIRROR_PV}")


def test_extend_pvdb_refuses_to_shadow_an_existing_name() -> None:
    """Contributing a name the model or a control PV already owns is fatal.

    Run in-process: the merge happens before any server is created, so nothing
    binds a port before the failure.
    """

    class ShadowRunner(Runner):
        def _extend_pvdb(self) -> dict[str, dict[str, Any]]:
            return {IN_A: {"type": "float"}, RESET_CONTROL_PV: {"type": "int"}}

    model = SeamModel(_MP.Event())
    config = Runner.generate_config(model, prefix="SHADOW:")
    config["protocol"] = ["ca"]

    with pytest.raises(RuntimeError) as excinfo:
        ShadowRunner(model=model, config=config)

    message = str(excinfo.value)
    assert IN_A in message
    assert RESET_CONTROL_PV in message


# --------------------------------------------------------------------------
# (d) ca_driver_cls: the replacement class is the one serving the database
# --------------------------------------------------------------------------


def test_ca_driver_cls_override_serves_the_database(serve) -> None:
    """A write lands on a PV no model describes, transformed by the override.

    The stock driver resolves a reason through ``pv_to_var`` and refuses
    anything it cannot find, so both the acceptance and the value are the
    override's doing.
    """
    prefix = serve("seam")

    _put(f"{prefix}{EXTRA_PV}", 4.0)

    assert _read(f"{prefix}{EXTRA_PV}") == pytest.approx(4.0 * EXTRA_GAIN)


def test_stock_driver_refuses_an_extended_pv(serve) -> None:
    """Without the override, the same write is refused by the stock driver."""
    prefix = serve("extend_only")

    epics.caput(f"{prefix}{EXTRA_PV}", 4.0, wait=True, timeout=OP_TIMEOUT)

    assert _read(f"{prefix}{EXTRA_PV}") == pytest.approx(0.0)


def test_ca_driver_cls_override_still_serves_the_model(serve) -> None:
    """Delegating to the base class keeps the model's own PVs writable."""
    prefix = serve("seam", echo_unconfirmed_writes=False)

    _put(f"{prefix}{IN_A}", -2.5)

    assert _read(f"{prefix}{IN_A}") == pytest.approx(-2.5)
    assert _read(f"{prefix}{OUT_DOUBLE}") == pytest.approx(-5.0)


# --------------------------------------------------------------------------
# (e) _post_outputs: the run loop's publishing step is the overridable one
# --------------------------------------------------------------------------


def test_post_outputs_override_publishes_from_the_run_loop(serve) -> None:
    prefix = serve("seam")

    _put(f"{prefix}{IN_A}", 3.0)

    assert _read(f"{prefix}{OUT_DOUBLE}") == pytest.approx(6.0)
    assert _read(f"{prefix}{MIRROR_PV}") == pytest.approx(6.0 + MIRROR_OFFSET)


def test_post_outputs_is_not_published_without_the_override(serve) -> None:
    prefix = serve("extend_only")

    _put(f"{prefix}{IN_A}", 3.0)

    assert _read(f"{prefix}{OUT_DOUBLE}") == pytest.approx(6.0)
    assert _read(f"{prefix}{MIRROR_PV}") == pytest.approx(0.0)


# --------------------------------------------------------------------------
# (f) control_pvs: whether the runner claims a name of its own
# --------------------------------------------------------------------------


def test_control_pvs_are_served_by_default(serve) -> None:
    """The default is what a runner setting nothing has always done."""
    prefix = serve()

    assert _read(f"{prefix}{RESET_CONTROL_PV}") == 0
    with Context("pva") as ctx:
        assert ctx.get(f"{prefix}{RESET_CONTROL_PV}", timeout=OP_TIMEOUT) is not None


def test_control_pvs_false_claims_no_reset_pv(serve) -> None:
    prefix = serve(control_pvs=False)

    # The model's own PVs are served as before...
    _read(f"{prefix}{IN_A}")
    with Context("pva") as ctx:
        assert ctx.get(f"{prefix}{IN_A}", timeout=OP_TIMEOUT) is not None

        # ...and neither transport claims the control PV.
        _absent(f"{prefix}{RESET_CONTROL_PV}")
        with pytest.raises(PvaTimeoutError):
            ctx.get(f"{prefix}{RESET_CONTROL_PV}", timeout=ABSENT_TIMEOUT)


def test_control_pvs_false_leaves_the_write_path_working(serve) -> None:
    """Suppressing the control PV must not disturb writes to the model."""
    prefix = serve(control_pvs=False, echo_unconfirmed_writes=False)

    _put(f"{prefix}{IN_A}", 1.5)

    assert _read(f"{prefix}{IN_A}") == pytest.approx(1.5)
    assert _read(f"{prefix}{OUT_DOUBLE}") == pytest.approx(3.0)


# --------------------------------------------------------------------------
# harness: a mixed-type model, and per-variable config overrides
# --------------------------------------------------------------------------


def test_mixed_model_serves_every_type_over_ca(serve) -> None:
    """Float, int, bool, str and enum each reach a CA client with their value."""
    prefix = serve("mixed")

    assert _read(f"{prefix}{MIXED_FLOAT_IN}") == pytest.approx(MIXED_DEFAULTS[MIXED_FLOAT_IN])
    assert _read(f"{prefix}{MIXED_FLOAT_OUT}") == pytest.approx(MIXED_DEFAULTS[MIXED_FLOAT_OUT])
    assert _read(f"{prefix}{MIXED_INT}") == MIXED_DEFAULTS[MIXED_INT]
    assert _read(f"{prefix}{MIXED_BOOL}") == 1
    str_value = epics.caget(
        f"{prefix}{MIXED_STR}", as_string=True, use_monitor=False, timeout=OP_TIMEOUT
    )
    assert str_value == MIXED_DEFAULTS[MIXED_STR]
    assert _read(f"{prefix}{MIXED_ENUM}") == MIXED_ENUM_OPTIONS.index(MIXED_DEFAULTS[MIXED_ENUM])

    ctrl = _ctrlvars(f"{prefix}{MIXED_ENUM}")
    assert list(ctrl["enum_strs"]) == MIXED_ENUM_OPTIONS


def test_mixed_model_serves_every_type_over_pva(serve) -> None:
    """The same five types reach a PVA client, each as its own normative type."""
    prefix = serve("mixed")

    with Context("pva") as ctx:

        def raw(name: str) -> Any:
            return ctx.get(f"{prefix}{name}", timeout=OP_TIMEOUT).raw.value

        assert raw(MIXED_FLOAT_IN) == pytest.approx(MIXED_DEFAULTS[MIXED_FLOAT_IN])
        assert raw(MIXED_FLOAT_OUT) == pytest.approx(MIXED_DEFAULTS[MIXED_FLOAT_OUT])
        assert raw(MIXED_INT) == MIXED_DEFAULTS[MIXED_INT]
        assert raw(MIXED_BOOL) is True
        assert raw(MIXED_STR) == MIXED_DEFAULTS[MIXED_STR]
        enum = raw(MIXED_ENUM)
        assert list(enum.choices) == MIXED_ENUM_OPTIONS
        assert enum.index == MIXED_ENUM_OPTIONS.index(MIXED_DEFAULTS[MIXED_ENUM])


def test_variable_override_reaches_the_config(serve) -> None:
    """``variables={name: {...}}`` changes that variable's entry and no other.

    Renaming the PV is observable from outside: the variable is served under
    the new name, no longer under its own, and its neighbour is untouched.
    """
    prefix = serve(variables={IN_A: {"pv": "RENAMED"}})

    assert _read(f"{prefix}RENAMED") == pytest.approx(0.0)
    _absent(f"{prefix}{IN_A}")
    _read(f"{prefix}{OUT_DOUBLE}")


# --------------------------------------------------------------------------
# display metadata: a configured precision and description reach the wire
# --------------------------------------------------------------------------


def _pva_raw(prefix: str, name: str) -> Any:
    """Read a PVA PV's whole structure, metadata included."""
    with Context("pva") as ctx:
        return ctx.get(f"{prefix}{name}", timeout=OP_TIMEOUT).raw


def test_a_configured_precision_reaches_ca_as_dbr_ctrl_precision(serve) -> None:
    """``precision: 3`` on a float is the ``precision`` a CA client's DBR_CTRL read reports."""
    prefix = serve("mixed", variables={MIXED_FLOAT_IN: {"precision": 3}})

    assert _ctrlvars(f"{prefix}{MIXED_FLOAT_IN}")["precision"] == 3


def test_a_configured_precision_reaches_pva_as_display_precision(serve) -> None:
    """``precision: 3`` on a float is served as ``display.precision``; a float
    configured without one keeps its precision-less display block."""
    prefix = serve("mixed", variables={MIXED_FLOAT_IN: {"precision": 3}})

    configured = _pva_raw(prefix, MIXED_FLOAT_IN)
    assert configured["display"]["precision"] == 3
    assert configured["value"] == pytest.approx(MIXED_DEFAULTS[MIXED_FLOAT_IN])

    plain = _pva_raw(prefix, MIXED_FLOAT_OUT)
    assert "precision" not in plain["display"].keys()


@pytest.mark.parametrize(
    "var_name",
    [
        pytest.param(MIXED_FLOAT_IN, id="float"),
        pytest.param(MIXED_INT, id="int"),
        pytest.param(MIXED_BOOL, id="bool"),
        pytest.param(MIXED_STR, id="str"),
        pytest.param(MIXED_ENUM, id="enum"),
    ],
)
def test_a_configured_description_reaches_pva_as_display_description(serve, var_name: str) -> None:
    """``description: "x"`` is served as ``display.description`` on every
    PVA type with a display block, and only on the variable configured with it."""
    prefix = serve("mixed", variables={var_name: {"description": "x"}})

    assert _pva_raw(prefix, var_name)["display"]["description"] == "x"
    assert _pva_raw(prefix, MIXED_FLOAT_OUT)["display"]["description"] == ""


def test_precision_and_description_leave_add_pv_signature_unchanged() -> None:
    """Subclasses call and override ``_add_pv``; the metadata travels through
    ``_pv_meta`` rather than new parameters, so its signature is 0.1.4's."""
    params = list(inspect.signature(Runner._add_pv).parameters)
    assert params == ["self", "pv", "var", "ro", "prefix", "handler"]


# --------------------------------------------------------------------------
# output_severity: an undefined output reaches both transports as INVALID/UDF
# --------------------------------------------------------------------------

# What a CA client reads for an undefined output: INVALID severity, UDF status.
CA_UDF = (int(pcaspy.Severity.INVALID_ALARM), int(pcaspy.Alarm.UDF_ALARM))
# The same on PVA: epicsAlarmSeverity INVALID_ALARM, epicsAlarmStatus UDF_STATUS.
PVA_UDF = (3, 6)


def _pva_alarm(prefix: str, name: str) -> tuple[int, int]:
    raw = _pva_raw(prefix, name)
    return raw["alarm"]["severity"], raw["alarm"]["status"]


def test_severity_udf_outputs_reach_ca_and_pva_as_invalid_udf(serve) -> None:
    prefix = serve("mixed_udf")

    _put(f"{prefix}{MIXED_FLOAT_IN}", MIXED_SEVERITY_TRIGGER)

    for name in (MIXED_FLOAT_OUT, MIXED_INT):
        assert _severity(f"{prefix}{name}") == CA_UDF, name
        assert _pva_alarm(prefix, name) == PVA_UDF, name
    # an undefined name still carries its type-valid value
    assert _read(f"{prefix}{MIXED_FLOAT_OUT}") == pytest.approx(2 * MIXED_SEVERITY_TRIGGER)
    assert _pva_raw(prefix, MIXED_FLOAT_OUT)["value"] == pytest.approx(2 * MIXED_SEVERITY_TRIGGER)
    # a name the hook did not report is untouched
    assert _severity(f"{prefix}{MIXED_BOOL}") == (0, 0)
    assert _pva_alarm(prefix, MIXED_BOOL) == (0, 0)


def test_severity_recovered_udf_outputs_leave_invalid_on_both_transports(serve) -> None:
    prefix = serve("mixed_udf")
    _put(f"{prefix}{MIXED_FLOAT_IN}", MIXED_SEVERITY_TRIGGER)
    assert _severity(f"{prefix}{MIXED_INT}") == CA_UDF

    _put(f"{prefix}{MIXED_FLOAT_IN}", 2.0)

    for name in (MIXED_FLOAT_OUT, MIXED_INT):
        assert _severity(f"{prefix}{name}") == (0, 0), name
        assert _pva_alarm(prefix, name) == (0, 0), name
    assert _read(f"{prefix}{MIXED_FLOAT_OUT}") == pytest.approx(4.0)


def test_severity_a_name_absent_from_the_reply_keeps_0_1_4_alarms(serve) -> None:
    """Only the int is reported: the float output, outside its range, keeps the
    PVA range rule (MAJOR) and CA's no-alarm-threshold behaviour."""
    prefix = serve("mixed_udf_int")

    _put(f"{prefix}{MIXED_FLOAT_IN}", MIXED_SEVERITY_TRIGGER)

    assert _severity(f"{prefix}{MIXED_INT}") == CA_UDF
    assert _pva_alarm(prefix, MIXED_INT) == PVA_UDF
    assert _pva_alarm(prefix, MIXED_FLOAT_OUT) == (2, 2)
    assert _severity(f"{prefix}{MIXED_FLOAT_OUT}") == (0, 0)


@pytest.mark.parametrize(
    "runner_key",
    [
        pytest.param("mixed_raise", id="hook-raises"),
        pytest.param("mixed_malformed", id="unknown-condition"),
    ],
)
def test_severity_a_failing_hook_posts_nothing_on_either_transport(serve, runner_key: str) -> None:
    prefix = serve(runner_key)
    _put(f"{prefix}{MIXED_FLOAT_IN}", 2.0)
    before = _pva_raw(prefix, MIXED_FLOAT_OUT)
    assert before["value"] == pytest.approx(4.0)

    # The cycle fails; the put still completes, with the error.
    epics.caput(f"{prefix}{MIXED_FLOAT_IN}", MIXED_SEVERITY_TRIGGER, wait=True, timeout=OP_TIMEOUT)

    after = _pva_raw(prefix, MIXED_FLOAT_OUT)
    assert after["value"] == pytest.approx(4.0)
    assert after["timeStamp"]["secondsPastEpoch"] == before["timeStamp"]["secondsPastEpoch"]
    assert after["timeStamp"]["nanoseconds"] == before["timeStamp"]["nanoseconds"]
    assert _read(f"{prefix}{MIXED_FLOAT_OUT}") == pytest.approx(4.0)
    assert _severity(f"{prefix}{MIXED_INT}") == (0, 0)

    # ...and the next good cycle publishes as usual.
    _put(f"{prefix}{MIXED_FLOAT_IN}", 1.0)
    assert _read(f"{prefix}{MIXED_FLOAT_OUT}") == pytest.approx(2.0)
    assert _pva_raw(prefix, MIXED_FLOAT_OUT)["value"] == pytest.approx(2.0)


def test_severity_hookless_model_with_an_extra_key_serves_as_0_1_4(serve) -> None:
    """No hook: an extra, wrongly typed key from ``_get`` changes nothing, and
    no name is ever reported undefined."""
    prefix = serve("mixed_extra")

    _put(f"{prefix}{MIXED_FLOAT_IN}", 2.0)
    assert _read(f"{prefix}{MIXED_FLOAT_OUT}") == pytest.approx(4.0)
    assert _pva_raw(prefix, MIXED_FLOAT_OUT)["value"] == pytest.approx(4.0)

    _put(f"{prefix}{MIXED_FLOAT_IN}", MIXED_SEVERITY_TRIGGER)
    assert _pva_alarm(prefix, MIXED_FLOAT_OUT) == (2, 2)
    assert _severity(f"{prefix}{MIXED_FLOAT_OUT}") == (0, 0)
    assert _severity(f"{prefix}{MIXED_INT}") == (0, 0)
    assert _pva_alarm(prefix, MIXED_INT) == (0, 0)


def test_severity_an_extra_key_does_not_disturb_the_udf_hook(serve) -> None:
    prefix = serve("mixed_udf_extra")

    _put(f"{prefix}{MIXED_FLOAT_IN}", MIXED_SEVERITY_TRIGGER)

    assert _severity(f"{prefix}{MIXED_INT}") == CA_UDF
    assert _pva_alarm(prefix, MIXED_INT) == PVA_UDF
    assert _read(f"{prefix}{MIXED_FLOAT_OUT}") == pytest.approx(2 * MIXED_SEVERITY_TRIGGER)


def test_severity_a_pva_put_to_a_udf_name_keeps_invalid_udf(serve) -> None:
    prefix = serve("mixed_udf")
    _put(f"{prefix}{MIXED_FLOAT_IN}", MIXED_SEVERITY_TRIGGER)

    with Context("pva") as ctx:
        ctx.put(f"{prefix}{MIXED_INT}", 7, timeout=OP_TIMEOUT, wait=True)

    raw = _pva_raw(prefix, MIXED_INT)
    assert raw["value"] == 7
    assert (raw["alarm"]["severity"], raw["alarm"]["status"]) == PVA_UDF


# --------------------------------------------------------------------------
# supports_ca cleared around _add_pv keeps one variable off CA
# --------------------------------------------------------------------------


def test_supports_ca_cleared_serves_pva_only(serve) -> None:
    """The variable added with ``supports_ca`` cleared is PVA-only; the rest keep CA."""
    prefix = serve("pva_only_variable")

    with Context("pva") as ctx:
        assert ctx.get(f"{prefix}{PVA_ONLY}", timeout=OP_TIMEOUT) is not None
        assert ctx.get(f"{prefix}{IN_A}", timeout=OP_TIMEOUT) is not None

    _absent(f"{prefix}{PVA_ONLY}")
    assert _read(f"{prefix}{IN_A}") == pytest.approx(0.0)
    assert _read(f"{prefix}{RESET_CONTROL_PV}") == 0

    # The output pass skips the name CA does not serve and still publishes it
    # over PVA.
    _put(f"{prefix}{IN_A}", 1.5)
    assert _pva_raw(prefix, PVA_ONLY)["value"] == pytest.approx(3.0)
