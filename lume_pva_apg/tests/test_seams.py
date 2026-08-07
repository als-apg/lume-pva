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

import itertools
import multiprocessing
import os
import threading
from collections.abc import Callable, Generator
from multiprocessing.synchronize import Event as mpEvent
from typing import Any

import pytest

# Keep all EPICS traffic on the loopback interface. Must be set before p4p,
# pyepics, or the pcaspy server (created in Runner.__init__) initialise.
os.environ.setdefault("EPICS_CA_ADDR_LIST", "127.0.0.1")
os.environ.setdefault("EPICS_CA_AUTO_ADDR_LIST", "NO")
os.environ.setdefault("EPICS_PVA_ADDR_LIST", "127.0.0.1")
os.environ.setdefault("EPICS_PVA_AUTO_ADDR_LIST", "NO")

import epics
from lume.model import LUMEModel
from lume.variables import ScalarVariable
from p4p.client.thread import Context
from p4p.client.thread import TimeoutError as PvaTimeoutError

from lume_pva_apg.runner import RESET_CONTROL_PV, Runner

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


_RUNNERS: dict[str, type[Runner]] = {
    "stock": Runner,
    "extend_only": ExtendOnlyRunner,
    "seam": SeamRunner,
}


def _serve(
    prefix: str,
    runner_key: str,
    overrides: dict[str, Any],
    started: mpEvent,
    ready: mpEvent,
) -> None:
    """Child-process entry point: serve a SeamModel under `prefix`.

    Must be importable at module top level so the ``spawn`` start method can
    locate it. Blocks forever once ready; the parent terminates the process.
    """
    model = SeamModel(started)
    config = Runner.generate_config(model, prefix=prefix)
    config["update_rate"] = 0.0
    config.update(overrides)

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
    """Yield a factory that starts a configured Runner and returns its prefix."""
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
        assert ready.wait(timeout=OP_TIMEOUT), "child Runner never became ready"
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
