"""Tests for the names and the metadata a :class:`lume_pva_apg.runner.Runner`
puts on the wire.

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


def _serve(prefix: str, overrides: dict[str, Any], started: mpEvent, ready: mpEvent) -> None:
    """Child-process entry point: serve a SeamModel under `prefix`.

    Must be importable at module top level so the ``spawn`` start method can
    locate it. Blocks forever once ready; the parent terminates the process.
    """
    model = SeamModel(started)
    config = Runner.generate_config(model, prefix=prefix)
    config["update_rate"] = 0.0
    config.update(overrides)

    runner = Runner(model=model, config=config)
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

    def _start(**overrides: Any) -> str:
        prefix = f"SEAM{next(_TAGS)}:"
        started = _MP.Event()
        ready = _MP.Event()
        proc = _MP.Process(target=_serve, args=(prefix, overrides, started, ready), daemon=True)
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
