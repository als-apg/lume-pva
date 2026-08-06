"""Write-path policy tests for :class:`lume_pva_apg.runner.Runner`.

Three policies are exercised, each configuration-selectable and each defaulting
to the behaviour of a runner that sets none of them:

``echo_unconfirmed_writes``
    Whether an input PV advertises a value the model has not accepted.
``update_rate == 0``
    Whether a client's write can be merged into another client's
    ``model.set()``.
``clamp_writes``
    Whether a write is clamped into the variable's ``value_range`` before it
    reaches the model.

Two properties of the transports shape these tests, because they are what makes
the naive assertions wrong:

- A value handed to ``pcaspy.Driver.setParam`` reaches a *monitoring* client
  only when ``updatePV`` is called, but reaches a *one-shot* ``caget``
  immediately, because ``SimplePV.getValue`` reads back through
  ``Driver.read``. The two clients therefore disagree until the commit is
  flushed. Reads here pass ``use_monitor=False`` so they go to the server and
  reflect what the driver actually holds; the one test that cares about the
  flush uses a subscription instead, and says so.
- Channel Access put-completion has no failure channel: ``callbackPV`` ends the
  asynchronous write with ``S_casApp_success`` unconditionally. A refused write
  is reported to the client as a successful put no matter what, which is why
  the policy is to withhold the echo rather than to signal an error.

Runners are started in independent subprocesses so each test gets a fresh
server and a fresh configuration. Each one names its variables uniquely, so no
two tests ever share a PV name and the client's channel cache stays valid for
the whole session. Clearing that cache between tests is the obvious
alternative and is not used here: it detaches and recreates the CA context
underneath PV objects that are still alive, which pyepics documents as a route
to a random SIGSEGV from inside the EPICS libraries, and which reliably crashed
this suite once it grew past a handful of tests.
"""

import itertools
import multiprocessing
import os
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
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
import pcaspy
from lume.model import LUMEModel
from lume.variables import IntVariable, ScalarVariable
from p4p.client.thread import Context

from lume_pva_apg.runner import Runner

# Generous upper bound for any single operation to complete.
OP_TIMEOUT = 10.0
# The value PolicyModel refuses. Inside the variable's value_range, so the
# refusal comes from the model rather than from anything the transport could
# have caught on its own.
REFUSED = 5.5
# Batching window for the test that asserts the default still coalesces. Long
# enough that two back-to-back puts cannot straddle it.
BATCH_WINDOW = 1.0

_MP = multiprocessing.get_context("spawn")
_TAGS = itertools.count()

# The variables PolicyModel serves, before its per-server tag is appended. The
# Runner's own `prefix` would be the natural way to separate one test's PVs
# from another's, but it is applied twice on the CA path -- once into the pvdb
# keys and again by SimpleServer.createPV -- so a prefixed runner serves names
# its own driver cannot resolve. Tagging the variables sidesteps that entirely.
INPUT_A = "input_a"
INPUT_B = "input_b"
INPUT_I = "input_i"
ECHO_ONLY = "echo_only"
SUM_OUTPUT = "sum_output"
BATCH_SIZE = "batch_size"
SET_CALLS = "set_calls"


def _tagged(name: str, tag: str) -> str:
    return f"{name}{tag}"


class PolicyModel(LUMEModel):
    """A model that reports, through its own outputs, how it was driven.

    ``batch_size`` and ``set_calls`` make the shape of each ``model.set()``
    observable over EPICS, so the isolation tests need no shared memory and no
    white-box access to the runner.
    """

    def __init__(self, tag: str, started: mpEvent) -> None:
        self.tag = tag
        self.a = _tagged(INPUT_A, tag)
        self._state: dict[str, Any] = {
            self.a: 0.0,
            _tagged(INPUT_B, tag): 0.0,
            _tagged(INPUT_I, tag): 0,
            _tagged(SUM_OUTPUT, tag): 0.0,
            _tagged(BATCH_SIZE, tag): 0.0,
            _tagged(SET_CALLS, tag): 0.0,
        }
        bounded = {"default_value": 0.0, "value_range": (-10.0, 10.0), "read_only": False}
        self._vars: dict[str, ScalarVariable | IntVariable] = {
            self.a: ScalarVariable(name=self.a, **bounded),
            _tagged(INPUT_B, tag): ScalarVariable(name=_tagged(INPUT_B, tag), **bounded),
            _tagged(INPUT_I, tag): IntVariable(
                name=_tagged(INPUT_I, tag), default_value=0, value_range=(-4, 4), read_only=False
            ),
            # Writable, but the model swallows the write and always reports
            # zero, so the cycle's output pass can never be the source of a
            # non-zero value here. The write path's own echo is the only thing
            # that can put one on this PV.
            _tagged(ECHO_ONLY, tag): ScalarVariable(name=_tagged(ECHO_ONLY, tag), **bounded),
            _tagged(SUM_OUTPUT, tag): ScalarVariable(
                name=_tagged(SUM_OUTPUT, tag), default_value=0.0, read_only=True
            ),
            _tagged(BATCH_SIZE, tag): ScalarVariable(
                name=_tagged(BATCH_SIZE, tag), default_value=0.0, read_only=True
            ),
            _tagged(SET_CALLS, tag): ScalarVariable(
                name=_tagged(SET_CALLS, tag), default_value=0.0, read_only=True
            ),
        }
        self.started = started

    @property
    def supported_variables(self) -> dict[str, ScalarVariable | IntVariable]:
        return self._vars

    def _get(self, names: list[str]) -> dict[str, Any]:
        # echo_only is never stored, so it always reads back as zero.
        return {n: self._state.get(n, 0.0) for n in names}

    def _set(self, values: dict[str, Any]) -> None:
        if values.get(self.a) == REFUSED:
            raise RuntimeError(f"model refuses {REFUSED}")
        if values:
            self._state[_tagged(SET_CALLS, self.tag)] += 1
            self._state[_tagged(BATCH_SIZE, self.tag)] = max(
                self._state[_tagged(BATCH_SIZE, self.tag)], float(len(values))
            )
        self._state.update({k: v for k, v in values.items() if k in self._state})
        self._state[_tagged(SUM_OUTPUT, self.tag)] = self._state[self.a] * 2.0
        self.started.set()

    def reset(self) -> None:
        for key in self._state:
            self._state[key] = 0 if key.startswith(INPUT_I) else 0.0
        self.started.set()


def _serve(tag: str, overrides: dict[str, Any], started: mpEvent, ready: mpEvent) -> None:
    """Child-process entry point: serve a PolicyModel with `overrides` applied.

    Must be importable at module top level so the ``spawn`` start method can
    locate it. Blocks forever once ready; the parent terminates the process.
    """
    model = PolicyModel(tag, started)
    config = Runner.generate_config(model)
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
def serve() -> Generator[Callable[..., Callable[[str], str]], None, None]:
    """Yield a factory that starts a configured Runner.

    The factory returns the name mapper for that server: it turns a base
    variable name into the PV this particular test's runner serves it as.
    """
    procs: list[Any] = []

    def _start(**overrides: Any) -> Callable[[str], str]:
        tag = f"_wp{next(_TAGS)}"
        started = _MP.Event()
        ready = _MP.Event()
        proc = _MP.Process(target=_serve, args=(tag, overrides, started, ready), daemon=True)
        proc.start()
        procs.append(proc)
        assert ready.wait(timeout=OP_TIMEOUT), "child Runner never became ready"
        return lambda name: _tagged(name, tag)

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


def _severity(name: str) -> tuple[int, int]:
    """Return (severity, status) as the server currently reports them."""
    pv = epics.get_pv(name, timeout=OP_TIMEOUT)
    pv.get(use_monitor=False, timeout=OP_TIMEOUT)
    return pv.severity, pv.status


def _put(name: str, value: Any) -> None:
    """Issue a completion-aware caput and assert the client was not left hanging."""
    rc = epics.caput(name, value, wait=True, timeout=OP_TIMEOUT)
    assert rc == 1, f"caput on {name} did not complete (rc={rc})"


@contextmanager
def _monitored(name: str) -> Generator[Any, None, None]:
    """A PV of our own with a live subscription, torn down deterministically.

    pyepics finalises a PV from ``__del__``, which clears the subscription
    through libca. A PV built directly rather than through the channel cache
    has to be disconnected explicitly rather than left to the collector.
    """
    pv = epics.PV(name, auto_monitor=True)
    try:
        assert pv.wait_for_connection(timeout=OP_TIMEOUT), f"{name} never connected"
        yield pv
    finally:
        pv.clear_auto_monitor()
        pv.disconnect()


# --------------------------------------------------------------------------
# (a) the echo may not advertise a value the model has not accepted
# --------------------------------------------------------------------------


def test_ca_echo_is_withheld_when_the_model_refuses(serve) -> None:
    p = serve(echo_unconfirmed_writes=False)

    _put(p("input_a"), 4.2)
    assert _read(p("input_a")) == pytest.approx(4.2)
    assert _read(p("sum_output")) == pytest.approx(8.4)

    # Refused by the model, but the client is still told the put completed --
    # put-completion has no way to say anything else.
    _put(p("input_a"), REFUSED)

    # The model kept its previous value, and so must the PV.
    assert _read(p("sum_output")) == pytest.approx(8.4)
    assert _read(p("input_a")) == pytest.approx(4.2)


def test_ca_echo_stands_on_refusal_by_default(serve) -> None:
    """The default reproduces the behaviour of a runner with no policy set."""
    p = serve()

    _put(p("input_a"), 4.2)
    _put(p("input_a"), REFUSED)

    # The model refused, so the readback is unchanged...
    assert _read(p("sum_output")) == pytest.approx(8.4)
    # ...but the input PV carries the value the client asked for regardless.
    assert _read(p("input_a")) == pytest.approx(REFUSED)


def test_ca_echo_still_lands_when_the_model_accepts(serve) -> None:
    """Withholding on refusal must not withhold on success."""
    p = serve(echo_unconfirmed_writes=False)

    _put(p("input_a"), -3.5)

    assert _read(p("input_a")) == pytest.approx(-3.5)
    assert _read(p("sum_output")) == pytest.approx(-7.0)


def test_ca_put_completion_follows_the_commit(serve) -> None:
    """A completed put must leave the committed value already published.

    The model swallows ``echo_only`` and reports zero for it, so the cycle's
    output pass publishes zero and the write path's echo is the only source of
    the written value. Both the value store and the monitor stream must carry
    it by the time the client is released: the monitoring client below never
    goes back to the server, so it holds the value only if ``updatePV`` ran
    before ``callbackPV``.
    """
    p = serve(echo_unconfirmed_writes=False)

    with _monitored(p("echo_only")) as monitored:
        assert monitored.get(timeout=OP_TIMEOUT) == pytest.approx(0.0)

        _put(p("echo_only"), 6.0)

        # Committed to the value store...
        assert _read(p("echo_only")) == pytest.approx(6.0)
        # ...and flushed to monitors, both before the put was signalled
        # complete. This read never goes back to the server.
        assert monitored.value == pytest.approx(6.0)


def test_ca_refused_write_raises_an_alarm_when_enabled(serve) -> None:
    p = serve(echo_unconfirmed_writes=False, alarm_on_refused_write=True)

    _put(p("input_a"), 4.2)
    assert _severity(p("input_a")) == (0, 0)

    _put(p("input_a"), REFUSED)
    assert _severity(p("input_a")) == (
        pcaspy.Severity.INVALID_ALARM,
        pcaspy.Alarm.WRITE_ALARM,
    )

    # An accepted write clears it again.
    _put(p("input_a"), 1.0)
    assert _severity(p("input_a")) == (0, 0)


def test_ca_refused_write_raises_no_alarm_by_default(serve) -> None:
    p = serve(echo_unconfirmed_writes=False)

    _put(p("input_a"), 4.2)
    _put(p("input_a"), REFUSED)

    assert _severity(p("input_a")) == (0, 0)


def test_pva_echo_is_withheld_when_the_model_refuses(serve) -> None:
    p = serve(echo_unconfirmed_writes=False)

    with Context("pva") as ctx:
        ctx.put(p("input_a"), 4.2, timeout=OP_TIMEOUT, wait=True)
        assert float(ctx.get(p("input_a"), timeout=OP_TIMEOUT)) == pytest.approx(4.2)

        # PVAccess, unlike Channel Access, can report the failure -- but that
        # alone never stopped the echo from standing.
        with pytest.raises(Exception, match=f"refuses {REFUSED}"):
            ctx.put(p("input_a"), REFUSED, timeout=OP_TIMEOUT, wait=True)

        assert float(ctx.get(p("input_a"), timeout=OP_TIMEOUT)) == pytest.approx(4.2)
        assert float(ctx.get(p("sum_output"), timeout=OP_TIMEOUT)) == pytest.approx(8.4)


def test_pva_echo_stands_on_refusal_by_default(serve) -> None:
    p = serve()

    with Context("pva") as ctx:
        ctx.put(p("input_a"), 4.2, timeout=OP_TIMEOUT, wait=True)
        with pytest.raises(Exception, match=f"refuses {REFUSED}"):
            ctx.put(p("input_a"), REFUSED, timeout=OP_TIMEOUT, wait=True)

        assert float(ctx.get(p("sum_output"), timeout=OP_TIMEOUT)) == pytest.approx(8.4)
        assert float(ctx.get(p("input_a"), timeout=OP_TIMEOUT)) == pytest.approx(REFUSED)


# --------------------------------------------------------------------------
# (b) one client's write may not be merged into another's model.set()
# --------------------------------------------------------------------------


def test_zero_update_rate_gives_one_write_per_model_set(serve) -> None:
    p = serve(update_rate=0.0)

    baseline = _read(p("set_calls"))

    # Two writes to two different variables, back to back and without waiting.
    # Both are on the queue well inside any batching window.
    epics.caput(p("input_a"), 1.0, wait=False)
    _put(p("input_b"), 2.0)

    # Each write drove a model.set() of its own...
    assert _read(p("set_calls")) == pytest.approx(baseline + 2)
    # ...and no model.set() ever carried more than one variable.
    assert _read(p("batch_size")) == pytest.approx(1.0)
    # Both landed.
    assert _read(p("sum_output")) == pytest.approx(2.0)
    assert _read(p("input_b")) == pytest.approx(2.0)


def test_nonzero_update_rate_still_coalesces(serve) -> None:
    """The batching window is the default and is left alone."""
    p = serve(update_rate=BATCH_WINDOW)

    epics.caput(p("input_a"), 1.0, wait=False)
    _put(p("input_b"), 2.0)

    assert _read(p("batch_size")) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# (c) clamping an incoming write into the variable's value_range
# --------------------------------------------------------------------------


def test_ca_write_is_clamped_when_enabled(serve) -> None:
    p = serve(clamp_writes=True, echo_unconfirmed_writes=False)

    _put(p("input_a"), 50.0)

    # The model was given the clamped value...
    assert _read(p("sum_output")) == pytest.approx(20.0)
    # ...and the client is echoed the same value the model was given.
    assert _read(p("input_a")) == pytest.approx(10.0)

    _put(p("input_a"), -50.0)
    assert _read(p("sum_output")) == pytest.approx(-20.0)
    assert _read(p("input_a")) == pytest.approx(-10.0)

    # In-range writes are untouched.
    _put(p("input_a"), 2.5)
    assert _read(p("input_a")) == pytest.approx(2.5)


def test_ca_write_is_not_clamped_by_default(serve) -> None:
    p = serve()

    _put(p("input_a"), 50.0)

    # LUMEModel.set does not enforce value_range, so an out-of-range value
    # reaches the model untouched. That is the behaviour the default keeps.
    assert _read(p("sum_output")) == pytest.approx(100.0)


def test_pva_write_is_clamped_when_enabled(serve) -> None:
    p = serve(clamp_writes=True, echo_unconfirmed_writes=False)

    with Context("pva") as ctx:
        ctx.put(p("input_a"), 50.0, timeout=OP_TIMEOUT, wait=True)
        assert float(ctx.get(p("sum_output"), timeout=OP_TIMEOUT)) == pytest.approx(20.0)
        assert float(ctx.get(p("input_a"), timeout=OP_TIMEOUT)) == pytest.approx(10.0)


def test_pva_write_is_not_clamped_by_default(serve) -> None:
    p = serve()

    with Context("pva") as ctx:
        ctx.put(p("input_a"), 50.0, timeout=OP_TIMEOUT, wait=True)
        assert float(ctx.get(p("sum_output"), timeout=OP_TIMEOUT)) == pytest.approx(100.0)


def test_integer_write_stays_integral_when_clamped(serve) -> None:
    p = serve(clamp_writes=True, echo_unconfirmed_writes=False)

    _put(p("input_i"), 99)

    value = _read(p("input_i"))
    assert value == 4
    assert isinstance(value, int)


# --------------------------------------------------------------------------
# clamp policy, without a server
# --------------------------------------------------------------------------


def _clamp_stub(clamp_writes: bool = True) -> Runner:
    runner = Runner.__new__(Runner)
    runner.clamp_writes = clamp_writes
    return runner


@pytest.mark.parametrize(
    ("value_range", "value", "expected"),
    [
        ((-10.0, 10.0), 50.0, 10.0),
        ((-10.0, 10.0), -50.0, -10.0),
        ((-10.0, 10.0), 2.5, 2.5),
        ((-10.0, 10.0), -10.0, -10.0),
        ((-4, 4), 99, 4),
        ((-4, 4), -99, -4),
    ],
)
def test_clamp_scalar(value_range: tuple, value: Any, expected: Any) -> None:
    var = ScalarVariable(name="v", default_value=0.0, value_range=value_range)
    assert _clamp_stub()._clamp_scalar(var, value) == expected


def test_clamp_scalar_leaves_unrangeable_values_alone() -> None:
    runner = _clamp_stub()
    var = ScalarVariable(name="v", default_value=0.0, value_range=(-10.0, 10.0))

    # Not a number: nothing sensible to clamp against.
    assert runner._clamp_scalar(var, "50") == "50"
    # bool is a Real, but clamping it into a numeric range is meaningless.
    assert runner._clamp_scalar(var, True) is True

    # No range at all.
    unbounded = ScalarVariable(name="u", default_value=0.0)
    assert runner._clamp_scalar(unbounded, 1e9) == 1e9


def test_clamp_write_is_a_no_op_when_disabled() -> None:
    runner = _clamp_stub(clamp_writes=False)
    var = ScalarVariable(name="v", default_value=0.0, value_range=(-10.0, 10.0))

    assert runner._clamp_write(var, 50.0) == 50.0


# --------------------------------------------------------------------------
# commit-before-signal ordering, without a server
# --------------------------------------------------------------------------


class _RecordingDriver(Runner.CaDriver):
    """A CaDriver that records its calls instead of reaching a pcaspy server.

    ``pcaspy.Driver.__init__`` registers with the server's PV manager, so it is
    deliberately not called here: the point of this driver is to observe the
    order in which :meth:`Runner.CaDriver.write` drives the driver API, with no
    server, no ports and no timing involved.
    """

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.calls: list[tuple] = []

    def setParam(self, reason, value, timestamp=None) -> None:
        self.calls.append(("setParam", reason, value))

    def setParamStatus(self, reason, alarm=None, severity=None) -> None:
        self.calls.append(("setParamStatus", reason, alarm, severity))

    def updatePV(self, reason) -> None:
        self.calls.append(("updatePV", reason))

    def callbackPV(self, reason) -> None:
        self.calls.append(("callbackPV", reason))


def _driver_stub(**policy: Any) -> tuple[_RecordingDriver, list]:
    """A driver wired to a Runner stub, plus the list of captured completions."""
    runner = Runner.__new__(Runner)
    # No tag: this model is never served, so its variables keep their base names.
    runner.model = PolicyModel("", _MP.Event())
    runner.reset_control_pv = "RESET"
    runner.pv_to_var = {INPUT_A: INPUT_A}
    runner.pvdb = {INPUT_A: {"type": "float"}}
    runner.echo_unconfirmed_writes = policy.get("echo_unconfirmed_writes", True)
    runner.alarm_on_refused_write = policy.get("alarm_on_refused_write", False)
    runner.clamp_writes = policy.get("clamp_writes", False)

    completions: list = []
    runner._enqueue = lambda values, done=None, reset=False: completions.append(done)

    return _RecordingDriver(runner), completions


def test_commit_is_flushed_before_put_completion_is_signalled() -> None:
    driver, completions = _driver_stub()

    assert driver.write("input_a", 4.2) is True
    # Nothing is driven at write time: the write is queued, and the value has
    # not reached the model yet.
    assert driver.calls == []

    completions[0](None)

    assert driver.calls == [
        ("setParam", "input_a", 4.2),
        ("updatePV", "input_a"),
        ("callbackPV", "input_a"),
    ]


def test_refusal_withholds_the_echo_but_still_completes_the_put() -> None:
    driver, completions = _driver_stub(echo_unconfirmed_writes=False)

    driver.write("input_a", REFUSED)
    completions[0]("model refused it")

    # No value was published, and the client was still released -- an
    # asynchronous write left dangling would block every later write to this
    # PV, since pcaspy postpones a second write while one is in flight.
    assert driver.calls == [
        ("updatePV", "input_a"),
        ("callbackPV", "input_a"),
    ]


def test_refusal_echoes_and_completes_by_default() -> None:
    """The default is the behaviour of a runner with no policy set.

    The refused value is recorded but deliberately not flushed, which is what
    keeps it off a monitoring client's stream exactly as before.
    """
    driver, completions = _driver_stub()

    driver.write("input_a", REFUSED)
    completions[0]("model refused it")

    assert driver.calls == [
        ("setParam", "input_a", REFUSED),
        ("callbackPV", "input_a"),
    ]


def test_refusal_alarms_after_withholding_the_echo() -> None:
    driver, completions = _driver_stub(echo_unconfirmed_writes=False, alarm_on_refused_write=True)

    driver.write("input_a", REFUSED)
    completions[0]("model refused it")

    assert driver.calls == [
        ("setParamStatus", "input_a", pcaspy.Alarm.WRITE_ALARM, pcaspy.Severity.INVALID_ALARM),
        ("updatePV", "input_a"),
        ("callbackPV", "input_a"),
    ]


def test_alarm_alone_is_flushed_even_with_the_historical_echo() -> None:
    """An alarm nobody is told about is not an alarm."""
    driver, completions = _driver_stub(alarm_on_refused_write=True)

    driver.write("input_a", REFUSED)
    completions[0]("model refused it")

    assert driver.calls == [
        ("setParam", "input_a", REFUSED),
        ("setParamStatus", "input_a", pcaspy.Alarm.WRITE_ALARM, pcaspy.Severity.INVALID_ALARM),
        ("updatePV", "input_a"),
        ("callbackPV", "input_a"),
    ]


def test_accepted_write_raises_no_alarm_even_with_alarms_enabled() -> None:
    driver, completions = _driver_stub(echo_unconfirmed_writes=False, alarm_on_refused_write=True)

    driver.write("input_a", 4.2)
    completions[0](None)

    assert driver.calls == [
        ("setParam", "input_a", 4.2),
        ("updatePV", "input_a"),
        ("callbackPV", "input_a"),
    ]
