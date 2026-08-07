"""
End-to-end tests for :class:`lume_pva_apg.runner.Runner`.
Runner interactions here should be performed using EPICS

Put completion test details:

- Relies on multiprocessing.Event gates to track the exact state of the model
in the Runner.
- Rough flow:
    1. Issue the put on a background thread.
    2. Wait until the simulation has entered ``_set`` (so the value was delivered
    and dequeued by the runner).
    3. While the gate is still closed, assert the put has **not** returned -- this
    is the property that breaks if put-completion is removed (the client would
    be signalled immediately).
    4. Open the gate, let the simulation finish, and assert the put now returns.
    5. Confirm the output PV already holds the freshly computed value, proving the
    put waited for the whole cycle

Two EPICS providers are exercised:
- PVA via ``p4p``
- CA via ``pyepics``

Runners are started up in independent subprocesses to ensure each test gets a
fresh Runner.
"""

import multiprocessing
import os
import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass
from multiprocessing.synchronize import Event as mpEvent
from typing import Any

import pytest
from lume.exceptions import ReadOnlyError
from lume.variables.variable import ConfigEnum, Variable

from lume_pva_apg.tests._requires import skip_if_absent

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
    from lume.model import LUMEModel
    from lume.variables import ScalarVariable
    from p4p.client.thread import Context

    from lume_pva_apg.runner import Runner
except ImportError as exc:
    skip_if_absent(exc)

# Generous upper bound for any single operation to complete.
OP_TIMEOUT = 10.0
# Window during which a blocked put is observed to *not* complete. The gate is
# genuinely closed for this whole window, so the simulation cannot finish and a
# correct put cannot return -- this is a lower bound, not a race.
BLOCK_WINDOW = 0.5

# pin process start method for consistency
_MP = multiprocessing.get_context("spawn")


def read_ca(pvname: str) -> float:
    """Read ``pvname`` off the wire, never out of pyepics' monitor cache.

    ``caget`` defaults to ``use_monitor=True``, which returns the value of the
    last monitor callback this client happened to have processed. On a channel
    an earlier read already subscribed to, that can be a *stale* value: a put
    completes, and the read that follows still reports the previous value
    because the monitor event has not been dispatched yet. The put is not at
    fault and retrying hides it, so every value asserted on here is read
    explicitly instead.
    """
    return float(epics.caget(pvname, timeout=OP_TIMEOUT, use_monitor=False))


class GatedModel(LUMEModel):
    """
    A model whose simulation blocks until the test opens the gate.

    ``sum_output`` is computed as ``2 * input_a`` so that a completed put leaves
    an unambiguous, checkable value on the output PV.
    """

    def __init__(
        self,
        release: mpEvent,
        entered: mpEvent,
        completed: mpEvent,
    ) -> None:
        self._state = {"input_a": 0.0, "sum_output": 0.0}
        self._vars = {
            "input_a": ScalarVariable(
                name="input_a",
                default_value=0.0,
                value_range=(-1e6, 1e6),
                read_only=False,
            ),
            "sum_output": ScalarVariable(
                name="sum_output",
                default_value=0.0,
                read_only=True,
            ),
        }
        # Set => _set proceeds. Cleared => _set blocks.
        self.release = release
        # Set when the simulation enters _set (value delivered + dequeued).
        self.entered = entered
        # Set when the simulation finishes a cycle.
        self.completed = completed

    @property
    def supported_variables(self) -> dict[str, ScalarVariable]:
        return self._vars

    def _get(self, names: list[str]) -> dict[str, float]:
        return {name: self._state[name] for name in names}

    def set(self, values: dict[str, Any]) -> None:
        # vendor LUMEModel.set, but error if validation fails
        # Validate input values
        for name in values.keys():
            if name not in self.supported_variables:
                raise ValueError(f"Variable '{name}' is not supported by the model.")
            else:
                variable = self.supported_variables[name]
                if not isinstance(variable, Variable):
                    raise ValueError(f"Variable '{name}' is not a valid Variable instance.")

                if variable.read_only:
                    raise ReadOnlyError(f"Variable '{name}' is read-only. Cannot be set.")
                try:
                    variable.validate_value(values[name], config=ConfigEnum.ERROR)
                except (ValueError, TypeError) as exc:
                    raise type(exc)(f"Validation failed for variable '{name}': {exc}") from exc

        # Set the control parameters of the simulator
        self._set(values)

    def _set(self, values: dict[str, float]) -> None:
        self.entered.set()
        if not self.release.wait(timeout=OP_TIMEOUT):
            raise TimeoutError("simulation gate was never opened")
        self._state.update(values)
        self._state["sum_output"] = self._state["input_a"] * 2.0
        self.completed.set()

    def reset(self) -> None:
        # Implemented to appease type-checker
        self._state = {"input_a": 0.0, "sum_output": 0.0}


def _serve(
    release: mpEvent,
    entered: mpEvent,
    completed: mpEvent,
    ready: mpEvent,
    echo_unconfirmed_writes: bool = True,
) -> None:
    """Child-process entry point: serve a gated model over CA and PVA.

    Must be importable at module top level so the ``spawn`` start method can
    locate it. Blocks forever once ready; the parent terminates the process.
    """
    model = GatedModel(release, entered, completed)
    config = Runner.generate_config(model)
    # No batching delay -- the cycle is driven purely by the gate.
    config["update_rate"] = 0.0
    config["echo_unconfirmed_writes"] = echo_unconfirmed_writes

    # Let the implicit startup cycle (Runner.__init__ enqueues an empty update)
    # pass freely before the parent arms the gate.
    release.set()

    runner = Runner(model=model, config=config)
    threading.Thread(target=runner._run, daemon=True).start()

    if not completed.wait(timeout=OP_TIMEOUT):
        raise RuntimeError("startup cycle never ran in child")

    # Server startup cycle complete
    ready.set()
    # Mutate state directly for cache testing (thread safety be damned)
    model._state = {"input_a": 1.0, "sum_output": -20.0}
    # block until terminated
    threading.Event().wait()


@dataclass
class RunnerHandle:
    release: mpEvent
    entered: mpEvent
    completed: mpEvent
    # The echo mode the served Runner was configured with, so a test can
    # assert the contract that mode actually promises.
    echo_unconfirmed_writes: bool


@pytest.fixture(scope="function")
def harness(request: pytest.FixtureRequest) -> Generator[RunnerHandle, None, None]:
    """
    Run a Runner in a child process and yield the shared gate + a PVA client.

    Each test gets a pristine child, and teardown reclaims the EPICS ports so
    tests stay independent.

    The returned harness provides `multiprocessing.Event` for inspecting the
    state of the model internals
    """
    release = _MP.Event()
    entered = _MP.Event()
    completed = _MP.Event()
    ready = _MP.Event()

    # Indirect parametrization selects the echo mode; unparametrized uses of
    # the fixture serve the Runner default (echo_unconfirmed_writes=True).
    echo_unconfirmed_writes = getattr(request, "param", True)

    proc = _MP.Process(
        target=_serve,
        args=(release, entered, completed, ready, echo_unconfirmed_writes),
        daemon=True,
    )
    proc.start()
    assert ready.wait(timeout=OP_TIMEOUT), "child Runner never became ready"

    handle = RunnerHandle(
        release=release,
        entered=entered,
        completed=completed,
        echo_unconfirmed_writes=echo_unconfirmed_writes,
    )
    try:
        yield handle
    finally:
        release.set()
        proc.terminate()
        proc.join(timeout=OP_TIMEOUT)
        # Drop pyepics channels bound to the now-dead server so the next test
        # connects fresh rather than to a stale, disconnected channel.
        epics.ca.clear_cache()


def clear_harness(harness: RunnerHandle) -> None:
    """Close gates so the next simulation cycle blocks until released."""
    harness.entered.clear()
    harness.completed.clear()
    harness.release.clear()


def _assert_put_completion(harness: RunnerHandle, putter: Callable[[], None]) -> None:
    """Run ``putter`` on a thread and assert it blocks until the sim completes.

    ``putter`` must perform a blocking, completion-aware put (PVA put, or
    ``caput(wait=True)``).
    """
    clear_harness(harness)

    put_returned = threading.Event()
    errors: list[BaseException] = []

    def _do_put() -> None:
        try:
            putter()
        except BaseException as exc:
            errors.append(exc)
        finally:
            put_returned.set()

    thread = threading.Thread(target=_do_put, daemon=True)
    thread.start()

    # The value reached the runner and the simulation is now running...
    assert harness.entered.wait(timeout=OP_TIMEOUT), "simulation never started"

    # ...so a completion-aware put must still be blocked. If this fires, the
    # client was signalled before the simulation finished (put-completion bug).
    assert not put_returned.wait(timeout=BLOCK_WINDOW), (
        "put reported completion before the simulation finished"
    )
    assert thread.is_alive()

    # Let the simulation finish; the put must now complete.
    harness.release.set()
    assert put_returned.wait(timeout=OP_TIMEOUT), "put never completed"
    # Finish the put action.  Thread will fail to join if put does not complete
    thread.join(timeout=OP_TIMEOUT)
    assert not errors, f"put raised: {errors[0]!r}"


def test_pva_put_waits_for_simulation(harness: RunnerHandle) -> None:
    with Context("pva") as ctx:
        _assert_put_completion(
            harness, lambda: ctx.put("input_a", 5.0, timeout=OP_TIMEOUT, wait=True)
        )

        assert harness.completed.is_set()
        assert float(ctx.get("sum_output", timeout=OP_TIMEOUT)) == pytest.approx(10.0)


def test_ca_put_waits_for_simulation(harness: RunnerHandle) -> None:
    # caput timeout needs to essentially run forever, to `_assert_put_completion` to check
    # if the thread has completed.
    _assert_put_completion(
        harness, lambda: epics.caput("input_a", 7.0, wait=True, timeout=10 * OP_TIMEOUT)
    )

    assert harness.completed.is_set()
    assert read_ca("sum_output") == pytest.approx(14.0)


def test_standard_sim(harness: RunnerHandle):
    # attempt set out of bounds, no need to wait
    epics.caput("input_a", 10.0)

    # assert model set has completed
    assert harness.completed.is_set()
    assert read_ca("sum_output") == pytest.approx(20.0)
    assert read_ca("input_a") == pytest.approx(10.0)


@pytest.mark.parametrize(
    "harness",
    [True, False],
    indirect=True,
    ids=["echo-unconfirmed", "echo-withheld"],
)
def test_failed_sim(harness: RunnerHandle):
    # Assert initial state
    assert read_ca("input_a") == pytest.approx(0.0)
    assert read_ca("sum_output") == pytest.approx(0.0)

    # Reasonable input. Clear the completion event first so the wait below can
    # only be satisfied by the cycle this put triggers, not a leftover from
    # startup.
    harness.completed.clear()
    epics.caput("input_a", 4.2, wait=True)
    assert harness.completed.wait(timeout=OP_TIMEOUT)
    assert read_ca("input_a") == pytest.approx(4.2)
    assert read_ca("sum_output") == pytest.approx(8.4)

    # Attempt a set out of bounds. The model refuses it and the runner reverts
    # to the cached pre-put state -- and that revert is itself a full gated
    # model.set() cycle, so waiting on `completed` observes the revert
    # finishing before anything is asserted.
    harness.completed.clear()
    epics.caput("input_a", 7.0e6, wait=True)
    assert harness.completed.wait(timeout=OP_TIMEOUT)

    # What the input PV now reads off the wire is the echo contract, not the
    # model state. The default (echo_unconfirmed_writes=True) posts the
    # requested value when the write is queued and leaves it standing even
    # though the cycle that consumed it failed; echo_unconfirmed_writes=False
    # withholds the echo, so the PV keeps the last value the model accepted.
    if harness.echo_unconfirmed_writes:
        assert read_ca("input_a") == pytest.approx(7.0e6)
    else:
        assert read_ca("input_a") == pytest.approx(4.2)

    # In both modes the failed cycle posts no outputs and the model reverts,
    # so the output PV still carries the last good cycle's value: the runner
    # is still operational.
    assert read_ca("sum_output") == pytest.approx(8.4)

    # And a subsequent valid write goes through normally on the same runner.
    harness.completed.clear()
    epics.caput("input_a", 2.0, wait=True)
    assert harness.completed.wait(timeout=OP_TIMEOUT)
    assert read_ca("input_a") == pytest.approx(2.0)
    assert read_ca("sum_output") == pytest.approx(4.0)


def test_pva_reset_calls_model_reset(harness: RunnerHandle) -> None:
    # Keep the model gate open so regular put/get traffic can flow freely.
    harness.release.set()

    with Context("pva") as ctx:
        ctx.put("input_a", 5.0, timeout=OP_TIMEOUT, wait=True)
        assert float(ctx.get("input_a", timeout=OP_TIMEOUT)) == pytest.approx(5.0)
        assert float(ctx.get("sum_output", timeout=OP_TIMEOUT)) == pytest.approx(10.0)

        # Any write to RESET must invoke model.reset() and publish the reset state.
        harness.completed.clear()
        ctx.put("RESET", 0, timeout=OP_TIMEOUT, wait=True)
        assert harness.completed.wait(timeout=OP_TIMEOUT)

        assert float(ctx.get("input_a", timeout=OP_TIMEOUT)) == pytest.approx(0.0)
        assert float(ctx.get("sum_output", timeout=OP_TIMEOUT)) == pytest.approx(0.0)
