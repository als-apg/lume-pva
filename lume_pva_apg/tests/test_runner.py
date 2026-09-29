"""Tests for lume_pva_apg.runner configuration.

Two things are covered: the configuration ``Runner.generate_config`` produces,
and what the constructor rejects when it is handed one by hand -- which is the
supported way to use it, since the generated configuration is documented as
something to edit before passing on.

Both run in-process against a stub model. ``Runner.__init__`` validates the
whole ``variables`` table before it creates either server, so a configuration
the constructor rejects binds no port; nothing here starts a server or makes a
network call.
"""

import time
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from lume_pva_apg.tests._requires import skip_if_absent

# No server is started here, but importing the runner still needs both
# transports. Guarded so an install missing one of them skips this file rather
# than failing collection, which would take the whole suite with it.
try:
    from lume.model import LUMEModel
    from lume.variables import (
        BoolVariable,
        EnumVariable,
        IntVariable,
        NDVariable,
        ParticleGroupVariable,
        ScalarVariable,
        StrVariable,
        Variable,
    )
    from p4p.server.thread import SharedPV

    from lume_pva_apg.runner import _CONDITIONS, Runner, _resolve_pv_meta
    from lume_pva_apg.tests._mixed_model import (
        FAIL_MALFORMED,
        FAIL_RAISE,
        FAIL_UDF,
        MIXED_BOOL,
        MIXED_ENUM,
        MIXED_EXTRA_KEY,
        MIXED_FLOAT_IN,
        MIXED_FLOAT_OUT,
        MIXED_INT,
        MIXED_SEVERITY_TRIGGER,
        MIXED_STR,
        MixedModel,
    )
    from lume_pva_apg.variables import (
        TORCH_AVAILABLE,
        TorchNDVariable,
        TorchScalarVariable,
        find_variable_handler,
    )
except ImportError as exc:
    skip_if_absent(exc)


class StubModel:
    """Minimal stand-in for a LUMEModel: generate_config only reads
    supported_variables."""

    def __init__(self, variables: dict[str, Variable]) -> None:
        self.supported_variables = variables


@pytest.fixture
def model() -> StubModel:
    return StubModel(
        {
            "input_a": ScalarVariable(name="input_a"),
            "output_b": ScalarVariable(name="output_b", read_only=True),
            "image": NDVariable(name="image", shape=(4, 4), dtype=np.float64, read_only=True),
        }
    )


def test_runner_defaults(model: StubModel) -> None:
    config = Runner.generate_config(model)

    for name, var_config in config["variables"].items():
        assert var_config["name"] == name
        assert var_config["pv"] == name

    assert set(config["variables"].keys()) == {"input_a", "output_b", "image"}
    # only one read write
    assert config["variables"]["input_a"]["mode"] == "rw"
    assert config["variables"]["output_b"]["mode"] == "ro"
    assert config["variables"]["image"]["mode"] == "ro"

    # No prefix
    assert config["prefix"] == ""


def test_set_prefix(model: StubModel) -> None:
    config = Runner.generate_config(model, prefix="TEST:")

    assert config["prefix"] == "TEST:"


def test_pv_name_transformer(model: StubModel) -> None:
    config = Runner.generate_config(
        model, name_transformer=lambda var, name: f"XFORM:{name.upper()}"
    )
    assert config["variables"]["input_a"]["pv"] == "XFORM:INPUT_A"
    # Variable names must remain untouched — only the PV name changes
    assert config["variables"]["input_a"]["name"] == "input_a"


def test_no_variables() -> None:
    empty_model = StubModel({})
    config = Runner.generate_config(empty_model)

    assert config["variables"] == {}


def test_a_variable_the_model_does_not_have_is_rejected(model: StubModel) -> None:
    """A configuration is edited by hand, so a name in it can be a typo.

    Accepted, it serves a PV backed by nothing: reads answer with the type's
    default and writes are silently discarded.
    """
    config = Runner.generate_config(model)
    config["variables"]["input_a"]["name"] = "input_typo"

    with pytest.raises(KeyError, match="input_typo"):
        Runner(model=model, config=config)


def test_an_unknown_pv_mode_is_rejected(model: StubModel) -> None:
    """Anything but 'rw' or 'ro'. Not defaulted: 'r', 'readonly' and 'w' all
    read as an intent to restrict, and defaulting them to the variable's own
    permission serves a writable PV to someone who asked for a read-only one."""
    config = Runner.generate_config(model)
    config["variables"]["input_a"]["mode"] = "readonly"

    with pytest.raises(KeyError, match="readonly"):
        Runner(model=model, config=config)


def test_a_writable_pv_for_a_read_only_variable_is_rejected(model: StubModel) -> None:
    """``mode`` is the configuration's claim and ``read_only`` is the model's.

    The model wins, and loudly: a PV served writable over a variable the model
    refuses to set accepts writes from a client and drops every one of them.
    """
    config = Runner.generate_config(model)
    assert config["variables"]["output_b"]["mode"] == "ro"
    config["variables"]["output_b"]["mode"] = "rw"

    with pytest.raises(ValueError, match="output_b"):
        Runner(model=model, config=config)


def test_a_variable_type_with_no_handler_is_rejected() -> None:
    """A type no handler claims cannot be put on the wire at all.

    Distinct from a *supported* type carrying an unservable dtype, which is
    logged and skipped: that leaves the rest of the model served, while a type
    the handler table does not know about means the caller is holding a
    variable this package cannot represent.
    """

    class UnhandledVariable(Variable):
        def validate_value(self, *args, **kwargs) -> None:
            # Abstract on Variable, and never reached: the runner rejects the
            # type before a value is ever offered to it.
            raise NotImplementedError

    model = StubModel({"odd": UnhandledVariable(name="odd")})
    config = Runner.generate_config(model)

    with pytest.raises(RuntimeError, match="UnhandledVariable"):
        Runner(model=model, config=config)


def _make_runner_control_stub(protocol: list[str]) -> Runner:
    runner = Runner.__new__(Runner)
    runner._config = {
        "prefix": "",
        "protocol": protocol,
    }
    runner.providers = {}
    runner.pvdb = {}
    runner.reset_control_pv = ""
    runner.supports_pva = "pva" in protocol
    runner.supports_ca = "ca" in protocol

    # _create_control_pvs wires a callback to this method; a simple stub is enough.
    runner._enqueue = lambda *args, **kwargs: None
    return runner


def test_control_pvs_do_not_create_pva_sharedpvs_for_ca_only() -> None:
    runner = _make_runner_control_stub(["ca"])

    runner._create_control_pvs()

    assert runner.reset_control_pv == "RESET"
    assert "RESET" not in runner.providers
    assert runner.pvdb["RESET"]["type"] == "int"


def test_control_pvs_create_pva_sharedpvs_when_pva_enabled() -> None:
    runner = _make_runner_control_stub(["pva"])

    runner._create_control_pvs()

    assert "RESET" in runner.providers
    assert "RESET" not in runner.pvdb


# --------------------------------------------------------------------------
# queue items carry jobs
# --------------------------------------------------------------------------


QUEUE_ITEM_KEYS = {"values", "done", "reset", "jobs"}


def _queue_owner() -> SimpleNamespace:
    """The only state ``_enqueue`` touches is ``self.queue``."""
    return SimpleNamespace(queue=Queue())


def _only_item(queue: Queue) -> dict:
    item = queue.get_nowait()
    assert queue.empty(), "expected exactly one queue item"
    return item


def _recording_job(calls: list[str], name: str):
    def job() -> None:
        calls.append(name)

    return job


def test_an_item_enqueued_with_jobs_carries_them_in_order() -> None:
    """The loop runs jobs in the order given, so the item must keep that order.

    Enqueueing is not running: a job that fired on the enqueuing thread would
    touch the model off the run loop, which is the thing jobs exist to avoid.
    """
    owner = _queue_owner()
    calls: list[str] = []
    first = _recording_job(calls, "first")
    second = _recording_job(calls, "second")

    Runner._enqueue(owner, {}, jobs=[first, second])

    item = _only_item(owner.queue)
    assert set(item) == QUEUE_ITEM_KEYS
    assert item["jobs"] == [first, second]
    assert calls == []


def test_an_item_enqueued_without_jobs_carries_an_empty_list() -> None:
    """Every caller that predates jobs passes none and must see ``[]``, not a
    missing key -- and each item gets its own list, never a shared default."""
    owner = _queue_owner()

    Runner._enqueue(owner, {"a": {"value": 1.0, "ts": 0.0}})
    Runner._enqueue(owner, {}, reset=True)

    one = owner.queue.get_nowait()
    two = owner.queue.get_nowait()
    assert set(one) == QUEUE_ITEM_KEYS
    assert one["jobs"] == []
    assert two["jobs"] == []
    assert one["jobs"] is not two["jobs"]


def test_jobs_from_a_one_shot_iterable_are_collected_at_enqueue() -> None:
    """A generator is consumed once; the item must hold the jobs themselves,
    not an iterator the loop would find already spent or never started."""
    owner = _queue_owner()
    calls: list[str] = []
    jobs = [_recording_job(calls, name) for name in ("a", "b", "c")]

    Runner._enqueue(owner, {}, jobs=(job for job in jobs))

    item = _only_item(owner.queue)
    assert isinstance(item["jobs"], list)
    assert item["jobs"] == jobs
    assert calls == []


def test_jobs_are_keyword_only() -> None:
    """Positionally, a list of jobs would land in ``reset`` and read as truthy:
    a caller meaning to queue work would reset the model instead."""
    owner = _queue_owner()

    with pytest.raises(TypeError):
        Runner._enqueue(owner, {}, None, False, [lambda: None])

    assert owner.queue.empty()


def test_jobs_travel_alongside_values_done_and_reset() -> None:
    """Jobs are an addition to a batch, not a separate kind of item."""
    owner = _queue_owner()
    done = lambda error: None  # noqa: E731
    job = lambda: None  # noqa: E731
    values = {"a": {"value": 2.0, "ts": 1.0}}

    Runner._enqueue(owner, values, done=done, reset=True, jobs=[job])

    item = _only_item(owner.queue)
    assert item == {"values": values, "done": [done], "reset": True, "jobs": [job]}


class _QueueRecordingDriver(Runner.CaDriver):
    """A CaDriver with no pcaspy server behind it.

    ``pcaspy.Driver.__init__`` registers with a server's PV manager, so it is
    not called; the driver-API calls ``write`` makes are absorbed, leaving the
    runner's real queue as the only observable effect.
    """

    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    def setParam(self, reason, value, timestamp=None) -> None:
        pass

    def callbackPV(self, reason) -> None:
        pass


def _runner_with_queue(protocol: list[str]) -> Runner:
    """A server-free Runner whose ``_enqueue`` is the real one."""
    runner = Runner.__new__(Runner)
    runner.queue = Queue()
    runner._config = {"prefix": "", "protocol": protocol}
    runner.providers = {}
    runner.pvdb = {}
    runner.reset_control_pv = ""
    runner.supports_pva = "pva" in protocol
    runner.supports_ca = "ca" in protocol
    runner.clamp_writes = False
    return runner


def test_a_ca_variable_write_enqueues_no_jobs() -> None:
    runner = _runner_with_queue(["ca"])
    runner.model = StubModel({"input_a": ScalarVariable(name="input_a")})
    runner.pv_to_var = {"input_a": "input_a"}
    runner.pvdb["input_a"] = {"type": "float"}
    driver = _QueueRecordingDriver(runner)

    assert driver.write("input_a", 4.2) is True

    item = _only_item(runner.queue)
    assert set(item) == QUEUE_ITEM_KEYS
    assert item["values"]["input_a"]["value"] == 4.2
    assert len(item["done"]) == 1
    assert item["reset"] is False
    assert item["jobs"] == []


def test_a_ca_reset_write_enqueues_no_jobs() -> None:
    runner = _runner_with_queue(["ca"])
    runner._create_control_pvs()
    driver = _QueueRecordingDriver(runner)

    assert driver.write(runner.reset_control_pv, 1) is True

    item = _only_item(runner.queue)
    assert item == {"values": {}, "done": [], "reset": True, "jobs": []}


def test_a_pva_reset_put_enqueues_no_jobs() -> None:
    runner = _runner_with_queue(["pva"])
    runner._create_control_pvs()
    # SharedPV's ``put`` decorator stores the handler on ``_handler``; calling
    # it directly drives the reset PV's put with no server behind it.
    on_put = runner.providers["RESET"]._handler.put
    op = SimpleNamespace(done=lambda error=None: None)

    on_put(runner.providers["RESET"], op)

    item = _only_item(runner.queue)
    assert item == {"values": {}, "done": [], "reset": True, "jobs": []}


# --------------------------------------------------------------------------
# one run-loop cycle: jobs first, then the model pass
# --------------------------------------------------------------------------

IN_X = "x"
OUT_Y = "y"
# Short enough to keep the batching tests quick, long enough that items already
# sitting in the queue are drained inside the window.
BATCH_WINDOW = 0.05


class CycleModel(LUMEModel):
    """One writable input and one read-only output; every model call is logged.

    ``events`` is shared with the jobs and the output step, so one list holds
    the order in which a cycle touched everything.
    """

    def __init__(self, events: list, *, fail_set: bool = False) -> None:
        self.events = events
        self.fail_set = fail_set
        self._state = {IN_X: 0.0, OUT_Y: 0.0}
        self._vars = {
            IN_X: ScalarVariable(name=IN_X, default_value=0.0),
            OUT_Y: ScalarVariable(name=OUT_Y, default_value=0.0, read_only=True),
        }

    @property
    def supported_variables(self) -> dict[str, ScalarVariable]:
        return self._vars

    def _get(self, names) -> dict[str, float]:
        self.events.append(("get", tuple(names)))
        return {name: self._state[name] for name in names}

    def _set(self, values: dict) -> None:
        self.events.append(("set", dict(values)))
        if self.fail_set:
            raise RuntimeError("set refused")
        self._state.update(values)
        self._state[OUT_Y] = 2.0 * self._state[IN_X]

    def reset(self) -> None:
        self.events.append(("reset",))
        self._state = {IN_X: 0.0, OUT_Y: 0.0}


def _cycle_runner(
    events: list,
    *,
    update_rate: float = 0.0,
    fail_set: bool = False,
    runner_cls: type[Runner] = Runner,
) -> Runner:
    """A server-free Runner that can run one cycle in-process.

    ``_post_outputs`` and ``_reset_to_cached_state`` are replaced on the instance
    so that each lands in ``events`` rather than on a wire. ``runner_cls`` lets a
    test run the cycle of a subclass that overrides a hook.
    """
    runner = runner_cls.__new__(runner_cls)
    runner.queue = Queue()
    runner.update_rate = update_rate
    runner.model = CycleModel(events, fail_set=fail_set)
    runner.pv_handlers = {}
    runner._cached_state = {}
    runner._post_outputs = lambda out_values, ts: events.append(("post", dict(out_values), ts))
    runner._reset_to_cached_state = lambda: events.append(("reset_to_cached",))
    return runner


def _job(events: list, name: str, *, raises: bool = False):
    def job() -> None:
        events.append(("job", name))
        if raises:
            raise RuntimeError(f"job {name} failed")

    return job


def _item(values=None, *, done=None, reset=False, jobs=()) -> dict:
    """A queue item shaped exactly as ``_enqueue`` builds one."""
    return {
        "values": dict(values or {}),
        "done": [done] if done is not None else [],
        "reset": reset,
        "jobs": list(jobs),
    }


def _write(value: float, ts: float) -> dict:
    return {"value": value, "ts": ts}


def _kinds(events: list) -> list:
    return [event[0] for event in events]


def _sets(events: list) -> list:
    return [event[1] for event in events if event[0] == "set"]


def test_a_jobs_only_cycle_skips_the_model_pass() -> None:
    """A batch holding nothing but jobs has nothing to set, so no snapshot,
    ``model.set``, output ``get`` or post may run -- and its puts still complete."""
    events: list = []
    completions: list = []
    runner = _cycle_runner(events)

    runner._run_cycle(_item(done=completions.append, jobs=[_job(events, "a")]))

    assert events == [("job", "a")]
    assert completions == [None]


def test_a_raising_job_does_not_stop_the_next_job_or_the_cycle_pass() -> None:
    """A job owns its own errors. One that raises anyway is logged and passed
    over: it does not cancel the jobs behind it, the batch's writes, or their
    completions, and it is not a failed cycle."""
    events: list = []
    completions: list = []
    runner = _cycle_runner(events)

    runner._run_cycle(
        _item(
            {IN_X: _write(1.5, 5.0)},
            done=completions.append,
            jobs=[_job(events, "boom", raises=True), _job(events, "after")],
        )
    )

    assert events[:2] == [("job", "boom"), ("job", "after")]
    assert _sets(events) == [{IN_X: 1.5}]
    assert events[-1] == ("post", {IN_X: 1.5, OUT_Y: 3.0}, 5.0)
    assert ("reset_to_cached",) not in events
    assert completions == [None]


def test_a_raising_job_in_a_jobs_only_cycle_never_rolls_the_model_back() -> None:
    """Rolling back belongs to the pass. A jobs-only cycle has no pass, so a
    raising job must leave the model untouched rather than restore a snapshot
    this cycle never took."""
    events: list = []
    completions: list = []
    runner = _cycle_runner(events)

    runner._run_cycle(_item(done=completions.append, jobs=[_job(events, "boom", raises=True)]))

    assert events == [("job", "boom")]
    assert completions == [None]


def test_a_mixed_cycle_runs_its_jobs_first_then_one_set() -> None:
    events: list = []
    runner = _cycle_runner(events)

    runner._run_cycle(_item({IN_X: _write(2.0, 1.0)}, jobs=[_job(events, "a"), _job(events, "b")]))

    assert events[:2] == [("job", "a"), ("job", "b")]
    assert "job" not in _kinds(events[2:])
    assert _sets(events) == [{IN_X: 2.0}]


def test_a_cycle_pass_snapshots_before_it_sets_and_reads_after() -> None:
    """The pass keeps its order: cache the settable state, set, read every
    variable, post the result stamped with the batch's newest timestamp."""
    events: list = []
    runner = _cycle_runner(events)

    runner._run_cycle(_item({IN_X: _write(1.0, 7.0)}, jobs=[_job(events, "a")]))

    assert events == [
        ("job", "a"),
        ("get", (IN_X,)),
        ("set", {IN_X: 1.0}),
        ("get", (IN_X, OUT_Y)),
        ("post", {IN_X: 1.0, OUT_Y: 2.0}, 7.0),
    ]
    assert runner._cached_state == {IN_X: 0.0}


def test_an_empty_cycle_item_still_runs_the_pass() -> None:
    """The item enqueued at start-up carries no values, no reset and no jobs.
    Its pass is what publishes the model's initial outputs."""
    events: list = []
    completions: list = []
    runner = _cycle_runner(events)

    runner._run_cycle(_item(done=completions.append))

    assert _sets(events) == [{}]
    assert _kinds(events)[-1] == "post"
    assert completions == [None]


def test_a_reset_cycle_with_jobs_still_runs_the_pass() -> None:
    events: list = []
    runner = _cycle_runner(events)

    runner._run_cycle(_item(reset=True, jobs=[_job(events, "a")]))

    kinds = _kinds(events)
    assert kinds[0] == "job"
    assert kinds.index("reset") < kinds.index("set")
    assert _sets(events) == [{}]
    assert kinds[-1] == "post"


def test_a_failed_cycle_pass_completes_its_puts_with_the_error() -> None:
    """Jobs ahead of a failing pass have already run; the pass still rolls the
    model back and hands its error to every waiting put."""
    events: list = []
    completions: list = []
    runner = _cycle_runner(events, fail_set=True)

    runner._run_cycle(
        _item({IN_X: _write(1.0, 1.0)}, done=completions.append, jobs=[_job(events, "a")])
    )

    assert events[0] == ("job", "a")
    assert events[-1] == ("reset_to_cached",)
    assert "post" not in _kinds(events)
    assert len(completions) == 1
    assert completions[0] is not None
    assert "set refused" in completions[0]


def test_a_raising_done_callback_does_not_stop_the_next_one_in_a_cycle() -> None:
    events: list = []
    completions: list = []

    def broken(error) -> None:
        raise RuntimeError("callback failed")

    runner = _cycle_runner(events)
    runner.queue.put(_item(done=completions.append))
    runner.update_rate = BATCH_WINDOW

    runner._run_cycle(_item(done=broken, jobs=[_job(events, "a")]))

    assert completions == [None]


def test_batching_a_cycle_preserves_job_order() -> None:
    """Items drained inside the window join the first one: their jobs run after
    its jobs in arrival order, their values merge into a single set, and every
    put they carry is completed."""
    events: list = []
    completions: list = []
    runner = _cycle_runner(events, update_rate=BATCH_WINDOW)
    runner.queue.put(
        _item(
            {IN_X: _write(2.0, 2.0)},
            done=lambda error: completions.append(("second", error)),
            jobs=[_job(events, "c")],
        )
    )
    runner.queue.put(
        _item(
            done=lambda error: completions.append(("third", error)),
            jobs=[_job(events, "d"), _job(events, "e")],
        )
    )

    runner._run_cycle(
        _item(
            {IN_X: _write(1.0, 1.0)},
            done=lambda error: completions.append(("first", error)),
            jobs=[_job(events, "a"), _job(events, "b")],
        )
    )

    assert runner.queue.empty()
    job_names = [event[1] for event in events if event[0] == "job"]
    assert job_names == ["a", "b", "c", "d", "e"]
    assert "job" not in _kinds(events[5:])
    assert _sets(events) == [{IN_X: 2.0}]
    assert events[-1][2] == 2.0
    assert completions == [("first", None), ("second", None), ("third", None)]


def test_batching_only_jobs_into_a_cycle_skips_the_pass() -> None:
    """Merging keeps the rule: a batch whose every item carries only jobs has
    nothing for the model."""
    events: list = []
    runner = _cycle_runner(events, update_rate=BATCH_WINDOW)
    runner.queue.put(_item(jobs=[_job(events, "b")]))

    runner._run_cycle(_item(jobs=[_job(events, "a")]))

    assert events == [("job", "a"), ("job", "b")]


def test_batching_values_behind_a_jobs_only_item_runs_the_cycle_pass() -> None:
    events: list = []
    runner = _cycle_runner(events, update_rate=BATCH_WINDOW)
    runner.queue.put(_item({IN_X: _write(3.0, 1.0)}))

    runner._run_cycle(_item(jobs=[_job(events, "a")]))

    assert events[0] == ("job", "a")
    assert _sets(events) == [{IN_X: 3.0}]


# --------------------------------------------------------------------------
# the post-cycle read: _cycle_output_names narrows it
# --------------------------------------------------------------------------

OUT_Z = "z"


class WideCycleModel(CycleModel):
    """``CycleModel`` with a second read-only output, so a narrowed read is
    distinguishable from a read of the whole roster."""

    def __init__(self, events: list) -> None:
        super().__init__(events)
        self._state[OUT_Z] = 0.0
        self._vars[OUT_Z] = ScalarVariable(name=OUT_Z, default_value=0.0, read_only=True)

    def _set(self, values: dict) -> None:
        super()._set(values)
        self._state[OUT_Z] = -self._state[IN_X]


class _NarrowRunner(Runner):
    def _cycle_output_names(self) -> list[str]:
        return [IN_X, OUT_Z]


class _SilentRunner(Runner):
    def _cycle_output_names(self) -> list[str]:
        return []


def _wide_cycle_runner(events: list, *, runner_cls: type[Runner] = Runner) -> Runner:
    runner = _cycle_runner(events, runner_cls=runner_cls)
    runner.model = WideCycleModel(events)
    return runner


def test_the_default_cycle_output_names_are_the_whole_roster() -> None:
    events: list = []
    runner = _wide_cycle_runner(events)

    assert runner._cycle_output_names() == [IN_X, OUT_Y, OUT_Z]

    runner._run_cycle(_item({IN_X: _write(1.0, 4.0)}))

    assert events == [
        ("get", (IN_X,)),
        ("set", {IN_X: 1.0}),
        ("get", (IN_X, OUT_Y, OUT_Z)),
        ("post", {IN_X: 1.0, OUT_Y: 2.0, OUT_Z: -1.0}, 4.0),
    ]


def test_a_subclass_narrows_the_post_cycle_get_to_its_names() -> None:
    """The hook narrows only the read that feeds the output step; the snapshot
    of the settable state a failed cycle rolls back to is taken as before."""
    events: list = []
    runner = _wide_cycle_runner(events, runner_cls=_NarrowRunner)

    runner._run_cycle(_item({IN_X: _write(1.0, 4.0)}))

    assert events == [
        ("get", (IN_X,)),
        ("set", {IN_X: 1.0}),
        ("get", (IN_X, OUT_Z)),
        ("post", {IN_X: 1.0, OUT_Z: -1.0}, 4.0),
    ]


def test_an_empty_cycle_output_list_skips_the_post_cycle_get() -> None:
    """Nothing to read means no ``model.get`` after the set at all. The output
    step still runs, with nothing to publish, and the put still completes."""
    events: list = []
    completions: list = []
    runner = _wide_cycle_runner(events, runner_cls=_SilentRunner)

    runner._run_cycle(_item({IN_X: _write(1.0, 4.0)}, done=completions.append))

    assert events == [
        ("get", (IN_X,)),
        ("set", {IN_X: 1.0}),
        ("post", {}, 4.0),
    ]
    assert completions == [None]


class _StopLoop(Exception):
    pass


def test_run_hands_each_queue_item_to_run_cycle_in_order() -> None:
    """``_run`` is the loop and nothing else; the cycle is where the work is."""
    runner = Runner.__new__(Runner)
    runner.queue = Queue()
    first, second = _item({IN_X: _write(1.0, 1.0)}), _item(reset=True)
    runner.queue.put(first)
    runner.queue.put(second)
    seen: list = []

    def cycle(item: dict) -> None:
        seen.append(item)
        if len(seen) == 2:
            raise _StopLoop

    runner._run_cycle = cycle

    with pytest.raises(_StopLoop):
        runner._run()

    assert seen[0] is first
    assert seen[1] is second


# --------------------------------------------------------------------------
# model info lists only the variables the configuration serves
# --------------------------------------------------------------------------


def _make_model_info_stub(model: StubModel, served: dict) -> Runner:
    runner = Runner.__new__(Runner)
    runner.model = model
    runner._config = {"prefix": "", "description": "stub", "variables": served}
    runner.types = {}
    runner.pvs = {}
    runner.providers = {}
    return runner


def test_model_info_lists_only_configured_variables(model: StubModel) -> None:
    """A model variable the configuration omits is served on no transport, and
    the model info PV describes what is served: it leaves that variable out
    instead of failing on its missing entry."""
    served = {"input_a": {"pv": "input_a", "mode": "rw"}}
    runner = _make_model_info_stub(model, served)

    runner._create_model_info()

    info = runner.pvs["MODEL_INFO"].current()
    listed = [(v["name"], v["pvname"], v["mode"]) for v in info["supported_variables"]]
    assert listed == [("input_a", "input_a", "rw")]
    assert "MODEL_INFO" in runner.providers


def test_published_values_carry_wall_clock_timestamps(model: StubModel) -> None:
    """A served value is stamped with UNIX time, never the monotonic clock.

    ``_generate_value`` without an explicit ``ts`` is the path every
    subclass-published value takes. Stamping it from the monotonic clock
    puts every timestamp a client sees in January 1970."""
    runner = _make_model_info_stub(model, {})
    var = model.supported_variables["input_a"]
    handler = find_variable_handler(type(var))
    runner.pv_handlers = {"input_a": handler}
    runner.types = {"input_a": handler.create_type(var)}
    before = time.time()
    value = runner._generate_value("input_a", 1.0)
    after = time.time()
    stamped = value["timeStamp"]["secondsPastEpoch"] + value["timeStamp"]["nanoseconds"] / 1e9
    assert before - 1.0 <= stamped <= after + 1.0


def _meta_stub(var: Variable, meta: dict | None = None) -> Runner:
    """A server-free runner packing ``var`` under the name ``x``, its type built
    the way ``__init__`` builds it for ``meta``."""
    runner = _make_model_info_stub(StubModel({"x": var}), {})
    handler = find_variable_handler(type(var))
    runner.pv_handlers = {"x": handler}
    if meta is not None:
        runner._pv_meta = {"x": meta}
    runner.types = {"x": handler.create_type(var, **runner._precision_kwargs("x"))}
    return runner


@pytest.mark.parametrize(
    "var, value",
    [
        pytest.param(ScalarVariable(name="x", value_range=(0.0, 10.0), unit="mm"), 2.5, id="float"),
        pytest.param(EnumVariable(name="x", options=["A", "B", "C"]), "B", id="enum"),
    ],
)
def test_generate_value_and_pack_value_agree(var: Variable, value: object) -> None:
    """``_generate_value`` delegates to ``_pack_value``: with the same fixed
    timestamp both produce the same fields and the same changed set."""
    runner = _meta_stub(var)
    ts = 1_700_000_000.25

    generated = runner._generate_value("x", value, ts)
    packed = runner._pack_value("x", value, ts)

    assert generated.todict() == packed.todict()
    assert generated.changedSet() == packed.changedSet()
    assert packed["timeStamp"]["secondsPastEpoch"] == 1_700_000_000


# --- precision / description metadata (F1.1) --------------------------------


class DescribedScalarVariable(ScalarVariable):
    """A ScalarVariable carrying lume-base 0.6's ``description`` field, declared
    here so the fallback is testable against a lume-base that predates it."""

    description: str | None = None


class DescribedNDVariable(NDVariable):
    description: str | None = None


def _meta(var: Variable, **entry: object) -> dict:
    handler = find_variable_handler(type(var))
    assert handler is not None
    return _resolve_pv_meta(var.name, {"name": var.name, **entry}, var, handler)


def _nd() -> NDVariable:
    return NDVariable(name="arr", shape=(2,), dtype=np.float64)


def _torch_meta_cases() -> list:
    if not TORCH_AVAILABLE:
        return []
    import torch

    return [
        pytest.param(TorchScalarVariable(name="ts"), id="torch-scalar"),
        pytest.param(TorchNDVariable(name="tnd", shape=(2,), dtype=torch.float32), id="torch-nd"),
    ]


def test_an_entry_without_meta_keys_resolves_to_none() -> None:
    assert _meta(ScalarVariable(name="x")) == {"precision": None, "description": None}


@pytest.mark.parametrize("precision", [0, 3, 17])
def test_a_precision_in_range_on_a_float_scalar_is_kept(precision: int) -> None:
    assert _meta(ScalarVariable(name="x"), precision=precision)["precision"] == precision


@pytest.mark.parametrize(
    "precision",
    [
        pytest.param(-1, id="negative"),
        pytest.param(18, id="above-17"),
        pytest.param(True, id="bool"),
        pytest.param(2.0, id="float"),
        pytest.param("3", id="str"),
        pytest.param(None, id="none"),
    ],
)
def test_a_malformed_precision_is_rejected(precision: object) -> None:
    """``type(p) is int`` and 0..17: a bool is an int to isinstance, but True
    is not a digit count, and an explicit None is a typo rather than an absence."""
    with pytest.raises(ValueError, match=r"x.*'precision'"):
        _meta(ScalarVariable(name="x"), precision=precision)


@pytest.mark.parametrize(
    "var",
    [
        pytest.param(IntVariable(name="v"), id="int"),
        pytest.param(BoolVariable(name="v"), id="bool"),
        pytest.param(StrVariable(name="v"), id="str"),
        pytest.param(EnumVariable(name="v", options=["A", "B"]), id="enum"),
        pytest.param(NDVariable(name="v", shape=(2,), dtype=np.float64), id="nd"),
        *_torch_meta_cases(),
    ],
)
def test_a_precision_on_anything_but_a_float_scalar_is_rejected(var: Variable) -> None:
    """Only a float NTScalar has digits after the decimal point to limit;
    IntVariable shares the float handler, so it is excluded by type."""
    with pytest.raises(ValueError, match=f"{var.name}.*'precision'"):
        _meta(var, precision=3)


@pytest.mark.parametrize(
    "var",
    [
        pytest.param(ScalarVariable(name="v"), id="float"),
        pytest.param(IntVariable(name="v"), id="int"),
        pytest.param(BoolVariable(name="v"), id="bool"),
        pytest.param(StrVariable(name="v"), id="str"),
        pytest.param(EnumVariable(name="v", options=["A", "B"]), id="enum"),
    ],
)
def test_a_config_description_is_kept_on_every_display_type(var: Variable) -> None:
    assert _meta(var, description="beam current")["description"] == "beam current"


@pytest.mark.parametrize("description", [pytest.param(3, id="int"), pytest.param(None, id="none")])
def test_a_non_str_description_is_rejected(description: object) -> None:
    with pytest.raises(ValueError, match=r"x.*'description'"):
        _meta(ScalarVariable(name="x"), description=description)


@pytest.mark.parametrize("var", [pytest.param(_nd(), id="nd"), *_torch_meta_cases()])
def test_a_config_description_on_an_array_or_torch_variable_is_rejected(var: Variable) -> None:
    """These PVA types have no display block to carry it; accepted, it would
    be silently dropped."""
    with pytest.raises(ValueError, match=f"{var.name}.*'description'"):
        _meta(var, description="beam current")


def test_an_empty_config_description_means_none_and_beats_the_variable() -> None:
    var = DescribedScalarVariable(name="x", description="from the model")
    assert _meta(var, description="")["description"] is None


def test_the_variable_description_is_the_fallback() -> None:
    var = DescribedScalarVariable(name="x", description="from the model")
    assert _meta(var)["description"] == "from the model"
    assert _meta(var, description="from config")["description"] == "from config"


def test_an_empty_variable_description_means_none() -> None:
    assert _meta(DescribedScalarVariable(name="x", description=""))["description"] is None


def test_a_variable_description_on_an_array_is_ignored_not_rejected() -> None:
    """A lume-base 0.6 model may describe an array; it must still boot."""
    var = DescribedNDVariable(
        name="arr", shape=(2,), dtype=np.dtype(np.float64), description="an image"
    )
    assert _meta(var) == {"precision": None, "description": None}


def test_resolving_meta_never_mutates_the_entry_or_the_variable() -> None:
    var = DescribedScalarVariable(name="x", description="from the model")
    entry = {"name": "x", "precision": 2}
    before = var.model_dump()
    _resolve_pv_meta("x", entry, var, find_variable_handler(type(var)))
    assert entry == {"name": "x", "precision": 2}
    assert var.model_dump() == before


def test_a_bad_meta_entry_is_rejected_before_any_server_exists(
    model: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    import p4p.server
    import pcaspy

    def no_server(*args: object, **kwargs: object) -> None:
        raise AssertionError("a server was created before validation failed")

    monkeypatch.setattr(p4p.server, "Server", no_server)
    monkeypatch.setattr(pcaspy, "SimpleServer", no_server)
    config = Runner.generate_config(model)
    config["variables"]["input_a"]["precision"] = 99

    with pytest.raises(ValueError, match=r"input_a.*'precision'"):
        Runner(model=model, config=config)


def test_the_runner_class_default_pv_meta_is_empty_and_read_only() -> None:
    runner = Runner.__new__(Runner)
    assert dict(runner._pv_meta) == {}
    with pytest.raises(TypeError):
        runner._pv_meta["x"] = {}  # type: ignore[index]


@pytest.mark.parametrize(
    "skipped",
    [
        pytest.param(ParticleGroupVariable(name="skipped"), id="particle-group"),
        pytest.param(
            NDVariable(name="skipped", shape=(2,), dtype=np.complex128), id="unsupported-nd"
        ),
    ],
)
def test_a_skipped_variable_carrying_a_description_is_skipped_not_rejected(
    skipped: Variable,
) -> None:
    """Validation runs after the skips, so 0.1.4's warn-and-skip is unchanged.

    A typo-named entry after the skipped one stops the constructor with a
    KeyError before any server exists: reaching it proves the skipped entry's
    description and precision raised nothing.
    """
    model = StubModel({"skipped": skipped})
    config = Runner.generate_config(model)
    config["variables"]["skipped"]["description"] = "not servable"
    config["variables"]["skipped"]["precision"] = 3
    config["variables"]["later"] = {"name": "later_typo", "pv": "later"}

    with pytest.raises(KeyError, match="later_typo"):
        Runner(model=model, config=config)


# --- precision / description wiring (F1.3, F1.4) ---------------------------


def test_a_configured_precision_is_packed_as_display_precision() -> None:
    runner = _meta_stub(ScalarVariable(name="x"), {"precision": 4, "description": None})

    packed = runner._pack_value("x", 1.25)

    assert packed["display"]["precision"] == 4
    assert "format" not in packed["display"].keys()


def test_a_float_without_precision_keeps_its_0_1_4_display_block() -> None:
    runner = _meta_stub(ScalarVariable(name="x"), {"precision": None, "description": None})

    packed = runner._pack_value("x", 1.25)

    assert "precision" not in packed["display"].keys()
    assert "format" in packed["display"].keys()


@pytest.mark.parametrize(
    "var, value",
    [
        pytest.param(ScalarVariable(name="x"), 1.0, id="float"),
        pytest.param(IntVariable(name="x"), 2, id="int"),
        pytest.param(BoolVariable(name="x"), True, id="bool"),
        pytest.param(StrVariable(name="x"), "s", id="str"),
        pytest.param(EnumVariable(name="x", options=["A", "B"]), "B", id="enum"),
    ],
)
def test_a_configured_description_is_packed_as_display_description(
    var: Variable, value: object
) -> None:
    runner = _meta_stub(var, {"precision": None, "description": "what it is"})

    packed = runner._pack_value("x", value)

    assert packed["display"]["description"] == "what it is"


def test_an_nd_variable_is_packed_without_display_metadata() -> None:
    """The ND handler's type has no display block; the runner must not write one."""
    runner = _meta_stub(_nd(), {"precision": None, "description": None})

    packed = runner._pack_value("x", np.zeros(2))

    assert "display" not in packed.keys()


def test_precision_is_passed_only_to_a_float_scalar_handler() -> None:
    """``create_type``/``ca_pvspec`` get a ``precision`` keyword only from a
    ScalarVariableHandler with one configured; every other call is 0.1.4's."""
    runner = _make_model_info_stub(StubModel({}), {})
    runner.pv_handlers = {
        "float": find_variable_handler(ScalarVariable),
        "plain": find_variable_handler(ScalarVariable),
        "enum": find_variable_handler(EnumVariable),
    }
    runner._pv_meta = {
        "float": {"precision": 3, "description": None},
        "plain": {"precision": None, "description": None},
        "enum": {"precision": 3, "description": None},
    }

    assert runner._precision_kwargs("float") == {"precision": 3}
    assert runner._precision_kwargs("plain") == {}
    assert runner._precision_kwargs("enum") == {}
    assert runner._precision_kwargs("unknown") == {}


@pytest.mark.parametrize(
    "var, meta, expected",
    [
        pytest.param(ScalarVariable(name="x"), {"precision": 2}, 2, id="configured"),
        pytest.param(ScalarVariable(name="x"), {"precision": None}, None, id="unconfigured"),
        pytest.param(IntVariable(name="x"), None, None, id="int-no-meta"),
    ],
)
def test_add_pv_puts_a_configured_precision_in_the_ca_pvspec(
    var: Variable, meta: dict | None, expected: int | None
) -> None:
    """``_add_pv`` reads the precision from ``_pv_meta``: ``prec`` is in the
    pvdb spec exactly when one is configured."""
    runner = _meta_stub(var, meta)
    runner.supports_pva = False
    runner.supports_ca = True
    runner.pvdb = {}
    runner.ca_pvs = {}

    runner._add_pv("X_PV", var, ro=False, prefix="", handler=runner.pv_handlers["x"])

    assert runner.pvdb["X_PV"].get("prec") == expected


# --------------------------------------------------------------------------
# undefined (UDF) alarm overlay
# --------------------------------------------------------------------------


def _alarm(value) -> tuple[int, int]:
    return value["alarm"]["severity"], value["alarm"]["status"]


def _udf_pv_stub(var: Variable) -> tuple[Runner, SharedPV]:
    """A server-free runner serving ``var`` as ``x`` on a real SharedPV, built
    the way ``_add_pv`` builds it."""
    runner = _meta_stub(var)
    pv = SharedPV(
        handler=Runner.Handler(variable=var, runner=runner, read_only=var.read_only),
        initial=runner._generate_value("x", None),
    )
    return runner, pv


def test_the_runner_class_default_udf_is_an_empty_frozenset() -> None:
    runner = Runner.__new__(Runner)
    assert runner._udf == frozenset()
    assert isinstance(runner._udf, frozenset)


def test_a_udf_name_posted_via_generate_value_carries_invalid_udf() -> None:
    runner, pv = _udf_pv_stub(ScalarVariable(name="x", value_range=(0.0, 10.0), read_only=True))
    runner._udf = frozenset({"x"})

    pv.post(runner._generate_value("x", 2.5))

    assert _alarm(pv.current()) == (3, 6)
    assert pv.current()["value"] == 2.5


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(2.5, (0, 0), id="in-range"),
        pytest.param(20.0, (2, 2), id="out-of-range"),
    ],
)
def test_a_name_not_in_udf_carries_the_handler_alarm_pair(value: float, expected: tuple) -> None:
    runner, pv = _udf_pv_stub(ScalarVariable(name="x", value_range=(0.0, 10.0), read_only=True))
    runner._udf = frozenset({"other"})

    pv.post(runner._generate_value("x", value))

    assert _alarm(pv.current()) == expected


def test_pack_value_never_applies_the_udf_overlay() -> None:
    runner = _meta_stub(ScalarVariable(name="x", value_range=(0.0, 10.0)))
    runner._udf = frozenset({"x"})
    ts = 1_700_000_000.25

    packed = runner._pack_value("x", 2.5, ts)
    generated = runner._generate_value("x", 2.5, ts)

    assert _alarm(packed) == (0, 0)
    assert _alarm(generated) == (3, 6)
    assert generated["value"] == packed["value"]


def test_the_add_pv_initial_value_with_an_empty_udf_is_unchanged_from_0_1_4() -> None:
    """With nothing undefined, the initial value is exactly what the handler
    packs -- the 0.1.4 value -- apart from the wall-clock timestamp."""
    var = ScalarVariable(name="x", value_range=(0.0, 10.0))
    runner = _meta_stub(var)
    runner._udf = frozenset()
    runner.supports_pva = True
    runner.supports_ca = False

    runner._add_pv("x", var, False, "", runner.pv_handlers["x"])

    initial = runner.pvs["x"].current()
    packed = runner._pack_value("x", None)
    strip = lambda d: {k: v for k, v in d.items() if k != "timeStamp"}  # noqa: E731
    assert strip(initial.todict()) == strip(packed.todict())
    assert _alarm(initial) == _alarm(packed)


def test_a_put_carrying_alarm_fields_leaves_a_standing_udf_alarm() -> None:
    """The client's alarm leaves are unmarked before either echo, so a put
    cannot overwrite the stored (INVALID, UDF) pair."""
    var = ScalarVariable(name="x", value_range=(-10.0, 10.0))
    runner, pv = _udf_pv_stub(var)
    runner._udf = frozenset({"x"})
    pv.post(runner._generate_value("x", 1.0))
    runner.echo_unconfirmed_writes = True
    runner.clamp_writes = False
    runner._enqueue = lambda values, done=None, **kw: done(None)

    put = runner.types["x"](
        {"value": 4.2, "alarm": {"severity": 0, "status": 0, "message": "client"}}
    )
    put.mark("value")
    put.mark("alarm.severity")
    put.mark("alarm.status")
    put.mark("alarm.message")
    finished = []
    op = SimpleNamespace(value=lambda: put, done=lambda error=None: finished.append(error))

    Runner.Handler(variable=var, runner=runner, read_only=False).put(pv, op)

    stored = pv.current()
    assert finished == [None]
    assert stored["value"] == 4.2
    assert _alarm(stored) == (3, 6)
    assert stored["alarm"]["message"] != "client"


# --------------------------------------------------------------------------
# output_severity hook evaluation
# --------------------------------------------------------------------------


def _severity_runner(**model_attrs) -> Runner:
    """A server-free runner whose model carries only ``model_attrs``."""
    runner = Runner.__new__(Runner)
    runner.model = SimpleNamespace(**model_attrs)
    return runner


def test_severity_conditions_table_maps_udf_to_the_pva_and_ca_pairs() -> None:
    import pcaspy

    assert set(_CONDITIONS) == {"udf"}
    pva, ca = _CONDITIONS["udf"]
    assert pva == (3, 6)
    assert ca == (pcaspy.Alarm.UDF_ALARM, pcaspy.Severity.INVALID_ALARM)


def test_severity_without_a_hook_is_an_empty_frozenset() -> None:
    runner = _severity_runner()
    assert runner._evaluate_severity(["a", "b"]) == frozenset()


def test_severity_with_a_non_callable_hook_attribute_is_an_empty_frozenset() -> None:
    runner = _severity_runner(output_severity={"a": {"condition": "udf"}})
    assert runner._evaluate_severity(["a"]) == frozenset()


def test_severity_valid_reply_returns_the_udf_names_and_calls_the_hook_once() -> None:
    calls = []

    def hook(names):
        calls.append(list(names))
        return {"a": {"condition": "udf"}, "c": {"condition": "udf"}}

    runner = _severity_runner(output_severity=hook)
    result = runner._evaluate_severity(["a", "b", "c"])

    assert result == frozenset({"a", "c"})
    assert isinstance(result, frozenset)
    assert calls == [["a", "b", "c"]]


def test_severity_empty_reply_returns_an_empty_frozenset() -> None:
    runner = _severity_runner(output_severity=lambda names: {})
    assert runner._evaluate_severity(["a"]) == frozenset()


def test_severity_reply_keys_outside_the_requested_names_are_ignored() -> None:
    # Keys outside ``names`` are ignored whatever their value, even malformed.
    runner = _severity_runner(
        output_severity=lambda names: {
            "a": {"condition": "udf"},
            "other": {"condition": "udf"},
            "junk": {"condition": "nonsense"},
            "worse": 42,
        }
    )
    assert runner._evaluate_severity(["a", "b"]) == frozenset({"a"})


def test_severity_unknown_condition_raises_naming_condition_and_variable() -> None:
    runner = _severity_runner(output_severity=lambda names: {"b": {"condition": "stale"}})
    with pytest.raises(ValueError) as info:
        runner._evaluate_severity(["a", "b"])
    assert "stale" in str(info.value)
    assert "b" in str(info.value)


def test_severity_unhashable_condition_raises_value_error() -> None:
    runner = _severity_runner(output_severity=lambda names: {"b": {"condition": ["udf"]}})
    with pytest.raises(ValueError, match="b"):
        runner._evaluate_severity(["b"])


def test_severity_entry_without_a_condition_raises_naming_the_variable() -> None:
    runner = _severity_runner(output_severity=lambda names: {"speed": {}})
    with pytest.raises(ValueError, match="speed"):
        runner._evaluate_severity(["speed"])


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(None, id="none"),
        pytest.param(["a"], id="list"),
        pytest.param("udf", id="str"),
        pytest.param({"a"}, id="set"),
    ],
)
def test_severity_non_dict_reply_raises(reply) -> None:
    runner = _severity_runner(output_severity=lambda names: reply)
    with pytest.raises(ValueError):
        runner._evaluate_severity(["a"])


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("udf", id="str"),
        pytest.param(None, id="none"),
        pytest.param(["udf"], id="list"),
    ],
)
def test_severity_non_dict_entry_raises_naming_the_variable(entry) -> None:
    runner = _severity_runner(output_severity=lambda names: {"speed": entry})
    with pytest.raises(ValueError, match="speed"):
        runner._evaluate_severity(["speed"])


def test_severity_exception_from_the_hook_propagates_unchanged() -> None:
    class HookFailure(RuntimeError):
        pass

    def hook(names):
        raise HookFailure("model cannot tell")

    runner = _severity_runner(output_severity=hook)
    with pytest.raises(HookFailure, match="model cannot tell"):
        runner._evaluate_severity(["a"])


def test_severity_evaluation_does_not_touch_the_runner_udf_state() -> None:
    runner = _severity_runner(output_severity=lambda names: {"a": {"condition": "udf"}})
    runner._udf = frozenset({"z"})
    runner._evaluate_severity(["a"])
    assert runner._udf == frozenset({"z"})


# --------------------------------------------------------------------------
# _post_outputs: two-phase publishing under an output_severity hook
# --------------------------------------------------------------------------

POST_TS = 1_700_000_000.5


class _PvSpy:
    """A real SharedPV that also logs every post into a shared event list."""

    def __init__(self, name: str, initial, events: list, *, fail: bool = False) -> None:
        self.name = name
        self.events = events
        self.fail = fail
        self.shared = SharedPV(initial=initial)

    def post(self, value) -> None:
        if self.fail:
            raise RuntimeError(f"post of {self.name} refused")
        self.events.append(("post", self.name, _alarm(value), value.changedSet()))
        self.shared.post(value)

    def current(self):
        return self.shared.current()


class _CaSpy:
    """Records the pcaspy driver calls ``_post_outputs`` makes, in order."""

    def __init__(self, events: list, *, fail_set: tuple = (), fail_update: bool = False) -> None:
        self.events = events
        self.fail_set = fail_set
        self.fail_update = fail_update

    def setParam(self, reason, value, timestamp=None) -> None:
        if reason in self.fail_set:
            raise RuntimeError(f"setParam of {reason} refused")
        self.events.append(("setParam", reason, value))

    def setParamStatus(self, reason, alarm, severity) -> None:
        self.events.append(("setParamStatus", reason, alarm, severity))

    def updatePVs(self) -> None:
        if self.fail_update:
            raise RuntimeError("updatePVs refused")
        self.events.append(("updatePVs",))


def _ca(name: str) -> str:
    return f"CA_{name}"


def _post_runner(model: MixedModel, events: list, **ca_kwargs) -> Runner:
    """A server-free runner serving every MixedModel variable on a spied PVA
    SharedPV and a recording CA driver, requesting the whole roster."""
    runner = Runner.__new__(Runner)
    runner.model = model
    runner.pv_handlers = {}
    runner.types = {}
    for name, var in model.supported_variables.items():
        handler = find_variable_handler(type(var))
        runner.pv_handlers[name] = handler
        runner.types[name] = handler.create_type(var)
    runner.pvs = {
        name: _PvSpy(name, runner._generate_value(name, None), events)
        for name in model.supported_variables
    }
    runner.ca_pvs = {name: _ca(name) for name in model.supported_variables}
    runner.ca_driver = _CaSpy(events, **ca_kwargs)
    runner._udf = frozenset()
    runner._cycle_requested_names = tuple(model.supported_variables)
    hook = getattr(model, "output_severity", None)
    if hook is not None:

        def logged(names):
            events.append(("hook", list(names)))
            return hook(names)

        model.output_severity = logged
    return runner


def _trigger(model: MixedModel) -> dict:
    """Drive the model to the severity trigger; return a whole-roster get."""
    model.set({MIXED_FLOAT_IN: MIXED_SEVERITY_TRIGGER})
    return model.get(list(model.supported_variables))


def _posts(events: list) -> dict:
    return {e[1]: e[2] for e in events if e[0] == "post"}


def _set_params(events: list) -> list:
    return [e[1] for e in events if e[0] == "setParam"]


def _ca_calls(events: list) -> list:
    return [e for e in events if e[0] in ("setParam", "setParamStatus", "updatePVs")]


def test_post_outputs_requested_names_class_default_is_empty() -> None:
    assert Runner.__new__(Runner)._cycle_requested_names == ()


def test_post_outputs_sees_the_names_run_cycle_requested() -> None:
    events: list = []
    seen: list = []
    runner = _wide_cycle_runner(events, runner_cls=_NarrowRunner)
    runner._post_outputs = lambda out_values, ts: seen.append(runner._cycle_requested_names)

    runner._run_cycle(_item({IN_X: _write(1.0, 4.0)}))

    assert seen == [(IN_X, OUT_Z)]


def test_post_outputs_without_a_hook_posts_as_0_1_4() -> None:
    """No hook: every Value is ``_generate_value``'s (no forced alarm pair), one
    setParam per name, one updatePVs, no setParamStatus, and ``_udf`` stays empty."""
    events: list = []
    model = MixedModel()
    runner = _post_runner(model, events)
    out = _trigger(model)
    expected = {name: runner._generate_value(name, value, POST_TS) for name, value in out.items()}

    runner._post_outputs(out, POST_TS)

    posts = [e for e in events if e[0] == "post"]
    assert [p[1] for p in posts] == list(out)
    for _, name, pair, changed in posts:
        assert pair == _alarm(expected[name])
        assert changed == expected[name].changedSet()
    # the range rule still holds for the out-of-range float output
    assert _posts(events)[MIXED_FLOAT_OUT] == (2, 2)
    assert _ca_calls(events)[-1] == ("updatePVs",)
    assert _set_params(events) == [_ca(n) for n in out]
    assert not [e for e in events if e[0] == "setParamStatus"]
    assert runner._udf == frozenset()


def test_post_outputs_without_a_hook_ignores_an_extra_key_as_0_1_4() -> None:
    """A hook-less model whose ``_get`` adds an unserved, wrongly typed key
    publishes exactly what the same model without the key publishes."""
    plain_events: list = []
    extra_events: list = []
    plain = MixedModel()
    extra = MixedModel(extra_key=True)
    plain_runner = _post_runner(plain, plain_events)
    extra_runner = _post_runner(extra, extra_events)
    extra_out = _trigger(extra)
    assert MIXED_EXTRA_KEY in extra_out

    plain_runner._post_outputs(_trigger(plain), POST_TS)
    extra_runner._post_outputs(extra_out, POST_TS)

    assert extra_events == plain_events
    assert extra_runner._udf == frozenset()


def test_post_outputs_calls_the_hook_once_with_the_cycle_names_before_posting() -> None:
    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF)
    runner = _post_runner(model, events)
    out = _trigger(model)

    runner._post_outputs(out, POST_TS)

    hooks = [e for e in events if e[0] == "hook"]
    assert hooks == [("hook", list(model.supported_variables))]
    assert events[0][0] == "hook"


def test_post_outputs_udf_names_carry_invalid_udf_on_both_transports() -> None:
    import pcaspy

    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF)
    runner = _post_runner(model, events)

    runner._post_outputs(_trigger(model), POST_TS)

    posts = _posts(events)
    assert posts[MIXED_FLOAT_OUT] == (3, 6)
    assert posts[MIXED_INT] == (3, 6)
    assert runner.pvs[MIXED_INT].current()["alarm"]["status"] == 6
    assert runner._udf == frozenset({MIXED_FLOAT_OUT, MIXED_INT})

    udf_ca = (pcaspy.Alarm.UDF_ALARM, pcaspy.Severity.INVALID_ALARM)
    ca = _ca_calls(events)
    for name in (MIXED_FLOAT_OUT, MIXED_INT):
        # the status directly follows that name's setParam
        set_at = [c[:2] for c in ca].index(("setParam", _ca(name)))
        assert ca[set_at + 1] == ("setParamStatus", _ca(name), *udf_ca)
    assert [c for c in ca if c[0] == "setParamStatus"] == [
        ("setParamStatus", _ca(MIXED_FLOAT_OUT), *udf_ca),
        ("setParamStatus", _ca(MIXED_INT), *udf_ca),
    ]
    assert ca[-1] == ("updatePVs",)
    assert ca.count(("updatePVs",)) == 1


def test_post_outputs_names_absent_from_the_reply_keep_the_range_rule_or_no_alarm() -> None:
    """With a hook every post carries an explicit pair: the handler's range
    rule where it marked one, (0, 0) where it marked none."""
    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF, udf_names=(MIXED_INT,))
    runner = _post_runner(model, events)

    runner._post_outputs(_trigger(model), POST_TS)

    posts = _posts(events)
    assert posts[MIXED_INT] == (3, 6)
    assert posts[MIXED_FLOAT_OUT] == (2, 2)  # -15.0, outside value_range
    assert posts[MIXED_FLOAT_IN] == (0, 0)  # inside value_range
    for name in (MIXED_BOOL, MIXED_STR, MIXED_ENUM):
        assert posts[name] == (0, 0)
    changed = {e[1]: e[3] for e in events if e[0] == "post"}
    for name in model.supported_variables:
        assert {"alarm.severity", "alarm.status"} <= changed[name], name


def test_post_outputs_recovered_udf_names_leave_invalid() -> None:
    """The explicit pair on the next post clears a stored (3, 6); ``_udf``
    empties; CA recovers through setParam with no setParamStatus."""
    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF)
    runner = _post_runner(model, events)
    runner._post_outputs(_trigger(model), POST_TS)
    assert _alarm(runner.pvs[MIXED_INT].current()) == (3, 6)

    events.clear()
    model.set({MIXED_FLOAT_IN: 2.0})
    runner._post_outputs(model.get(list(model.supported_variables)), POST_TS + 1)

    assert _alarm(runner.pvs[MIXED_INT].current()) == (0, 0)
    assert _alarm(runner.pvs[MIXED_FLOAT_OUT].current()) == (0, 0)
    assert runner._udf == frozenset()
    assert not [e for e in events if e[0] == "setParamStatus"]


def test_post_outputs_a_name_the_cycle_did_not_read_keeps_its_udf_state() -> None:
    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF)
    runner = _post_runner(model, events)
    runner._udf = frozenset({MIXED_STR, MIXED_INT})
    runner._cycle_requested_names = (MIXED_FLOAT_IN, MIXED_INT)
    model.set({MIXED_FLOAT_IN: 2.0})

    runner._post_outputs(model.get([MIXED_FLOAT_IN, MIXED_INT]), POST_TS)

    # MIXED_INT was read and reported clean; MIXED_STR was not read at all.
    assert runner._udf == frozenset({MIXED_STR})
    assert set(_posts(events)) == {MIXED_FLOAT_IN, MIXED_INT}


def test_post_outputs_an_extra_key_never_reaches_the_hook_udf_or_phase_1() -> None:
    """An unserved extra key and a served but unrequested one, both wrongly
    typed: neither is shown to the hook, posted, converted or put in ``_udf``."""
    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF, extra_key=True)
    runner = _post_runner(model, events)
    requested = [n for n in model.supported_variables if n != MIXED_BOOL]
    runner._cycle_requested_names = tuple(requested)
    model.set({MIXED_FLOAT_IN: MIXED_SEVERITY_TRIGGER})
    out = model.get(requested)
    assert MIXED_EXTRA_KEY in out
    out[MIXED_BOOL] = object()

    runner._post_outputs(out, POST_TS)

    assert [e for e in events if e[0] == "hook"] == [("hook", requested)]
    assert set(_posts(events)) == set(requested)
    assert _ca(MIXED_BOOL) not in _set_params(events)
    assert runner._udf == frozenset({MIXED_FLOAT_OUT, MIXED_INT})


@pytest.mark.parametrize(
    "fail_mode, error",
    [
        pytest.param(FAIL_RAISE, RuntimeError, id="hook-raises"),
        pytest.param(FAIL_MALFORMED, ValueError, id="unknown-condition"),
    ],
)
def test_post_outputs_a_failing_severity_hook_posts_nothing(fail_mode: str, error: type) -> None:
    events: list = []
    model = MixedModel(fail_mode=fail_mode)
    runner = _post_runner(model, events)
    runner._udf = frozenset({MIXED_STR})

    with pytest.raises(error):
        runner._post_outputs(_trigger(model), POST_TS)

    assert [e[0] for e in events] == ["hook"]
    assert runner._udf == frozenset({MIXED_STR})


def test_post_outputs_a_ca_conversion_error_in_phase_1_posts_nothing() -> None:
    """An enum value outside its options fails ``value_to_native`` in phase 1:
    no PVA post happens either, unlike 0.1.4's partial set."""
    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF)
    runner = _post_runner(model, events)
    out = _trigger(model)
    out[MIXED_ENUM] = "NOT-AN-OPTION"

    with pytest.raises(Exception):
        runner._post_outputs(out, POST_TS)

    assert [e[0] for e in events] == ["hook"]
    assert runner._udf == frozenset()


def _pack_failing_for(runner: Runner, bad: str) -> None:
    original = runner._pack_value

    def pack(pv, value, ts=None):
        if pv == bad:
            raise RuntimeError(f"cannot pack {pv}")
        return original(pv, value, ts)

    runner._pack_value = pack


def test_post_outputs_a_pack_error_on_a_non_udf_name_skips_only_its_pva_post() -> None:
    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF)
    runner = _post_runner(model, events)
    _pack_failing_for(runner, MIXED_BOOL)

    runner._post_outputs(_trigger(model), POST_TS)

    posts = _posts(events)
    assert MIXED_BOOL not in posts
    assert posts[MIXED_INT] == (3, 6)
    # its CA value is still written
    assert _ca(MIXED_BOOL) in _set_params(events)
    assert runner._udf == frozenset({MIXED_FLOAT_OUT, MIXED_INT})


def test_post_outputs_a_pack_error_on_a_udf_name_fails_phase_1() -> None:
    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF)
    runner = _post_runner(model, events)
    _pack_failing_for(runner, MIXED_INT)

    with pytest.raises(RuntimeError, match="cannot pack"):
        runner._post_outputs(_trigger(model), POST_TS)

    assert [e[0] for e in events] == ["hook"]
    assert runner._udf == frozenset()


def test_post_outputs_phase_2_errors_are_logged_per_name_and_never_raise() -> None:
    import pcaspy

    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF)
    runner = _post_runner(model, events, fail_set=(_ca(MIXED_FLOAT_OUT),), fail_update=True)
    runner.pvs[MIXED_INT].fail = True

    runner._post_outputs(_trigger(model), POST_TS)

    posts = _posts(events)
    assert MIXED_INT not in posts
    assert posts[MIXED_FLOAT_OUT] == (3, 6)
    # the refused setParam skips only that name; the next udf name still gets its status
    udf_ca = (pcaspy.Alarm.UDF_ALARM, pcaspy.Severity.INVALID_ALARM)
    assert ("setParamStatus", _ca(MIXED_INT), *udf_ca) in events
    assert ("setParamStatus", _ca(MIXED_FLOAT_OUT), *udf_ca) not in events
    assert runner._udf == frozenset({MIXED_FLOAT_OUT, MIXED_INT})


def test_post_outputs_with_a_hook_skips_a_none_value() -> None:
    events: list = []
    model = MixedModel(fail_mode=FAIL_UDF)
    runner = _post_runner(model, events)
    out = _trigger(model)
    out[MIXED_STR] = None

    runner._post_outputs(out, POST_TS)

    assert MIXED_STR not in _posts(events)
    assert _ca(MIXED_STR) not in _set_params(events)
    assert ("hook", list(out)) in events


def test_post_outputs_a_udf_failure_rolls_the_cycle_back_and_completes_puts() -> None:
    """A phase-1 failure takes ``_run_cycle``'s existing error path: rollback,
    and every done callback receives the error."""
    completions: list = []
    rolled_back: list = []
    events: list = []
    model = MixedModel(fail_mode=FAIL_MALFORMED)
    runner = _post_runner(model, events)
    runner.queue = Queue()
    runner.update_rate = 0.0
    runner._cached_state = {}
    runner._reset_to_cached_state = lambda: rolled_back.append(True)

    runner._run_cycle(
        _item({MIXED_FLOAT_IN: _write(MIXED_SEVERITY_TRIGGER, 4.0)}, done=completions.append)
    )

    assert rolled_back == [True]
    assert len(completions) == 1 and "no-such-condition" in completions[0]
    assert [e[0] for e in events] == ["hook"]
