import logging
import math
import numbers
import os
import platform
import threading
import time
from collections.abc import Callable, Iterable
from queue import Empty, Queue
from types import MappingProxyType
from typing import Any, NotRequired, TypedDict

from lume.model import LUMEModel, Variable
from lume.variables import IntVariable, ParticleGroupVariable

from lume_pva_apg._optional import missing_extra

try:
    import p4p.server
    from p4p import Type, Value
    from p4p.nt import NTScalar
    from p4p.server import ServerOperation
    from p4p.server.thread import SharedPV
except ImportError as exc:
    raise missing_extra("p4p", "pva", exc) from exc

try:
    import pcaspy
    import pcaspy.cas
except ImportError as exc:
    raise missing_extra("pcaspy", "ca", exc) from exc

from lume_pva_apg.epics import epicsAlarmSeverity, epicsAlarmStatus
from lume_pva_apg.variables import (
    EnumVariableHandler,
    ScalarVariableHandler,
    SimpleScalarHandler,
    VariableHandler,
    find_variable_handler,
)

try:
    from ._version import version as lume_pva_version
except ImportError:
    lume_pva_version = "HEAD"

LOG = logging.getLogger("LumePva")
logging.getLogger("pcaspy").setLevel(logging.WARNING)

VALID_PV_MODES = ["rw", "ro"]

DEFAULT_PV_MODE = "rw"

# Base name of the reset control PV, served as f"{prefix}{RESET_CONTROL_PV}".
RESET_CONTROL_PV = "RESET"


class RunnerVariable(TypedDict):
    """
    Attributes
    ----------
    name : str
        Name of the input or output. Must match one of the model's supported variables.
    pv : str
        Name of the PV to serve or consume. If not provided, it will be defaulted to 'name'
    mode : str
        Operation mode of the PV. May be one of:
        - 'ro': Read-only PV served by this server
        - 'rw': Read-write PV served by this server. Errors if Variable.read_only
        Default is 'rw'
    precision : int, optional
        Number of digits after the decimal point a display client shows. An int
        in 0..17 (the significant digits a double carries), allowed only on a
        float ``ScalarVariable``.
    description : str, optional
        Text served as the PV's description. Overrides the variable's own
        ``description`` (lume-base 0.6); ``""`` means none. Rejected on array
        and Torch variables, whose PVs carry no display block.
    """

    name: str
    pv: str
    mode: str
    precision: NotRequired[int]
    description: NotRequired[str]


# Upper bound on a configured precision: the significant decimal digits an
# IEEE double carries. A policy bound -- CA's dbr_short field would take more,
# but no digit past the 17th means anything.
MAX_PRECISION = 17

# Handlers whose PVA type carries a display block, and so can serve a
# description. Compared by exact type: TorchScalarVariableHandler and
# NDVariableHandler are deliberately absent.
_DESCRIBED_HANDLERS = (ScalarVariableHandler, SimpleScalarHandler, EnumVariableHandler)

# Conditions a model's output_severity hook may report, each mapped to the
# alarm it raises: the PVA (severity, status) pair and the CA (alarm, severity)
# pair handed to pcaspy's setParamStatus.
_CONDITIONS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "udf": (
        (int(epicsAlarmSeverity.INVALID_ALARM), int(epicsAlarmStatus.UDF_STATUS)),
        (pcaspy.Alarm.UDF_ALARM, pcaspy.Severity.INVALID_ALARM),
    ),
}
# The alarm an undefined output carries on each transport.
_UDF_PVA, _UDF_CA = _CONDITIONS["udf"]


def _resolve_pv_meta(
    name: str,
    entry: RunnerVariable | dict,
    variable: Variable,
    handler: VariableHandler,
) -> dict[str, Any]:
    """Validate a config entry's ``precision`` and ``description``.

    Parameters
    ----------
    name : str
        Variable name, used in error messages.
    entry : RunnerVariable
        The variable's config entry. Only read.
    variable : Variable
        The model's variable. Only read.
    handler : VariableHandler
        The handler resolved for ``variable``; never None.

    Returns
    -------
    dict
        ``{"precision": int | None, "description": str | None}``. A
        description of ``""`` -- configured or from the variable -- is None.

    Raises
    ------
    ValueError
        A key is malformed or not allowed on this variable's type.
    """
    handler_type = type(handler)

    precision = entry.get("precision")
    if "precision" in entry:
        # type() rather than isinstance: bool is an int subclass, and True is
        # not a precision.
        if type(precision) is not int or not 0 <= precision <= MAX_PRECISION:
            raise ValueError(
                f"Variable {name}: 'precision' must be an int in 0..{MAX_PRECISION}, "
                f"got {precision!r}"
            )
        # IntVariable shares ScalarVariableHandler but is served as an integer
        # NTScalar, where digits after the decimal point mean nothing.
        if handler_type is not ScalarVariableHandler or isinstance(variable, IntVariable):
            raise ValueError(
                f"Variable {name}: 'precision' is only allowed on a float scalar variable, "
                f"not {type(variable).__name__}"
            )

    described = handler_type in _DESCRIBED_HANDLERS
    if "description" in entry:
        description = entry["description"]
        if not isinstance(description, str):
            raise ValueError(
                f"Variable {name}: 'description' must be a str, got {type(description).__name__}"
            )
        if not described:
            raise ValueError(
                f"Variable {name}: 'description' is not supported on {type(variable).__name__}"
            )
    elif described:
        # lume-base 0.6 added Variable.description; older versions lack it.
        description = getattr(variable, "description", None)
    else:
        # A model-side description on an array or Torch variable is ignored,
        # so a lume-base 0.6 model carrying one still boots.
        description = None

    return {"precision": precision, "description": description or None}


class RunnerConfig(TypedDict):
    """
    Attributes
    ----------
    prefix : str
        Additional prefix to append to PV names. May be None if you don't need any.
    variables : Dict[str, RunnerVariable]
        List of model variables
    protocol : list[str]
        List of supported protocols
    update_rate : float
        Length in seconds of the window during which incoming writes are batched
        into a single ``model.set()``. Zero disables the window: every queued
        write gets a ``model.set()`` of its own, so one client's write is never
        merged with another's. Default 0.1.
    echo_unconfirmed_writes : bool
        Whether an input PV publishes the requested value before the model has
        accepted it. True (the default) posts the echo as soon as the write is
        queued -- and leaves it standing even if the simulation cycle that
        consumed it failed. False defers the echo until the model has accepted
        the write and withholds it entirely if the model refused, so the PV
        keeps the last value the model actually took.
    alarm_on_refused_write : bool
        Whether a refused write raises WRITE_ALARM/INVALID_ALARM on the CA PV.
        Channel Access put-completion carries no failure channel -- it can only
        ever report success -- so an alarm is the only way to tell a CA client
        its write did not land. Default False.
    clamp_writes : bool
        Whether an incoming write is clamped into the target variable's
        ``value_range`` before being handed to ``model.set()``. Default False,
        which passes the requested value through unchanged and leaves any
        range enforcement to the model.
    control_pvs : bool
        Whether the runner serves its own control PVs -- ``{prefix}RESET`` --
        alongside the model's variables. Default True. Set it to False when the
        runner shares its prefix with another server, or when the surrounding
        deployment offers reset through a channel of its own, so the runner
        claims no name the model did not ask for.
    tick_interval_s : float | None
        Period in seconds of the runner's own passes: every interval, a pass
        with no input values is queued, so a model whose outputs move on their
        own is published without a client write. Absent or None, the default,
        queues none. At most one such pass is ever waiting in the queue, so a
        model slower than the interval is never buried under a backlog.
        Otherwise a finite int or float greater than zero; anything else is
        rejected with a ValueError.
    """

    prefix: str
    variables: dict[str, RunnerVariable]
    protocol: list[str]
    update_rate: float
    echo_unconfirmed_writes: bool
    alarm_on_refused_write: bool
    clamp_writes: bool
    control_pvs: bool
    tick_interval_s: NotRequired[float | None]


class Runner:
    """Simple runner for LUMEModel derived models"""

    pvs: dict[str, SharedPV]
    ca_pvs: dict[str, str]
    pv_handlers: dict[str, VariableHandler]
    # List of all output PVs that need to be updated after simulation
    outputs: list[str]
    values: dict[str, Value]
    # Validated precision/description per served variable name. The class
    # default lets a Runner built without __init__ read it; __init__ replaces it.
    _pv_meta: MappingProxyType | dict[str, dict[str, Any]] = MappingProxyType({})
    # Names whose served value is undefined: _generate_value overlays
    # (INVALID_ALARM, UDF_STATUS) on them. Immutable, and empty by default so a
    # Runner built without __init__ publishes exactly as 0.1.4 did.
    _udf: frozenset[str] = frozenset()
    # The names this cycle asked ``model.get`` for. ``_post_outputs`` evaluates
    # severity over these only: ``LUMEModel.get`` validates just the requested
    # names, so any extra key a model's ``_get`` returned is unvalidated.
    _cycle_requested_names: tuple[str, ...] = ()
    # Periodic-pass state (``tick_interval_s``). ``_tick_pending`` is true while
    # a tick item sits in the queue, so at most one is ever waiting;
    # ``_pass_is_tick`` tells a subclass the current pass is a pure tick. The
    # defaults describe a runner with no ticker, so a Runner built without
    # __init__ behaves as one that never ticks.
    _tick_interval_s: float | None = None
    _tick_pending: bool = False
    _pass_is_tick: bool = False
    _ticker: threading.Thread | None = None
    _ticker_stop: threading.Event | None = None

    class Handler:
        """
        Handles PUT and RPC requests to a specific PV
        """

        model: LUMEModel
        variable: Variable

        def __init__(self, variable: Variable, runner: "Runner", read_only: bool):
            self.model = runner.model
            self.variable = variable
            self.runner = runner
            self.ro = read_only

        def put(self, pv: SharedPV, op: ServerOperation):
            if self.ro:
                op.done(error="Read only PV")
                return

            value = self.runner._clamp_write(self.variable, op.value())
            # The alarm is the server's to set, never the client's: a put that
            # carried alarm fields must not overwrite a standing alarm (such as
            # an undefined output's INVALID/UDF) through either echo below.
            # SharedPV.post stores only marked fields, so unmarking them keeps
            # the stored alarm. Each leaf is unmarked on its own because
            # unmarking the "alarm" parent leaves the leaves marked.
            for field in ("alarm.severity", "alarm.status", "alarm.message"):
                value.mark(field, False)

            def _complete(error: str | None) -> None:
                # The echo is the value the model was given, so it may only be
                # published once the model has taken it. A cycle that failed
                # left the model on its previous value; publishing the request
                # anyway would leave the PV advertising a value that was never
                # applied.
                if error is None and not self.runner.echo_unconfirmed_writes:
                    pv.post(value)
                op.done(error=error)

            # Update PVs in simulator
            self.runner._enqueue(
                {self.variable.name: {"value": value, "ts": time.time()}},
                done=_complete,
            )
            if self.runner.echo_unconfirmed_writes:
                pv.post(value)
            LOG.debug(f"Setting PVA: {self.variable.name} -> {value}")

        def rpc(self, op: ServerOperation):
            op.done()

    class CaDriver(pcaspy.Driver):
        """ChannelAccess driver handling operations on behalf of the Runner class"""

        def __init__(self, runner: "Runner"):
            super().__init__()
            self.runner = runner

        def write(self, reason, value) -> bool:
            if reason == self.runner.reset_control_pv:
                self.runner._enqueue({}, reset=True)
                self.setParam(reason, value)
                self.callbackPV(reason)
                return True

            # Lookup variable based on name
            vn = self.runner.pv_to_var.get(reason, None)
            if vn is None:
                return False

            var: Variable = self.runner.model.supported_variables.get(vn, None)
            if var is None:
                raise NameError(f"No variable named {vn} associated with pv {reason}")

            # Reject writes to read-only PVs
            if var.read_only:
                return False

            nv = value

            # Transform int -> str for enums. Must be done before we submit it to the variable queue
            desc = self.runner.pvdb[reason]
            if desc.get("type") == "enum":
                # Check range
                if value < 0 or value >= len(desc["enums"]):
                    LOG.info(f"{reason}: Rejected invalid enum value {value} for")
                    return False
                nv = desc["enums"][value]
            else:
                nv = self.runner._clamp_write(var, value)
                value = nv

            # Insert into update queue
            def _complete_put(error: str | None) -> None:
                accepted = error is None

                # The echo advertises the value the model was given. A failed
                # cycle left the model on its previous value, so recording the
                # request leaves the PV holding a value that never landed.
                if accepted or self.runner.echo_unconfirmed_writes:
                    self.setParam(reason, value)
                # setParam recomputes the alarm from the value, which would
                # clear a standing UDF the model reported on the last cycle.
                # Restated whether or not setParam ran, and before the refusal
                # alarm so a refusal still reads as the latest word.
                if vn in self.runner._udf:
                    self.setParamStatus(reason, *_UDF_CA)
                if not accepted and self.runner.alarm_on_refused_write:
                    # Put-completion can only ever report success, so an alarm
                    # is the sole channel available to tell the client its write
                    # did not take. Must follow setParam, which recomputes the
                    # alarm from the value.
                    self.setParamStatus(
                        reason, pcaspy.Alarm.WRITE_ALARM, pcaspy.Severity.INVALID_ALARM
                    )

                # Commit, then signal. A value handed to setParam reaches a
                # monitoring client only when updatePV is called, so without
                # this flush a client unblocking on put-completion is told the
                # write finished while its monitor still carries the value that
                # write replaced.
                #
                # The single unflushed case is a refusal with neither policy
                # enabled: the value recorded just above was never taken by the
                # model, and publishing it is the very thing this change
                # exists to prevent.
                if accepted or not self.runner.echo_unconfirmed_writes:
                    self.updatePV(reason)
                elif self.runner.alarm_on_refused_write:
                    self.updatePV(reason)

                self.callbackPV(reason)

            self.runner._enqueue(
                {vn: {"value": nv, "ts": time.time()}},
                done=_complete_put,
            )
            return True

    #: Driver class instantiated to serve the CA database. Override in a
    #: subclass to serve reasons the stock driver knows nothing about; the
    #: replacement is constructed with the runner as its only argument.
    ca_driver_cls: type[pcaspy.Driver] = CaDriver

    def __init__(
        self,
        model: LUMEModel,
        prefix="",
        config: RunnerConfig | None = None,
    ):
        """
        Init a Runner for the specified model

        Parameters
        ----------
        model : LUMEModel
            A LUMEModel object implementing the LUMEModel interface
        prefix: str
            Prefix to append to PV names. Only applies to PVs served by the Runner
        config: RunnerConfig|None
            Configuration for this runner. If 'None' a default configuration is generated.
            Note that you may call Runner.generate_config yourself to get+modify a configuration.
            Overrides the 'prefix' parameter.
        """
        self.model = model
        self.pvs = {}
        self.pv_handlers = {}
        self.queue = Queue()
        self.new_values = {}
        self.outputs = []
        self.types = {}
        self.providers = {}  # Just for renaming
        self.pvdb = {}  # For pcaspy
        self.pv_to_var: dict[str, str] = {}  # Map pv name -> variable name
        self.var_to_pv = {}
        self.ca_pvs = {}
        self._pv_meta = {}
        # Must be set before the PVs are built: _add_pv's initial value goes
        # through _generate_value, which reads it.
        self._udf = frozenset()
        # Tick state exists whether or not tick_interval_s is set, so the queue
        # and the run loop never have to ask which kind of runner they serve.
        self._tick_lock = threading.Lock()
        self._tick_pending = False
        self._pass_is_tick = False
        # Base name of the reset control PV -- the key it holds in the pvdb, and
        # the reason the CA driver is called back with. Empty when control PVs
        # are suppressed.
        self.reset_control_pv = ""
        self.ca_server: pcaspy.SimpleServer | None = None
        self.ca_driver: pcaspy.Driver | None = None

        # Cache for previous state, value per name
        self._cached_state: dict[str, Any] = {}

        # Generate default config
        if config is None:
            config = self.generate_config(model, prefix)
        self._config = config

        # Grab list of supported protocols
        self.protos = self._config.get("protocol", ["ca", "pva"])
        self.supports_ca = "ca" in self.protos
        self.supports_pva = "pva" in self.protos

        self.update_rate = config.get("update_rate", 0.1)

        # Checked before any PV is built, so a bad value fails the whole start.
        # A bool is refused even though it is an int: True reads as "yes, tick",
        # not as one second.
        tick_interval_s = config.get("tick_interval_s")
        if tick_interval_s is not None and (
            type(tick_interval_s) not in (int, float)
            or not math.isfinite(tick_interval_s)
            or tick_interval_s <= 0
        ):
            raise ValueError(
                "tick_interval_s must be a finite number of seconds greater than zero, "
                f"or None; got {tick_interval_s!r}"
            )
        self._tick_interval_s = tick_interval_s

        # Write-path policy. Every default here reproduces the behaviour of a
        # runner that sets none of them.
        self.echo_unconfirmed_writes = bool(config.get("echo_unconfirmed_writes", True))
        self.alarm_on_refused_write = bool(config.get("alarm_on_refused_write", False))
        self.clamp_writes = bool(config.get("clamp_writes", False))

        # Configure CA environment
        os.environ["EPICS_CA_MAX_ARRAY_BYTES"] = self.config.get("max_array_bytes", 80000000)

        # Setup PVs
        for c in self.config["variables"].values():
            # Set default PV name if not provided
            if "pv" not in c:
                c["pv"] = c["name"]
            pv = c["pv"]

            # Validate some other things first
            if c["name"] not in self.model.supported_variables:
                raise KeyError(f'Variable "{c["name"]}" not found in model variables')
            if "mode" in c and c["mode"] not in VALID_PV_MODES:
                raise KeyError(
                    f'Variable "{c["name"]} has invalid mode "{c["mode"]}". Must be one of {VALID_PV_MODES}'
                )

            # Lookup variable based on name
            var = self.model.supported_variables[c["name"]]

            # Determine a default mode, if there is none
            if "mode" not in c:
                c["mode"] = "ro" if var.read_only else "rw"

            # Validate r/w setting
            if c["mode"] == "rw" and var.read_only:
                raise ValueError(
                    f"Variable {c['name']} was configured with read-write permissions, but the variable is read-only"
                )

            handler = find_variable_handler(type(var))
            if handler is None:
                if isinstance(var, ParticleGroupVariable):
                    continue  # ParticleGroupVariable is a special case that doesn't have a handler
                raise RuntimeError(f'Unknown type "{type(var)}"')

            # Skip unsupported variable types
            if not handler.is_supported(var):
                LOG.warning(f'Unsupported variable "{var.name}". Skipping.')
                continue

            # Validated only for a variable that is served: a skipped one keeps
            # its warn-and-skip, whatever its entry carries.
            self._pv_meta[var.name] = _resolve_pv_meta(c["name"], c, var, handler)

            # Cache handler and type for later
            self.pv_handlers[var.name] = handler
            self.types[var.name] = handler.create_type(var, **self._precision_kwargs(var.name))

            self.pv_to_var[pv] = var.name
            self.var_to_pv[var.name] = pv

            # Generate a PV to be served
            self._add_pv(
                pv,
                var,
                ro=c["mode"] == "ro",
                prefix=self.config.get("prefix", ""),
                handler=handler,
            )

        # Create an informational PV (i.e. including list of variables, etc.)
        # Only supported for PVA since it uses structures
        if self.supports_pva:
            self._create_model_info()

        # Create additional control PVs
        if self.config.get("control_pvs", True):
            self._create_control_pvs()

        # Let a subclass add PVs of its own to the served CA database. Done
        # before the server is created, since createPV takes the database whole.
        if self.supports_ca:
            self._merge_pvdb(self._extend_pvdb())

        # Start the server
        self.server = p4p.server.Server(providers=[self.providers])

        # Start the CA server under the shared async context
        if len(self.pvdb.keys()) > 0:
            self.ca_server = pcaspy.SimpleServer()
            self.ca_server.createPV(self.config.get("prefix", ""), self.pvdb)
            self.ca_driver = self.ca_driver_cls(self)

            # Spin up a thread to run the pcaspy update loop
            self.ca_thread = threading.Thread(target=self._run_pcaspy, daemon=True)

            self.ca_thread.start()

        # Kick off an initial update to propagate any defaults the model may have set
        self._enqueue({})

    def _run_pcaspy(self):
        """Run pcaspy forever"""
        while True:
            self.ca_server.process(0.1)

    @staticmethod
    def generate_config(
        model: LUMEModel,
        prefix: str = "",
        name_transformer: Callable[[Variable, str], str] | None = None,
    ) -> RunnerConfig:
        """
        Generate a configuration for the specified model.

        Parameters
        ----------
        model : LUMEModel
            Instance of a LUMEModel object
        prefix : str
            PV name prefix
        name_transformer: Callable[[Variable, str], str] | None
            A callable that transforms a variable's name into a new PV name. by default it just maps variable.name -> pv_name

        Returns
        -------
        RunnerConfig :
            A new configuration built based on the supplied parameters. May be tweaked as you wish before
            passing to the Runner() constructor.
        """
        config = {
            "description": "",
            "prefix": prefix,
            "max_array_bytes": os.environ.get("EPICS_CA_MAX_ARRAY_BYTES", "80000000"),
            "variables": {},
        }
        for k, v in model.supported_variables.items():
            mode = "ro" if v.read_only else "rw"
            if name_transformer is not None:
                pv = name_transformer(v, v.name)
            else:
                pv = k
            config["variables"][k] = {
                "name": k,
                "pv": pv,
                "mode": mode,
            }
        return config

    def _enqueue(
        self,
        values: dict[str, Any],
        done: Callable[[str | None], None] | None = None,
        reset: bool = False,
        *,
        jobs: Iterable[Callable[[], None]] = (),
    ) -> None:
        """
        Enqueue a batch of PV updates to be applied to the model.

        Parameters
        ----------
        values : Dict[str, Any]
            Mapping of variable name -> {"value": ..., "ts": ...}
        done : Callable[[str | None], None] | None
            Optional completion callback. Invoked once the simulation that
            consumes these values has finished (or failed). Receives an error
            string on failure, or None on success. Used to defer signalling
            put-completion to clients until results are actually available.
        reset : bool
            When true, request model.reset() before applying this batch.
        jobs : Iterable[Callable[[], None]]
            Keyword-only. Callables the run loop invokes on its own thread, in
            the order given. A job owns its operation end to end, including its
            own error handling: the loop passes it no arguments and reads
            nothing back from it. Collected into a list when the batch is
            enqueued, so a one-shot iterable is safe to pass. Empty by default.
        """
        self.queue.put(
            {
                "values": values,
                "done": [done] if done is not None else [],
                "reset": reset,
                "jobs": list(jobs),
                "tick": False,
            }
        )

    def _tick(self) -> None:
        """
        Queue one periodic pass, unless one is already waiting.

        Called by the ticker thread every ``tick_interval_s``. The pending flag
        is checked and set under ``_tick_lock``, set before the item is queued,
        and cleared only by ``_run_cycle`` when it takes the tick off the queue,
        so a model slower than the interval sees at most one tick waiting. The
        item is built fresh on every call: the run loop mutates the items it
        merges, so no two ticks may share one.
        """
        with self._tick_lock:
            if self._tick_pending:
                return
            self._tick_pending = True
            self.queue.put({"values": {}, "done": [], "reset": False, "jobs": [], "tick": True})

    def _clamp_write(self, variable: Variable, value: Any) -> Any:
        """
        Clamp an incoming write into the variable's ``value_range``.

        A no-op unless the ``clamp_writes`` configuration key is set, so range
        enforcement stays the model's job by default. Values the range cannot
        describe -- non-numerics, arrays, a variable with no ``value_range`` --
        are passed through untouched.

        Applied where the write enters the server rather than where it reaches
        the model, so the value the client is echoed is the same value the
        model was given.

        Parameters
        ----------
        variable : Variable
            The variable being written.
        value : Any
            The requested value. A p4p ``Value`` is clamped through its
            ``value`` field, in place; anything else is treated as the raw
            value.

        Returns
        -------
        Any :
            The clamped value, or `value` unchanged.
        """
        if not self.clamp_writes:
            return value

        if isinstance(value, Value):
            try:
                raw = value["value"]
            except (KeyError, TypeError):
                return value
            clamped = self._clamp_scalar(variable, raw)
            if clamped is not raw:
                value["value"] = clamped
            return value

        return self._clamp_scalar(variable, value)

    def _clamp_scalar(self, variable: Variable, value: Any) -> Any:
        """Clamp a native scalar into `variable`'s ``value_range``."""
        value_range = getattr(variable, "value_range", None)
        if value_range is None:
            return value
        # bool is a Real, and clamping a boolean into a numeric range is not a
        # meaningful operation.
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return value

        low, high = value_range[0], value_range[1]
        clamped = min(max(value, low), high)
        if clamped == value:
            return value

        # An integer variable must stay integral even if its range is not.
        if isinstance(value, numbers.Integral):
            clamped = int(clamped)

        LOG.info(f"{variable.name}: clamped write {value} into {tuple(value_range)} -> {clamped}")
        return clamped

    def _add_pv(
        self, pv: str, var: Variable, ro: bool, prefix: str, handler: VariableHandler
    ) -> None:
        """
        Create a new PV for CA and/or PVA

        Parameters
        ----------
        pv : str
            Name of the PV
        var : Variable
            LUME variable this PV is implementing
        ro : bool
            True if read-only
        prefix : str
            String to prefix the PV name with. Applies to the PVA provider name
            only: pcaspy prefixes the CA names itself, in ``createPV``, from the
            base names the pvdb is keyed by.
        handler : VariableHandler
            The variable handler for this variable type
        """
        if self.supports_pva:
            LOG.debug(f"Creating PVA PV: pv={pv}")
            pvobj = SharedPV(
                handler=Runner.Handler(variable=var, runner=self, read_only=ro),
                initial=self._generate_value(var.name, None),
            )
            self.pvs[var.name] = pvobj
            self.providers[f"{prefix}{pv}"] = pvobj

        if self.supports_ca:
            # Generate a default value suitable for pcaspy
            default_value = handler.default_value(var, flatten=True, native_python=True)

            # String arrays are not really supported in channel access. Skip it.
            if isinstance(default_value, list) and isinstance(default_value[0], str):
                return

            LOG.debug(f"Creating CA PV: pv={pv}")
            spec = handler.ca_pvspec(var, **self._precision_kwargs(var.name))

            # Keyed by the base name: SimpleServer.createPV prepends the prefix
            # to build the served name, and every callback into the driver --
            # write's `reason`, setParam, updatePV -- names the PV by this key.
            self.pvdb[pv] = spec
            self.pvdb[pv].update({"asyn": True})
            # enable async for put-completion
            self.ca_pvs[var.name] = pv

    def _create_model_info(self):
        """Creates a model info PV for PVA"""
        pv = "MODEL_INFO"

        envs = [
            "EPICS_CA_ADDR_LIST",
            "EPICS_CA_AUTO_ADDR_LIST",
            "EPICS_CA_SERVER_PORT",
            "EPICS_CA_CONN_TMO",
            "EPICS_CA_MAX_ARRAY_BYTES",
            "EPICS_CA_REPEATER_PORT",
            "EPICS_PVA_ADDR_LIST",
            "EPICS_PVA_AUTO_ADDR_LIST",
            "EPICS_PVA_SERVER_PORT",
            "EPICS_PVA_CONN_TMO",
            "EPICS_PVA_BROADCAST_PORT",
        ]

        self.types[pv] = Type(
            [
                ("class", "s"),
                ("description", "s"),
                ("lume_pva_version", "s"),
                ("hostname", "s"),
                (
                    "env",
                    ("S", None, [(x, "s") for x in envs]),
                ),
                (
                    "supported_variables",
                    (
                        "aS",
                        None,
                        [
                            ("name", "s"),
                            ("pvname", "s"),
                            ("type", "s"),
                            ("read_only", "?"),
                            ("mode", "s"),
                        ],
                    ),
                ),
            ]
        )

        val = Value(self.types[pv])
        val["class"] = self.model.__class__.__name__
        val["description"] = self.config["description"]
        val["lume_pva_version"] = lume_pva_version
        val["hostname"] = platform.node()

        for e in envs:
            val["env"][e] = os.environ.get(e, "")

        # The info describes what is served. A model variable the
        # configuration omits has no channel on either transport, so it has
        # no place here either.
        vars = []
        for k, v in self.model.supported_variables.items():
            spec = self.config["variables"].get(k)
            if spec is None:
                continue
            info = {
                "name": v.name,
                "read_only": v.read_only,
                "pvname": spec["pv"],
                "type": v.__class__.__name__,
                "mode": spec["mode"],
            }
            vars.append(info)

        val["supported_variables"] = vars

        self.pvs[pv] = SharedPV(initial=val)
        self.providers[f"{self.config['prefix']}{pv}"] = self.pvs[pv]

    def _cycle_output_names(self) -> list[str]:
        """
        Names of the variables read back from the model after each cycle.

        Called on every model pass, after ``model.set``; :meth:`_post_outputs`
        receives exactly these variables' values. The base implementation reads
        the model's whole roster, which is what the stock output step publishes.
        A subclass that publishes only part of the model, or publishes it
        elsewhere, narrows the read here so a cycle does not compute values
        nothing will use. An empty list skips the read entirely, and
        :meth:`_post_outputs` is handed an empty dict.

        Only the post-cycle read is narrowed. The snapshot of the settable state
        that a failed cycle rolls back to is still taken in full.

        Returns
        -------
        list[str] :
            Variable names to pass to ``model.get``, each a key of
            ``model.supported_variables``.
        """
        return list(self.model.supported_variables)

    def _evaluate_severity(self, names: list[str]) -> frozenset[str]:
        """
        Ask the model which of ``names`` are undefined this cycle.

        A model opts in by defining a callable ``output_severity(names)``; it is
        called once and must return a dict mapping a variable name to
        ``{"condition": <condition>}``, where the only known condition is
        ``"udf"`` (see ``_CONDITIONS``). Keys outside ``names`` are ignored --
        the model may report on variables this cycle does not publish. The
        result depends only on the model's reply; runner state is not touched.

        Parameters
        ----------
        names : list[str]
            Variable names published this cycle.

        Returns
        -------
        frozenset[str] :
            The names reported ``"udf"``; empty when the model has no hook.

        Raises
        ------
        ValueError
            The reply is not a dict, or an entry for a requested name is not a
            dict carrying a known condition. An exception raised by the hook
            itself propagates unchanged.
        """
        hook = getattr(self.model, "output_severity", None)
        if not callable(hook):
            return frozenset()

        reply = hook(names)
        if not isinstance(reply, dict):
            raise ValueError(
                f"output_severity must return a dict, got {type(reply).__name__}: {reply!r}"
            )

        requested = set(names)
        udf = set()
        for name, entry in reply.items():
            if name not in requested:
                continue
            if not isinstance(entry, dict):
                raise ValueError(
                    f"output_severity entry for variable '{name}' must be a dict "
                    f"with a 'condition' key, got {entry!r}"
                )
            condition = entry.get("condition")
            # isinstance first: an unhashable condition would make the
            # membership test raise TypeError instead of this ValueError.
            if not isinstance(condition, str) or condition not in _CONDITIONS:
                raise ValueError(
                    f"output_severity reported unknown condition {condition!r} for "
                    f"variable '{name}'; known conditions: {sorted(_CONDITIONS)}"
                )
            udf.add(name)
        return frozenset(udf)

    def _extend_pvdb(self) -> dict[str, dict[str, Any]]:
        """
        Additional entries to serve alongside the model's own CA PVs.

        Called once, before the CA server is created, and only when the CA
        protocol is enabled. The base implementation contributes nothing; a
        subclass returning entries here serves PVs the model does not describe,
        and is expected to supply a :attr:`ca_driver_cls` that knows how to read
        and write them -- the stock driver resolves a reason through the model's
        variables and refuses anything it cannot find.

        Returns
        -------
        dict[str, dict[str, Any]] :
            pcaspy database entries, keyed by base PV name exactly as the pvdb
            is: ``prefix`` is applied by the server, not here.
        """
        return {}

    def _merge_pvdb(self, entries: dict[str, dict[str, Any]]) -> None:
        """
        Merge `entries` into the served database, refusing to shadow a PV.

        A name already in the pvdb belongs to a model variable or to a control
        PV, and silently replacing it would serve something other than what the
        model describes under a name the model owns.
        """
        if not entries:
            return

        conflicts = sorted(set(entries) & set(self.pvdb))
        if conflicts:
            raise RuntimeError(f"Fatal name conflict: {', '.join(conflicts)} already exist!")

        self.pvdb.update(entries)

    def _create_control_pvs(self):
        """Create any required control PVs"""
        # Create a reset PV, used to reset the model.
        #
        # The CA database is keyed by base name -- pcaspy prepends the prefix
        # itself, and names the PV by its base name in every driver callback --
        # while a PVA provider is registered under the full name.
        reset_reason = RESET_CONTROL_PV
        reset_pvname = f"{self.config['prefix']}{reset_reason}"
        self.reset_control_pv = reset_reason

        # Create a PVA shared PV for reset if PVA is enabled
        if self.supports_pva:
            if reset_pvname in self.providers:
                raise RuntimeError(
                    f"Fatal name conflict: {reset_pvname} for the reset PV already exists!"
                )

            self.providers[reset_pvname] = SharedPV(initial=NTScalar("i").wrap(0))

            @self.providers[reset_pvname].put
            def onResetPut(pv, op):
                self._enqueue(
                    {},
                    reset=True,
                )
                op.done()

        # Create the CA reset PV if CA is enabled
        if self.supports_ca:
            if reset_reason in self.pvdb:
                raise RuntimeError(
                    f"Fatal name conflict: {reset_reason} for the CA reset PV already exists!"
                )

            self.pvdb[reset_reason] = {
                "type": "int",
                "value": 0,
                "asyn": False,
            }

        return None

    def _precision_kwargs(self, name: str) -> dict[str, int]:
        """Return ``{"precision": p}`` for a float scalar configured with one.

        Only ``ScalarVariableHandler`` (compared by exact type) accepts the
        keyword; every other handler's ``create_type``/``ca_pvspec`` is called
        exactly as in 0.1.4, and so is a scalar configured without a precision.
        """
        precision = self._pv_meta.get(name, {}).get("precision")
        if precision is None or type(self.pv_handlers.get(name)) is not ScalarVariableHandler:
            return {}
        return {"precision": precision}

    def _generate_value(self, pv: str, value: Any | None, ts: float | None = None) -> Value:
        """
        Generates a new value for posting to the PV.
        Handles alarm updates, timestamp updates, and generating the value in the first place. This handles the
        'common' metadata that the variable handlers shouldn't need to handle.

        A name in ``self._udf`` is published with (INVALID_ALARM, UDF_STATUS)
        in place of the handler's alarm pair. :meth:`_pack_value` never applies
        that overlay.
        """
        v = self._pack_value(pv, value, ts)
        if pv in self._udf:
            v["alarm"]["severity"], v["alarm"]["status"] = _UDF_PVA
        return v

    def _pack_value(self, pv: str, value: Any | None, ts: float | None = None) -> Value:
        """
        Packs ``value`` for ``pv`` with its handler and stamps it with ``ts``
        (the current UNIX time when omitted).
        """
        handler = self.pv_handlers[pv]
        variable = self.model.supported_variables[pv]
        v = handler.pack_value(variable, self.types[pv], value)

        # Display metadata from the configuration. Read with .get so a runner
        # built without __init__ (class default: empty) packs as 0.1.4 did.
        meta = self._pv_meta.get(pv, {})
        precision = meta.get("precision")
        if precision is not None:
            v["display"]["precision"] = precision
        if type(handler) in _DESCRIBED_HANDLERS:
            handler.set_display_metadata(variable, v, description=meta.get("description"))

        # Ensure timestamp is current
        self._update_timestamp(v, ts=ts)
        return v

    def _update_timestamp(self, value: Value, ts=None) -> None:
        """
        Helper to update timestamp on a value
        """
        if ts is None:
            ts = time.time()
        value["timeStamp"]["nanoseconds"] = math.fmod(ts, 1.0) * 1e9
        value["timeStamp"]["secondsPastEpoch"] = int(ts)

    @property
    def config(self) -> RunnerConfig:
        """Access the underlying config"""
        return self._config

    def _ticker_join_timeout(self) -> float:
        """Bound on waiting for a stopping ticker: two intervals, never under 5 s."""
        return max(2 * (self._tick_interval_s or 0.0), 5.0)

    def _start_ticker(self) -> None:
        """
        Start the thread that queues a periodic pass every ``tick_interval_s``.

        A no-op without an interval, or while a ticker is running and has not
        been asked to stop. A ticker that is stopping is joined first, so two
        never run at once. The pending flag is reset before the new thread
        starts: an interrupt between a tick's dequeue and its clear would
        otherwise leave it set, and no later tick would ever be queued.

        The stop event is held in the thread's closure rather than read from
        ``self``, so a restart that replaces ``_ticker_stop`` cannot leave an
        old thread waiting on an event nobody will set.
        """
        interval = self._tick_interval_s
        if interval is None:
            return
        ticker, stop = self._ticker, self._ticker_stop
        if ticker is not None and ticker.is_alive():
            if stop is not None and not stop.is_set():
                return
            ticker.join(timeout=self._ticker_join_timeout())
        with self._tick_lock:
            self._tick_pending = False
        stop = threading.Event()

        def tick_loop() -> None:
            while not stop.wait(interval):
                self._tick()

        ticker = threading.Thread(target=tick_loop, name="lume-pva-ticker", daemon=True)
        self._ticker_stop = stop
        self._ticker = ticker
        ticker.start()

    def _stop_ticker(self) -> None:
        """
        Stop the ticker thread, if there is one, and wait for it to end.

        The references are dropped only once the thread is dead; a thread still
        alive after the bounded join is logged and kept, so the next
        :meth:`_start_ticker` sees it stopping and joins it again.
        """
        ticker, stop = self._ticker, self._ticker_stop
        if ticker is None:
            return
        if stop is not None:
            stop.set()
        ticker.join(timeout=self._ticker_join_timeout())
        if ticker.is_alive():
            LOG.warning("tick thread did not stop within its join timeout; leaving it to exit")
            return
        self._ticker = None
        self._ticker_stop = None

    def _run(self):
        """
        Runs the simulation, blocks forever.
        Dequeues PV updates from the updater thread, sets values on the model, and updates outputs.
        The ticker, when ``tick_interval_s`` is set, runs for exactly as long as
        this loop does, so a second ``run()`` after an interrupt ticks again.
        """
        self._start_ticker()
        try:
            while True:
                # Wait for new data to come in
                self._run_cycle(self.queue.get())
        finally:
            self._stop_ticker()

    def _run_cycle(self, item: dict) -> None:
        """
        Run one cycle of the run loop for a dequeued item.

        A cycle has two parts. First, every job in the batch runs, in arrival
        order. Then the model pass runs: snapshot the settable state, reset if
        requested, ``model.set`` the batch's values, read back the variables
        :meth:`_cycle_output_names` lists and publish them. The pass runs only when the batch has something for the
        model -- values, a reset, a tick, or no jobs at all (the empty start-up
        item publishes the initial outputs). ``_pass_is_tick`` is true only
        when every item of the batch is a tick and it carries no values and no
        reset. A batch of jobs alone touches the model
        only through its jobs.

        A job owns its operation and its reply, so a job that raises anyway is
        logged and passed over. That does not fail the cycle: the remaining
        jobs and the pass still run. Only a failed pass rolls the model back to
        the cached state and hands its error to the batch's completion
        callbacks, which receive ``None`` otherwise, including when the pass was
        skipped.

        Parameters
        ----------
        item : dict
            A queue item as built by :meth:`_enqueue`. Items drained during the
            batching window are merged into it in place.
        """
        value_data: dict = item["values"]
        done_callbacks: list = item["done"]
        reset_requested: bool = item.get("reset", False)
        jobs: list = item.get("jobs", [])

        # A tick taken off the queue clears the pending flag, so the ticker may
        # queue the next one; the same holds for every tick merged below.
        any_tick: bool = item.get("tick", False)
        all_tick: bool = any_tick
        if any_tick:
            self._tick_pending = False

        # Wait for a time window of 'update_rate' seconds to pass before
        # continuing, batching whatever else arrives into the same cycle.
        #
        # An update_rate of zero skips the window entirely: this item is the
        # whole batch, so each queued write gets a model.set() of its own and
        # can never be merged with another client's. Writes that arrived
        # while the previous cycle ran stay queued and are served in order,
        # one cycle each.
        if self.update_rate > 0:
            until = time.monotonic() + self.update_rate
            while time.monotonic() < until:
                try:
                    next_update = self.queue.get_nowait()
                    value_data.update(next_update["values"])
                    done_callbacks.extend(next_update["done"])
                    reset_requested = reset_requested or next_update.get("reset", False)
                    jobs.extend(next_update.get("jobs", []))
                    next_tick = next_update.get("tick", False)
                    if next_tick:
                        self._tick_pending = False
                    any_tick = any_tick or next_tick
                    all_tick = all_tick and next_tick
                except Empty:
                    pass

        # A merged tick makes the batch a tick. Only a batch of ticks alone,
        # with no values and no reset, is a pure tick pass: the subclass signal
        # is set here, before the jobs run, and holds for the whole cycle.
        item["tick"] = any_tick
        self._pass_is_tick = all_tick and not value_data and not reset_requested

        for job in jobs:
            try:
                job()
            except Exception as exc:
                LOG.exception(f"Run-loop job failed: ({exc}); continuing the cycle")

        run_pass = bool(value_data) or reset_requested or not jobs or any_tick

        if run_pass:
            new_values = {}
            latest_ts = 0.0
            for k, g in value_data.items():
                v = g["value"]
                ts = g["ts"]

                # Record newest timestamp from the inputs
                if ts > latest_ts:
                    latest_ts = ts

                # If needed, unpack value and add it to the new list of PVs
                if isinstance(v, Value):
                    new_values[k] = self.pv_handlers[k].unpack_value(
                        self.model.supported_variables[k], v
                    )
                else:
                    new_values[k] = v

            # Use current time if we're missing a latest timestamp
            if latest_ts <= 0:
                latest_ts = time.time()

            # Stash previous state
            settable_var_names = [
                key for key, var in self.model.supported_variables.items() if not var.read_only
            ]
            self._set_cached_state(self.model.get(settable_var_names))

        # Set and simulate
        sim_error = None
        try:
            if run_pass:
                if reset_requested:
                    reset_start = time.perf_counter()
                    LOG.info("Reset requested through RESET control PV")
                    self.model.reset()
                    LOG.debug(
                        f"Model reset() took {(time.perf_counter() - reset_start) * 1000.0:.3f} ms"
                    )

                set_start = time.perf_counter()
                LOG.debug(f"Setting model with new values: {new_values}")
                self.model.set(new_values)
                LOG.debug(f"Model set() took {(time.perf_counter() - set_start) * 1000.0:.3f} ms")

                # Get new simulated values
                output_names = self._cycle_output_names()
                self._cycle_requested_names = tuple(output_names)
                out_values = {}
                if output_names:
                    get_start = time.perf_counter()
                    out_values = self.model.get(output_names)
                    LOG.debug(
                        f"Model get() took {(time.perf_counter() - get_start) * 1000.0:.3f} ms"
                    )

                # Update output PVs with new values
                pv_update_start = time.perf_counter()
                self._post_outputs(out_values, latest_ts)

                LOG.debug(
                    f"PV update loop took {(time.perf_counter() - pv_update_start) * 1000.0:.3f} ms"
                )
        except Exception as exc:
            sim_error = str(exc)
            LOG.error(f"Simulation Cycle Failed: ({sim_error}), resetting to cached value")
            self._reset_to_cached_state()
        finally:
            # With simulation completed, signal put completion to any waiting clients
            for cb in done_callbacks:
                try:
                    cb(sim_error)
                except Exception as excp:
                    LOG.error(f"Error signalling put-completion: {excp}")

    def _post_outputs(self, out_values: dict[str, Any], ts: float) -> None:
        """
        Publish a completed cycle's values to the PVs that carry them.

        The output step of :meth:`_run`, factored out so that a subclass can
        publish elsewhere, publish more, or publish nothing at all. Raising from
        here fails the cycle exactly as raising inline did: the model is rolled
        back to its cached state and every waiting put is completed with the
        error.

        A model without a callable ``output_severity`` is published exactly as
        0.1.4 published it. A model with one is published in two phases, all or
        nothing. Phase 1 -- the only phase that may raise -- asks the model
        which names are undefined, builds every PVA Value with an explicit alarm
        pair and every CA native value, then swaps ``self._udf``. Phase 2
        publishes what phase 1 built and never raises. A phase-1 failure
        therefore posts nothing on either transport and leaves ``self._udf`` as
        it was. Only names this cycle requested are considered:
        ``LUMEModel.get`` validates those alone, so an extra key a model's
        ``_get`` returned is neither shown to the hook nor published.

        Parameters
        ----------
        out_values : dict[str, Any]
            Variable name -> value, as returned by ``model.get``.
        ts : float
            Timestamp to stamp the published values with.
        """
        if not callable(getattr(self.model, "output_severity", None)):
            LOG.debug(f"writing {len(out_values)} PVs")
            for k, v in out_values.items():
                # The model may return None for an output; there is nothing
                # meaningful to post, and passing it downstream would either
                # silently substitute the variable default (PVA path) or raise
                # in value_to_native (CA path). Skip and warn instead.
                if v is None:
                    LOG.warning(f"Model returned None for output '{k}'; skipping update")
                    continue

                # Update PVA component
                pv = self.pvs.get(k)
                if pv is not None:
                    try:
                        pv.post(self._generate_value(k, v, ts))
                    except Exception as e:
                        LOG.error(f"Error posting value for {k}: {e}")

                # Update CA component
                capv = self.ca_pvs.get(k)
                if capv is not None and self.ca_driver is not None:
                    # pcaspy can only understand native python types, not necessarily what the model gives us.
                    nv = self.pv_handlers[k].value_to_native(self.model.supported_variables[k], v)

                    self.ca_driver.setParam(
                        capv,
                        nv,
                        pcaspy.cas.epicsTimeStamp.fromPosixTimeStamp(ts),
                    )

            if self.ca_driver is not None:
                self.ca_driver.updatePVs()
            return

        requested = set(self._cycle_requested_names)
        names = [n for n in out_values if n in requested]

        # ---- phase 1: evaluate, build, swap ----
        udf = self._evaluate_severity(names)

        pva_values: list[tuple[str, Any]] = []
        ca_values: list[tuple[str, Any, bool]] = []
        for k in names:
            v = out_values[k]
            if v is None:
                LOG.warning(f"Model returned None for output '{k}'; skipping update")
                continue

            if self.pvs.get(k) is not None:
                try:
                    packed = self._pack_value(k, v, ts)
                    # Write the pair on every post: p4p's SharedPV.post stores
                    # only marked fields, so an unwritten pair would leave a
                    # recovered output stuck at the previous INVALID.
                    if k in udf:
                        severity, status = _UDF_PVA
                    elif packed.changed("alarm.severity"):
                        severity = packed["alarm"]["severity"]
                        status = packed["alarm"]["status"]
                    else:
                        severity, status = 0, 0
                    packed["alarm"]["severity"] = severity
                    packed["alarm"]["status"] = status
                    pva_values.append((k, packed))
                except Exception as e:
                    # An undefined name must reach clients as undefined; one
                    # that cannot be packed fails the cycle instead.
                    if k in udf:
                        raise
                    LOG.error(f"Error posting value for {k}: {e}")

            capv = self.ca_pvs.get(k)
            if capv is not None and self.ca_driver is not None:
                nv = self.pv_handlers[k].value_to_native(self.model.supported_variables[k], v)
                ca_values.append((capv, nv, k in udf))

        # Names this cycle did not read keep their state.
        self._udf = (self._udf - set(names)) | udf

        # ---- phase 2: publish; nothing below may raise ----
        LOG.debug(f"writing {len(names)} PVs")
        for k, packed in pva_values:
            try:
                self.pvs[k].post(packed)
            except Exception as e:
                LOG.error(f"Error posting value for {k}: {e}")

        if self.ca_driver is None:
            return

        for capv, nv, is_udf in ca_values:
            try:
                self.ca_driver.setParam(
                    capv,
                    nv,
                    pcaspy.cas.epicsTimeStamp.fromPosixTimeStamp(ts),
                )
                # setParam recomputes the alarm, so the UDF status follows it.
                if is_udf:
                    self.ca_driver.setParamStatus(capv, *_UDF_CA)
            except Exception as e:
                LOG.error(f"Error writing CA value for {capv}: {e}")

        try:
            self.ca_driver.updatePVs()
        except Exception as e:
            LOG.error(f"Error flushing CA values: {e}")

    def run(self):
        """
        Runs the simulation, blocks forever (until there's a keyboard interrupt)
        Dequeues PV updates from the updater thread, sets values on the model, and updates outputs.
        """
        try:
            self._run()
        except KeyboardInterrupt:
            return
        except Exception as e:
            raise e

    def _reset_to_cached_state(self) -> None:
        """Apply cached values to the model"""
        LOG.debug(f"Resetting model with new values: {self._cached_state}")
        self.model.set(self._cached_state)

    def _set_cached_state(self, state: dict[str, Any]) -> None:
        """Save `state` to the cache"""
        LOG.debug(f"Caching model state: {state}")
        self._cached_state = state
