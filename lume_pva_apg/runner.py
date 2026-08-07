import logging
import math
import numbers
import os
import threading
import time
from collections.abc import Callable
from queue import Empty, Queue
from typing import Any, TypedDict

from lume.model import LUMEModel, Variable
from lume.variables import ParticleGroupVariable

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

from lume_pva_apg.variables import VariableHandler, find_variable_handler

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
    """

    name: str
    pv: str
    mode: str


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
    """

    prefix: str
    variables: dict[str, RunnerVariable]
    protocol: list[str]
    update_rate: float
    echo_unconfirmed_writes: bool
    alarm_on_refused_write: bool
    clamp_writes: bool


class Runner:
    """Simple runner for LUMEModel derived models"""

    pvs: dict[str, SharedPV]
    ca_pvs: dict[str, str]
    pv_handlers: dict[str, VariableHandler]
    # List of all output PVs that need to be updated after simulation
    outputs: list[str]
    values: dict[str, Value]

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
                {self.variable.name: {"value": value, "ts": time.monotonic()}},
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
                {vn: {"value": nv, "ts": time.monotonic()}},
                done=_complete_put,
            )
            return True

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
        # Base name of the reset control PV -- the key it holds in the pvdb, and
        # the reason the CA driver is called back with.
        self.reset_control_pv = ""
        self.ca_server: pcaspy.SimpleServer | None = None
        self.ca_driver: Runner.CaDriver | None = None

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

            # Cache handler and type for later
            self.pv_handlers[var.name] = handler
            self.types[var.name] = handler.create_type(var)

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
        self._create_control_pvs()

        # Start the server
        self.server = p4p.server.Server(providers=[self.providers])

        # Start the CA server under the shared async context
        if len(self.pvdb.keys()) > 0:
            self.ca_server = pcaspy.SimpleServer()
            self.ca_server.createPV(self.config.get("prefix", ""), self.pvdb)
            self.ca_driver = Runner.CaDriver(self)

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
        """
        self.queue.put(
            {
                "values": values,
                "done": [done] if done is not None else [],
                "reset": reset,
            }
        )

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
            spec = handler.ca_pvspec(var)

            # Keyed by the base name: SimpleServer.createPV prepends the prefix
            # to build the served name, and every callback into the driver --
            # write's `reason`, setParam, updatePV -- names the PV by this key.
            self.pvdb[pv] = spec
            self.pvdb[pv].update({"asyn": True})
            # enable async for put-completion
            self.ca_pvs[var.name] = pv

    def _create_model_info(self):
        """Creates a model info PV for PVA"""
        pv = "model_info"

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

        for e in envs:
            val["env"][e] = os.environ.get(e, "")

        vars = []
        for k, v in self.model.supported_variables.items():
            info = {
                "name": v.name,
                "read_only": v.read_only,
                "pvname": self.config["variables"][k]["pv"],
                "type": v.__class__.__name__,
                "mode": self.config["variables"][k]["mode"],
            }
            vars.append(info)

        val["supported_variables"] = vars

        self.pvs[pv] = SharedPV(initial=val)
        self.providers[f"{self.config['prefix']}{pv}"] = self.pvs[pv]

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

    def _generate_value(self, pv: str, value: Any | None, ts: float | None = None) -> Value:
        """
        Generates a new value for posting to the PV.
        Handles alarm updates, timestamp updates, and generating the value in the first place. This handles the
        'common' metadata that the variable handlers shouldn't need to handle.
        """
        v = self.pv_handlers[pv].pack_value(
            self.model.supported_variables[pv], self.types[pv], value
        )

        # Ensure timestamp is current
        self._update_timestamp(v, ts=ts)
        return v

    def _update_timestamp(self, value: Value, ts=None) -> None:
        """
        Helper to update timestamp on a value
        """
        if ts is None:
            ts = time.monotonic()
        value["timeStamp"]["nanoseconds"] = math.fmod(ts, 1.0) * 1e9
        value["timeStamp"]["secondsPastEpoch"] = int(ts)

    @property
    def config(self) -> RunnerConfig:
        """Access the underlying config"""
        return self._config

    def _run(self):
        """
        Runs the simulation, blocks forever.
        Dequeues PV updates from the updater thread, sets values on the model, and updates outputs.
        """
        while True:
            # Wait for new data to come in
            item = self.queue.get()

            value_data: dict = item["values"]
            done_callbacks: list = item["done"]
            reset_requested: bool = item.get("reset", False)

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
                    except Empty:
                        pass

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
                latest_ts = time.monotonic()

            # Stash previous state
            settable_var_names = [
                key for key, var in self.model.supported_variables.items() if not var.read_only
            ]
            self._set_cached_state(self.model.get(settable_var_names))

            # Set and simulate
            sim_error = None
            try:
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
                get_start = time.perf_counter()
                out_values = self.model.get(self.model.supported_variables)
                LOG.debug(f"Model get() took {(time.perf_counter() - get_start) * 1000.0:.3f} ms")

                # Update output PVs with new values
                pv_update_start = time.perf_counter()
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
                            pv.post(self._generate_value(k, v, latest_ts))
                        except Exception as e:
                            LOG.error(f"Error posting value for {k}: {e}")

                    # Update CA component
                    capv = self.ca_pvs.get(k)
                    if capv is not None and self.ca_driver is not None:
                        # pcaspy can only understand native python types, not necessarily what the model gives us.
                        nv = self.pv_handlers[k].value_to_native(
                            self.model.supported_variables[k], v
                        )

                        self.ca_driver.setParam(
                            capv,
                            nv,
                            pcaspy.cas.epicsTimeStamp.fromPosixTimeStamp(latest_ts),
                        )

                if self.ca_driver is not None:
                    self.ca_driver.updatePVs()

                LOG.debug(
                    f"PV update loop took {(time.perf_counter() - pv_update_start) * 1000.0:.3f} ms"
                )
            except Exception as exc:
                sim_error = str(exc)
                LOG.error(f"Simulation Cycle Failed: ({sim_error}), resetting to cached value")
                self._reset_to_cached_state()
            finally:
                # With simulation compoleted, signal put completion to any waitihng clients
                for cb in done_callbacks:
                    try:
                        cb(sim_error)
                    except Exception as excp:
                        LOG.error(f"Error signalling put-completion: {excp}")

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
