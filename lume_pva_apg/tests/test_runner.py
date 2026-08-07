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

import numpy as np
import pytest

from lume_pva_apg.tests._requires import skip_if_absent

# No server is started here, but importing the runner still needs both
# transports. Guarded so an install missing one of them skips this file rather
# than failing collection, which would take the whole suite with it.
try:
    from lume.variables import NDVariable, ScalarVariable, Variable

    from lume_pva_apg.runner import Runner
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
