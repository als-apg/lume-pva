"""Tests for the variable handlers in lume_pva_apg.variables.

These tests exercise the public handler interface (create_type, pack_value,
unpack_value, default_value, value_to_native, native_to_value, is_supported,
ca_pvspec) against in-memory p4p Value objects. No servers are started and no
network or disk I/O is performed.

The Torch* variable types live behind the 'torch' extra, and the cases for them
are collected only when it is installed. Everything else here covers the
variable types the core carries, so it runs on any install with the 'pva'
extra.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from lume.variables import (
    BoolVariable,
    EnumVariable,
    IntVariable,
    NDVariable,
    ScalarVariable,
    StrVariable,
    Variable,
)

from lume_pva_apg.tests._requires import optional_module, skip_if_absent

# The value layer is expressed in p4p types, so it needs the 'pva' extra even
# where nothing is served. Guarded so a core-only install skips this file rather
# than failing collection, which would take the whole suite with it.
try:
    from p4p import Type, Value
    from p4p.nt import NTEnum, NTNDArray, NTScalar

    from lume_pva_apg.epics import epicsAlarmSeverity, epicsAlarmStatus
    from lume_pva_apg.variables import (
        EnumVariableHandler,
        NDVariableHandler,
        ScalarVariableHandler,
        SimpleScalarHandler,
        VariableHandler,
        find_variable_handler,
    )
except ImportError as exc:
    skip_if_absent(exc)

torch = optional_module("torch", "torch")
TORCH_INSTALLED = torch is not None and optional_module("lume_torch", "torch") is not None

if TORCH_INSTALLED:
    from lume_torch.variables import TorchNDVariable, TorchScalarVariable

    from lume_pva_apg.variables import TorchScalarVariableHandler


def _torch_cases(build: Callable[[], list]) -> list:
    """The parametrisation cases that need the torch extra, or none without it.

    Takes a callable rather than a list because a parametrisation evaluates its
    cases at collection: a list built eagerly from ``torch`` objects would raise
    while the module is being imported, which is the collection failure the
    guard above exists to avoid.
    """
    return build() if TORCH_INSTALLED else []


class DerivedScalarVariable(ScalarVariable):
    pass


class DerivedBoolVariable(BoolVariable):
    pass


@pytest.mark.parametrize(
    ("variable", "value", "expected", "expected_type"),
    [
        pytest.param(ScalarVariable(name="x"), 3.5, 3.5, float, id="float"),
        pytest.param(IntVariable(name="i"), 4, 4, int, id="int"),
        pytest.param(IntVariable(name="i"), 3.9, 3, int, id="int-truncates-float"),
        pytest.param(ScalarVariable(name="x"), -123.25, -123.25, float, id="negative"),
        pytest.param(ScalarVariable(name="x"), 1e308, 1e308, float, id="very-large"),
        pytest.param(StrVariable(name="s"), "hello", "hello", str, id="string"),
        pytest.param(StrVariable(name="s"), "", "", str, id="empty-string"),
        pytest.param(BoolVariable(name="b"), True, True, bool, id="bool-true"),
        pytest.param(BoolVariable(name="b"), False, False, bool, id="bool-false"),
        pytest.param(
            EnumVariable(name="e", options=["x", "y", "z"]),
            "x",
            "x",
            str,
            id="enum_name",
        ),
        pytest.param(
            EnumVariable(name="e", options=["x", "y", "z"]),
            2,
            "z",
            str,
            id="enum_index",
        ),
        *_torch_cases(
            lambda: [
                pytest.param(
                    TorchScalarVariable(name="x"),
                    torch.tensor(2.5),
                    2.5,
                    float,
                    id="torch_to_float",
                ),
            ]
        ),
    ],
)
def test_value_pack_unpack_roundtrip(
    variable: Variable,
    value: Any,
    expected: Any,
    expected_type: type,
) -> None:
    handler = find_variable_handler(type(variable))
    assert isinstance(handler, VariableHandler)
    type_ = handler.create_type(variable)

    unpacked = handler.unpack_value(variable, handler.pack_value(variable, type_, value))

    assert unpacked == expected
    assert isinstance(unpacked, expected_type)


@pytest.mark.parametrize(
    ("variable", "value", "expected"),
    [
        (
            NDVariable(name="arr", shape=(2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
        ),
        # TODO: make type-conversion behavior consistent, scalar handlers coerce but
        # array handlers do not
        # (NDVariable(name="arr", shape=(2, 3), dtype=np.float64),
        #  np.array(((2.45, 3.2),(1.0, 0.0)), dtype=np.float64),
        #  np.array(((2, 3),(1, 0)), dtype=np.int64),),
    ],
)
def test_numpy_array_roundtrip(
    variable: NDVariable,
    value: np.ndarray,
    expected: np.ndarray,
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None
    type_ = handler.create_type(variable)

    unpacked = handler.unpack_value(variable, handler.pack_value(variable, type_, value))

    assert unpacked.shape == expected.shape
    assert unpacked.shape == expected.shape
    assert unpacked.dtype is expected.dtype
    np.testing.assert_allclose(unpacked, expected)


@pytest.mark.parametrize(
    ("variable", "value", "expected"),
    _torch_cases(
        lambda: [
            (
                TorchNDVariable(name="tarr", shape=(2, 3), dtype=torch.float32),
                torch.ones(2, 3, dtype=torch.float32),
                torch.ones(2, 3, dtype=torch.float32),
            ),
            (
                TorchNDVariable(name="tarr", shape=(2, 2, 3), dtype=torch.float32),
                torch.arange(0, 1.2, 0.1, dtype=torch.float32).reshape(2, 2, 3),
                torch.arange(0, 1.2, 0.1, dtype=torch.float32).reshape(2, 2, 3),
            ),
        ]
    ),
)
def test_torch_array_roundtrip(
    variable: NDVariable,
    value: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None
    type_ = handler.create_type(variable)

    unpacked = handler.unpack_value(variable, handler.pack_value(variable, type_, value))

    assert unpacked.shape == variable.shape
    assert unpacked.shape == expected.shape
    assert unpacked.dtype is expected.dtype
    torch.testing.assert_close(unpacked, expected)


@pytest.mark.parametrize(
    "variable",
    [
        pytest.param(ScalarVariable(name="x"), id="scalar"),
        pytest.param(IntVariable(name="i"), id="int"),
        pytest.param(BoolVariable(name="b"), id="bool"),
        pytest.param(StrVariable(name="s"), id="str"),
        pytest.param(NDVariable(name="nd", shape=(2, 2), dtype=np.int16), id="nd"),
        pytest.param(EnumVariable(name="enum", options=["A", "B", "C"]), id="enum"),
        *_torch_cases(lambda: [pytest.param(TorchScalarVariable(name="ts"), id="torchscalar")]),
    ],
)
def test_valid_p4p_type(variable: Variable) -> None:
    handler = find_variable_handler(type(variable))
    assert isinstance(handler.create_type(variable), Type)


@pytest.mark.parametrize(
    ("variable", "code"),
    [
        pytest.param(IntVariable(name="i"), "i", id="int"),
        pytest.param(ScalarVariable(name="x"), "d", id="float"),
    ],
)
def test_scalar_type_wire_code(variable: Variable, code: str) -> None:
    """An int variable is served as an int32 NTScalar, not as a double.

    IntVariable subclasses ScalarVariable, so a handler that tests for the base
    class first serves every integer variable as a double.
    """
    handler = find_variable_handler(type(variable))
    assert handler is not None
    type_ = handler.create_type(variable)

    assert type_["value"] == code
    assert type_.aspy() == NTScalar.buildType(code, control=True, display=True).aspy()
    assert "display" in type_.keys()
    assert "control" in type_.keys()


def test_int_variable_pack_unpack_keeps_int() -> None:
    variable = IntVariable(name="i", value_range=(-3, 9), unit="counts")
    handler = find_variable_handler(type(variable))
    assert handler is not None
    type_ = handler.create_type(variable)

    packed = handler.pack_value(variable, type_, 7)

    assert packed["value"] == 7
    assert isinstance(packed["value"], int)
    assert packed.display.limitLow == -3
    assert packed.display.limitHigh == 9
    assert packed.control.limitLow == -3
    assert packed.control.limitHigh == 9
    assert packed.display.units == "counts"
    assert packed.alarm.severity == int(epicsAlarmSeverity.NO_ALARM)

    unpacked = handler.unpack_value(variable, packed)

    assert unpacked == 7
    assert type(unpacked) is int
    variable.validate_value(unpacked)


def test_int_variable_unpacks_client_put() -> None:
    """A value a PVA client writes to an int PV unpacks to an accepted int."""
    variable = IntVariable(name="i", value_range=(-4, 4))
    handler = find_variable_handler(type(variable))
    assert handler is not None
    put = Value(handler.create_type(variable), {"value": -2})

    unpacked = handler.unpack_value(variable, put)

    assert unpacked == -2
    assert type(unpacked) is int
    variable.validate_value(unpacked)


# timestamp?  display/controls metadata mismatched?
# Enum variable metadata unset? (display)
@pytest.mark.parametrize(
    ("variable", "value", "ctrl_dict", "disp_dict", "alarm_dict"),
    [
        (
            ScalarVariable(name="x", value_range=(0.0, 10.0), unit="mm", default_value=5.0),
            3.5,
            {"limitLow": 0.0, "limitHigh": 10.0, "minStep": 0.0},
            {
                "limitLow": 0.0,
                "limitHigh": 10.0,
                "description": "",
                "format": "",
                "units": "mm",
            },
            {"severity": 0, "status": 0, "message": ""},
        ),
        # (
        #     EnumVariableHandler(),
        #     EnumVariable(name="enum", options=["A", "B", "C"]),
        #     "A",
        #     {'limitLow': 0.0, 'limitHigh': 10.0, 'minStep': 0.0},
        #     {'limitLow': 0.0, 'limitHigh': 0.0, 'description': '', 'format': '', 'units': 'mm'},
        #     {'severity': 0, 'status': 0, 'message': ''}
        # ),
    ],
)
def test_control_limits_and_units_metadata(
    variable, value, ctrl_dict, disp_dict, alarm_dict
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None
    type_ = handler.create_type(variable)

    packed = handler.pack_value(variable, type_, value)

    assert packed.display.todict() == disp_dict
    assert packed.control.todict() == ctrl_dict
    assert packed.alarm.todict() == alarm_dict


@pytest.mark.parametrize(
    (
        "variable",
        "value",
        "size",
    ),
    [
        (NDVariable(name="n", shape=(2, 3)), np.zeros((2, 3)), [2, 3]),
        (NDVariable(name="n", shape=(2, 2, 3)), np.zeros((2, 2, 3)), [2, 2, 3]),
        *_torch_cases(
            lambda: [
                (TorchNDVariable(name="n", shape=(2, 3)), torch.zeros((2, 3)), [2, 3]),
                (
                    TorchNDVariable(name="n", shape=(2,)),
                    torch.zeros((2,)),
                    [
                        2,
                    ],
                ),
            ]
        ),
    ],
)
def test_dimension_size_metadata(variable, value, size):
    handler = find_variable_handler(type(variable))
    assert handler is not None
    type_ = handler.create_type(variable)
    packed = handler.pack_value(variable, type_, value)

    assert [d["size"] for d in packed["dimension"]] == size
    assert packed["compressedSize"] == value.nbytes
    assert packed["uncompressedSize"] == value.nbytes


@pytest.mark.parametrize(
    (
        "variable",
        "expected_value",
    ),
    [
        (ScalarVariable(name="scalar", default_value=5.0), 5.0),
        (StrVariable(name="str", default_value="hi"), "hi"),
        (IntVariable(name="str", default_value=1), 1),
        (BoolVariable(name="bool", default_value=True), True),
        (EnumVariable(name="enum", options=["A", "B", "C"], default_value="B"), "B"),
        (EnumVariable(name="enum", options=["A", "B", "C"]), "A"),
        # (NDVariable(name="nd", shape=(2, 3), dtype=np.int64), np.array(((1,2,3), (4,5,6))) ),
        *_torch_cases(
            lambda: [
                (TorchScalarVariable(name="torchscalar", default_value=1.0), 1.0),
                (TorchScalarVariable(name="torchscalar"), 0),
            ]
        ),
    ],
)
def test_default_value_passthrough(
    variable: Variable,
    expected_value: Any,
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None

    type_ = handler.create_type(variable)
    packed = handler.pack_value(variable, type_, None)

    # Assert
    assert handler.unpack_value(variable, packed) == expected_value


@pytest.mark.parametrize(
    (
        "variable",
        "expected_value",
    ),
    [
        (
            NDVariable(
                name="nd",
                shape=(2, 3),
                dtype=np.int64,
                default_value=np.array(((1, 2, 3), (4, 5, 6))),
            ),
            np.array(((1, 2, 3), (4, 5, 6))),
        ),
        (
            NDVariable(name="nd", shape=(4, 5), dtype=np.int64),
            np.zeros((4, 5)),
        ),  # zero filled with no default
        # string arrays fail to roundtrip
        # (NDVariable(name="nd", shape=(2,), dtype=np.dtypes.StringDType(),),
        #  np.array(["", ""], dtype=np.dtypes.StringDType())),
        *_torch_cases(
            lambda: [
                (
                    TorchNDVariable(
                        name="ndtorch",
                        shape=(2, 3),
                        dtype=torch.int64,
                        default_value=torch.tensor(np.array(((1, 2, 3), (4, 5, 6)))),
                    ),
                    torch.tensor(np.array(((1, 2, 3), (4, 5, 6)))),
                ),
                (
                    TorchNDVariable(
                        name="ndtorch",
                        shape=(5, 5),
                        dtype=torch.int64,
                    ),
                    torch.zeros(5, 5),
                ),
            ]
        ),
    ],
)
def test_default_array_value_passthrough(
    variable: Variable,
    expected_value: Any,
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None

    type_ = handler.create_type(variable)
    packed = handler.pack_value(variable, type_, None)

    bool_mask = handler.unpack_value(variable, packed) == expected_value
    assert bool_mask.all()


@pytest.mark.parametrize(
    ("variable", "expected_default"),
    [
        pytest.param(ScalarVariable(name="scalar", default_value=5.0), 5.0, id="scalar"),
        pytest.param(IntVariable(name="int", default_value=2), 2, id="int"),
        pytest.param(BoolVariable(name="bool", default_value=True), True, id="bool"),
        pytest.param(StrVariable(name="str", default_value="hi"), "hi", id="str"),
        pytest.param(
            EnumVariable(name="enum", options=["A", "B", "C"], default_value="B"),
            "B",
            id="enum",
        ),
        *_torch_cases(
            lambda: [
                pytest.param(
                    TorchScalarVariable(name="torchscalar", default_value=1.0),
                    1.0,
                    id="torchscalar",
                ),
            ]
        ),
    ],
)
def test_default_value_method_scalar_like(
    variable: Variable,
    expected_default: Any,
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None

    assert handler.default_value(variable) == expected_default


@pytest.mark.parametrize(
    ("variable", "expected_default"),
    [
        pytest.param(
            NDVariable(
                name="nd",
                shape=(2, 3),
                dtype=np.int64,
                default_value=np.array(((1, 2, 3), (4, 5, 6))),
            ),
            np.array(((1, 2, 3), (4, 5, 6))),
            id="nd",
        ),
        *_torch_cases(
            lambda: [
                pytest.param(
                    TorchNDVariable(
                        name="ndtorch",
                        shape=(2, 3),
                        dtype=torch.int64,
                        default_value=torch.tensor(np.array(((1, 2, 3), (4, 5, 6)))),
                    ),
                    torch.tensor(np.array(((1, 2, 3), (4, 5, 6)))),
                    id="torchnd",
                ),
            ]
        ),
    ],
)
def test_default_value_method_array_like(
    variable: Variable,
    expected_default: Any,
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None

    actual = handler.default_value(variable)
    bool_mask = actual == expected_default
    assert bool_mask.all()


@pytest.mark.parametrize(
    "variable",
    [
        pytest.param(ScalarVariable(name="scalar", default_value=5.0), id="scalar"),
        pytest.param(IntVariable(name="int", default_value=2), id="int"),
        pytest.param(BoolVariable(name="bool", default_value=True), id="bool"),
        pytest.param(StrVariable(name="str", default_value="hi"), id="str"),
        pytest.param(
            EnumVariable(name="enum", options=["A", "B", "C"], default_value="B"),
            id="enum",
        ),
        pytest.param(
            NDVariable(
                name="nd",
                shape=(2, 3),
                dtype=np.int64,
                default_value=np.array(((1, 2, 3), (4, 5, 6))),
            ),
            id="nd",
        ),
        *_torch_cases(
            lambda: [
                pytest.param(
                    TorchScalarVariable(name="torchscalar", default_value=1.0),
                    id="torchscalar",
                ),
                pytest.param(
                    TorchNDVariable(
                        name="ndtorch",
                        shape=(2, 3),
                        dtype=torch.int64,
                        default_value=torch.tensor(np.array(((1, 2, 3), (4, 5, 6)))),
                    ),
                    id="torchnd",
                ),
            ]
        ),
    ],
)
def test_pack_value_none_matches_default_value(variable: Variable) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None

    type_ = handler.create_type(variable)
    packed = handler.pack_value(variable, type_, None)
    unpacked = handler.unpack_value(variable, packed)
    expected = handler.default_value(variable)

    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(unpacked, expected)
    elif TORCH_INSTALLED and isinstance(expected, torch.Tensor):
        torch.testing.assert_close(unpacked, expected)
    else:
        assert unpacked == expected


@pytest.mark.parametrize(
    ("variable"),
    [
        pytest.param(
            ScalarVariable(name="x", value_range=(0.0, 10.0), default_value=5.0),
            id="scalarvar",
        ),
        *_torch_cases(
            lambda: [
                pytest.param(
                    TorchScalarVariable(name="x", value_range=(0.0, 10.0), default_value=5.0),
                    id="torchscalarvar",
                ),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    ("value", "expected_severity", "expected_status"),
    [
        pytest.param(
            3.5,
            epicsAlarmSeverity.NO_ALARM,
            epicsAlarmStatus.NO_STATUS,
            id="within_range",
        ),
        pytest.param(
            0.0,
            epicsAlarmSeverity.NO_ALARM,
            epicsAlarmStatus.NO_STATUS,
            id="at_lower_boundary",
        ),
        pytest.param(
            10.0,
            epicsAlarmSeverity.NO_ALARM,
            epicsAlarmStatus.NO_STATUS,
            id="at_upper_boundary",
        ),
        pytest.param(
            -1.0,
            epicsAlarmSeverity.MAJOR_ALARM,
            epicsAlarmStatus.DRIVER_STATUS,
            id="below_range",
        ),
        pytest.param(
            11.0,
            epicsAlarmSeverity.MAJOR_ALARM,
            epicsAlarmStatus.DRIVER_STATUS,
            id="above_range",
        ),
    ],
)
def test_alarm_metadata_from_value_range(
    variable: ScalarVariable | TorchScalarVariable,
    value: float,
    expected_severity: epicsAlarmSeverity,
    expected_status: epicsAlarmStatus,
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None
    type_ = handler.create_type(variable)

    packed = handler.pack_value(variable, type_, value)

    assert packed["alarm"]["severity"] == int(expected_severity)
    assert packed["alarm"]["status"] == int(expected_status)


@pytest.mark.parametrize(
    ("variable", "value", "expected_val", "expected_type"),
    [
        (ScalarVariable(name="s"), np.float64(2.5), 2.5, float),
        (IntVariable(name="i"), 2.5, 2, int),
        (
            NDVariable(name="n", shape=(2, 3)),
            np.arange(6, dtype=np.float64).reshape(2, 3),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            list,
        ),
    ],
)
def test_value_to_native(
    variable: Variable,
    value: Any,
    expected_val: Any,
    expected_type: Any,
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None
    native = handler.value_to_native(variable, value)

    assert native == expected_val
    assert isinstance(native, expected_type)


@pytest.mark.parametrize(
    ("variable", "value", "expected_exception"),
    [
        pytest.param(
            ScalarVariable(name="x"),
            "not-a-number",
            TypeError,
            id="non_numeric_value",
        ),
        pytest.param(
            ScalarVariable(
                name="strict",
                value_range=(0.0, 10.0),
                default_validation_config="error",
            ),
            50.0,
            ValueError,
            id="out_of_range",
        ),
        pytest.param(
            BoolVariable(name="b"),
            "not-a-bool",
            TypeError,
            id="non_bool_value",
        ),
        pytest.param(
            StrVariable(name="s"),
            123,
            TypeError,
            id="non_str_value",
        ),
        pytest.param(
            EnumVariable(name="x", options=["A", "B", "C"]),
            "not-an-option",
            ValueError,
            id="invalid_option",
        ),
        pytest.param(
            NDVariable(name="arr", shape=(2, 3)),
            np.zeros((3, 3), dtype=np.float64),
            ValueError,
            id="wrong_arr_shape",
        ),
        pytest.param(
            NDVariable(name="arr", shape=(2, 3)),
            np.zeros((2, 3), dtype=np.int32),
            ValueError,
            id="wrong_dtype",
        ),
        pytest.param(
            NDVariable(name="arr", shape=(2, 3)),
            [[1, 2, 3], [4, 5, 6]],
            TypeError,
            id="not_array",
        ),
        *_torch_cases(
            lambda: [
                pytest.param(
                    TorchScalarVariable(name="x"),
                    "not-a-torch-number",
                    TypeError,
                    id="non_torch_numeric_value",
                ),
            ]
        ),
    ],
)
def test_raise_packing_invalid_value(
    variable: Variable,
    value: Any,
    expected_exception: type[Exception],
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None
    type_ = handler.create_type(variable)

    with pytest.raises(expected_exception):
        handler.pack_value(variable, type_, value)


@pytest.mark.parametrize(
    ("variable", "expected_spec"),
    [
        pytest.param(
            StrVariable(name="s"),
            {"type": "char", "count": 1024},
            id="string_waveform",
        ),
        pytest.param(BoolVariable(name="b"), {}, id="bool_no_extras"),
        pytest.param(
            ScalarVariable(name="x"),
            {
                "unit": None,
                "type": "float",
                "lolim": 0,
                "hilim": 0,
            },
            id="scalar_no_extras",
        ),
        # A range and a unit that are neither absent nor symmetric, so a spec
        # that dropped them, zeroed them, or swapped the two limits differs
        # from a spec that carried them through.
        pytest.param(
            ScalarVariable(name="x", value_range=(-2.5, 7.5), unit="mm"),
            {
                "unit": "mm",
                "type": "float",
                "lolim": -2.5,
                "hilim": 7.5,
            },
            id="scalar_with_range_and_unit",
        ),
        pytest.param(
            IntVariable(name="i", value_range=(-3, 9), unit="counts"),
            {
                "unit": "counts",
                "type": "int",
                "lolim": -3,
                "hilim": 9,
            },
            id="int_with_range_and_unit",
        ),
        pytest.param(
            NDVariable(name="x", shape=(2, 3)),
            {"count": 6, "type": "float"},
            id="nd_no_extras",
        ),
        pytest.param(
            EnumVariable(name="e", options=["A", "B"]),
            {
                "type": "enum",
                "enums": ["A", "B"],
            },
            id="enum_mbbi",
        ),
        *_torch_cases(
            lambda: [
                pytest.param(TorchScalarVariable(name="x"), {}, id="tscalar_no_extras"),
                pytest.param(
                    TorchNDVariable(name="x", shape=(2, 3)),
                    {"count": 6, "type": "float"},
                    id="tnd",
                ),
            ]
        ),
    ],
)
def test_ca_pvspec(
    variable: Variable,
    expected_spec: dict[str, Any],
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None

    assert handler.ca_pvspec(variable) == expected_spec


@pytest.mark.parametrize(
    "variable",
    [
        pytest.param(ScalarVariable(name="x", value_range=(-2.5, 7.5)), id="scalar"),
        pytest.param(IntVariable(name="i", value_range=(-3, 9)), id="int"),
    ],
)
def test_ca_pvspec_publishes_no_alarm_thresholds(variable: Variable) -> None:
    """A range is a display limit on CA and nothing else.

    pcaspy evaluates a numeric alarm only where a lolo/hihi pair is present and
    lolo < hihi, and it compares them inclusively, so a threshold taken from the
    variable's own range alarms on a value driven to either end of its legal
    span. Omitting the keys is what disables the check -- a pair of zeros is
    still a defined threshold -- so the assertion is on their absence, not on
    their value.
    """
    handler = find_variable_handler(type(variable))
    assert handler is not None

    spec = handler.ca_pvspec(variable)

    assert {"lolo", "hihi", "low", "high"}.isdisjoint(spec)
    assert (spec["lolim"], spec["hilim"]) == variable.value_range


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.float64, True),
        (np.float32, True),
        (np.int16, True),
        (np.int32, True),
        (np.int64, True),
        (np.uint16, True),
        (np.uint32, True),
        (np.uint64, True),
        (np.str_, True),
        (np.dtypes.StringDType(), True),
        (np.complex128, False),
    ],
    ids=str,
)
def test_handler_report_dtype_support(dtype: type[np.generic], expected: bool) -> None:
    var = NDVariable(name="v", shape=(2,), dtype=np.dtype(dtype))
    handler = find_variable_handler(NDVariable)
    assert handler is not None

    assert handler.is_supported(var) is expected


@pytest.mark.parametrize(
    ("variable_type", "expected_handler_type"),
    [
        (ScalarVariable, ScalarVariableHandler),
        (IntVariable, ScalarVariableHandler),
        (NDVariable, NDVariableHandler),
        (BoolVariable, SimpleScalarHandler),
        (StrVariable, SimpleScalarHandler),
        (EnumVariable, EnumVariableHandler),
        *_torch_cases(
            lambda: [
                (TorchScalarVariable, TorchScalarVariableHandler),
                (TorchNDVariable, NDVariableHandler),
            ]
        ),
    ],
)
def test_should_return_matching_handler_for_each_variable_type(
    variable_type: type[Variable],
    expected_handler_type: type[VariableHandler],
) -> None:
    handler = find_variable_handler(variable_type)

    assert isinstance(handler, expected_handler_type)
    assert isinstance(handler, VariableHandler)


def test_should_return_none_for_unknown_type() -> None:
    assert find_variable_handler(dict) is None


@pytest.mark.parametrize(
    ("variable_type", "expected_handler_type"),
    [
        (DerivedScalarVariable, ScalarVariableHandler),
        (DerivedBoolVariable, SimpleScalarHandler),
    ],
)
def test_should_resolve_handler_for_variable_subclasses(
    variable_type: type[Variable],
    expected_handler_type: type[VariableHandler],
) -> None:
    handler = find_variable_handler(variable_type)

    assert isinstance(handler, expected_handler_type)
    assert isinstance(handler, VariableHandler)


# --- Display metadata: the 0.1.5 wire structure of str, bool and enum PVs ----


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        pytest.param(
            StrVariable(name="s"), lambda: NTScalar.buildType("s", display=True), id="str"
        ),
        pytest.param(
            BoolVariable(name="b"), lambda: NTScalar.buildType("?", display=True), id="bool"
        ),
        pytest.param(
            EnumVariable(name="e", options=["A", "B"]),
            lambda: NTEnum.buildType(display=True),
            id="enum",
        ),
    ],
)
def test_simple_and_enum_types_carry_display_block(variable: Variable, expected: Callable) -> None:
    """Str, bool and enum PVs are served with a display block, so a client can read a description."""
    handler = find_variable_handler(type(variable))
    assert handler is not None

    type_ = handler.create_type(variable)

    assert type_.aspy() == expected().aspy()
    assert "display" in type_.keys()
    assert "description" in type_["display"].keys()
    assert "units" in type_["display"].keys()


def test_simple_scalar_display_has_only_description_and_units() -> None:
    """A non-numeric NTScalar display carries description and units, and no limits or format."""
    type_ = SimpleScalarHandler().create_type(StrVariable(name="s"))

    assert sorted(type_["display"].keys()) == ["description", "units"]


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        pytest.param(ScalarVariable(name="x", unit="mm"), 1.5, id="float"),
        pytest.param(IntVariable(name="i", unit="counts"), 3, id="int"),
        pytest.param(StrVariable(name="s"), "hello", id="str"),
        pytest.param(BoolVariable(name="b"), True, id="bool"),
        pytest.param(EnumVariable(name="e", options=["A", "B"]), "B", id="enum"),
    ],
)
def test_set_display_metadata_writes_description(variable: Variable, value: Any) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None
    packed = handler.pack_value(variable, handler.create_type(variable), value)

    handler.set_display_metadata(variable, packed, description="Beam energy")

    assert packed["display"]["description"] == "Beam energy"


@pytest.mark.parametrize("description", [None, ""], ids=["none", "empty"])
@pytest.mark.parametrize(
    "variable",
    [
        pytest.param(ScalarVariable(name="x"), id="float"),
        pytest.param(StrVariable(name="s"), id="str"),
        pytest.param(EnumVariable(name="e", options=["A", "B"]), id="enum"),
    ],
)
def test_set_display_metadata_leaves_description_unset_when_absent(
    variable: Variable, description: str | None
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None
    packed = handler.pack_value(variable, handler.create_type(variable), None)

    handler.set_display_metadata(variable, packed, description=description)

    assert packed["display"]["description"] == ""
    assert "display.description" not in packed.changedSet()


def test_set_display_metadata_defaults_to_no_description() -> None:
    variable = StrVariable(name="s")
    handler = SimpleScalarHandler()
    packed = handler.pack_value(variable, handler.create_type(variable), "a")

    handler.set_display_metadata(variable, packed)

    assert packed["display"]["description"] == ""


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        pytest.param(ScalarVariable(name="x", unit="mm"), 1.5, id="float"),
        pytest.param(IntVariable(name="i", unit="counts"), 3, id="int"),
    ],
)
def test_set_display_metadata_writes_units_when_variable_has_one(
    variable: Variable, value: Any
) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None
    type_ = handler.create_type(variable)
    fresh = Value(type_, {"value": value})

    handler.set_display_metadata(variable, fresh)

    assert fresh["display"]["units"] == variable.unit


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        pytest.param(StrVariable(name="s"), "hello", id="str"),
        pytest.param(BoolVariable(name="b"), False, id="bool"),
        pytest.param(EnumVariable(name="e", options=["A", "B"]), "A", id="enum"),
        pytest.param(ScalarVariable(name="x"), 1.0, id="float-no-unit"),
    ],
)
def test_set_display_metadata_skips_units_without_a_unit(variable: Variable, value: Any) -> None:
    """Str, bool and enum variables have no unit field; a missing or empty unit writes nothing."""
    handler = find_variable_handler(type(variable))
    assert handler is not None
    packed = handler.pack_value(variable, handler.create_type(variable), value)

    handler.set_display_metadata(variable, packed, description="d")

    assert packed["display"]["units"] == ""


def test_set_display_metadata_does_not_collide_with_scalar_set_metadata() -> None:
    """The static ScalarVariableHandler.set_metadata that the torch handler calls keeps its signature."""
    assert isinstance(ScalarVariableHandler.__dict__["set_metadata"], staticmethod)
    assert "set_display_metadata" in VariableHandler.__dict__
    assert "set_display_metadata" not in ScalarVariableHandler.__dict__


def test_nd_type_is_untouched_by_display_metadata() -> None:
    variable = NDVariable(name="nd", shape=(2, 2), dtype=np.float64)
    handler = find_variable_handler(type(variable))
    assert handler is not None

    type_ = handler.create_type(variable)
    packed = handler.pack_value(variable, type_, np.zeros((2, 2)))

    assert type_.aspy() == NTNDArray.buildType().aspy()
    assert "display" not in type_.keys()
    assert "display" not in packed.keys()


@pytest.mark.skipif(not TORCH_INSTALLED, reason="needs the torch extra")
def test_torch_scalar_type_is_untouched_by_display_metadata() -> None:
    variable = TorchScalarVariable(name="ts")
    handler = find_variable_handler(type(variable))
    assert isinstance(handler, TorchScalarVariableHandler)

    type_ = handler.create_type(variable)
    packed = handler.pack_value(variable, type_, 2.0)

    assert type_.aspy() == NTScalar.buildType("d", control=True, display=True).aspy()
    assert packed["display"]["description"] == ""


@pytest.mark.parametrize(
    "variable",
    [
        pytest.param(StrVariable(name="s"), id="str"),
        pytest.param(BoolVariable(name="b"), id="bool"),
        pytest.param(EnumVariable(name="e", options=["A", "B"]), id="enum"),
    ],
)
def test_one_argument_create_type_and_ca_pvspec_still_work(variable: Variable) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None

    assert isinstance(handler.create_type(variable), Type)
    assert isinstance(handler.ca_pvspec(variable), dict)


def test_float_with_precision_type_carries_display_precision() -> None:
    variable = ScalarVariable(name="x", unit="mm", value_range=(0.0, 10.0))
    handler = find_variable_handler(type(variable))
    assert isinstance(handler, ScalarVariableHandler)

    type_ = handler.create_type(variable, precision=3)

    assert type_.aspy() == NTScalar.buildType("d", control=True, display=True, form=True).aspy()
    display_keys = type_["display"].keys()
    assert "precision" in display_keys
    assert "form" in display_keys
    assert "format" not in display_keys


def test_float_with_precision_type_packs_and_holds_precision() -> None:
    """A form=True type still packs through the unchanged pack_value and set_metadata."""
    variable = ScalarVariable(name="x", unit="mm", value_range=(0.0, 10.0))
    handler = find_variable_handler(type(variable))
    assert handler is not None

    packed = handler.pack_value(variable, handler.create_type(variable, precision=3), 1.5)
    packed["display"]["precision"] = 3

    assert packed["value"] == 1.5
    assert packed["display"]["units"] == "mm"
    assert packed["display"]["precision"] == 3


def test_float_with_zero_precision_still_builds_form_type() -> None:
    """precision=0 is a real precision, not an absent one."""
    variable = ScalarVariable(name="x")
    handler = find_variable_handler(type(variable))
    assert handler is not None

    type_ = handler.create_type(variable, precision=0)

    assert "precision" in type_["display"].keys()
    assert handler.ca_pvspec(variable, precision=0)["prec"] == 0


def test_float_with_precision_ca_pvspec_adds_prec() -> None:
    variable = ScalarVariable(name="x", unit="mm", value_range=(0.0, 10.0))
    handler = find_variable_handler(type(variable))
    assert handler is not None

    assert handler.ca_pvspec(variable, precision=3) == {
        "unit": "mm",
        "type": "float",
        "lolim": 0.0,
        "hilim": 10.0,
        "prec": 3,
    }


@pytest.mark.parametrize(
    ("variable", "code", "type_name"),
    [
        pytest.param(
            ScalarVariable(name="x", unit="mm", value_range=(0.0, 10.0)), "d", "float", id="float"
        ),
        pytest.param(IntVariable(name="i", unit="ct", value_range=(0, 10)), "i", "int", id="int"),
    ],
)
def test_without_precision_type_and_pvspec_are_unchanged(
    variable: Variable, code: str, type_name: str
) -> None:
    """precision=None (explicit or omitted) reproduces 0.1.4's type and pvspec exactly."""
    handler = find_variable_handler(type(variable))
    assert handler is not None
    expected_type = NTScalar.buildType(code, control=True, display=True).aspy()
    expected_spec = {
        "unit": variable.unit,
        "type": type_name,
        "lolim": variable.value_range[0],
        "hilim": variable.value_range[1],
    }

    for kwargs in ({}, {"precision": None}):
        type_ = handler.create_type(variable, **kwargs)
        assert type_.aspy() == expected_type
        assert "precision" not in type_["display"].keys()
        assert "format" in type_["display"].keys()
        assert handler.ca_pvspec(variable, **kwargs) == expected_spec


def test_precision_is_keyword_only() -> None:
    variable = ScalarVariable(name="x")
    handler = find_variable_handler(type(variable))
    assert handler is not None

    with pytest.raises(TypeError):
        handler.create_type(variable, 3)  # type: ignore[misc]
    with pytest.raises(TypeError):
        handler.ca_pvspec(variable, 3)  # type: ignore[misc]


@pytest.mark.parametrize(
    "variable",
    [
        pytest.param(StrVariable(name="s"), id="str"),
        pytest.param(BoolVariable(name="b"), id="bool"),
        pytest.param(EnumVariable(name="e", options=["A", "B"]), id="enum"),
        pytest.param(NDVariable(name="nd", shape=(2,), dtype=np.float64), id="nd"),
    ],
)
def test_other_handlers_take_no_precision_keyword(variable: Variable) -> None:
    handler = find_variable_handler(type(variable))
    assert handler is not None

    with pytest.raises(TypeError):
        handler.create_type(variable, precision=3)  # type: ignore[call-arg]


@pytest.mark.skipif(not TORCH_INSTALLED, reason="needs the torch extra")
def test_torch_scalar_handler_takes_no_precision_keyword() -> None:
    variable = TorchScalarVariable(name="ts")
    handler = find_variable_handler(type(variable))
    assert isinstance(handler, TorchScalarVariableHandler)

    with pytest.raises(TypeError):
        handler.create_type(variable, precision=3)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        handler.ca_pvspec(variable, precision=3)  # type: ignore[call-arg]
