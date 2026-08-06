# lume-pva-apg

`lume-pva-apg` serves EPICS PVs based on a LUMEModel subclass and its supported
variables. It is a distribution of [lume-pva](https://github.com/lume-science/lume-pva)
maintained as a staging area for changes intended to land upstream; see
[Relationship to lume-pva](#relationship-to-lume-pva) below.

Features:
* Model outputs served over PVAccess (PVA) and/or ChannelAccess (CA).
    * PVA PVs support a subset of the EPICS NormativeTypes metadata.
    * CA PVs support a subset of the standard EPICS CA meta, such as alarms, display limits and control limits.
* Model inputs served as writable PVs over PVA and/or CA.

## Installation

The core distribution is pure-Python and carries no EPICS transport, so it
installs from wheels on every platform without triggering a native build.
Serving requires at least one transport extra:

```sh
pip install 'lume-pva-apg[ca,pva]'   # both transports
pip install 'lume-pva-apg[pva]'      # PVAccess only
```

| Extra | Pulls in | Needed for |
| --- | --- | --- |
| `pva` | `p4p` | PVAccess serving, and the NormativeTypes value layer |
| `ca` | `pcaspy>=0.8.1` | Channel Access serving |
| `torch` | `torch`, `lume-torch` | the `TorchScalarVariable` / `TorchNDVariable` types |

The `pva` extra is required for any serving at all: the value layer in
`lume_pva_apg.variables` is expressed in p4p types and is used by the CA path
too. A CA-only install is therefore not currently possible. Importing a module
whose transport is absent raises an `ImportError` naming the extra to install.

`pcaspy>=0.8.1` is a floor rather than an exact pin. On 0.8.0 the asynchronous
write path passes a `casClientInfo` where a `casCtx` is required, which raises
inside the SWIG director and terminates the server process; a client issuing a
put with callback is told the write succeeded while the value never lands.
0.8.1 fixes it, and the fix depends on a binding 0.8.0 does not compile, so it
cannot be worked around from Python. 0.8.1 publishes no linux/aarch64 wheel, so
the bound is left open: arm64 consumers must stay free to build from sdist or
to take a later release that restores the wheel.

## Basic Usage

```py
from lume_pva_apg.runner import Runner

myModel = MyLUMEModel()
r = Runner(model=myModel)
r.run()
```

## Operational Description

### Model Outputs

Model outputs can be served over PVA and/or CA. If the output is solely an output, it will be configured as a read-only PV.

### Model Inputs

Model inputs are served by the `Runner` class and can be interacted with using pvput, caput or other
CA/PVA tools on the command line.

Like model outputs, inputs can be served over CA or PVA, depending on the `Runner` configuration.

### Control PVs

The runner always exposes a control PV:
* `{prefix}RESET`: any write requests `model.reset()` and publishes the reset state to output PVs.

The control PV is served over PVA, and is also available over CA when `protocol` includes `"ca"`.

`prefix` is passed to the constructor of the `Runner` class and defines a prefix to prepend to the start of PV names.

### Configuration

`Runner.generate_config()` can be used to generate a `dict` describing the default configuration for the model.
You can either edit this on the fly, or serialize it and edit it by hand later.

```py
print(Runner.generate_config(model=myModel))
```

An example configuration:
```py
{
    'prefix': 'MY_PV_PREFIX:',
    'update_rate': 0.1, # Update period under which PVs will be batched together into one model.set(). Set to 0 to disable the window.
    'protocol': ['ca', 'pva'], # Serve this as both CA and PVA (the default)
    'variables': {
        'input_a': {
            'name': 'input_a',
            'pv': 'input_a_pv',
            'mode': 'rw' # 'rw' means we can read and write this PV
        },
        'output_b': {
            'name': 'output_b',
            'pv': 'output_b_pv',
            'mode': 'ro' # 'ro' means the PV is read-only
        }
    }
}
```

## Relationship to lume-pva

This distribution exists so that changes can be exercised against a real
deployment before they are proposed upstream. It is a staging area, not a
long-lived divergence: each entry below is either an intended upstream pull
request or a packaging consequence of publishing under a second name, and the
distribution retires once the upstream ones are merged and released.

### Intended upstream pull requests

| Change | Rationale |
| --- | --- |
| Declare `p4p` as a first-class dependency behind a `pva` extra | `p4p` is imported unconditionally by `variables.py` and `runner.py` but was declared only under the `dev` extra, so a non-dev install of the project could not import its own runner. |
| Split the EPICS transports into `ca` and `pva` extras, leaving a pure-Python core | Installing the project currently forces a `pcaspy` build on any platform without a matching wheel, even for consumers that only generate configuration or only speak one protocol. |
| Raise the `pcaspy` floor to `>=0.8.1` | On 0.8.0 an asynchronous CA write raises inside the SWIG director and terminates the server process, while the client is told the put succeeded. The fix needs a binding 0.8.0 does not compile, so a version floor is the only remedy. |
| Replace the `lume-torch` VCS reference with the released `lume-torch>=3.0.0` | PyPI rejects direct-URL dependencies in any extra, so the `torch` extra as written makes the project unpublishable. |
| Adopt the PEP 639 SPDX licence expression and drop the deprecated `[project.license]` table | Published metadata is immutable, so the deprecated form has to go before a release rather than after one. |
| Drop `pydantic` and `pyyaml` from the dependency list | Neither is imported anywhere in the package. |
| Move CI from conda to `uv`, pin the interpreter per matrix leg, and assert it at runtime | A matrix that silently degrades to one interpreter tests the same leg repeatedly and stays green. |
| Run the test suite from outside the checkout, against the installed distribution | Run from the source tree, the tests import the adjacent package regardless of what the packaging metadata ships, so a packaging mistake cannot fail the build. |
| Add a job asserting the core install is pure-Python on Linux and macOS | The pure-Python core is a property consumers depend on to provision a host without an EPICS toolchain. Stated only in a comment it decays; the job checks the built wheel is `py3-none-any` and that no transport reaches the environment. |
| Add a job rejecting direct-URL dependencies in the metadata | PyPI refuses a distribution carrying one, in any extra. Without the gate this is discovered by the release that fails, after the tag is spent. |

### Fork-local changes

| Change | Rationale |
| --- | --- |
| Import package renamed to `lume_pva_apg`, distribution to `lume-pva-apg` | Two distributions cannot share an import package; the rename is what makes the two installable side by side. Retired when the fork is. |
| CI job failing on a facility-specific reference in the package | The staging fork is edited alongside a facility deployment, which is exactly the condition under which a site-specific name gets committed by accident. The gate keeps every change here shaped as something upstream can take. |
| Dropped the `no-commit-to-branch` pre-commit hook | Changes land on `main` here before they are proposed upstream, so the hook blocks the fork's only workflow. |
| Removed the `pvua` dependency and the remote-input mode it backed | `pvua` is only available as a VCS reference, which makes the project unpublishable, and the remote mode is unused here. Dropped rather than reworked: upstream owns `pvua` and should keep the feature, so this is not offered as a pull request. |

Removing the remote-input mode also removed what depended on it: the `remote`
PV mode, the `remote_model_mode` config key, the `remote_inputs` argument to
`generate_config`, snapshot mode, and the `{prefix}SNAPSHOT` control PV, whose
only purpose was to trigger a pull from remote inputs.

## Supported Variables

Supported variable types and their metadata fields.

### `ScalarVariable` and `IntVariable`

Represented as **NTScalar** with a `double` or `int` value field (depending on variable type).

Supported metadata:
* `timestamp`
* `display.units`
    * `ScalarVariable.unit`
* `control.limitLow`
    * `ScalarVariable.value_range[0]`
* `control.limitHigh`
    * `ScalarVariable.value_range[1]`
* `alarm.severity` and `alarm.status`
    * Set based on the value in relation to `value_range`. Out of range values trigger alarms.

### `NDVariable`

Represented as **NTNDArray** with data representation matching the numpy shape and dtype.

Supported metadata:
* `timestamp`

### `TorchScalarVariable`

Requires the `torch` extra (`pip install 'lume-pva-apg[torch]'`).

Represented as **NTScalar** with a `double` value field.

Supported metadata:
* `timestamp`

### `TorchNDVariable`

Requires the `torch` extra (`pip install 'lume-pva-apg[torch]'`).

Represented as **NTNDArray** with data representation matching the Tensor shape and dtype.

Supported metadata:
* `timestamp`

### `BoolVariable`

Represented as **NTScalar** with a `bool` value field.

Supported metadata:
* `timestamp`

### `StrVariable`

Represented as **NTScalar** with a `str` value field.

Supported metadata:
* `timestamp`
