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

### The Write Path

A write to an input PV is queued, applied to the model on the next simulation
cycle, and only then acknowledged: a `caput -c` or a PVA put with `wait=True`
stays blocked until the cycle that consumed it has finished. Three properties
of what happens in between are selectable, and each defaults to the behaviour
the runner has always had.

`echo_unconfirmed_writes` (default `True`) decides whether an input PV may
advertise a value the model has not accepted. By default the requested value is
published immediately on PVA, and recorded on CA whether or not the cycle
succeeded, so a client reading back an input after a failed cycle can see a
value the model never took. Set it to `False` and the echo waits for the model
to accept the write, and is withheld entirely if the model refuses it, leaving
the PV on the last value that actually landed.

`alarm_on_refused_write` (default `False`) raises `WRITE_ALARM`/`INVALID_ALARM`
on the CA PV when the model refuses a write. Channel Access put-completion
carries no failure channel — it can only ever report success — so an alarm is
the only way to tell a CA client its write did not land. PVAccess reports the
failure through the put itself and needs no equivalent.

`clamp_writes` (default `False`) clamps a written value into the variable's
`value_range` before handing it to the model. `LUMEModel.set` does not enforce
`value_range`, so by default an out-of-range write reaches the model unchanged
and it is the model's job to reject it. The clamp is applied where the write
enters the server, so the value echoed back to the client is the same value the
model was given.

`update_rate` is the length in seconds of the window during which arriving
writes are batched into a single `model.set()`. Set it to zero and the window
is skipped: every queued write drives a `model.set()` of its own, so one
client's write is never merged into another's.

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
    'echo_unconfirmed_writes': True, # Echo a written value before the model has accepted it
    'alarm_on_refused_write': False, # Alarm the CA PV when the model refuses a write
    'clamp_writes': False, # Clamp a written value into the variable's value_range
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
| Flush the input echo before signalling put-completion | pcaspy publishes a value to monitoring clients only when `updatePV` runs, so a client released by put-completion was told its write had finished while its own monitor still carried the value that write replaced. |
| Withhold the input echo when the model refuses a write (`echo_unconfirmed_writes`) | A failed simulation cycle leaves the model on its previous value, but the requested value is recorded on the input PV regardless, so a read-back reports a value the model never took. Put-completion cannot report the failure, so withholding the echo is the only signal available. |
| Alarm a refused write (`alarm_on_refused_write`) | Channel Access put-completion ends an asynchronous write with `S_casApp_success` unconditionally; there is no failure channel. An alarm is the only way to tell a CA client its write did not land. |
| Skip the batching window when `update_rate` is zero | The window was skipped only because its deadline had already elapsed by the time it was tested, making per-write isolation an accident of the clock rather than something the documented `update_rate` of zero guarantees. |
| Clamp a write into the variable's `value_range` (`clamp_writes`) | `LUMEModel.set` does not enforce `value_range`, so an out-of-range write reaches the model unchallenged and, on failure, costs a whole simulation cycle. Applied at the point the write enters the server, so the echo matches what the model was given. |
| Publish `value_range` as CA display limits only, not as alarm thresholds | pcaspy compares `lolo`/`hihi` with `<=`/`>=`, so thresholds taken from the variable's own range put a value driven to either end of its legal span into MAJOR alarm — where the PVA path, which compares strictly, reports no alarm for the same value. This does not preserve the old behaviour for a value *outside* the range, which no longer alarms on CA at all; pcaspy's inclusive comparison cannot express a threshold that alarms outside the range without also alarming at it. |
| Apply the PV name prefix exactly once on the Channel Access path | `prefix` was written into the pvdb keys and then applied again by `SimpleServer.createPV`, so a runner configured with `PFX:` served `PFX:PFX:name`. The driver names a PV by its pvdb key, so the same mistake left the cycle's output pass calling `setParam` with a name the database did not hold: every cycle raised `KeyError` and was logged as a failed simulation. Keying the database by base name leaves the prefix to the server, which is also what names a PV in every driver callback. Invisible at `prefix=""`, which is what every existing test used. |

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
* `alarm.severity` and `alarm.status` (PVA only)
    * Set from the value's position relative to `value_range`: a value strictly outside it is a MAJOR alarm, a value anywhere within it — including at either limit — is no alarm.

On CA, `value_range` is served as the display limits (`lolim`/`hilim`) and is
not served as an alarm threshold. pcaspy compares its `lolo`/`hihi` thresholds
inclusively, so a range published as an alarm limit puts a value driven to
either end of its own legal span into MAJOR alarm. A CA client therefore sees
no range-derived alarm at all, while a PVA client still sees one for a value
outside the range.

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
