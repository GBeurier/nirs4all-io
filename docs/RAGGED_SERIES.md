# Variable-length series in Python

`RaggedSeriesSource` preserves raw multichannel series whose lengths differ
between samples. It is a separate source type from the dense `TensorSource`.
This is the host IO contract; parsing external sequence files and choosing ML
encoders remain separate concerns.

```python
import numpy as np
from nirs4all_io import MultimodalDataset, RaggedSeriesSource

series = RaggedSeriesSource(
    np.array([[10.0, 20.0], [11.0, 21.0], [30.0, 40.0]]),
    offsets=[0, 2, 3],
    sample_ids=["plant-b", "plant-a"],
    time_coordinates=np.array([0.0, 1.5, 4.0]),
    channel_names=["temperature", "humidity"],
    time_unit="h",
)
cohort = MultimodalDataset(
    {"weather": series}, sample_ids=["plant-a", "plant-b"],
    y=[1.0, 2.0], task_type="regression",
)
batch = cohort.source_values()[0]
assert batch.lengths.tolist() == [1, 2]
assert batch.shape == (2, None, 2)
assert batch[0].shape == (1, 2)
```

The packed `values` array has shape `(total_points, channels)`. Integer offsets
start at zero, end at `total_points`, and contain one boundary per sample plus
the final boundary. Offsets normalize exactly to int64. Measurement and time
coordinate dtypes are preserved, including byte order. Buffers are copied and
read-only. No padding, truncation, interpolation, unit conversion or learned
processing occurs during assembly.

`source.values` is a `RaggedSeriesBatch`. Its `.values`, `.offsets`,
`.time_coordinates` and `.lengths` expose the packed buffers and boundaries.
`batch.take_rows(indices)` accepts integer positions or an exact-length boolean
mask. `source.take(ids)` and `cohort.take(ids)` select by unique sample IDs.
`np.asarray(batch)` raises an explicit error: an encoder must support the
ragged type. Channel names are also exposed through `source.feature_names`;
their count describes channels, not a flattened feature width.

Present series must contain at least one point and finite measurements. A
`presence_mask=False` row may be empty or retain opaque hidden measurements.
Source presence and target validity are separate masks. A completely absent
source can use `values=np.empty((0, channels))`, `offsets=[0]`, and
`sample_ids=[]`; explicit `source_alignment="left"` creates empty absent rows
for the canonical cohort IDs. Foreign source IDs always fail validation.
Present rows with missing individual measurements are outside this profile.

Optional time coordinates contain one real finite value per packed point.
They must increase strictly within each sample; samples may have different
time origins and spacing. Unspecified time units remain `None`. Providing
coordinates does not request interpolation or time-weighted statistics.

The source uses DAG-ML Data's existing `series_mv` representation, rank 3, with
`ragged=True` and an unsized variable time axis. Replay schemas bind measurement
dtype, channels, channel names, units, and the presence/dtype of time
coordinates. Sample counts, offsets, lengths and individual time coordinates
are data values and are excluded from the schema. New series lengths can
therefore satisfy the same input contract.

`MultimodalDataset.to_dict()` / `from_dict()` preserve the packed arrays, times,
masks and identities through the version-1 JSON format. Ragged records have
`source_kind="ragged_series"`; existing dense records are unchanged. Pickle
reconstructs through validated constructors and restores read-only buffers.
The same JSON dtype restrictions apply as for dense sources.

`DataProvider` can return a ragged cohort or add ragged sources to an explicit
fixed base. Provider checkpoints compare the complete serialized content,
including masks, offsets, times and hidden buffers, after explicit deterministic
regeneration. This exact checkpoint identity is distinct from any downstream
ML policy that excludes hidden observations from model inputs or cache keys.

`SklearnProviderAdapter(cohort, source=None)` returns a dictionary whose ragged
entries remain copied, read-only `RaggedSeriesBatch` objects, including offsets
and time coordinates. This also applies to its sample batches. A consumer must
choose an explicit ragged-aware encoder before an ordinary matrix estimator.
Selecting `source="weather"` as a single sklearn matrix still refuses rank-3
series; selecting another dense rank-2 source works. Missing sources or labels
require `return_metadata=True`, which retains both masks alongside the typed
batch.

Sklearn array selection resolves the requested unique IDs to row positions once.
Selecting a matrix source copies only that source's rows and the requested
target/metadata fields; it does not project other images or ragged series or
construct a temporary cohort. `batches()` reuses the selection positions across
its batches, applying `start` and `drop_last` within the requested ID order.
Returned dense arrays remain mutable copies; ragged batches remain typed and
read-only. Missingness outside the sources and rows being returned does not
reject an otherwise complete tuple result. IDs must remain unique and known.

`TorchMapDataset` and `TorchIterableDataset` reject ragged sources by default,
including equal-length series and empty selections. Choose packed collation
explicitly and retain the metadata:

This tightens the behavior of IO 0.2.0, which could silently discard time
coordinates when passing ragged rows to Torch's default collator. Existing
callers must select `ragged_policy="packed"` and `return_metadata=True`.

```python
from torch.utils.data import DataLoader
from nirs4all_io.provider_adapters import (
    TorchMapDataset, TorchRaggedSeriesBatch, collate_provider_samples,
)

dataset = TorchMapDataset(cohort, ragged_policy="packed", return_metadata=True)
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_provider_samples)
item = next(iter(loader))
packed = item["X"]["weather"]
assert isinstance(packed, TorchRaggedSeriesBatch)
assert item["sample_id"] == ["plant-a", "plant-b"]
assert packed.offsets.tolist() == [0, 1, 3]
assert packed.lengths.tolist() == [1, 2]
assert packed.time_coordinates.tolist() == [4.0, 0.0, 1.5]
assert packed.presence_mask.tolist() == [True, True]
assert packed.channel_names == ("temperature", "humidity")
assert packed.time_unit == "h"
```

Each uncollated ragged entry is a one-row `RaggedSeriesSource`; its times and
declarations survive sample extraction. `collate_provider_samples` packs these
entries into `TorchRaggedSeriesBatch`. It contains `values` with shape
`(total_points, channels)`, int64 `offsets` and `lengths`, optional
`time_coordinates`, boolean `presence_mask`, and unchanged `channel_names`
and `time_unit`. Sample IDs, source masks, target masks, labels, groups and
partitions remain in the existing metadata envelope. In particular,
`packed.presence_mask` matches `item["source_masks"][source_name]`.

No series are sorted, padded, truncated or dropped. Absent rows may have zero
length or contain opaque hidden measurements; the mask remains authoritative.
An entirely absent batch has `(0, channels)` values and repeated zero offsets.
This container is not a Torch RNN `PackedSequence`. Consumers decide how their
models handle packed rows, missing sources and partially observed targets.

Packed collation requires identical source schemas across samples, including
channel count/names, time unit, measurement dtype, coordinate presence and
coordinate dtype. It refuses inconsistent IDs or presence metadata. Numeric
dtypes are preserved where Torch can represent them; non-native byte order
and unsupported types such as extended-precision floats are explicitly
rejected before conversion. There is no implicit float32 cast. Unknown policy
values fail at adapter construction. Dense sources keep their previous
collation behavior, including mixed tables and string labels.

`packed.to(device, non_blocking=False)` returns a batch with its tensors moved
to that device, preserving dtypes and declarations. It accepts no dtype
conversion overload. `packed.pin_memory()` returns the same structure using
Torch's pinned-memory allocator, and integrates with `DataLoader(pin_memory=True)`.
Neither method mutates the IO source or the batch's fields; a move to its
existing device may reuse tensor storage, as in Torch. Actual pinning and
accelerator transfers depend on the installed Torch build and hardware.
CPU-only qualification does not establish physical pinned-memory or GPU support.

The datasets and packed batches support pickle and DataLoader spawn workers.
Adapters capture the materialized cohort without its generator callback.
Map datasets retain sampler order, including repeated samples. Iterable
workers keep disjoint position shards: every sample is emitted once, while
the global order can differ with worker count. Each packed batch remains
aligned with its own `sample_id` sequence. Neither adapter provides a
DataLoader prefetch checkpoint or mid-epoch resume. Importing the adapter
module or the packed batch type does not import Torch; requesting a Torch
dataset class or collating samples requires that optional dependency.

The public cross-language envelope `nirs4all.dataset.v2` (schema version 2)
retains the existing inner `nirs4all.multimodal-dataset` version 1 record. It
admits packed `source_kind: "ragged_series"` records and explicit target masks.
Offsets are an int64 vector of length samples + 1, from zero to packed row
count; coordinates increase strictly within each sample. IO aligns packed
segments and presence by sample identity, including empty missing segments
under explicit left alignment. Channel names, time dtype and time units remain
part of the predictor input contract. IO performs no ragged encoding.

A false target mask admits JSON null or finite numeric storage and normalizes
its storage to zero; the mask retains the distinction from observed truth.
True target cells must be finite. Native `normalize_dataset`, Python
`nirs4all_io.Dataset.from_dict`, and JavaScript `new Dataset(record)` validate
the same synthetic fixture under `tests/fixtures/public-dataset-v2.json`.
Version 1 continues to reject ragged public records and nullable targets.

Complete matrix workflows use native `matrix_dataset_package(record, source)`
or Python `Dataset.to_matrix_regression(source)` / JavaScript
`Dataset.toMatrixRegression(source)`. They retain numeric multi-target columns,
target names, task type and sample order. Classification is explicit mono-y
int64; this first f32 matrix execution profile refuses labels that cannot be
represented exactly rather than recoding classes. Legacy dense regression
continues to refuse multi-target and classification input.

Masked consumers must opt into native
`masked_matrix_dataset_package(record, source)`; it returns the package and a
validated projection containing sample IDs, target names, the target mask and
its canonical content fingerprint. They must join the mask by sample IDs and
bind it to native fit, refit and scoring. Python
`Dataset.to_masked_matrix_regression(source)` and JavaScript
`Dataset.toMaskedMatrixRegression(source)` expose the mask explicitly. The
complete projection refuses incomplete targets so existing consumers cannot
silently treat placeholder zero as truth.

Native `projected_matrix_dataset(record, projections)` returns a dense
`native_features` record and source provenance after Methods has produced each
projection. Projection records declare `source_id`, `sample_ids`, a finite
float matrix, unique `feature_names`, and boolean `presence_encoded`; their
inventory follows IO source order. IO joins projected rows by identity and
concatenates columns with source-qualified names. Missing input presence is
refused unless the native recipe explicitly declares its encoding. Source
schemas, input presence, projection content fingerprints and input content
fingerprints are retained for Core/DAG to bind to the persisted recipe.
Python `nirs4all_io.projected_matrix_dataset` delegates to this native owner.
JavaScript `projectedMatrixDataset(record, projections, digest)` exposes the
same assembly and provenance contract, with the caller's native content digest.
False target sentinels are normalized before dtype/f32 conversion, including
finite values too large for f32; observed target overflow remains an error.

The opt-in masked matrix projection also preserves rank-2 int64 classification
targets. Each named column represents an independent classifier target; consumers
must select its matching observation mask and target index before native fitting
and scoring. Class IDs retain their integer values and must be exactly
representable in float32 storage. Unobserved placeholders become zero before this
check. The complete classification projection still requires one int64 vector.
