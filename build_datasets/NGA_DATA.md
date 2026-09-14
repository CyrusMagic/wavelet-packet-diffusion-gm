# NGA-West2 Data Access and Local HDF5 Preparation

This repository does **not** distribute NGA-West2 waveforms or metadata. NGA-West2 is distributed by the Pacific Earthquake Engineering Research Center (PEER), and users should obtain the data directly from PEER and prepare local HDF5 files with the scripts in `build_datasets/`.

The NGA-West2 database used in this study is available from the PEER Data & Databases page:

```text
https://ngawest2.berkeley.edu/
```

## Why the data are not included

NGA-West2 data are not publicly redistributable as part of an open-source GitHub repository. Therefore, all `*.h5` datasets are expected to be built locally and are intentionally excluded from this repo.

## Expected input file: `datasets/NGAW2_acc_meta.h5`

After downloading and organizing NGA-West2 records locally, the data are expected to be available as a single HDF5 file:

```text
datasets/NGAW2_acc_meta.h5
```

Below is the **observed HDF5 structure** (obtained by running
`/Users/cyrus/Nutstore/CyrusMagic/NGA-EDM-Sa-zh/scripts/check_h5_structure.py` on a local copy of `NGAW2_acc_meta.h5`):

```bash
python /Users/cyrus/Nutstore/CyrusMagic/NGA-EDM-Sa-zh/scripts/check_h5_structure.py \
  datasets/NGAW2_acc_meta.h5
```

```text
/ (root)
  events/ (Group)
    <event_id>/ (Group) attrs: magnitude, mechanism, year
      stations/ (Group)
        <station_id>/ (Group) attrs: station_name, vs30
          acceleration/ (Group) attrs: dt, npts, hypd, rjb, rsn
            H1 (Dataset) float32, shape (npts,)
            H2 (Dataset) float32, shape (npts,)
            V  (Dataset) float32, shape (npts,)

  metadata/ (Group)
    events_table   (Dataset) shape (422,)
      dtype fields: event_name (S50), magnitude (float32), mechanism (S20), year (S10)
    records_index  (Dataset) shape (58887,)
      dtype fields: rsn (S10), event (S50), station (S50), component (S10),
                   dt (float32), npts (int32), magnitude (float32),
                   vs30 (float32), hypd (float32)
    stations_table (Dataset) shape (25693,)
      dtype fields: event_station (S100), station_name (S100), vs30 (float32)
```

Notes:
- `events/<event_id>/stations/<station_id>/acceleration/<component>` stores a single component time history in units of **g** (float32).
- This repo assumes three components: `H1`, `H2`, and `V`.
- Observed scale (example file): 422 events, 25699 stations, 58887 records.

## Build local training/evaluation datasets

1) Create a local datasets directory:

```bash
mkdir -p datasets
```

2) Place your prepared `NGAW2_acc_meta.h5` at:

```text
datasets/NGAW2_acc_meta.h5
```

3) Build the step-1 dataset (100 Hz resampling + 5% damping response spectrum + optional db6 wavelet packet export):

```bash
python build_datasets/build_dataset1_nga_100hz.py \
  --source datasets/NGAW2_acc_meta.h5 \
  --mode h1
```

By default this produces (auto-generated):

```text
datasets/step1_NGAH1_len16k_symmetric_freq.h5
```

4) Build the step-2 dataset (compute CDF-normalized IM features):

```bash
python build_datasets/build_dataset2_pga_cdf.py \
  --source datasets/step1_NGAH1_len16k_symmetric_freq.h5
```

By default this produces:

```text
datasets/step2_NGAH1_len16k_symmetric_freq.h5
```
