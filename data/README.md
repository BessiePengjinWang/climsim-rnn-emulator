# Data

`data/raw/` holds two days of ClimSim low-resolution E3SM-MMF snapshots (`day1_*`, `day2_*`), each a set of 72 NetCDF files spaced 1200 s apart across 384 grid columns and 60 vertical levels (288 files, about 417 MB in total). The files are committed to this repository unmodified, so a clone contains everything the notebook needs. Cloning is therefore about 300 MB.

## Source and license

- Source: [LEAP/ClimSim_low-res on Hugging Face](https://huggingface.co/datasets/LEAP/ClimSim_low-res), part of the ClimSim dataset ([paper](https://arxiv.org/abs/2306.08754): "ClimSim: An open large-scale dataset for training high-resolution physics emulators in hybrid multi-scale climate simulators").
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit goes to the ClimSim authors. The files here are an unmodified subset: the two simulated days 0001-02-01 and 0001-02-02.
- The MIT license in the repository root covers the code in this repository, not this data.

## Layout

```
data/raw/
├── day1_input/   E3SM-MMF.mli.0001-02-01-*.nc   (72 files)
├── day1_output/  E3SM-MMF.mlo.0001-02-01-*.nc   (72 files)
├── day2_input/   E3SM-MMF.mli.0001-02-02-*.nc   (72 files)
└── day2_output/  E3SM-MMF.mlo.0001-02-02-*.nc   (72 files)
```

Day 1 is used for training and validation (a random 90/10 split of its samples); day 2 is held out entirely and used only for the final evaluation – see [`notebooks/01_rnn_climate_tendency_emulator.ipynb`](../notebooks/01_rnn_climate_tendency_emulator.ipynb).
