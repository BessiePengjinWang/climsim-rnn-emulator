from pathlib import Path

import xarray as xr

TIMES_PER_DAY = list(range(0, 86400, 1200))  # 72 snapshots/day, one every 1200 s

DAYS = {
    "day1": "0001-02-01",
    "day2": "0001-02-02",
}


def _snapshot_files(directory: Path, tag: str, date: str) -> list[Path]:
    return [directory / f"E3SM-MMF.{tag}.{date}-{t:05d}.nc" for t in TIMES_PER_DAY]


def load_day(raw_dir: Path, day: str) -> tuple[xr.Dataset, xr.Dataset]:
    """Load one day of ClimSim mli/mlo snapshots as (input, output) datasets.

    Both datasets are concatenated along a new `time` dimension of length 72,
    matching the 384-column x 60-level grid of the low-resolution ClimSim data.
    """
    if day not in DAYS:
        raise ValueError(f"Unknown day '{day}', expected one of {list(DAYS)}")
    date = DAYS[day]
    raw_dir = Path(raw_dir)

    input_files = _snapshot_files(raw_dir / f"{day}_input", "mli", date)
    output_files = _snapshot_files(raw_dir / f"{day}_output", "mlo", date)

    missing = [f for f in input_files + output_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} ClimSim snapshot file(s), e.g. {missing[0]}. "
            "See data/README.md for how to obtain the raw data."
        )

    ds_in = xr.concat([xr.open_dataset(f) for f in input_files], dim="time")
    ds_out = xr.concat([xr.open_dataset(f) for f in output_files], dim="time")
    return ds_in, ds_out
