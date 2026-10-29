"""The surface climatology of raw FVCOM output, on its own mesh, for analyze().

CDO cannot read FVCOM's unstructured variables (see oceanval.fvcom), so the
nctoolkit chain the rest of Analyze uses cannot work them out. Here they are
read with netCDF4, the top sigma layer (siglay 0, FVCOM's first layer being the
one at the surface) of a block of times at a time, so a file is never read
whole: for each month of each year, the sum of each node's (or element's)
values and how many there are. The monthly mean of each year is taken first,
then the climatology is the mean of each month of the year over the years, and
the annual mean the mean of those, as analysis.climatology_of does.

The files can be shared out to `cores` processes, each running this file as a
script by its path (python -I), as it imports nothing of OceanVal: a spawned
worker of a module of the package would import the package, and the user's
script with it, which would then run again.
"""

import json
import os
import subprocess
import sys
import tempfile

import cftime
import numpy as np

# the location of a variable on the mesh: its nodes or its elements
NODE = "node"
NELE = "nele"
# the vertical dimension of a variable on sigma layers, whose first is the surface
SIGLAY = "siglay"
# the mesh, as the results files keep it, so a map can be drawn of them alone
MESH = ("lon", "lat", "lonc", "latc", "x", "y", "nv")
# the most values read at once, from one variable
BLOCK_VALUES = 20_000_000


def variables(path):
    """The variables of the FVCOM file at path that can be mapped (those on
    its nodes or elements, other than the mesh's own), as analysis' streams
    gives them: {name: {"long_name", "units", "levels"}}, levels the sigma
    layers or levels, or 1."""
    import netCDF4

    from oceanval.viewer_mesh import MESH_VARIABLES

    found = {}
    with netCDF4.Dataset(path) as handle:
        for name, variable in handle.variables.items():
            dims = variable.dimensions
            if name in MESH_VARIABLES or not ({NODE, NELE} & set(dims)) or "time" not in dims:
                continue
            levels = [len(handle.dimensions[dim]) for dim in dims if dim in (SIGLAY, "siglev")]
            found[str(name)] = {
                "long_name": str(getattr(variable, "long_name", "")),
                "units": str(getattr(variable, "units", "")),
                "levels": int(levels[0]) if levels else 1,
            }
    return found


def location(path, parts):
    """NODE or NELE: where every one of the variables in parts is on the mesh
    in the file at path. Raises ValueError if they differ, or are on neither."""
    import netCDF4

    found = {}
    with netCDF4.Dataset(path) as handle:
        for name in parts:
            dims = handle.variables[name].dimensions
            found[name] = NODE if NODE in dims else NELE if NELE in dims else None
    if any(where is None for where in found.values()):
        missing = [name for name, where in found.items() if where is None]
        raise ValueError(
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} on neither the nodes nor the "
            "elements of the FVCOM mesh, so cannot be mapped."
        )
    if len(set(found.values())) > 1:
        raise ValueError(
            " and ".join(parts) + " are not all on the same part of the FVCOM mesh (some are on its nodes, "
            "some on its elements), so cannot be summed."
        )
    return next(iter(found.values()))


def _times(handle, time_shift):
    """The (year, month) of each time in an open FVCOM file, moved by
    time_shift days."""
    time = handle.variables["time"]
    calendar = getattr(time, "calendar", "standard")
    dates = cftime.num2date(np.asarray(time[:], dtype="float64"), time.units, calendar)
    dates = np.atleast_1d(dates)
    if time_shift:
        import datetime

        dates = [date + datetime.timedelta(days=time_shift) for date in dates]
    return [(int(date.year), int(date.month)) for date in dates]


def _missing(values, as_missing):
    """values with those as_missing says (a value, or [low, high]) made NaN."""
    if as_missing is None:
        return values
    if isinstance(as_missing, (list, tuple)):
        low, high = as_missing if len(as_missing) == 2 else (as_missing[0], as_missing[0])
    else:
        low = high = as_missing
    values[(values >= low) & (values <= high)] = np.nan
    return values


def _surface(variable, start, stop):
    """The values of variable at times start to stop, at the surface, as a
    (time, cells) float64 array with NaN where they are missing."""
    dims = variable.dimensions
    index = []
    for dim in dims:
        if dim == "time":
            index.append(slice(start, stop))
        elif dim in (NODE, NELE):
            index.append(slice(None))
        else:
            # siglay (or siglev): the first, at the surface
            index.append(0)
    variable.set_auto_maskandscale(True)
    values = np.ma.filled(np.ma.asarray(variable[tuple(index)]).astype("float64"), np.nan)
    return values.reshape(stop - start, -1)


def file_sums(path, parts, years, time_shift=None, as_missing=None):
    """The sums of the values of the sum of parts at the surface in one FVCOM
    file, and how many there are, in each month of years ([first, last]):
    {(year, month): (sums, counts)}, each an array with a value for each
    cell. A cell missing in any part is missing in the sum, as CDO sums."""
    import netCDF4

    found = {}
    with netCDF4.Dataset(path) as handle:
        months = _times(handle, time_shift)
        if not months:
            return found
        first = handle.variables[parts[0]]
        cells = int(np.prod([len(handle.dimensions[dim]) for dim in first.dimensions if dim in (NODE, NELE)]))
        block = max(1, BLOCK_VALUES // max(cells, 1))
        for start in range(0, len(months), block):
            stop = min(start + block, len(months))
            wanted = [i for i in range(start, stop) if years[0] <= months[i][0] <= years[1]]
            if not wanted:
                continue
            values = None
            for name in parts:
                part = _missing(_surface(handle.variables[name], start, stop), as_missing)
                values = part if values is None else values + part
            for i in wanted:
                row = values[i - start]
                sums, counts = found.setdefault(months[i], (np.zeros(row.size), np.zeros(row.size, dtype="int64")))
                ok = np.isfinite(row)
                sums[ok] += row[ok]
                counts += ok
    return found


def _add(total, found):
    for month, (sums, counts) in found.items():
        if month in total:
            total[month][0][:] += sums
            total[month][1][:] += counts
        else:
            total[month] = (sums.copy(), counts.copy())


def _run_jobs(jobs, out_dir):
    """Work out the file_sums of each of jobs ([path, parts, years,
    time_shift, as_missing]), saving each as out_dir/<n>.npz: the script
    each worker process runs."""
    for number, (path, parts, years, time_shift, as_missing) in jobs:
        found = file_sums(path, parts, years, time_shift, as_missing)
        arrays = {}
        for (year, month), (sums, counts) in found.items():
            arrays[f"sums_{year}_{month}"] = sums
            arrays[f"counts_{year}_{month}"] = counts
        np.savez(os.path.join(out_dir, f"{number}.npz"), **arrays)


def _read_saved(path):
    found = {}
    with np.load(path) as saved:
        for name in saved.files:
            if name.startswith("sums_"):
                _, year, month = name.split("_")
                found[(int(year), int(month))] = (saved[name], saved[f"counts_{year}_{month}"])
    return found


def monthly_sums(files, parts, years, time_shift=None, as_missing=None, cores=1):
    """The file_sums of every one of files, added: {(year, month): (sums,
    counts)}. With cores > 1 the files are shared out to that many processes
    running this file as a script."""
    total = {}
    workers = max(1, min(int(cores), len(files)))
    if workers == 1:
        for path in files:
            _add(total, file_sums(path, parts, years, time_shift, as_missing))
        return total
    jobs = [[number, [path, list(parts), list(years), time_shift, as_missing]] for number, path in enumerate(files)]
    with tempfile.TemporaryDirectory(prefix="oceanval_fvcom_") as out_dir:
        running = []
        for worker in range(workers):
            share = jobs[worker::workers]
            running.append(
                subprocess.Popen(
                    [sys.executable, "-I", os.path.abspath(__file__), out_dir],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
            running[-1].stdin.write(json.dumps(share))
            running[-1].stdin.close()
        errors = []
        for process in running:
            error = process.stderr.read()
            process.wait()
            if process.returncode != 0:
                errors.append(error[-3000:])
        if errors:
            raise RuntimeError("The FVCOM output could not be read:\n" + errors[0])
        for number, _ in jobs:
            _add(total, _read_saved(os.path.join(out_dir, f"{number}.npz")))
    return total


def climatology(total):
    """The months of the year in total (see monthly_sums), and the
    climatology of each: the mean over the years of each year's monthly mean,
    as ([month, ...], array (months, cells)). A cell with no values is NaN."""
    by_month = {}
    for (year, month), (sums, counts) in sorted(total.items()):
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        by_month.setdefault(month, []).append(mean)
    months = sorted(by_month)
    fields = []
    for month in months:
        stack = np.vstack(by_month[month])
        with np.errstate(invalid="ignore"):
            seen = np.isfinite(stack).sum(axis=0)
            fields.append(np.where(seen > 0, np.nansum(stack, axis=0) / np.maximum(seen, 1), np.nan))
    return months, np.vstack(fields) if fields else np.zeros((0, 0))


def _mesh(path):
    """The variables of the mesh of the FVCOM file at path, as xarray
    variables, and the calendar of its times."""
    import netCDF4
    import xarray as xr

    mesh = {}
    with netCDF4.Dataset(path) as handle:
        for name in MESH:
            if name in handle.variables:
                variable = handle.variables[name]
                attrs = {key: variable.getncattr(key) for key in variable.ncattrs() if key != "_FillValue"}
                mesh[name] = xr.Variable(variable.dimensions, np.asarray(variable[:]), attrs)
        calendar = getattr(handle.variables["time"], "calendar", "standard") if "time" in handle.variables else "standard"
    return mesh, calendar


FILL = np.float32(-1e20)
TIME_UNITS = "days since 1900-01-01 00:00:00"


def save(path, key, where, months, fields, year, mesh_from, units, long_name):
    """Save fields (a row for each of months) as variable key on the mesh
    (where: NODE or NELE) of the FVCOM file mesh_from, at path: a time for
    each month, on its first day in year, as CDO dates a climatology, and the
    month of each in "month"."""
    import xarray as xr

    mesh, calendar = _mesh(mesh_from)
    dates = [cftime.datetime(year, month, 1, calendar=calendar) for month in months]
    time = cftime.date2num(dates, TIME_UNITS, calendar)
    ds = xr.Dataset(
        {
            key: (("time", where), np.asarray(fields, dtype="float32"), {"units": units, "long_name": long_name}),
            "month": (("time",), np.asarray(months, dtype="int32"), {"long_name": "month of the year"}),
            **mesh,
        },
        coords={"time": ("time", np.asarray(time, dtype="float64"), {"units": TIME_UNITS, "calendar": calendar})},
        attrs={"source": "oceanval", "mesh": where},
    )
    ds[key].encoding["_FillValue"] = FILL
    if os.path.exists(path):
        os.remove(path)
    ds.to_netcdf(path)
    return path


def work_out(files, parts, years, key, monthly, annual, units, long_name, time_shift=None, as_missing=None, cores=1):
    """The surface climatology of the sum of parts in the FVCOM files, over
    years ([first, last]): its monthly climatology saved at monthly, and its
    annual mean, the mean of the months, at annual. Returns (where, months):
    the part of the mesh it is on, and how many months the climatology has.
    Raises ValueError if there is no output in years."""
    where = location(files[0], parts)
    total = monthly_sums(files, parts, years, time_shift, as_missing, cores)
    if not total:
        raise ValueError(f"There is no output of {'+'.join(parts)} in {years[0]}-{years[1]}.")
    months, fields = climatology(total)
    last = max(year for year, _ in total)
    save(monthly, key, where, months, fields, last, files[0], units, long_name)
    with np.errstate(invalid="ignore"):
        seen = np.isfinite(fields).sum(axis=0)
        mean = np.where(seen > 0, np.nansum(fields, axis=0) / np.maximum(seen, 1), np.nan)
    save(annual, key, where, [months[0]], mean[None, :], last, files[0], units, long_name)
    return where, len(months)


if __name__ == "__main__":
    _run_jobs(json.loads(sys.stdin.read()), sys.argv[1])
