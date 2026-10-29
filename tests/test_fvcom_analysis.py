"""Analyze with raw FVCOM output (see oceanval.fvcom_climatology and
oceanval.mesh_maps): so far only the climatology (at the surface, the bottom
and through the water column) is worked out, on the mesh itself, and
everything else is refused, by analyze() and by the window."""

import ast
import datetime
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

import oceanval  # noqa: E402
from oceanval import analysis, fvcom_climatology, mesh_maps, missing_probe  # noqa: E402
from oceanval.app import App  # noqa: E402
from test_app import get_json, leftover_dir, post, wait_for  # noqa: E402, F401 (leftover_dir is autouse)
from viewer_simulations import fvcom_file, fvcom_mesh, fvcom_months  # noqa: E402

MONTHS = ((2012, 1), (2012, 2), (2012, 3), (2012, 4), (2013, 1), (2013, 2))
SURFACE = {"start": 2012, "end": 2013, "surface": True}


@pytest.fixture
def fvcom_sim(tmp_path):
    return fvcom_months(str(tmp_path / "fvcom_sim"), MONTHS)


def days_of(year, month):
    """The days since 2012-01-01 of each day of a month, as fvcom_file dates them."""
    first = datetime.date(year, month, 1)
    after = datetime.date(year + month // 12, month % 12 + 1, 1)
    return np.arange((first - datetime.date(2012, 1, 1)).days, (after - datetime.date(2012, 1, 1)).days)


def expected_temperature(lon, months=MONTHS):
    """The monthly climatology of temp at the surface (10 + lon + the day)
    and its annual mean, by hand: each year's monthly mean, then the mean of
    each month of the year over the years."""
    by_month = {}
    for year, month in months:
        by_month.setdefault(month, []).append(10 + lon + days_of(year, month).mean())
    climatology = np.array([np.mean(by_month[month], axis=0) for month in sorted(by_month)])
    return sorted(by_month), climatology, climatology.mean(axis=0)


def analyze(sim, out, **arguments):
    arguments.setdefault("variables", {"Temperature": "temp"})
    arguments.setdefault("climatology", {name: SURFACE for name in arguments["variables"]})
    return oceanval.analyze(sim_dir=sim, n_dirs_down=1, cores=1, view=False, out_dir=str(out), **arguments)


def results_of(out):
    return os.path.join(str(out), analysis.FOLDER, analysis.RESULTS)


class TestClimatology:
    def test_the_surface_climatology_is_worked_out_on_the_mesh(self, fvcom_sim, tmp_path, no_report):
        analyze(fvcom_sim, tmp_path / "out")
        info = no_report[-1]
        assert info["fvcom"] is True
        entry = info["variables"][0]
        layer = entry["climatology"]["layers"]["surface"]
        assert layer["mesh"] == "node" and layer["months"] == 4 and layer["units"] == "degrees_C"
        assert entry["levels"] == 5 and entry["time_res_text"] == "daily"
        lon, _, _ = fvcom_mesh()
        months, climatology, annual = expected_temperature(lon)
        results = results_of(tmp_path / "out")
        with xr.open_dataset(os.path.join(results, layer["monthly"])) as ds:
            assert ds["temperature"].dims == ("time", "node")
            assert list(ds["month"].values) == months
            assert np.allclose(ds["temperature"].values, climatology, atol=1e-4)
            # the mesh is kept, so the maps are drawn of the results alone
            assert {"lon", "lat", "nv"} <= set(ds.variables)
            assert ds["temperature"].attrs["long_name"] == "Temperature, surface"
        with xr.open_dataset(os.path.join(results, layer["annual"])) as ds:
            assert np.allclose(ds["temperature"].values[0], annual, atol=1e-4)

    def test_values_on_the_elements_and_sums(self, fvcom_sim, tmp_path, no_report):
        analyze(fvcom_sim, tmp_path / "out", variables={"Velocity": "u", "Plus": "temp+zeta"})
        info = no_report[-1]
        velocity, plus = info["variables"]
        assert velocity["climatology"]["layers"]["surface"]["mesh"] == "nele"
        results = results_of(tmp_path / "out")
        with xr.open_dataset(os.path.join(results, velocity["climatology"]["layers"]["surface"]["annual"])) as ds:
            # u is each element's number, at every time
            assert np.allclose(ds["velocity"].values[0], np.arange(ds.sizes["nele"]))
        lon, _, _ = fvcom_mesh()
        with xr.open_dataset(os.path.join(results, plus["climatology"]["layers"]["surface"]["monthly"])) as ds:
            assert np.allclose(ds["plus"].values, expected_temperature(lon)[1] + 0.5, atol=1e-4)

    def test_the_years_and_what_is_missing(self, fvcom_sim, tmp_path, no_report):
        analyze(
            fvcom_sim, tmp_path / "out", variables={"Velocity": "u", "Temperature": "temp"},
            climatology={"Velocity": SURFACE, "Temperature": {"start": 2013, "end": 2013, "surface": True}},
            as_missing=[4.5, 7.5],
        )
        velocity, temperature = no_report[-1]["variables"]
        results = results_of(tmp_path / "out")
        with xr.open_dataset(os.path.join(results, velocity["climatology"]["layers"]["surface"]["annual"])) as ds:
            values = ds["velocity"].values[0]
        assert np.isnan(values[5:8]).all() and np.isfinite(np.delete(values, [5, 6, 7])).all()
        # 2013 only: January and February
        lon, _, _ = fvcom_mesh()
        months, climatology, _ = expected_temperature(lon, MONTHS[4:])
        with xr.open_dataset(os.path.join(results, temperature["climatology"]["layers"]["surface"]["monthly"])) as ds:
            assert list(ds["month"].values) == months == [1, 2]
            assert np.allclose(ds["temperature"].values, climatology, atol=1e-4)

    def test_the_times_are_shifted(self, tmp_path):
        path = fvcom_file(str(tmp_path / "one.nc"), first_day=31, n_times=3)
        unshifted = fvcom_climatology.file_sums(path, ["temp"], [2012, 2012])
        shifted = fvcom_climatology.file_sums(path, ["temp"], [2012, 2012], time_shift=-1)
        assert set(unshifted) == {(2012, 2)} and set(shifted) == {(2012, 1), (2012, 2)}
        assert list(shifted[(2012, 1)][1][:1]) == [1] and list(shifted[(2012, 2)][1][:1]) == [2]

    def test_more_cores_give_the_same(self, fvcom_sim):
        files = sorted(os.path.join(root, name) for root, _, names in os.walk(fvcom_sim) for name in names)
        one = fvcom_climatology.monthly_sums(files, ["temp"], [2012, 2013], cores=1)
        two = fvcom_climatology.monthly_sums(files, ["temp"], [2012, 2013], cores=2)
        assert sorted(one) == sorted(two) == sorted(MONTHS)
        for month in one:
            assert np.allclose(one[month][0], two[month][0]) and np.array_equal(one[month][1], two[month][1])

    def test_parts_on_the_nodes_and_the_elements_cannot_be_summed(self, fvcom_sim, tmp_path, no_report):
        with pytest.raises(ValueError, match="not all on the same part of the FVCOM mesh"):
            analyze(fvcom_sim, tmp_path / "out", variables={"Both": "temp+u"})
        # before anything is worked out
        assert not os.path.exists(os.path.join(str(tmp_path / "out"), analysis.FOLDER))

    def test_the_streams_hold_the_mesh_s_variables_only(self, fvcom_sim):
        found = analysis.streams(fvcom_sim, 1)
        assert len(found) == 1 and found[0]["fvcom"] is True
        variables = found[0]["variables"]
        assert set(variables) == {"temp", "u", "zeta"}
        assert variables["temp"] == {"long_name": "temperature", "units": "degrees_C", "levels": 5}
        assert variables["zeta"]["levels"] == 1


def simulation_files(sim):
    return sorted(os.path.join(root, name) for root, _, names in os.walk(sim) for name in names)


def depth_of_water():
    """h + zeta of fvcom_file on its nodes, and on its elements (the mean of their nodes')."""
    lon, _, nv = fvcom_mesh()
    depth = (20 + 200 * (lon + 5)).astype("float32").astype("float64") + 0.5
    return depth, depth[nv - 1].mean(axis=0)


def mean_of(found, month):
    sums, counts = found[month]
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


class TestBottomAndWaterColumn:
    """fvcom_file's temp is 10 + lon + the layer + the day, on five layers of
    the same thickness, h is 20 + 200 (lon + 5) and zeta 0.5."""

    def test_the_bottom_and_water_column_with_no_thickness(self, fvcom_sim, tmp_path, no_report):
        years = {"start": 2012, "end": 2013}
        analyze(
            fvcom_sim, tmp_path / "out", variables={"Temperature": "temp", "Heat": "temp", "Velocity": "u"},
            climatology={
                "Temperature": dict(years, bottom=True, water_column="mean"),
                "Heat": dict(years, water_column="integral"),
                "Velocity": dict(years, bottom=True, water_column="integral"),
            },
        )
        temperature, heat, velocity = no_report[-1]["variables"]
        results = results_of(tmp_path / "out")
        lon, _, _ = fvcom_mesh()
        _, climatology, _ = expected_temperature(lon)
        on_nodes, on_elements = depth_of_water()

        def values(entry, layer, key):
            saved = entry["climatology"]["layers"][layer]
            with xr.open_dataset(os.path.join(results, saved["monthly"])) as ds:
                return saved, ds[key].values, ds[key].attrs["long_name"]

        saved, bottom, long_name = values(temperature, "bottom", "temperature")
        assert saved["mesh"] == "node" and saved["title"] == "Bottom" and long_name == "Temperature, bottom"
        assert np.allclose(bottom, climatology + 4, atol=1e-4)
        saved, mean, _ = values(temperature, "mean", "temperature")
        assert saved["units"] == "degrees_C" and np.allclose(mean, climatology + 2, atol=1e-4)
        saved, integral, long_name = values(heat, "integral", "heat")
        assert saved["units"] == analysis.integral_units("degrees_C") == "degrees_C m"
        assert long_name == "Heat, water column: depth integral"
        assert np.allclose(integral, (climatology + 2) * on_nodes, rtol=1e-5)
        # u is each element's number in every layer
        saved, integral, _ = values(velocity, "integral", "velocity")
        assert saved["mesh"] == "nele" and saved["units"] == "meters s-1 m"
        assert np.allclose(integral, np.arange(len(on_elements)) * on_elements, rtol=1e-5)
        assert np.allclose(values(velocity, "bottom", "velocity")[1], np.arange(len(on_elements)))

    def test_each_layer_is_weighted_by_its_thickness_at_each_time(self, tmp_path):
        import netCDF4

        path = fvcom_file(str(tmp_path / "uneven.nc"), n_times=3)
        levels = np.array([0.0, -0.1, -0.3, -0.6, -0.8, -1.0])
        zeta = np.array([0.0, 1.0, 5.0])
        with netCDF4.Dataset(path, "a") as ds:
            nodes = ds.dimensions["node"].size
            ds.variables["siglev"][:] = np.repeat(levels[:, None], nodes, axis=1)
            ds.variables["zeta"][:] = np.repeat(zeta[:, None], nodes, axis=1)
            h = ds.variables["h"][:].astype("float64")
            temp = ds.variables["temp"][:].astype("float64")
        shares = -np.diff(levels)
        thickness = shares[None, :, None] * (h[None, None, :] + zeta[:, None, None])
        found = {
            layer: mean_of(fvcom_climatology.file_sums(path, ["temp"], [2012, 2012], layer=layer), (2012, 1))
            for layer in ("mean", "integral")
        }
        # the mean of each time's column: the thickness changes with zeta, but not the shares
        assert np.allclose(found["mean"], (temp * shares[None, :, None]).sum(axis=1).mean(axis=0), rtol=1e-6)
        assert np.allclose(found["integral"], (temp * thickness).sum(axis=1).mean(axis=0), rtol=1e-6)
        # not the even split
        assert not np.allclose(found["mean"], temp.mean(axis=(0, 1)))

    def test_what_is_missing_is_left_out(self, tmp_path):
        import netCDF4

        path = fvcom_file(str(tmp_path / "gaps.nc"), n_times=2)
        with netCDF4.Dataset(path, "a") as ds:
            temp = ds.variables["temp"][:]
            temp[:, 0, 0] = np.ma.masked  # the top layer of node 0
            temp[:, :, 1] = np.ma.masked  # every layer of node 1
            temp[:, 4, 2] = np.ma.masked  # the bottom layer of node 2
            ds.variables["temp"][:] = temp
            h = ds.variables["h"][:].astype("float64")
            full = ds.variables["temp"][:].data.astype("float64")
        lon, _, _ = fvcom_mesh()

        def layer(name):
            return mean_of(fvcom_climatology.file_sums(path, ["temp"], [2012, 2012], layer=name), (2012, 1))

        top = 10 + lon + 0.5  # the mean of days 0 and 1
        mean, integral, bottom = layer("mean"), layer("integral"), layer("bottom")
        # node 0: the four layers below the top, each a fifth of the depth
        assert np.isclose(mean[0], top[0] + 2.5) and np.isclose(integral[0], (top[0] + 2.5) * 0.8 * (h[0] + 0.5))
        assert np.isnan(mean[1]) and np.isnan(integral[1]) and np.isnan(bottom[1])
        # node 2: the deepest layer with a value
        assert np.isclose(bottom[2], top[2] + 3) and np.isclose(bottom[3], top[3] + 4)
        assert np.isclose(mean[3], top[3] + 2) and np.isfinite(full).all()

    def test_more_cores_give_the_same(self, fvcom_sim):
        files = simulation_files(fvcom_sim)
        for layer in ("bottom", "integral"):
            one = fvcom_climatology.monthly_sums(files, ["temp"], [2012, 2013], cores=1, layer=layer)
            two = fvcom_climatology.monthly_sums(files, ["temp"], [2012, 2013], cores=2, layer=layer)
            assert sorted(one) == sorted(two) == sorted(MONTHS)
            for month in one:
                assert np.allclose(one[month][0], two[month][0]) and np.array_equal(one[month][1], two[month][1])

    def test_a_sum_with_a_surface_variable(self, fvcom_sim, tmp_path, no_report):
        years = {"start": 2012, "end": 2013}
        # the bottom of each part, added, as CDO takes it
        analyze(fvcom_sim, tmp_path / "out", variables={"Plus": "temp+zeta"},
                climatology={"Plus": dict(years, bottom=True)})
        saved = no_report[-1]["variables"][0]["climatology"]["layers"]["bottom"]
        lon, _, _ = fvcom_mesh()
        with xr.open_dataset(os.path.join(results_of(tmp_path / "out"), saved["monthly"])) as ds:
            assert np.allclose(ds["plus"].values, expected_temperature(lon)[1] + 4.5, atol=1e-4)
        # but zeta has no water column
        with pytest.raises(ValueError, match="the water column needs every part of it on the sigma layers"):
            analyze(fvcom_sim, tmp_path / "again", variables={"Plus": "temp+zeta"},
                    climatology={"Plus": dict(years, water_column="mean")})

    def test_what_has_no_water_column_is_refused(self, fvcom_sim, tmp_path, no_report):
        import netCDF4

        for path in simulation_files(fvcom_sim):
            with netCDF4.Dataset(path, "a") as ds:
                km = ds.createVariable("km", "f4", ("time", "siglev", "node"))
                km[:] = 1.0
                km.units = "m2 s-1"
        years = {"start": 2012, "end": 2013}
        with pytest.raises(ValueError, match=r"km is not \(on the sigma levels, siglev"):
            analyze(fvcom_sim, tmp_path / "out", variables={"Viscosity": "km"},
                    climatology={"Viscosity": dict(years, water_column="integral")})
        with pytest.raises(ValueError, match="has only one vertical level"):
            analyze(fvcom_sim, tmp_path / "out", variables={"Elevation": "zeta"},
                    climatology={"Elevation": dict(years, bottom=True)})
        assert not os.path.exists(os.path.join(str(tmp_path / "out"), analysis.FOLDER))
        # the bottom of what is on the sigma levels is the seabed's
        analyze(fvcom_sim, tmp_path / "out", variables={"Viscosity": "km"},
                climatology={"Viscosity": dict(years, bottom=True)})
        assert no_report[-1]["variables"][0]["climatology"]["layers"]["bottom"]["mesh"] == "node"


class TestEverythingElseIsRefused:
    def test_analyze_says_what_is_not_yet_available(self, fvcom_sim, tmp_path, no_report):
        years = {"start": 2012, "end": 2012}
        with pytest.raises(ValueError) as raised:
            analyze(
                fvcom_sim, tmp_path / "out",
                climatology={"Temperature": dict(years, surface=True, bottom=True, water_column="mean")},
                phenology={"Temperature": years},
                long_term={"Temperature": dict(years, surface=True)},
                seasonal={"Temperature": dict(years, surface=True)},
                map_changes={"Temperature": {"historical": years, "future": [{"start": 2013, "end": 2013}],
                                             "plot": "absolute", "change": "absolute", "surface": True}},
                transects={"Temperature": [{"start": [-5.0, 50.5], "end": [-4.0, 50.5], "climatology": years}]},
            )
        text = str(raised.value)
        for what in ("phenology", "long-term time series", "seasonal cycles", "map temporal changes", "transects"):
            assert what in text
        # the climatology's bottom and water column can be worked out
        assert "bottom is" not in text and "vertical mean" not in text
        assert "not yet available for FVCOM output" in text and "only work out a climatology of FVCOM output" in text
        assert not os.path.exists(os.path.join(str(tmp_path / "out"), analysis.FOLDER))

    def test_compare_simulations_still_refuses_fvcom(self, fvcom_sim, tmp_path, no_report):
        with pytest.raises(ValueError, match="compare_simulations does not work with FVCOM output yet"):
            oceanval.compare_simulations(
                simulations={"One": {"sim_dir": fvcom_sim, "n_dirs_down": 1}, "Two": {"sim_dir": fvcom_sim, "n_dirs_down": 1}},
                variables={"Temperature": {"One": "temp", "Two": "temp"}},
                climatology={"Temperature": SURFACE}, cores=1, view=False, out_dir=str(tmp_path / "out"),
            )


class TestMaps:
    @pytest.fixture
    def results(self, fvcom_sim, tmp_path, no_report):
        analyze(fvcom_sim, tmp_path / "out", variables={"Temperature": "temp", "Velocity": "u"})
        return results_of(tmp_path / "out")

    def test_the_maps_are_drawn_on_the_mesh(self, results):
        from oceanval import analysis_plots as ap

        for function in (ap.annual_map, ap.monthly_maps):
            for key in ("temperature", "velocity"):
                drawn = function(results, key, "surface")
                assert not drawn.regridded and "to_latlon" not in drawn.code and "pub_plot" not in drawn.code
                assert "from oceanval.mesh_maps import" in drawn.code
                assert any(ax.collections for ax in plt.gcf().axes)
                plt.close("all")
                ast.parse(drawn.code)

    def test_a_triangle_with_a_value_missing_is_not_drawn(self, tmp_path):
        path = fvcom_file(str(tmp_path / "one.nc"), n_times=1)
        whole = mesh_maps.mesh_plot(path, "temp")
        drawn = [c for c in whole.collections if hasattr(c, "get_array") and c.get_array() is not None]
        plt.close("all")
        import netCDF4

        with netCDF4.Dataset(path, "a") as ds:
            values = ds.variables["temp"][:]
            values[0, 0, 0] = np.ma.masked
            ds.variables["temp"][:] = values
        ax = mesh_maps.mesh_plot(path, "temp")
        mask = next(c for c in ax.collections if hasattr(c, "_triangulation"))._triangulation.mask
        plt.close("all")
        _, _, nv = fvcom_mesh()
        assert drawn and mask is not None and mask.sum() == ((nv - 1) == 0).any(axis=0).sum()

    def test_a_mesh_in_metres_is_drawn_on_plain_axes(self, tmp_path):
        path = fvcom_file(str(tmp_path / "flat.nc"), n_times=1, cartesian=True)
        ax = mesh_maps.mesh_plot(path, "u", title="Flat")
        assert ax.get_xlabel() == "x (m)" and not hasattr(ax, "projection")
        plt.close("all")
        fig = mesh_maps.mesh_panel_plot(path, "temp", ["One"], ncol=2)
        assert len([ax for ax in fig.axes if ax.get_title()]) == 1
        plt.close("all")


class TestMissingProbe:
    def test_the_surface_of_fvcom_output_is_read(self, fvcom_sim):
        files = sorted(os.path.join(root, name) for root, _, names in os.walk(fvcom_sim) for name in names)
        slices = missing_probe._slices(files[:2], "temp", 2)
        lon, _, _ = fvcom_mesh()
        assert len(slices) == 2 and slices[0].shape == lon.shape
        assert np.allclose(slices[0], 10 + lon, atol=1e-4)


# ---- the window ----


@pytest.fixture
def app(tmp_path):
    app = App(cwd=str(tmp_path))
    app.start()
    yield app
    app.close()


def to_options(app):
    assert post(app, "/api/choose", {"action": "analyze"})[0] == 200
    form = dict(app.analyze_setup_form, simdir="fvcom_sim", ndown="1")
    assert post(app, "/api/setup", {"form": form}) == (200, {"ok": True})
    wait_for(lambda: app.analyze_found["status"] != "reading")
    assert app.analyze_found["status"] == "done", app.analyze_found["error"]
    rows = [{"name": "Temperature", "model_variable": "temp"}]
    assert post(app, "/api/analyze_variables", {"rows": rows}) == (200, {"ok": True})
    wait_for(lambda: app.analyze_check is not None and app.analyze_check.status != "running", timeout=60)
    assert post(app, "/api/analyze_files") == (200, {"ok": True})
    return get_json(app, "/api/state")[1]["analyze"]


class TestWindow:
    def test_only_the_climatology_can_be_chosen(self, app, fvcom_sim, monkeypatch):
        started = []
        monkeypatch.setattr(app, "start_run", lambda args, label: started.append(args))
        state = to_options(app)
        assert state["fvcom"] is True and "only map its climatology, at the surface, the bottom" in state["fvcom_note"]
        # the dates are checked as matchup(fvcom=True) reads them
        assert app.analyze_check.request["fvcom"] is True and app.analyze_check.status == "done"
        years = {"use": True, "start": "2012", "end": "2013"}
        status, reply = post(app, "/api/analyze_option", {"kind": "phenology", "form": {"rows": {"Temperature": years}}})
        assert status == 400 and "Phenology is not yet available for FVCOM output." == reply["errors"]["rows"]
        # the bottom and the water column, with no thickness asked for: it is worked out from the mesh
        column = {"rows": {"Temperature": dict(years, surface=True, bottom=True, water_column=True, how="mean")},
                  "thickness": ""}
        assert post(app, "/api/analyze_option", {"kind": "climatology", "form": column}) == (200, {"ok": True})
        for kind, form in (("cores", {"cores": "1"}), ("missing", {"from": "", "to": ""}), ("figure_code", {"choice": "no"})):
            assert post(app, "/api/analyze_option", {"kind": kind, "form": form})[0] == 200
        assert post(app, "/api/analyze_run") == (200, {"ok": True})
        script = os.path.join(app.cwd, "analysis.py")
        call = ast.parse(open(script).read()).body[-1].value
        given = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
        assert given["climatology"] == {"Temperature": {"start": 2012, "end": 2013, "surface": True, "bottom": True,
                                                        "water_column": "mean"}}
        assert given["phenology"] is None and given["thickness"] is None and started


def test_the_other_analyses_are_greyed_out_in_a_browser(browser, tmp_path):
    fvcom_months(str(tmp_path / "fvcom_sim"), MONTHS[:2])
    app = App(cwd=str(tmp_path))
    url = app.start()
    errors = []
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        to_options(app)
        page.reload()
        page.wait_for_selector("#view-analyze-options:not([hidden])")
        assert page.is_visible("#ao-fvcom-note") and "FVCOM" in page.inner_text("#ao-fvcom-note")
        assert page.is_enabled("#ao-climatology")
        for kind in ("phenology", "long_term", "seasonal", "map_changes", "transects"):
            assert page.is_disabled(f"#ao-{kind}")
            assert page.inner_text(f"#ao-{kind}-text") == "Not yet available for FVCOM output"
        page.click("#ao-climatology")
        page.wait_for_selector("#clim-dialog:not([hidden])")
        cells = page.locator("#clim-body tr").first.locator("td")
        # the bottom and the water column can be chosen, and the thickness is not asked for
        bottom, water = (cells.nth(n).locator("input[type=checkbox]").first for n in (4, 5))
        assert bottom.is_enabled() and water.is_enabled()
        assert "Not yet available" not in cells.nth(4).inner_text() + cells.nth(5).inner_text()
        water.check()
        assert page.is_hidden("#clim-thickness-row") and page.is_visible("#clim-fvcom-thickness")
        assert "siglev" in page.inner_text("#clim-fvcom-thickness")
        assert errors == []
    finally:
        app.close()
