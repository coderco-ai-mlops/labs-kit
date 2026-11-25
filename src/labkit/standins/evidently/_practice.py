"""The working parts of the practice version of Evidently: the data drift report of Evidently 0.5.1.

For every numerical column the report compares the current data with the reference data using a
statistical test, the same way the real one chooses it:

- more than 1000 reference rows: the Wasserstein distance, divided by the standard deviation of the
  reference column. A score of 0.1 or more counts as drift.
- 1000 reference rows or fewer: the Kolmogorov-Smirnov test. A p-value of 0.05 or less counts as drift.

The dataset as a whole has drifted when the share of drifted columns reaches `drift_share` (0.5).
"""

import html
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

GO_FURTHER = "If the spelling is right, it is in the real Evidently: see Go further in this lab."
VERSION = "0.5.1"


def not_included(what: str) -> str:
    return f"The practice version of Evidently on the labs site does not include {what}. {GO_FURTHER}"


def module_getattr(module: str):
    """A module-level __getattr__ that explains, in a plain sentence, that a name is not in the practice version."""
    def __getattr__(name: str):
        if name.startswith("__"):
            raise AttributeError(f"module '{module}' has no attribute '{name}'")
        raise ImportError(not_included(f"`{module}.{name}`"))
    return __getattr__


# ---- the statistical tests ---------------------------------------------------

def _wasserstein(reference: pd.Series, current: pd.Series, threshold: float):
    norm = max(float(np.std(reference)), 0.001)
    score = float(stats.wasserstein_distance(reference, current)) / norm
    return score, bool(score >= threshold)


def _ks(reference: pd.Series, current: pd.Series, threshold: float):
    p_value = float(stats.ks_2samp(reference, current)[1])
    return p_value, bool(p_value <= threshold)


_TESTS = {"wasserstein": ("Wasserstein distance (normed)", 0.1, _wasserstein), "ks": ("K-S p_value", 0.05, _ks)}


def _pick_test(name, reference: pd.Series):
    if name is None:
        name = "wasserstein" if reference.shape[0] > 1000 else "ks"
    if not isinstance(name, str) or name not in _TESTS:
        raise NotImplementedError(not_included(f"the statistical test {name!r}. It has \"wasserstein\" and \"ks\""))
    return _TESTS[name]


def _distribution(values: pd.Series) -> dict:
    y, x = np.histogram(values, bins=10, density=True)
    return {"small_distribution": {"x": [float(v) for v in x], "y": [float(v) for v in y]}}


def _frame(value, name: str) -> pd.DataFrame:
    if value is None:
        raise ValueError(f"{'Reference' if name == 'reference_data' else 'Current'} dataset should be present")
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"`{name}` should be a pandas DataFrame. It is a {type(value).__name__}.")
    return value


def _drift(reference: pd.DataFrame, current: pd.DataFrame, settings: dict) -> dict:
    columns = settings["columns"]
    if columns is None:
        columns = sorted(str(c) for c in reference.columns)
    for frame in (current, reference):
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"{missing} not in index")
    by_column = {}
    for column in columns:
        ref, cur = reference[column].dropna(), current[column].dropna()
        if not (pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur)) or pd.api.types.is_bool_dtype(ref):
            raise NotImplementedError(not_included(f"drift tests for columns that are not numbers. The column {column!r} holds {ref.dtype} values"))
        if pd.concat([ref, cur]).nunique() <= 5:
            raise NotImplementedError(not_included(f"drift tests for columns with 5 or fewer different values, such as {column!r}"))
        label, default_threshold, test = _pick_test(settings["stattest"], ref)
        threshold = default_threshold if settings["threshold"] is None else settings["threshold"]
        score, drifted = test(ref, cur, threshold)
        by_column[column] = {"column_name": column, "column_type": "num", "stattest_name": label, "stattest_threshold": threshold,
                             "drift_score": score, "drift_detected": drifted,
                             "current": _distribution(cur), "reference": _distribution(ref)}
    drifted_columns = sum(1 for item in by_column.values() if item["drift_detected"])
    share = drifted_columns / len(by_column)
    return {"number_of_columns": len(by_column), "number_of_drifted_columns": drifted_columns, "share_of_drifted_columns": share,
            "dataset_drift": bool(share >= settings["drift_share"]), "drift_by_columns": by_column,
            "current_fi": None, "reference_fi": None}


# ---- metrics and the preset --------------------------------------------------

_OTHER_ARGS = ("embeddings", "embeddings_drift_method", "cat_stattest", "text_stattest", "per_column_stattest", "cat_stattest_threshold",
               "text_stattest_threshold", "per_column_stattest_threshold")


class _DriftSettings:
    def __init__(self, columns=None, drift_share: float = 0.5, stattest=None, num_stattest=None, stattest_threshold=None,
                 num_stattest_threshold=None, **other):
        for name, value in other.items():
            if name not in _OTHER_ARGS:
                raise TypeError(f"{type(self).__name__}.__init__() got an unexpected keyword argument '{name}'")
            if value is not None:
                raise NotImplementedError(not_included(f"the `{name}` setting of {type(self).__name__}"))
        if columns is not None and (isinstance(columns, str) or not all(isinstance(c, str) for c in columns)):
            raise TypeError(f"`columns` should be a list of column names, for example [\"MedInc\", \"HouseAge\"]. It is {columns!r}.")
        self.settings = {"columns": None if columns is None else list(columns), "drift_share": drift_share,
                         "stattest": num_stattest if num_stattest is not None else stattest,
                         "threshold": num_stattest_threshold if num_stattest_threshold is not None else stattest_threshold}

    def __repr__(self):
        return f"{type(self).__name__}()"


class DatasetDriftMetric(_DriftSettings):
    """Has the dataset drifted as a whole? The share of drifted columns against `drift_share`."""

    def _result(self, drift: dict) -> dict:
        return {"drift_share": self.settings["drift_share"], "number_of_columns": drift["number_of_columns"],
                "number_of_drifted_columns": drift["number_of_drifted_columns"],
                "share_of_drifted_columns": drift["share_of_drifted_columns"], "dataset_drift": drift["dataset_drift"]}


class DataDriftTable(_DriftSettings):
    """The drift result for every column: the test used, its threshold, the score and whether it drifted."""

    def _result(self, drift: dict) -> dict:
        # As in the real one, the table always judges the dataset against a share of 0.5.
        return {**drift, "dataset_drift": bool(drift["share_of_drifted_columns"] >= 0.5)}


class DataDriftPreset(_DriftSettings):
    """The two metrics above together. `Report(metrics=[DataDriftPreset()])` gives both, in that order."""

    def _metrics(self) -> list:
        parts = []
        for kind in (DatasetDriftMetric, DataDriftTable):
            metric = kind.__new__(kind)
            metric.settings = self.settings
            parts.append(metric)
        return parts


# ---- the report --------------------------------------------------------------

_QUIET_REPORT_ARGS = ("options", "timestamp", "id", "metadata", "tags", "model_id", "reference_id", "batch_size", "dataset_id", "name")


class Report:
    def __init__(self, metrics, **other):
        for name in other:
            if name not in _QUIET_REPORT_ARGS:
                raise TypeError(f"Report.__init__() got an unexpected keyword argument '{name}'")
        if isinstance(metrics, (_DriftSettings, type)) or not isinstance(metrics, (list, tuple)):
            raise TypeError(f"`metrics` should be a list, for example Report(metrics=[DataDriftPreset()]). It is {metrics!r}.")
        self._metrics = []
        for item in metrics:
            if isinstance(item, type):
                raise TypeError(f"`metrics` holds the class {item.__name__} itself. Call it to create one: {item.__name__}().")
            if isinstance(item, DataDriftPreset):
                self._metrics += item._metrics()
            elif isinstance(item, (DatasetDriftMetric, DataDriftTable)):
                self._metrics.append(item)
            else:
                raise NotImplementedError(not_included(f"the metric {item!r}. It has DataDriftPreset, DatasetDriftMetric and DataDriftTable"))
        self._results = None
        self._timestamp = None

    def run(self, *, reference_data, current_data, column_mapping=None, **other):
        for name in other:
            if name not in ("engine", "additional_data", "timestamp"):
                raise TypeError(f"Report.run() got an unexpected keyword argument '{name}'")
        if column_mapping is not None:
            raise NotImplementedError(not_included("`column_mapping`. Pass frames that hold only the columns you want compared"))
        current, reference = _frame(current_data, "current_data"), _frame(reference_data, "reference_data")
        self._results = [(type(metric).__name__, metric._result(_drift(reference, current, metric.settings))) for metric in self._metrics]
        self._timestamp = datetime.now()

    def as_dict(self, **_ignored) -> dict:
        """The results as plain dictionaries and lists. Before `run` there are none, as in the real one."""
        return {"metrics": [{"metric": name, "result": json.loads(json.dumps(result))} for name, result in self._results or []]}

    def json(self, **_ignored) -> str:
        return json.dumps({"version": VERSION, **self.as_dict(), "timestamp": str(self._timestamp or datetime.now())})

    def save_json(self, filename, **_ignored) -> None:
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(self.json())

    def save_html(self, filename, **_ignored) -> None:
        """Write the report as a small web page: one table row per column. The real report also draws the distributions."""
        if self._results is None:
            raise ValueError("The report has no results yet. Call report.run(reference_data=..., current_data=...) before saving it.")
        parts = ["<!doctype html><html><head><meta charset=\"utf-8\"><title>Data drift report</title>",
                 "<style>body{font-family:sans-serif;margin:2em}td,th{padding:4px 12px;text-align:left;border-bottom:1px solid #ddd}"
                 ".drift{color:#b00020;font-weight:bold}</style></head><body><h1>Data drift report</h1>"]
        for name, result in self._results:
            parts.append(f"<h2>{html.escape(name)}</h2>")
            if "drift_by_columns" not in result:
                state = "detected" if result["dataset_drift"] else "not detected"
                parts.append(f"<p>Dataset drift is {state}. {result['number_of_drifted_columns']} of {result['number_of_columns']} columns drifted "
                             f"(share {result['share_of_drifted_columns']:.3f}). The threshold for the dataset is {result['drift_share']}.</p>")
                continue
            parts.append("<table><tr><th>Column</th><th>Type</th><th>Test</th><th>Threshold</th><th>Score</th><th>Drift</th></tr>")
            for item in result["drift_by_columns"].values():
                mark = "<td class=\"drift\">Detected</td>" if item["drift_detected"] else "<td>Not detected</td>"
                parts.append(f"<tr><td>{html.escape(str(item['column_name']))}</td><td>{item['column_type']}</td><td>{item['stattest_name']}</td>"
                             f"<td>{item['stattest_threshold']}</td><td>{item['drift_score']:.6f}</td>{mark}</tr>")
            parts.append("</table>")
        parts.append("</body></html>")
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write("".join(parts))

    def show(self, *_args, **_kwargs):
        raise NotImplementedError(not_included("`report.show()`, which draws the report in a notebook. Use `report.as_dict()`, or `report.save_html(...)` for a page"))

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(not_included(f"`Report.{name}`"))
