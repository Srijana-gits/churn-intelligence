"""Generate a portable Tableau workbook from the reproducible dashboard CSVs.

Uses Tableau's XML workbook format and Hyper extracts. The .twbx embeds the .twb,
extracts and CSVs; the adjacent .twb reads extracts from dashboard/data.
"""

from copy import deepcopy
import json
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
from tableauhyperapi import (
    Connection, CreateMode, HyperProcess, Inserter, SqlType, TableDefinition, TableName, Telemetry,
)

from features import PROJECT_ROOT

DASHBOARD = PROJECT_ROOT / "dashboard"
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#162b40"
BUCKET_COLORS = {"Contact first": ORANGE, "Low-cost outreach": "#f2a07e",
                 "Monitor": BLUE, "Leave alone": "#b5bec8"}
CAPTIONS = {"customer_count": "Customers", "churn_actual": "Observed churn rate",
            "MonthlyCharges": "Monthly bill ($)", "churn_probability": "Churn score",
            "monthly_bill_at_risk": "Score × monthly bill (ranking measure)",
            "priority_rank": "Rank", "customerID": "Customer ID", "bucket": "Priority bucket",
            "contact_share": "Share of customers contacted", "recall": "Share of churners found",
            "precision": "Churn rate in contacted group", "mean_probability": "Mean predicted score",
            "observed_churn_rate": "Observed churn rate", "InternetService": "Internet service",
            "PaymentMethod": "Payment method"}
SOURCES = [("customers_scored", "Full population • 7,043 customers"),
           ("priority_list", "Eligible current customers • retrospective OOF scores"),
           ("capacity_metrics", "Held-out evaluation • contact capacity"),
           ("calibration", "Held-out evaluation • score calibration")]


def make_extracts():
    """Build Public-compatible extracts with telemetry disabled."""
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
                      parameters={"default_database_version": "2", "log_file_max_count": "0"}) as hyper:
        for name, _ in SOURCES:
            frame = pd.read_csv(DASHBOARD / "data" / f"{name}.csv")
            types = {"i": SqlType.big_int(), "u": SqlType.big_int(), "f": SqlType.double(),
                     "b": SqlType.bool()}
            table = TableDefinition(TableName("Extract", "Extract"), [
                TableDefinition.Column(col, types.get(frame[col].dtype.kind, SqlType.text()))
                for col in frame.columns])
            with Connection(hyper.endpoint, DASHBOARD / "data" / f"{name}.hyper",
                            CreateMode.CREATE_AND_REPLACE) as connection:
                connection.catalog.create_schema("Extract")
                connection.catalog.create_table(table)
                with Inserter(connection, table) as inserter:
                    inserter.add_rows([[None if pd.isna(v) else v for v in row]
                                       for row in frame.itertuples(index=False, name=None)])
                    inserter.execute()
                count = connection.execute_scalar_query('SELECT COUNT(*) FROM "Extract"."Extract"')
                if count != len(frame):
                    raise ValueError(f"Extract row count mismatch for {name}")


def element(parent, tag, text=None, **attrs):
    node = ET.SubElement(parent, tag, {k.replace('_', '-'): str(v) for k, v in attrs.items()})
    node.text = text
    return node


def field(name, derivation="None", kind=None):
    if kind is None:
        kind = "nk" if derivation == "None" else "qk"
    return f"[{derivation.lower()}:{name}:{kind}]"


def reference(source, name, derivation="None", kind=None):
    return f"[{source}].{field(name, derivation, kind)}"


class Workbook:
    def __init__(self):
        self.root = ET.Element("workbook", {"locale": "en_US", "version": "18.1",
                                           "source-platform": "mac", "source-build": "0.0.0 (0000.0.0.0)"})
        element(self.root, "document-format-change-manifest")
        element(self.root, "preferences")
        self.sources = element(self.root, "datasources")
        self.columns = {}
        self.captions = {}
        self.worksheets = element(self.root, "worksheets")
        self.dashboards = element(self.root, "dashboards")
        self.windows = element(self.root, "windows")

    def datasource(self, name, caption):
        frame = pd.read_csv(DASHBOARD / "data" / f"{name}.csv")
        self.captions[name] = caption
        source = element(self.sources, "datasource", caption=caption, inline="true", name=name, version="18.1")
        connection = element(source, "connection", **{"class": "federated"})
        named = element(element(connection, "named-connections"), "named-connection", name=f"hyper.{name}", caption=caption)
        element(named, "connection", **{"class": "hyper", "dbname": f"data/{name}.hyper",
                                        "authentication": "auth-none", "author-locale": "en_US",
                                        "default-settings": "yes", "username": "tableau_internal_user", "port": "", "sslmode": ""})
        element(connection, "relation", connection=f"hyper.{name}", name="Extract",
                table="[Extract].[Extract]", type="table")
        metadata = element(connection, "metadata-records")
        self.columns[name] = {}
        for ordinal, col in enumerate(frame.columns):
            dtype = frame[col].dtype.kind
            datatype = "integer" if dtype in "iu" else "real" if dtype == "f" else "boolean" if dtype == "b" else "string"
            dimension = datatype in ["string", "boolean"] or col in ["SeniorCitizen", "priority_rank"]
            record = element(metadata, "metadata-record", **{"class": "column"})
            for tag, value in {"remote-name": col, "remote-type": {"integer": 20, "real": 5, "boolean": 11, "string": 129}[datatype],
                               "local-name": f"[{col}]", "parent-name": "[Extract]", "remote-alias": col,
                               "ordinal": ordinal, "local-type": datatype,
                               "aggregation": "Count" if dimension else "Sum", "contains-null": "true"}.items():
                element(record, tag, str(value))
            attrs = {"datatype": datatype, "name": f"[{col}]", "role": "dimension" if dimension else "measure",
                     "type": "nominal" if dimension else "quantitative", "caption": CAPTIONS.get(col, col.replace('_', ' ').capitalize())}
            if col in ["churn_actual", "churn_probability", "contact_share", "recall", "precision", "mean_probability", "observed_churn_rate"]:
                attrs["default-format"] = "p0.0%"
            elif datatype in ["integer", "real"]:
                attrs["default-format"] = 'n#,##0.00' if datatype == "real" else 'n#,##0'
            node = element(source, "column", **attrs)
            self.columns[name][col] = node
        if name == "priority_list":
            style = element(element(source, "style"), "style-rule", element="mark")
            encoding = element(style, "encoding", attr="color", field=field("bucket"), type="palette")
            for bucket, color in BUCKET_COLORS.items():
                element(element(encoding, "map", to=color), "bucket", f'"{bucket}"')

    def sheet(self, name, source, rows=(), cols=(), mark="Bar", text=None, color=None, lod=None, filters=()):
        """Field specs are (column, aggregation[, kind]); filters initially include all."""
        sheet = element(self.worksheets, "worksheet", name=name)
        title = element(element(element(sheet, "layout-options"), "title"), "formatted-text")
        element(title, "run", name, fontsize="12", fontcolor=INK, bold="true")
        table = element(sheet, "table")
        view = element(table, "view")
        element(element(view, "datasources"), "datasource", name=source, caption=self.captions[source])
        deps = element(view, "datasource-dependencies", datasource=source)
        specs = list(rows) + list(cols) + [s for s in [text, color, lod] if s]
        specs += [(col, "None") for col in filters]
        for col in dict.fromkeys(s[0] for s in specs):
            deps.append(deepcopy(self.columns[source][col]))
        for spec in dict.fromkeys(specs):
            col, deriv = spec[:2]
            kind = spec[2] if len(spec) > 2 else None
            instance = field(col, deriv, kind)
            element(deps, "column-instance", column=f"[{col}]", derivation=deriv, name=instance,
                    pivot="key", type={"nk": "nominal", "ok": "ordinal", "qk": "quantitative"}[instance[-3:-1]])
        for col in filters:
            filt = element(view, "filter", **{"class": "categorical", "column": reference(source, col)})
            element(filt, "groupfilter", function="level-members", level=field(col))
        if filters:
            slices = element(view, "slices")
            for col in filters:
                element(slices, "column", reference(source, col))
        element(view, "aggregation", value="true")
        style = element(table, "style")
        rule = element(style, "style-rule", element="worksheet")
        element(rule, "format", attr="font-family", value="Tableau Book")
        element(rule, "format", attr="font-size", value="10")
        pane = element(element(table, "panes"), "pane", selection_relaxation_option="selection-relaxation-allow")
        element(element(pane, "view"), "breakdown", value="auto")
        element(pane, "mark", **{"class": mark})
        encodings = element(pane, "encodings")
        for tag, spec in [("text", text), ("color", color), ("lod", lod)]:
            if spec:
                element(encodings, tag, column=reference(source, *spec))
        if mark == "Text":
            rule = element(element(pane, "style"), "style-rule", element="mark")
            element(rule, "format", attr="color", value=INK)
            element(rule, "format", attr="font-size", value="24" if not rows else "10")
        elif color is None:
            rule = element(element(pane, "style"), "style-rule", element="mark")
            element(rule, "format", attr="color", value=BLUE)
        for axis, specs in [("rows", rows), ("cols", cols)]:
            value = " / ".join(reference(source, *s) for s in specs)
            element(table, axis, f"({value})" if len(specs) > 1 else value)
        window = element(self.windows, "window", **{"class": "worksheet", "name": name})
        element(window, "cards")
        element(element(window, "viewpoint"), "zoom", type="fit-width" if name == "Ranked contact table" else "entire-view")
        return name

    def dashboard(self, name, subtitle, tiles, filters=(), note=""):
        dash = element(self.dashboards, "dashboard", name=name)
        element(element(element(dash, "layout-options"), "title"), "formatted-text")
        element(dash, "size", maxheight="900", maxwidth="1300", minheight="900", minwidth="1300")
        zones = element(dash, "zones")
        base = element(zones, "zone", id="0", type_v2="layout-basic", x="0", y="0", w="100000", h="100000")
        counter = 0

        def zone(x, y, w, h, **attrs):
            nonlocal counter
            counter += 1
            return element(base, "zone", id=counter, x=x, y=y, w=w, h=h, **attrs)

        def text_zone(text, y, h, size, color):
            node = zone(1800, y, 96400, h, type_v2="text")
            element(element(node, "formatted-text"), "run", text, fontsize=size, fontcolor=color)

        text_zone("CHURN INTELLIGENCE  /  " + name.upper(), 1800, 4600, 22, INK)
        text_zone(subtitle, 6800, 5000, 10, "#566579")
        for sheet, x, y, w, h in tiles:
            node = zone(x, y, w, h, name=sheet)
            style = element(node, "zone-style")
            element(style, "format", attr="margin", value="10")
            element(style, "format", attr="background-color", value="#ffffff")
        for sheet, source, col, x, y, w in filters:
            zone(x, y, w, 7000, type_v2="filter", name=sheet,
                 param=reference(source, col), mode="checkdropdown")
        text_zone(note, 94500, 5000, 10, "#566579")
        window = element(self.windows, "window", **{"class": "dashboard", "name": name})
        viewpoints = element(window, "viewpoints")
        for sheet, *_ in tiles:
            element(element(viewpoints, "viewpoint", name=sheet), "zoom",
                    type="fit-width" if sheet == "Ranked contact table" else "entire-view")
        element(window, "active", id="-1")
        element(element(window, "device-preview"), "device", is_portrait="true", name="Generic Phone", type="Phone")


def build():
    meta = json.loads((DASHBOARD / "data" / "metadata.json").read_text())
    evaluation = json.loads((PROJECT_ROOT / "reports" / "evaluation.json").read_text())
    make_extracts()
    wb = Workbook()
    for name, caption in SOURCES:
        wb.datasource(name, caption)
    wb.sheet("All customers", "customers_scored", mark="Text", text=("customer_count", "Sum"))
    wb.sheet("Observed churn", "customers_scored", mark="Text", text=("churn_actual", "Avg"))
    wb.sheet("Monthly billing", "customers_scored", mark="Text", text=("MonthlyCharges", "Sum"))
    for dimension, name in [("Contract", "Churn by contract"), ("InternetService", "Churn by internet service"),
                            ("PaymentMethod", "Churn by payment method")]:
        wb.sheet(name, "customers_scored", rows=((dimension, "None"),), cols=(("churn_actual", "Avg"),),
                 text=("churn_actual", "Avg"))
    wb.sheet("Eligible current customers", "priority_list", mark="Text", text=("customer_count", "Sum"))
    wb.sheet("Customers by priority bucket", "priority_list", rows=(("bucket", "None"),),
             cols=(("customer_count", "Sum"),), color=("bucket", "None"), text=("customer_count", "Sum"))
    wb.sheet("Score × monthly bill by bucket", "priority_list", rows=(("bucket", "None"),),
             cols=(("monthly_bill_at_risk", "Sum"),), color=("bucket", "None"))
    wb.sheet("Risk and monthly bill", "priority_list", rows=(("MonthlyCharges", "Avg"),),
             cols=(("churn_probability", "Avg"),), mark="Circle", color=("bucket", "None"), lod=("customerID", "None"))
    wb.sheet("Ranked contact table", "priority_list", mark="Text",
             rows=(("priority_rank", "None", "ok"), ("customerID", "None"), ("bucket", "None"),
                   ("Contract", "None"), ("InternetService", "None"),
                   ("churn_probability", "Avg", "ok"), ("MonthlyCharges", "Avg", "ok")),
             text=("monthly_bill_at_risk", "Avg"), filters=("bucket", "Contract", "InternetService"))
    wb.sheet("Churners reached by contact capacity", "capacity_metrics", rows=(("recall", "Avg"),),
             cols=(("contact_share", "Avg"),), mark="Line", lod=("contact_share", "None", "ok"))
    wb.sheet("Precision at each contact capacity", "capacity_metrics", rows=(("precision", "Avg"),),
             cols=(("contact_share", "Avg"),), mark="Line", lod=("contact_share", "None", "ok"))
    wb.sheet("Calibration on unseen customers", "calibration", rows=(("observed_churn_rate", "Avg"),),
             cols=(("mean_probability", "Avg"),), mark="Line", lod=("mean_probability", "None", "ok"))
    wb.dashboard("Overview", "Full snapshot: 7,043 customers • 1,869 churned • descriptive views include the 11 zero-tenure customers.", [
        ("All customers", 1800, 13000, 30000, 13000), ("Observed churn", 34000, 13000, 30000, 13000),
        ("Monthly billing", 66200, 13000, 32000, 13000), ("Churn by contract", 1800, 29000, 46000, 30000),
        ("Churn by internet service", 50000, 29000, 48200, 30000),
        ("Churn by payment method", 1800, 62000, 96400, 30000)],
        note="Associations describe this snapshot; they do not show why customers left or whether an offer would retain them.")
    quality = evaluation["quality"]
    wb.dashboard("Model performance", f"Held-out customers: {evaluation['test_rows']:,} • ROC-AUC {quality['roc_auc']:.3f} • average precision {quality['average_precision']:.3f} • Brier score {quality['brier_score']:.3f}", [
        ("Churners reached by contact capacity", 1800, 15000, 46000, 35000),
        ("Precision at each contact capacity", 50000, 15000, 48200, 35000),
        ("Calibration on unseen customers", 1800, 53000, 96400, 39000)],
        note=f"Threshold {meta['risk_cutoff']:.8f} selected on training OOF F1. Similar CV/test performance; no future-month validation.")
    wb.dashboard("Customer priorities", f"Retrospective OOF scores • {meta['eligible_current_customers']:,} eligible current customers • 11 zero-tenure customers excluded • table filters apply only to the contact table.", [
        ("Customers by priority bucket", 1800, 13000, 46000, 22000),
        ("Score × monthly bill by bucket", 50000, 13000, 48200, 22000),
        ("Risk and monthly bill", 1800, 37000, 96400, 21000),
        ("Ranked contact table", 1800, 68000, 96400, 25000)],
        filters=[("Ranked contact table", "priority_list", col, x, 60000, 30000)
                 for col, x in [("bucket", 1800), ("Contract", 34000), ("InternetService", 66200)]],
        note=f"Cutoffs: score ≥ {meta['risk_cutoff']:.8f}; bill ≥ ${meta['value_cutoff']:.2f}. Score × bill is a ranking measure, not losses or savings.")
    # Tableau activates the first window; open on Overview rather than a KPI sheet.
    windows = list(wb.windows)
    for window in windows:
        wb.windows.remove(window)
    for window in sorted(windows, key=lambda w: w.get("class") != "dashboard"):
        wb.windows.append(window)
    ET.indent(wb.root, space="  ")
    path = DASHBOARD / "churn_intelligence.twb"
    ET.ElementTree(wb.root).write(path, encoding="utf-8", xml_declaration=True)
    # Stable archive timestamps avoid ZIP-only changes; Hyper bytes may vary.
    with zipfile.ZipFile(path.with_suffix(".twbx"), "w", zipfile.ZIP_DEFLATED) as package:
        for file in [path] + sorted((DASHBOARD / "data").glob("*")):
            info = zipfile.ZipInfo(file.relative_to(DASHBOARD).as_posix(), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            package.writestr(info, file.read_bytes())
    print("Built dashboard/churn_intelligence.twb and .twbx")


if __name__ == "__main__":
    build()
