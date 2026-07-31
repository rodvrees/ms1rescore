"""Pixelated Dash GUI for MSI-PICASSO (palette derived from the logo).

A thin wrapper: it writes a TOML config, launches the ``picasso`` CLI as a
subprocess, tails its log, and renders the on-disk results (TSV tables +
interactive Plotly plots + an on-demand ion-image viewer). It never
re-implements pipeline logic.

Run:  ``picasso-gui``            (browser at http://127.0.0.1:8050)
      ``picasso-gui --desktop``  (native window, needs the ``desktop`` extra)
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, ALL, no_update

# --- Logo-derived palette (must match assets/picasso.css) ---
NAVY, PURPLE, BLUE, CYAN = "#001050", "#3d0f8f", "#0060c0", "#00a0b0"
GREEN, YELLOW, ORANGE, RED = "#4fb020", "#f0b000", "#f08000", "#d82814"
PAPER, INK = "#f5f7fb", "#001050"
PIXEL_FONT = "VT323, PressStart2P, monospace"
# Sequential ion ramp: white -> cyan -> blue -> navy (monotonic lightness, blue-family).
ION_COLORSCALE = [[0.0, "#ffffff"], [0.33, CYAN], [0.66, BLUE], [1.0, NAVY]]

_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
_PKG_DATA = os.path.join(os.path.dirname(__file__), "package_data")
_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
_CONSOLE_LOG = "gui_console.log"

# ponytail: single-user local app — one job at a time, module-global handle.
_JOB: dict = {"proc": None, "outdir": None, "cancelled": False}

# --------------------------------------------------------------------------
# Config schema → field metadata (data-driven form; no hand-listing of types)
# --------------------------------------------------------------------------
def _load_schema_props() -> dict:
    try:
        s = json.load(open(os.path.join(_PKG_DATA, "config_schema.json")))
        return s["properties"]["MSI-PICASSO"]["properties"]
    except Exception:
        return {}


_SCHEMA = _load_schema_props()

# File paths / core choices handled by their own widgets in "Essentials".
_ESSENTIAL_KEYS = ("maldi_d", "lcms_peptides", "output_dir", "model", "decoy_method")

# Common knobs shown as plain fields in "Settings".
_STANDARD_KEYS = (
    "maldi_query_raw", "single_round", "lcms_id_format", "psm_utils_reader",
    "im2deep_calibration", "ppm_tolerance", "matching_ppm", "features_preset",
    "use_protein_level_feats", "use_spatial_ranker_features", "region_coloc",
    "mob_coloc", "match_ccs", "verbose",
)

# Revealed by the "Advanced settings" disclosure.
_ADVANCED_KEYS = (
    "substitution_n_residues", "substitution_mass_shift_min_da",
    "rbf_svm_c", "rbf_svm_gamma", "svm_c", "n_interaction_features",
    "init_fdr", "train_fdr", "max_iter", "r2_seed_percentile", "min_seed_positives",
    "init_ppm_threshold", "winner_percentile", "lcms_prior_weight", "spatial_prior_weight",
    "ccs_window_multiplier", "region_coloc_k", "coloc_tic_quantile",
    "coloc_common_mode", "coloc_tic_normalize", "drop_zero_signal", "mob_window_multiplier",
)

_FORM_KEYS = set(_ESSENTIAL_KEYS) | set(_STANDARD_KEYS) | set(_ADVANCED_KEYS)
_N_EXTRA_ROWS = 8  # ponytail: fixed spare key/value slots; TOML upload covers bulk.


def _scalar_keys() -> list[str]:
    """Top-level keys that render as a single field (skip nested dicts / list types)."""
    out = []
    for k, meta in _SCHEMA.items():
        if k in ("config_file", "im2deep", "maldi_extraction"):
            continue
        t = meta.get("type", [])
        t = [t] if isinstance(t, str) else t
        if "object" in t or "array" in t:
            continue
        out.append(k)
    return sorted(out)


def _field_kind(key: str) -> tuple[str, list | None]:
    meta = _SCHEMA.get(key, {})
    enum = [e for e in (meta.get("enum") or []) if e is not None]
    t = meta.get("type", ["string"])
    t = [t] if isinstance(t, str) else t
    if enum:
        return "enum", enum
    if "boolean" in t:
        return "bool", None
    if "integer" in t or "number" in t:
        return "number", None
    return "string", None


def _render_field(key: str):
    kind, enum = _field_kind(key)
    label = html.Label(key.replace("_", " "))
    fid = {"role": "cfg", "key": key}
    if kind == "enum":
        ctrl = dcc.Dropdown(id=fid, options=[{"label": e, "value": e} for e in enum], clearable=True)
    elif kind == "bool":
        ctrl = dcc.Dropdown(id=fid, options=[{"label": "true", "value": True},
                                             {"label": "false", "value": False}], clearable=True)
    elif kind == "number":
        ctrl = dcc.Input(id=fid, type="number", debounce=True)
    else:
        ctrl = dcc.Input(id=fid, type="text", debounce=True)
    return html.Div([label, ctrl])


# --------------------------------------------------------------------------
# Plotly template
# --------------------------------------------------------------------------
def _picasso_template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family=PIXEL_FONT, size=16, color=INK),
        colorway=[BLUE, ORANGE, GREEN, PURPLE, RED, CYAN, YELLOW],
        margin=dict(l=60, r=24, t=48, b=52),
        xaxis=dict(linecolor=NAVY, linewidth=2, gridcolor="#dde3ef", zeroline=False, ticks="outside"),
        yaxis=dict(linecolor=NAVY, linewidth=2, gridcolor="#dde3ef", zeroline=False, ticks="outside"),
        legend=dict(bgcolor="#ffffff", bordercolor=NAVY, borderwidth=2),
    )
    return t


TEMPLATE = _picasso_template()


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def _read_tsv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return None


def _score_r2_col(df: pd.DataFrame) -> str | None:
    cols = [c for c in df.columns if c.endswith("_score_r2")]
    return cols[0] if cols else None


def _cfg_get(cfg: dict, key: str, default=None):
    """Read a config value tolerant of hyphen/underscore key spelling."""
    return cfg.get(key, cfg.get(key.replace("_", "-"), default))


def _resolved_maldi_d(outdir: Path) -> str | None:
    p = outdir / ".full_config.json"
    if p.exists():
        try:
            return _cfg_get(json.loads(p.read_text())["MSI-PICASSO"], "maldi_d")
        except Exception:
            pass
    p = outdir / "gui_config.toml"
    if p.exists():
        try:
            return _cfg_get(tomllib.loads(p.read_text())["MSI-PICASSO"], "maldi_d")
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------
# Figure builders (from ms1rescore_matches.tsv — no pipeline re-run)
# --------------------------------------------------------------------------
def build_ids_vs_fdr(matches: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, title="IDs vs FDR",
                      xaxis_title="reweighted q-value", yaxis_title="cumulative target IDs")
    if "is_tdc_winner" not in matches or "reweighted_q_value" not in matches:
        return fig
    w = matches[matches["is_tdc_winner"] & ~matches["is_decoy"]].copy()
    w = w.dropna(subset=["reweighted_q_value"]).sort_values("reweighted_q_value")
    if w.empty:
        return fig
    y = np.arange(1, len(w) + 1)
    fig.add_trace(go.Scatter(x=w["reweighted_q_value"], y=y, mode="lines",
                             line=dict(color=BLUE, width=3), name="target IDs",
                             hovertemplate="q=%{x:.4f}<br>IDs=%{y}<extra></extra>"))
    fig.add_vline(x=0.01, line=dict(color=ORANGE, width=2, dash="dot"),
                  annotation_text="1% FDR", annotation_font_color=ORANGE)
    fig.update_xaxes(range=[0, min(0.1, float(w["reweighted_q_value"].max()) + 1e-3)])
    return fig


def build_score_dist(matches: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    col = _score_r2_col(matches)
    fig.update_layout(template=TEMPLATE, title="Score distribution (round 2)",
                      xaxis_title=col or "score", yaxis_title="candidates", barmode="overlay")
    if col is None:
        return fig
    tgt = matches.loc[~matches["is_decoy"], col].dropna()
    dec = matches.loc[matches["is_decoy"], col].dropna()
    # blue=target, orange=decoy — CVD-safe pair from the logo; legend = 2nd encoding.
    fig.add_trace(go.Histogram(x=tgt, name="target", marker_color=BLUE, opacity=0.7, nbinsx=60))
    fig.add_trace(go.Histogram(x=dec, name="decoy", marker_color=ORANGE, opacity=0.7, nbinsx=60))
    return fig


def build_ion_image(img: np.ndarray, mz: float) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=img, colorscale=ION_COLORSCALE, colorbar=dict(outlinecolor=NAVY, outlinewidth=2)))
    fig.update_layout(template=TEMPLATE, title=f"Ion image  m/z {mz:.4f}",
                      yaxis=dict(scaleanchor="x", autorange="reversed"))
    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, visible=False)
    return fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, annotations=[dict(text=msg, showarrow=False,
                      font=dict(size=18, color=INK), xref="paper", yref="paper", x=0.5, y=0.5)],
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
def _preset_options():
    if not _CONFIGS_DIR.exists():
        return []
    return [{"label": p.name, "value": str(p)} for p in sorted(_CONFIGS_DIR.glob("*.toml"))]


def _labeled(label, comp):
    return html.Div([html.Label(label), comp])


def _path_field(field_id, label):
    return _labeled(label, html.Div([
        dcc.Input(id=f"in-{field_id}", type="text", value="", debounce=True,
                  style={"width": "78%", "display": "inline-block"}),
        html.Button("Browse", id=f"browse-{field_id}", n_clicks=0,
                    className="btn btn-small", style={"marginLeft": "8px"}),
    ]))


def _header():
    return html.Div([
        html.Div(className="picasso-header", children=[
            html.Img(src="/assets/MSI-PICASSO-logo-t.png", alt="MSI-PICASSO"),
        ]),
        html.Div(className="rainbow-rule"),
    ])


def _extra_rows():
    opts = [{"label": k, "value": k} for k in _scalar_keys()]
    rows = []
    for i in range(_N_EXTRA_ROWS):
        rows.append(html.Div(className="row", style={"marginBottom": "6px"}, children=[
            html.Div(className="col", children=dcc.Dropdown(
                id={"role": "extra-key", "idx": i}, options=opts, placeholder="parameter…", clearable=True)),
            html.Div(className="col", children=dcc.Input(
                id={"role": "extra-val", "idx": i}, type="text", placeholder="value", style={"width": "100%"})),
        ]))
    return rows


def _tab_configure():
    return html.Div(className="content", children=[
        html.Div(className="card", children=[
            html.H2("1 · Load a config"),
            html.Div(className="row", children=[
                html.Div(className="col", children=_labeled("Preset", dcc.Dropdown(
                    id="preset", options=_preset_options(), placeholder="Load a preset…", clearable=False))),
                html.Div(className="col", children=_labeled("Upload a .toml", dcc.Upload(
                    id="upload-toml", className="upload-box", children="⬆ Drop or click to upload a .toml"))),
            ]),
            html.P("A preset or uploaded file fills the fields below. You can then edit anything.",
                   className="hint"),
        ]),
        html.Div(className="card", children=[
            html.H2("2 · Essentials"),
            _path_field("maldi_d", "MALDI .d directory"),
            _path_field("lcms_peptides", "LC-MS/MS PSM table"),
            _path_field("output_dir", "Output directory"),
            html.Div(className="row", children=[
                html.Div(className="col", children=_labeled("Model", dcc.Dropdown(
                    id="in-model", clearable=False,
                    options=[{"label": m, "value": m} for m in ("lda", "qda", "svm", "rbf_svm", "gbt")]))),
                html.Div(className="col", children=_labeled("Decoy method", dcc.Dropdown(
                    id="in-decoy_method", clearable=False,
                    options=[{"label": m, "value": m} for m in
                             ("substitution", "shuffle", "mz_shift", "mz_shuffle",
                              "entrapment", "balanced_shuffle", "paired_shuffle")]))),
            ]),
        ]),
        html.Div(className="card", children=[
            html.H2("3 · Settings"),
            html.Div(className="field-grid", children=[_render_field(k) for k in _STANDARD_KEYS]),
            html.Details([
                html.Summary("Advanced settings"),
                html.Div(className="details-body field-grid",
                         children=[_render_field(k) for k in _ADVANCED_KEYS]),
            ]),
            html.Details([
                html.Summary("More parameters"),
                html.Div(className="details-body", children=[
                    html.P("Set any remaining parameter by name — no TOML needed. Values are parsed "
                           "as numbers / true / false where possible.", className="hint"),
                    *_extra_rows(),
                ]),
            ]),
        ]),
        html.Div(className="card", children=[
            html.H2("4 · Run"),
            html.Button("▶ RUN PICASSO", id="run-btn", n_clicks=0, className="btn btn-run"),
            html.Span(id="run-msg", className="hint", style={"marginLeft": "16px"}),
        ]),
    ])


def _tab_log():
    return html.Div(className="content", children=[
        html.Div(className="card", children=[
            html.H2("Live log"),
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px"}, children=[
                html.Div(id="log-status", className="log-status idle", children="idle"),
                html.Button("■ CANCEL RUN", id="cancel-btn", className="btn", disabled=True),
            ]),
            html.Pre(id="log-pre", className="log-pre", children="No run started yet."),
        ]),
        dcc.Interval(id="log-interval", interval=2000, n_intervals=0),
    ])


def _tab_results():
    return html.Div(className="content", children=[
        html.Div(className="card", children=[
            html.H2("Results"),
            _labeled("Results directory", html.Div([
                dcc.Input(id="results-dir", type="text", value="",
                          style={"width": "70%", "display": "inline-block"}),
                html.Button("Browse", id="browse-results-dir", n_clicks=0,
                            className="btn btn-small", style={"marginLeft": "8px"}),
                html.Button("Load", id="load-results", n_clicks=0,
                            className="btn btn-small", style={"marginLeft": "8px"}),
            ])),
            html.Div(id="results-summary"),
        ]),
        html.Div(id="results-body"),
    ])


def make_layout():
    return html.Div([
        dcc.Store(id="loaded-config"),
        dcc.Store(id="fb-target"), dcc.Store(id="fb-cwd"), dcc.Store(id="fb-entries"),
        _header(),
        dcc.Tabs(id="tabs", value="cfg", className="picasso-tabs", children=[
            dcc.Tab(label="CONFIGURE & RUN", value="cfg", className="tab", selected_className="tab--selected",
                    children=_tab_configure()),
            dcc.Tab(label="LIVE LOG", value="log", className="tab", selected_className="tab--selected",
                    children=_tab_log()),
            dcc.Tab(label="RESULTS", value="res", className="tab", selected_className="tab--selected",
                    children=_tab_results()),
        ]),
        html.Div(id="fb-modal-wrap"),
    ])


app = Dash(__name__, assets_folder=_ASSETS, title="MSI-PICASSO", suppress_callback_exceptions=True)
app.layout = make_layout()


# --------------------------------------------------------------------------
# Preset / upload → loaded-config store
# --------------------------------------------------------------------------
@app.callback(
    Output("loaded-config", "data", allow_duplicate=True),
    Input("preset", "value"),
    prevent_initial_call=True,
)
def preset_to_store(path):
    if not path:
        return no_update
    try:
        return tomllib.loads(Path(path).read_text()).get("MSI-PICASSO", {})
    except Exception:
        return no_update


@app.callback(
    Output("loaded-config", "data", allow_duplicate=True),
    Output("run-msg", "children", allow_duplicate=True),
    Input("upload-toml", "contents"),
    prevent_initial_call=True,
)
def upload_to_store(contents):
    if not contents:
        return no_update, no_update
    try:
        _, b64 = contents.split(",", 1)
        text = base64.b64decode(b64).decode("utf-8")
        cfg = tomllib.loads(text).get("MSI-PICASSO", {})
        return cfg, "Config uploaded — fields populated."
    except Exception as exc:
        return no_update, f"Could not read TOML: {exc}"


@app.callback(
    Output("in-maldi_d", "value"), Output("in-lcms_peptides", "value"),
    Output("in-output_dir", "value"), Output("in-model", "value"),
    Output("in-decoy_method", "value"),
    Output({"role": "cfg", "key": ALL}, "value"),
    Output({"role": "extra-key", "idx": ALL}, "value"),
    Output({"role": "extra-val", "idx": ALL}, "value"),
    Input("loaded-config", "data"),
    State({"role": "cfg", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def populate_fields(cfg, cfg_ids):
    cfg = cfg or {}
    ess = [_cfg_get(cfg, k) for k in _ESSENTIAL_KEYS]
    # standard + advanced fields, in the DOM order Dash gives us
    cfg_vals = [_cfg_get(cfg, cid["key"]) for cid in cfg_ids]
    # leftover scalar keys → the "More parameters" rows
    leftover = [k for k in _scalar_keys()
                if k not in _FORM_KEYS and _cfg_get(cfg, k) is not None]
    ek = [None] * _N_EXTRA_ROWS
    ev = [None] * _N_EXTRA_ROWS
    for i, k in enumerate(leftover[:_N_EXTRA_ROWS]):
        ek[i], ev[i] = k, str(_cfg_get(cfg, k))
    return (*ess, cfg_vals, ek, ev)


# --------------------------------------------------------------------------
# File browser
# --------------------------------------------------------------------------
def _list_dir(path: str):
    entries, p = [], Path(path)
    try:
        for e in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if e.name.startswith("."):
                continue
            kind = "d" if (e.is_dir() and e.name.endswith(".d")) else ("dir" if e.is_dir() else "file")
            entries.append({"name": e.name, "path": str(e), "kind": kind})
    except PermissionError:
        pass
    return entries


def _fb_view(cwd, entries):
    rows = [html.Div([html.Span("⬆"), html.Span(".. (parent)")], className="fb-entry is-dir",
                     id={"role": "fb-entry", "idx": -1})]
    for i, e in enumerate(entries):
        icon = {"dir": "📁", "d": "🗂", "file": "📄"}[e["kind"]]
        cls = {"dir": "fb-entry is-dir", "d": "fb-entry is-d", "file": "fb-entry"}[e["kind"]]
        rows.append(html.Div([html.Span(icon), html.Span(e["name"])], className=cls,
                             id={"role": "fb-entry", "idx": i}))
    return html.Div(className="fb-modal-backdrop", children=[
        html.Div(className="fb-modal", children=[
            html.Div("Select a file or folder", style={"fontFamily": PIXEL_FONT}),
            html.Div(cwd, className="fb-path"),
            html.Div(rows, className="fb-list"),
            html.Div(className="fb-actions", children=[
                html.Button("Choose this folder", id="fb-choose-dir", n_clicks=0, className="btn btn-small"),
                html.Button("Cancel", id="fb-cancel", n_clicks=0, className="btn btn-small"),
            ]),
        ]),
    ])


@app.callback(
    Output("fb-modal-wrap", "children", allow_duplicate=True),
    Output("fb-target", "data"), Output("fb-cwd", "data"), Output("fb-entries", "data"),
    Input("browse-maldi_d", "n_clicks"), Input("browse-lcms_peptides", "n_clicks"),
    Input("browse-output_dir", "n_clicks"), Input("browse-results-dir", "n_clicks"),
    State("in-maldi_d", "value"), State("in-output_dir", "value"),
    prevent_initial_call=True,
)
def open_browser(n1, n2, n3, n4, maldi_v, out_v):
    target = {"browse-maldi_d": "maldi_d", "browse-lcms_peptides": "lcms_peptides",
              "browse-output_dir": "output_dir", "browse-results-dir": "results-dir"}[ctx.triggered_id]
    start = maldi_v or out_v or str(Path.cwd())
    start = str(Path(start).parent if Path(start).is_file() else Path(start))
    if not Path(start).is_dir():
        start = str(Path.cwd())
    entries = _list_dir(start)
    return _fb_view(start, entries), target, start, entries


@app.callback(
    Output("fb-modal-wrap", "children", allow_duplicate=True),
    Output("fb-cwd", "data", allow_duplicate=True), Output("fb-entries", "data", allow_duplicate=True),
    Output("in-maldi_d", "value", allow_duplicate=True),
    Output("in-lcms_peptides", "value", allow_duplicate=True),
    Output("in-output_dir", "value", allow_duplicate=True),
    Output("results-dir", "value", allow_duplicate=True),
    Input({"role": "fb-entry", "idx": ALL}, "n_clicks"),
    Input("fb-choose-dir", "n_clicks"), Input("fb-cancel", "n_clicks"),
    State("fb-cwd", "data"), State("fb-entries", "data"), State("fb-target", "data"),
    prevent_initial_call=True,
)
def browser_action(entry_clicks, choose, cancel, cwd, entries, target):
    trig = ctx.triggered_id
    keep = (no_update,) * 4

    def selection(path):
        vals = {"maldi_d": 0, "lcms_peptides": 1, "output_dir": 2, "results-dir": 3}
        out = [no_update, no_update, no_update, no_update]
        out[vals[target]] = path
        return tuple(out)

    if trig == "fb-cancel":
        return (None, no_update, no_update, *keep)
    if trig == "fb-choose-dir":
        return (None, no_update, no_update, *selection(cwd))
    if not any(entry_clicks or []):
        return (no_update,) * 7
    idx = trig["idx"]
    if idx == -1:
        new = str(Path(cwd).parent)
    else:
        e = entries[idx]
        if e["kind"] in ("d", "file"):
            return (None, no_update, no_update, *selection(e["path"]))
        new = e["path"]
    ent = _list_dir(new)
    return _fb_view(new, ent), new, ent, *keep


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def _coerce(val: str):
    """Best-effort parse of a free-text extra value into bool/number/str."""
    s = val.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


@app.callback(
    Output("run-msg", "children"), Output("run-btn", "disabled", allow_duplicate=True),
    Output("results-dir", "value", allow_duplicate=True),
    Input("run-btn", "n_clicks"),
    State("loaded-config", "data"),
    State("in-maldi_d", "value"), State("in-lcms_peptides", "value"),
    State("in-output_dir", "value"), State("in-model", "value"), State("in-decoy_method", "value"),
    State({"role": "cfg", "key": ALL}, "value"), State({"role": "cfg", "key": ALL}, "id"),
    State({"role": "extra-key", "idx": ALL}, "value"), State({"role": "extra-val", "idx": ALL}, "value"),
    prevent_initial_call=True,
)
def run_picasso(n, base_cfg, maldi_d, lcms, outdir, model, decoy, cfg_vals, cfg_ids, ekeys, evals):
    if _JOB["proc"] is not None and _JOB["proc"].poll() is None:
        return "A run is already in progress.", True, no_update
    if not outdir:
        return "Set an output directory first.", False, no_update

    # Base config = whatever preset/upload provided (long-tail keys + nested tables).
    section = dict(base_cfg or {})

    def put(key, val):
        if val is None or val == "":
            return
        section.pop(key.replace("-", "_"), None)
        section.pop(key.replace("_", "-"), None)
        section[key.replace("_", "-")] = val  # write hyphen form (matches presets)

    for k, v in zip(_ESSENTIAL_KEYS, (maldi_d, lcms, outdir, model, decoy)):
        put(k, v)
    for cid, v in zip(cfg_ids, cfg_vals):
        put(cid["key"], v)
    for k, v in zip(ekeys or [], evals or []):
        if k and v not in (None, ""):
            put(k, _coerce(v))

    import tomli_w
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "gui_config.toml"
    try:
        cfg_path.write_text(tomli_w.dumps({"MSI-PICASSO": section}))
    except Exception as exc:
        return f"Could not write config: {exc}", False, no_update

    console = open(out / _CONSOLE_LOG, "w")
    # Launch via the installed `picasso` console script, NOT `python -m msi_picasso.cli`:
    # `-m` prepends the cwd to sys.path, so running from the repo root lets the vendored
    # source dirs (psm_utils/, ms2pip/, deeplc/ …) shadow the installed packages and break
    # imports. Console scripts don't add cwd; cwd=out is a neutral dir as a further guard.
    picasso_exe = Path(sys.executable).with_name("picasso")
    cmd = ([str(picasso_exe), "-c", str(cfg_path)] if picasso_exe.exists()
           else [sys.executable, "-m", "msi_picasso.cli", "-c", str(cfg_path)])
    proc = subprocess.Popen(cmd, stdout=console, stderr=subprocess.STDOUT, cwd=str(out))
    _JOB.update(proc=proc, outdir=str(out), cancelled=False)
    return f"Launched (pid {proc.pid}). Watch the Live Log tab.", True, str(out)


@app.callback(
    Output("log-status", "children", allow_duplicate=True),
    Output("log-status", "className", allow_duplicate=True),
    Output("run-btn", "disabled", allow_duplicate=True),
    Output("cancel-btn", "disabled", allow_duplicate=True),
    Input("cancel-btn", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_run(_n):
    proc = _JOB["proc"]
    if proc is None or proc.poll() is not None:
        return no_update, no_update, no_update, True
    _JOB["cancelled"] = True
    proc.terminate()  # SIGTERM; poll_log reports "cancelled" once it exits.
    return "cancelling…", "log-status failed", False, True


@app.callback(
    Output("log-status", "children"), Output("log-status", "className"),
    Output("log-pre", "children"), Output("run-btn", "disabled", allow_duplicate=True),
    Output("cancel-btn", "disabled"),
    Input("log-interval", "n_intervals"),
    prevent_initial_call=True,
)
def poll_log(_n):
    proc, outdir = _JOB["proc"], _JOB["outdir"]
    if proc is None or outdir is None:
        return "idle", "log-status idle", "No run started yet.", False, True
    log_path = Path(outdir) / _CONSOLE_LOG
    text = ""
    if log_path.exists():
        text = "\n".join(log_path.read_text(errors="replace").splitlines()[-400:]) or "(waiting for output…)"
    rc = proc.poll()
    if rc is None:  # running → cancel enabled, run disabled
        return "running…", "log-status running", text, True, False
    if _JOB.get("cancelled"):
        return "cancelled ✗", "log-status failed", text, False, True
    if rc == 0:
        return "finished ✓", "log-status finished", text, False, True
    return f"failed (exit {rc})", "log-status failed", text, False, True


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
def _stat(value, label):
    return html.Div(className="stat", children=[
        html.Div(f"{value}", className="stat-value"), html.Div(label, className="stat-label")])


def _table(df: pd.DataFrame | None):
    if df is None or df.empty:
        return html.P("(empty)", className="hint")
    show = df.copy()
    for c in show.select_dtypes("float").columns:
        show[c] = show[c].round(4)
    return dash_table.DataTable(
        data=show.to_dict("records"), columns=[{"name": c, "id": c} for c in show.columns],
        page_size=12, sort_action="native", filter_action="native", style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "VT323, monospace", "fontSize": "16px",
                    "border": f"2px solid {NAVY}", "padding": "4px 8px", "backgroundColor": "#ffffff"},
        style_header={"fontFamily": PIXEL_FONT, "fontSize": "15px", "backgroundColor": NAVY,
                      "color": "#ffffff", "border": f"2px solid {NAVY}"})


def _gallery(debug_dir: Path):
    if not debug_dir.is_dir():
        return html.Div()
    pngs = sorted(glob.glob(str(debug_dir / "*.png")))
    if not pngs:
        return html.Div()
    figs = []
    for p in pngs:
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        figs.append(html.Figure([html.Img(src=f"data:image/png;base64,{b64}"),
                                 html.Figcaption(Path(p).name)]))
    return html.Div(className="card", children=[html.H2("Debug figures"),
                                                html.Div(figs, className="gallery")])


@app.callback(
    Output("results-summary", "children"), Output("results-body", "children"),
    Input("load-results", "n_clicks"), State("results-dir", "value"),
    prevent_initial_call=True,
)
def load_results(_n, outdir):
    if not outdir or not Path(outdir).is_dir():
        return html.P("Set a valid results directory.", className="hint"), None
    out = Path(outdir)
    matches = _read_tsv(out / "ms1rescore_matches.tsv")
    peptides = _read_tsv(out / "ms1rescore_peptides.tsv")
    if matches is None:
        return html.P("No ms1rescore_matches.tsv found here.", className="hint"), None

    n_pep = 0 if peptides is None else len(peptides)
    winners = matches[matches.get("is_tdc_winner", False) & ~matches["is_decoy"]] \
        if "is_tdc_winner" in matches else matches[~matches["is_decoy"]]
    n_1pct = int((winners.get("reweighted_q_value", pd.Series(dtype=float)) <= 0.01).sum())
    summary = html.Div(className="stat-row", children=[
        _stat(len(matches), "candidates"), _stat(int((~matches["is_decoy"]).sum()), "targets"),
        _stat(n_1pct, "target winners ≤1% FDR"), _stat(n_pep, "peptide winners (file)")])

    plots = html.Div(className="card", children=[html.H2("Plots"), html.Div(className="row", children=[
        html.Div(className="col", children=dcc.Graph(figure=build_ids_vs_fdr(matches))),
        html.Div(className="col", children=dcc.Graph(figure=build_score_dist(matches))),
    ])])

    wsort = winners.dropna(subset=["reweighted_q_value"]).sort_values("reweighted_q_value") \
        if "reweighted_q_value" in winners else winners
    mzs = wsort["feature_mz"].dropna().unique()[:200] if "feature_mz" in wsort else []
    ion_card = html.Div(className="card", children=[
        html.H2("Ion-image viewer"),
        html.P("Extracts the ion image on demand from the raw .d (needs alphatims/imzy).", className="hint"),
        dcc.Dropdown(id="ion-mz", options=[{"label": f"{mz:.4f}", "value": float(mz)} for mz in mzs],
                     placeholder="Select an identified m/z…"),
        dcc.Loading(dcc.Graph(id="ion-graph", figure=_empty_fig("Select an m/z above."))),
    ])

    tables = html.Div(className="card", children=[html.H2("Tables"),
        html.H3("Peptide winners"), _table(peptides),
        html.H3("All candidates (matches)"), _table(matches)])

    return summary, html.Div([plots, ion_card, tables, _gallery(out / "debug")])


@app.callback(
    Output("ion-graph", "figure"), Input("ion-mz", "value"), State("results-dir", "value"),
    prevent_initial_call=True,
)
def show_ion_image(mz, outdir):
    if mz is None:
        return _empty_fig("Select an m/z above.")
    maldi_d = _resolved_maldi_d(Path(outdir)) if outdir else None
    if not maldi_d or not Path(maldi_d).exists():
        return _empty_fig("Raw .d path not found for this run.")
    try:
        from msi_picasso.maldi_query import query_raw_maldi
        _, images, _, _, _ = query_raw_maldi(maldi_d, np.array([float(mz)]), extraction_ppm=25.0)
        return build_ion_image(np.asarray(images)[0], float(mz))
    except Exception as exc:
        return _empty_fig(f"Could not extract ion image: {exc}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(prog="picasso-gui", description="MSI-PICASSO graphical interface")
    parser.add_argument("--desktop", action="store_true",
                        help="Open in a native desktop window (needs the 'desktop' extra).")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if args.desktop:
        import threading
        import webview  # ponytail: ~10-line thread + pywebview wrapper, no Electron.
        threading.Thread(target=lambda: app.run(host=args.host, port=args.port), daemon=True).start()
        webview.create_window("MSI-PICASSO", f"http://{args.host}:{args.port}")
        webview.start()
    else:
        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
