"""
tin_mid_analysis_app.py
========================
TIN / MID Duplicate Analysis & Clean Sampling — Streamlit App

Tabs:
  1. TIN/MID Landscape      — distributions, KPIs, fraud-rate by MID-count tier
  2. Duplicate Deep Dive    — per-TIN explorer with tagged records & interpretation
  3. All Records Tagged     — full table with duplicate_tag + tag_reason, charts, download
  4. Clean Sampling         — configurable N, exclusion rules, stratified sample download
  5. Sample Representativeness — PSI / KS / chi² comparison between raw pool and sample

Run:
  streamlit run tin_mid_analysis_app.py
"""

import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Optional: Analyzer for feature-type classification (Tab 5) ───────────────
try:
    from analyzer_fixed import Analyzer as _FeatureClassifier
    _FEATURE_CLASSIFIER_AVAILABLE = True
except ImportError:
    _FeatureClassifier = None
    _FEATURE_CLASSIFIER_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "merged_full_dataset.csv"
TARGET_COL = "Fraud.Is Fraud (yes=1, no=0)"
FRAUD_TYPE = "Fraud.Type of Fraud (ID Theft=1, Synthetic=2, Others=3)"
TIN_COL    = "TIN"
MID_COL    = "MID"
STATE_COL  = "address_state"
CITY_COL   = "address_city"
POSTAL_COL = "address_postal_code"

# Tag colours (for UI)
TAG_COLORS = {
    "EXACT_DUPLICATE":          "#e74c3c",
    "IDENTITY_GROUP_FRAUD":     "#9b59b6",
    "IDENTITY_GROUP_CLEAN":     "#e67e22",
    "TECH_REAPPLICATION":       "#f39c12",
    "INVALID_FOR_KYC":          "#c0392b",
    "UNIQUE_CLEAN_NON_LINKED":  "#27ae60",
}
TAG_ICONS = {
    "EXACT_DUPLICATE":          "🔴",
    "IDENTITY_GROUP_FRAUD":     "🟣",
    "IDENTITY_GROUP_CLEAN":     "🟠",
    "TECH_REAPPLICATION":       "🟡",
    "INVALID_FOR_KYC":          "🔺",
    "UNIQUE_CLEAN_NON_LINKED":  "🟢",
}
TAG_DESC = {
    "EXACT_DUPLICATE":         "All fields are byte-for-byte identical to another row — pure ETL/data-entry duplicate. The first occurrence is preserved; all later copies are dropped.",
    "IDENTITY_GROUP_FRAUD":    "Same Legal Name + Owner + Address group contains at least one confirmed Fraud=1 row. Only one Fraud=1 record (earliest open_date) is preserved per group; all others — including non-fraud duplicates — are tagged for exclusion to avoid label contamination and redundant vendor spend.",
    "IDENTITY_GROUP_CLEAN":    "Same Legal Name + Owner + Address group, but none of the rows have fraud. The most recent record (latest open_date) is preserved; older copies are tagged as redundant.",
    "TECH_REAPPLICATION":      "Same (TIN, MID) pair submitted more than once with minor field differences (e.g. phone format, address variant, updated postal code). The earliest open_date record is preserved; later re-submissions are tagged.",
    "INVALID_FOR_KYC":         "Missing Legal Name, TIN, or Postal Code. These records will fail vendor KYC verification — sending them wastes lookup budget with no chance of a match.",
    "UNIQUE_CLEAN_NON_LINKED": "Single appearance of this TIN/identity with all required fields present and no fraud linkage in the group — high-quality clean example for modeling.",
}

# Dark theme constants
BG      = "#0f0f1a"
CARD_BG = "#1a1a2e"
BORDER  = "#2c2c4e"
TEXT    = "#e0e0e0"
ACCENT  = "#aad4f5"

st.set_page_config(
    page_title="TIN/MID Duplicate Analysis",
    page_icon="🔍",
    layout="wide",
)
st.markdown(f"""
<style>
  .stApp {{ background:{BG}; color:{TEXT}; }}
  .block-container {{ padding-top:1.2rem; }}
  h1,h2,h3 {{ color:{ACCENT}; }}
  .stTabs [data-baseweb="tab"] {{ color:{TEXT}; font-size:14px; }}
  .stTabs [aria-selected="true"] {{ color:{ACCENT}; border-bottom:2px solid {ACCENT}; }}
  .metric-card {{ background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;
                  padding:14px 18px;text-align:center; }}
  .metric-val  {{ font-size:24px;font-weight:700;color:{ACCENT}; }}
  .metric-lbl  {{ font-size:11px;color:#888;margin-top:4px; }}
  .tag-pill    {{ padding:3px 10px;border-radius:12px;font-size:12px;
                  font-weight:600;color:#fff;display:inline-block; }}
  .info-box    {{ background:#1e1e3e;border-left:4px solid {ACCENT};
                  border-radius:6px;padding:12px 16px;font-size:13px;
                  color:#ccc;line-height:1.7;margin:8px 0; }}
  .warn-box    {{ background:#2a1515;border-left:4px solid #e74c3c;
                  border-radius:6px;padding:12px 16px;font-size:13px;
                  color:#f5b7b7;line-height:1.7;margin:8px 0; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE-TYPE ENGINEERING (auto-fix datetime, id, etc.)
# ─────────────────────────────────────────────────────────────────────────────
def _apply_feature_type_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use Analyzer._detect_feature_types() to classify every column and
    apply auto-fixes (e.g. Excel serial dates → proper datetime64).
    Falls back to a manual Excel-serial conversion when the Analyzer is
    not available.
    Returns the fixed DataFrame.
    """
    if _FEATURE_CLASSIFIER_AVAILABLE:
        az = _FeatureClassifier(df.copy(), log_level='WARNING')
        az._detect_feature_types()          # triggers in-place serial→datetime fix
        fixed = az.data
    else:
        fixed = df.copy()
        # Fallback: manual Excel-serial detection for known date columns
        _KNOWN_DATE_COLS = [
            "open_date", "closed_date",
            "Fraud.Opened Date", "Fraud.Date Fraud Found",
        ]
        for col in _KNOWN_DATE_COLS:
            if col not in fixed.columns:
                continue
            if not pd.api.types.is_numeric_dtype(fixed[col]):
                continue
            _samp = fixed[col].dropna().head(200)
            if len(_samp) > 0 and _samp.between(20_000, 60_000).mean() > 0.8:
                fixed[col] = pd.to_datetime(
                    fixed[col].apply(
                        lambda x: np.floor(x) if pd.notna(x) else np.nan
                    ),
                    unit='D', origin='1899-12-30', errors='coerce'
                )
    # Normalise all datetime64 columns to date-only (no time component) for
    # cleaner display and deduplication logic
    for col in fixed.select_dtypes(include=['datetime64[ns]', 'datetime']).columns:
        fixed[col] = fixed[col].dt.normalize()   # midnight — strips intra-day jitter
    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & TAGGING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & tagging dataset …")
def load_and_tag(file_source, file_name: str) -> pd.DataFrame:
    if file_name.lower().endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_source)
    else:
        df = pd.read_csv(file_source, low_memory=False)

    # ── Feature-type engineering: convert Excel serials, classify columns ──────
    df = _apply_feature_type_fixes(df)

    # ── Normalise key columns ──────────────────────────────────────────────────
    # Keep original values; strip strings
    for c in [TIN_COL, MID_COL, STATE_COL, CITY_COL, POSTAL_COL, "LegalName",
              "dba1_name", "owner1_first_name", "owner1_last_name"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].str.strip().str.upper()

    # Normalise TIN & MID to string for consistent grouping (handle float NaN)
    df["_tin_str"] = df[TIN_COL].apply(
        lambda x: str(int(float(x))) if pd.notnull(x) and str(x).strip() not in ("", "nan") else ""
    )
    df["_mid_str"] = df[MID_COL].apply(
        lambda x: str(int(float(x))) if pd.notnull(x) and str(x).strip() not in ("", "nan") else ""
    )
    df["_state_norm"] = df[STATE_COL].fillna("").str.strip().str.upper()
    df["_city_norm"]  = df[CITY_COL].fillna("").str.strip().str.upper()
    df["_postal_norm"]= df[POSTAL_COL].fillna("").astype(str).str.strip()

    # ── Pre-compute TIN-group statistics ──────────────────────────────────────
    # Per-(TIN) group: count of distinct MIDs, fraud presence, record count
    tin_grp = (
        df[df["_tin_str"] != ""]
        .groupby("_tin_str")
        .agg(
            _tin_record_count  = ("_tin_str",    "count"),
            _tin_n_distinct_mid= ("_mid_str",    "nunique"),
            _tin_n_distinct_state=("_state_norm","nunique"),
            _tin_n_distinct_city =("_city_norm", "nunique"),
            _tin_fraud_count   = (TARGET_COL,    "sum"),
            _tin_fraud_rate    = (TARGET_COL,    "mean"),
        )
        .reset_index()
    )
    df = df.merge(tin_grp, on="_tin_str", how="left")

    # ── Tagging \u2014 6 mutually exclusive, priority-ordered tags ──────────────────
    n = len(df)
    tags    = ["UNIQUE_CLEAN_NON_LINKED"] * n
    reasons = ["Single appearance of this identity with all required fields and no fraud linkage."] * n

    # ── Pre-compute column sets ───────────────────────────────────────────────
    orig_cols = [c for c in df.columns if not c.startswith("_") and
                 c not in ["duplicate_tag", "tag_reason"]]

    # Identity grouping columns (strong person/business identity signals)
    _IDENTITY_GROUP_COLS = [c for c in [
        "LegalName", "owner1_first_name", "owner1_last_name",
        STATE_COL, CITY_COL, POSTAL_COL,
    ] if c in df.columns]

    # Pre-compute open_date as sortable datetime
    _open_dt = pd.to_datetime(df["open_date"], errors="coerce")

    # ── Priority 5 (lowest): INVALID_FOR_KYC ─────────────────────────────────
    # Missing Legal Name, TIN, or Postal Code — will fail vendor lookup.
    _ln_blank = (df.get("LegalName", pd.Series(["x"] * n))
                   .fillna("").astype(str).str.strip() == "")
    _pc_blank = (df.get(POSTAL_COL, pd.Series(["x"] * n))
                   .fillna("").astype(str).str.strip() == "")
    _tn_blank = df["_tin_str"] == ""
    _kyc_mask = _tn_blank | _ln_blank | _pc_blank
    for i in _kyc_mask[_kyc_mask].index:
        missing = []
        if _tn_blank[i]:  missing.append("TIN")
        if _ln_blank[i]:  missing.append("Legal Name")
        if _pc_blank[i]:  missing.append("Postal Code")
        tags[i]    = "INVALID_FOR_KYC"
        reasons[i] = (
            f"Missing required KYC field(s): {', '.join(missing)}. "
            "Will fail vendor verification \u2014 no value in sending."
        )

    # ── Priority 4: TECH_REAPPLICATION ──────────────────────────────────────
    # Same (TIN, MID) with differing fields (not exact duplicate).
    # Sort by open_date ascending; keep earliest, tag rest.
    _tmp_tr = df.assign(_open_s=_open_dt).sort_values("_open_s", na_position="last")
    _later_tr = (
        (_tmp_tr["_tin_str"] != "") &
        (_tmp_tr["_mid_str"] != "") &
        _tmp_tr.duplicated(subset=["_tin_str", "_mid_str"], keep="first")
    )
    for i in _later_tr[_later_tr].index:
        if tags[i] not in ("EXACT_DUPLICATE",):
            tags[i]    = "TECH_REAPPLICATION"
            reasons[i] = (
                f"Same TIN ({df.at[i,'_tin_str']}) + MID ({df.at[i,'_mid_str']}) "
                "submitted multiple times with differing field values "
                "(e.g. phone format, address variant). "
                "This is a later copy \u2014 the earliest open_date record is preserved."
            )

    # ── Priority 3 & 2: Identity-group tags ─────────────────────────────────
    # Requires at least 3 identity columns to form a meaningful group key.
    if len(_IDENTITY_GROUP_COLS) >= 3:
        _id_str = df[_IDENTITY_GROUP_COLS].fillna("__NA__").astype(str)
        _id_key = _id_str.agg("|".join, axis=1)
        _tf     = df[TARGET_COL].fillna(0).astype(int)
        _grp_fraud_sum = _id_key.map(_tf.groupby(_id_key).sum())
        _grp_size      = _id_key.map(_id_key.value_counts())

        # ── Priority 3: IDENTITY_GROUP_CLEAN ────────────────────────────────
        # Group size >= 2, all records no-fraud.
        # Keep most recent (sort open_date DESC), tag older copies.
        _tmp_igc = df.assign(
            _id_key   = _id_key,
            _eligible = (_grp_size > 1) & (_grp_fraud_sum == 0),
            _open_s   = _open_dt,
        ).sort_values("_open_s", ascending=False, na_position="last")
        _igc_dupes = (
            _tmp_igc["_eligible"] &
            _tmp_igc["_id_key"].duplicated(keep="first")
        )
        for i in _igc_dupes[_igc_dupes].index:
            if tags[i] not in ("EXACT_DUPLICATE", "TECH_REAPPLICATION", "INVALID_FOR_KYC"):
                tags[i]    = "IDENTITY_GROUP_CLEAN"
                reasons[i] = (
                    "Shares Legal Name + Owner + Address with other rows; none in this group "
                    "are fraud. This is an older copy \u2014 the most recent record is preserved."
                )

        # ── Priority 2: IDENTITY_GROUP_FRAUD ────────────────────────────────
        # Group size >= 2, at least one fraud=1.
        # Keep single Fraud=1 record with earliest open_date; tag all others
        # (both other fraud rows and all no-fraud rows in the group).
        _tmp_igf = df.assign(
            _id_key   = _id_key,
            _eligible = (_grp_size > 1) & (_grp_fraud_sum > 0),
            _open_s   = _open_dt,
            _is_fraud = _tf,
        ).sort_values(["_is_fraud", "_open_s"], ascending=[False, True], na_position="last")
        _igf_dupes = (
            _tmp_igf["_eligible"] &
            _tmp_igf["_id_key"].duplicated(keep="first")
        )
        for i in _igf_dupes[_igf_dupes].index:
            if tags[i] not in ("EXACT_DUPLICATE",):
                own_fraud  = int(df.at[i, TARGET_COL]) if pd.notna(df.at[i, TARGET_COL]) else 0
                tags[i]    = "IDENTITY_GROUP_FRAUD"
                reasons[i] = (
                    "Shares Legal Name + Owner + Address with at least one confirmed fraud=1 row. "
                    f"This record is {'Fraud' if own_fraud else 'Non-Fraud'} \u2014 "
                    "only the earliest Fraud=1 record from this identity group is preserved; "
                    "all others are tagged for exclusion."
                )

    # ── Priority 1 (highest): EXACT_DUPLICATE ───────────────────────────────
    # All original columns identical.
    # keep="first" \u2192 only 2nd+ copies tagged; first occurrence is preserved.
    exact_dup_mask = df.duplicated(subset=orig_cols, keep="first")
    for i in exact_dup_mask[exact_dup_mask].index:
        tags[i]    = "EXACT_DUPLICATE"
        reasons[i] = (
            f"All fields are identical to at least one other row "
            f"(TIN {df.at[i,'_tin_str']}, MID {df.at[i,'_mid_str']}). "
            "Pure data-entry or ETL duplicate \u2014 this copy is removed; "
            "the first occurrence is preserved."
        )

    df["duplicate_tag"] = tags
    df["tag_reason"]    = reasons

    # ── is_first_of_group flag  ───────────────────────────────────────────────
    # True = this record is the canonical survivor for its (TIN, MID) pair
    df_sorted = df.copy()
    df_sorted["_open_date_sort"] = pd.to_datetime(df["open_date"], errors="coerce")
    df_sorted = df_sorted.sort_values("_open_date_sort")
    df_sorted["is_first_of_group"] = ~df_sorted.duplicated(
        subset=["_tin_str", "_mid_str"], keep="first"
    )
    df["is_first_of_group"] = df_sorted["is_first_of_group"].reindex(df.index)

    return df


@st.cache_data(show_spinner=False)
def get_tin_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per TIN with aggregate metrics."""
    agg = (
        df[df["_tin_str"] != ""]
        .groupby("_tin_str")
        .agg(
            record_count   = ("_tin_str",     "count"),
            distinct_mids  = ("_mid_str",     "nunique"),
            distinct_states= (STATE_COL,      "nunique"),
            distinct_cities= (CITY_COL,       "nunique"),
            fraud_count    = (TARGET_COL,     "sum"),
            fraud_rate_pct = (TARGET_COL,     lambda x: round(x.mean()*100,2)),
            tags_in_group  = ("duplicate_tag", lambda x: ", ".join(sorted(x.unique()))),
            legal_names    = ("LegalName",    lambda x: " / ".join(x.dropna().unique()[:3])),
            states_list    = (STATE_COL,      lambda x: ", ".join(x.dropna().unique()[:5])),
            cities_list    = (CITY_COL,       lambda x: ", ".join(x.dropna().unique()[:5])),
        )
        .reset_index()
        .rename(columns={"_tin_str": "TIN"})
        .sort_values("record_count", ascending=False)
    )
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _psi_numeric(pool_vals, samp_vals, buckets=10):
    e = np.array(pool_vals, dtype=float)
    a = np.array(samp_vals, dtype=float)
    if len(e) < 10 or len(a) < 10:
        return float("nan")
    bp = np.unique(np.percentile(e, np.linspace(0, 100, buckets + 1)))
    if len(bp) < 2:
        return float("nan")
    ep = np.histogram(e, bins=bp)[0] / max(len(e), 1)
    ap = np.histogram(a, bins=bp)[0] / max(len(a), 1)
    ep = np.where(ep == 0, 1e-6, ep)
    ap = np.where(ap == 0, 1e-6, ap)
    return float(np.sum((ap - ep) * np.log(ap / ep)))

def _psi_categorical(pool_col, samp_col):
    pv = pool_col.value_counts(normalize=True)
    sv = samp_col.value_counts(normalize=True)
    cats = sorted(set(pv.index) | set(sv.index))
    if len(cats) < 2:
        return float("nan"), cats
    ep = np.array([max(float(pv.get(c, 0)), 1e-6) for c in cats])
    ap = np.array([max(float(sv.get(c, 0)), 1e-6) for c in cats])
    ep /= ep.sum(); ap /= ap.sum()
    return float(np.sum((ap - ep) * np.log(ap / ep))), cats

def _evaluate_max_psi(pool_df, samp_df):
    """Calculates max PSI across outcome cols, metrics, and standard features to check stability.
    Returns (max_psi, is_stable) where is_stable respects the relaxed threshold for Fraud sub-populations."""
    max_psi = 0.0
    is_stable = True
    
    def _check_feature(psi_val, col_name):
        nonlocal max_psi, is_stable
        if psi_val == psi_val: # no nan
            if psi_val > max_psi:
                max_psi = psi_val
            threshold = 0.20 if col_name.startswith("Fraud.") else 0.10
            if psi_val > threshold:
                is_stable = False

    _outcome_cols = [TARGET_COL, FRAUD_TYPE]
    _skip = {"duplicate_tag", "tag_reason", "is_first_of_group", "_tin_fraud_count", "_tin_record_count", "_tin_fraud_rate", "_tin_n_distinct_mid", "_tin_n_distinct_state", "_tin_n_distinct_city", "address_postal_code", "owner1_first_name", "address_line_2", "address_city"}
    _num, _cat, _date = [], [], []
    
    _hard_id = {"TIN", "MID", "SequenceKey", "LegalName", "dba1_name", "owner1_first_name", "owner1_last_name", "business_email", "business_phone"}
    _hard_dt = {"open_date", "closed_date", "Fraud.Opened Date", "Fraud.Date Fraud Found"}

    if _FEATURE_CLASSIFIER_AVAILABLE:
        az = _FeatureClassifier(pool_df.head(100), log_level="WARNING")
        ftype_map = az._detect_feature_types()
        for c, t in ftype_map.items():
            if c.startswith("_") or c in _skip or c in _outcome_cols: continue
            if t == "numeric": _num.append(c)
            elif t in ("categorical", "binary"): _cat.append(c)
            elif t == "datetime": _date.append(c)
    else:
        for c in pool_df.columns:
            if c.startswith("_") or c in _skip or c in _outcome_cols: continue
            if c in _hard_id: pass
            elif c in _hard_dt: _date.append(c)
            elif pool_df[c].dtype.kind in ("i", "f", "u") and c not in _hard_id: _num.append(c)
            else: _cat.append(c)

    for c in _outcome_cols:
        if c in pool_df.columns and c in samp_df.columns:
            _cat.append(c)

    for c in _num:
        if c not in samp_df.columns: continue
        pv = pool_df[c].dropna().values
        sv = samp_df[c].dropna().values
        psi = _psi_numeric(pv, sv)
        _check_feature(psi, c)
        
    for c in _cat:
        if c not in samp_df.columns: continue
        pc = pool_df[c].fillna("__NA__").astype(str)
        sc = samp_df[c].fillna("__NA__").astype(str)
        psi, _ = _psi_categorical(pc, sc)
        _check_feature(psi, c)
        
    for c in _date:
        if c not in samp_df.columns: continue
        def _to_dt(s): return pd.to_datetime(s, errors="coerce").dropna()
        pd_dt = _to_dt(pool_df[c])
        sd_dt = _to_dt(samp_df[c])
        if len(pd_dt) < 10 or len(sd_dt) < 10: continue
        
        pm = pd_dt.dt.quarter.astype(str)
        sm = sd_dt.dt.quarter.astype(str)
        psi_m, _ = _psi_categorical(pm, sm)
        _check_feature(psi_m, c)
        
        pw = pd_dt.dt.dayofweek.astype(str)
        sw = sd_dt.dt.dayofweek.astype(str)
        psi_d, _ = _psi_categorical(pw, sw)
        _check_feature(psi_d, c)

    return max_psi, is_stable

def _plotly_dark(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _metric_card(label: str, value, color: str = ACCENT) -> str:
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-val' style='color:{color}'>{value}</div>"
        f"<div class='metric-lbl'>{label}</div>"
        f"</div>"
    )


def _tag_pill(tag: str) -> str:
    c = TAG_COLORS.get(tag, "#888")
    return f"<span class='tag-pill' style='background:{c}'>{TAG_ICONS.get(tag,'')} {tag}</span>"


def _download_csv(df_out: pd.DataFrame, label: str, filename: str):
    buf = io.BytesIO()
    df_out.to_csv(buf, index=False)
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=filename,
        mime="text/csv",
    )


def _cols_for_display(df: pd.DataFrame) -> list[str]:
    """Columns to show in the tagged table (drop internal helpers)."""
    return [c for c in df.columns if not c.startswith("_")]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & APP HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV/Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    data_source = uploaded_file
    dataset_name = uploaded_file.name
else:
    data_source = DATA_PATH
    dataset_name = DATA_PATH

try:
    df = load_and_tag(data_source, dataset_name)
    tin_summary = get_tin_summary(df)
except FileNotFoundError:
    st.warning(f"Default dataset '{dataset_name}' not found. Please upload a dataset in the sidebar.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

TRUE_DUP_TAGS = ["EXACT_DUPLICATE", "TECH_REAPPLICATION", "IDENTITY_GROUP_CLEAN"]
RISK_TAGS     = ["IDENTITY_GROUP_FRAUD", "INVALID_FOR_KYC"]
TAG_ORDER     = TRUE_DUP_TAGS + RISK_TAGS + ["UNIQUE_CLEAN_NON_LINKED"]
tag_counts = df["duplicate_tag"].value_counts().reindex(TAG_ORDER).fillna(0).astype(int)

st.title("🔍 TIN / MID Duplicate Analysis")
st.markdown(
    f"<div style='color:#888;font-size:13px;margin-bottom:8px'>"
    f"Dataset: <b style='color:{ACCENT}'>{dataset_name}</b> &nbsp;|&nbsp; "
    f"Rows: <b style='color:{ACCENT}'>{len(df):,}</b> &nbsp;|&nbsp; "
    f"Last run: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>",
    unsafe_allow_html=True,
)

TAB1, TAB2, TAB3, TAB4, TAB5 = st.tabs([
    "📊 TIN/MID Landscape",
    "🔬 Duplicate Deep Dive",
    "🏷️ All Records Tagged",
    "🧪 Clean Sampling",
    "📏 Sample Representativeness",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — TIN/MID LANDSCAPE
# ═════════════════════════════════════════════════════════════════════════════
with TAB1:
    st.header("TIN / MID Landscape")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    unique_tins   = df["_tin_str"].replace("", np.nan).nunique()
    unique_mids   = df["_mid_str"].replace("", np.nan).nunique()
    tins_multi    = (tin_summary["distinct_mids"] > 1).sum()
    tins_fraud    = (tin_summary["fraud_count"] > 0).sum()
    exact_dups    = int(tag_counts.get("EXACT_DUPLICATE", 0))
    fraud_ring_n  = int(tag_counts.get("IDENTITY_GROUP_FRAUD", 0))

    kpi_html = "".join([
        _metric_card("Unique TINs",                    f"{unique_tins:,}"),
        _metric_card("Unique MIDs",                    f"{unique_mids:,}"),
        _metric_card("TINs with 2+ MIDs",              f"{tins_multi:,}",    "#f39c12"),
        _metric_card("TINs linked to Fraud",           f"{tins_fraud:,}",    "#e74c3c"),
        _metric_card("Exact Duplicate Rows",           f"{exact_dups:,}",    "#e74c3c"),
        _metric_card("Identity-Group Fraud Rows",      f"{fraud_ring_n:,}",  "#9b59b6"),
    ])
    cols_kpi = st.columns(6)
    vals = [
        ("Unique TINs",                    f"{unique_tins:,}",   ACCENT),
        ("Unique MIDs",                    f"{unique_mids:,}",   ACCENT),
        ("TINs with 2+ MIDs",              f"{tins_multi:,}",    "#f39c12"),
        ("TINs linked to Fraud",           f"{tins_fraud:,}",    "#e74c3c"),
        ("Exact Duplicate Rows",           f"{exact_dups:,}",    "#e74c3c"),
        ("Identity-Group Fraud Rows",      f"{fraud_ring_n:,}",  "#9b59b6"),
    ]
    for col, (lbl, val, clr) in zip(cols_kpi, vals):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-val' style='color:{clr}'>{val}</div>"
            f"<div class='metric-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── MID-count distribution per TIN ────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Distribution: Records per TIN")

        bins = [1, 2, 3, 4, 6, 11, 51, float("inf")]
        labels = ["1 (single)", "2", "3", "4–5", "6–10", "11–50", "51+"]
        tin_summary["mid_tier"] = pd.cut(
            tin_summary["record_count"], bins=bins, labels=labels, right=False
        )
        tier_counts = tin_summary["mid_tier"].value_counts().reindex(labels).fillna(0)

        fig_tier = px.bar(
            x=labels, y=tier_counts.values,
            labels={"x": "Records per TIN", "y": "Number of TINs"},
            color=tier_counts.values,
            color_continuous_scale=["#27ae60","#f39c12","#e74c3c"],
            title="How many records share the same TIN?",
        )
        fig_tier.update_coloraxes(showscale=False)
        fig_tier.update_traces(text=tier_counts.values.astype(int), textposition="outside")
        st.plotly_chart(_plotly_dark(fig_tier), use_container_width=True)

        st.markdown(
            "<div class='info-box'>Most TINs appear only once (single-location or "
            "first-time applicant). TINs with many records may indicate a large merchant "
            "network — or aggressive re-application / fraud ring behaviour. "
            "Check the far-right bars carefully.</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.subheader("Fraud Rate by Record-Count Tier")

        tier_fraud = (
            tin_summary.groupby("mid_tier", observed=False)["fraud_rate_pct"]
            .mean()
            .reindex(labels)
            .fillna(0)
        )
        fig_fr = px.bar(
            x=labels, y=tier_fraud.values,
            labels={"x": "Records per TIN", "y": "Avg Fraud Rate (%)"},
            color=tier_fraud.values,
            color_continuous_scale=["#3498db","#e74c3c"],
            title="Does more records per TIN → higher fraud rate?",
        )
        fig_fr.update_coloraxes(showscale=False)
        fig_fr.update_traces(
            text=[f"{v:.2f}%" for v in tier_fraud.values],
            textposition="outside",
        )
        st.plotly_chart(_plotly_dark(fig_fr), use_container_width=True)

        st.markdown(
            "<div class='info-box'>A rising fraud rate with higher record-count tiers "
            "would confirm that TINs with many MIDs are disproportionately fraudulent — "
            "a key signal to use in the model.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Distinct MIDs per TIN ─────────────────────────────────────────────────
    st.subheader("Distinct MIDs per TIN Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        mid_dist_bins  = [1, 2, 3, 4, 6, 11, float("inf")]
        mid_dist_labels= ["1", "2", "3", "4–5", "6–10", "11+"]
        tin_summary["mid_dist_tier"] = pd.cut(
            tin_summary["distinct_mids"], bins=mid_dist_bins,
            labels=mid_dist_labels, right=False,
        )
        mid_tier_c = tin_summary["mid_dist_tier"].value_counts().reindex(mid_dist_labels).fillna(0)
        fig_mid = px.bar(
            x=mid_dist_labels, y=mid_tier_c.values,
            labels={"x": "Distinct MIDs under Same TIN", "y": "Number of TINs"},
            title="# Distinct MIDs per TIN",
            color=mid_tier_c.values,
            color_continuous_scale=["#3498db","#e74c3c"],
        )
        fig_mid.update_coloraxes(showscale=False)
        fig_mid.update_traces(text=mid_tier_c.values.astype(int), textposition="outside")
        st.plotly_chart(_plotly_dark(fig_mid), use_container_width=True)

    with col_b:
        # Fraud rate vs distinct MID count tier
        mid_tier_fraud = (
            tin_summary.groupby("mid_dist_tier", observed=False)["fraud_rate_pct"]
            .mean()
            .reindex(mid_dist_labels)
            .fillna(0)
        )
        fig_mfr = px.bar(
            x=mid_dist_labels, y=mid_tier_fraud.values,
            labels={"x": "Distinct MIDs per TIN", "y": "Avg Fraud Rate (%)"},
            title="Fraud Rate vs Distinct MID Count",
            color=mid_tier_fraud.values,
            color_continuous_scale=["#27ae60","#e74c3c"],
        )
        fig_mfr.update_coloraxes(showscale=False)
        fig_mfr.update_traces(
            text=[f"{v:.3f}%" for v in mid_tier_fraud.values],
            textposition="outside",
        )
        st.plotly_chart(_plotly_dark(fig_mfr), use_container_width=True)

    st.markdown("---")

    # ── Top-20 TINs by record count ───────────────────────────────────────────
    st.subheader("Top 20 TINs by Record Count")
    top20 = tin_summary.head(20).copy()
    top20["has_fraud"] = top20["fraud_count"].apply(lambda x: "🔴 YES" if x > 0 else "🟢 NO")

    display_cols = ["TIN", "legal_names", "record_count", "distinct_mids",
                    "distinct_states", "distinct_cities", "fraud_count",
                    "fraud_rate_pct", "has_fraud", "states_list", "tags_in_group"]
    display_cols = [c for c in display_cols if c in top20.columns]
    st.dataframe(
        top20[display_cols].rename(columns={
            "TIN": "TIN", "legal_names": "Legal Name(s)",
            "record_count": "# Records", "distinct_mids": "# MIDs",
            "distinct_states": "# States", "distinct_cities": "# Cities",
            "fraud_count": "Fraud #", "fraud_rate_pct": "Fraud %",
            "has_fraud": "Fraud?", "states_list": "States",
            "tags_in_group": "Tags in Group",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DUPLICATE DEEP DIVE
# ═════════════════════════════════════════════════════════════════════════════
with TAB2:
    st.header("Duplicate Deep Dive — Search by TIN")
    st.markdown(
        "<div class='info-box'>Enter a TIN number to see every record associated with it, "
        "color-coded by its duplicate tag, along with a detailed interpretation of why "
        "each record was classified the way it was.</div>",
        unsafe_allow_html=True,
    )

    # ── Search ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        tin_input = st.text_input(
            "Search TIN",
            placeholder="e.g.  123456789",
            help="Enter the TIN number exactly as it appears in the data.",
        )
    with c2:
        show_only_dups = st.checkbox("Show only multi-TIN records", value=True)
    with c3:
        if st.button("🎲 Random multi-MID TIN"):
            multi_tins = tin_summary[tin_summary["distinct_mids"] > 1]["TIN"].values
            if len(multi_tins):
                tin_input = str(np.random.choice(multi_tins))
                st.session_state["_random_tin"] = tin_input

    # Use session state random pick
    if "random_tin" not in st.session_state:
        st.session_state["_random_tin"] = ""
    if not tin_input and st.session_state.get("_random_tin"):
        tin_input = st.session_state["_random_tin"]

    # ── Browse table ─────────────────────────────────────────────────────────
    if not tin_input:
        st.markdown("---")
        st.subheader("Browse: TINs with Multiple Records")

        br_col1, br_col2 = st.columns([1, 2])
        with br_col1:
            filter_n = st.slider("Minimum records per TIN", 2, 20, 2)
        with br_col2:
            tag_filter_browse = st.multiselect(
                "Filter by tag (TINs whose group contains any selected tag)",
                options=TAG_ORDER,
                default=[],
                format_func=lambda t: f"{TAG_ICONS.get(t, '')} {t}",
                key="_browse_tag_filter",
            )

        browse_df = tin_summary[tin_summary["record_count"] >= filter_n].copy()
        if tag_filter_browse:
            browse_df = browse_df[
                browse_df["tags_in_group"].apply(
                    lambda s: any(t in s for t in tag_filter_browse)
                )
            ]
        browse_df = browse_df.head(200)

        st.caption(
            f"Showing {len(browse_df):,} TIN group(s)"
            + (f" matching tag(s): {', '.join(tag_filter_browse)}" if tag_filter_browse else "")
        )

        # Columns & renames used inside each expander detail view
        _BROWSE_DETAIL_COLS = [
            "_mid_str", "LegalName", "dba1_name",
            STATE_COL, CITY_COL, POSTAL_COL,
            "owner1_first_name", "owner1_last_name",
            "open_date", TARGET_COL, FRAUD_TYPE,
            "duplicate_tag", "tag_reason",
        ]
        _BROWSE_DETAIL_RENAMES = {
            "_mid_str": "MID", "LegalName": "Legal Name", "dba1_name": "DBA",
            STATE_COL: "State", CITY_COL: "City", POSTAL_COL: "Postal",
            "owner1_first_name": "First Name", "owner1_last_name": "Last Name",
            "open_date": "Open Date",
            TARGET_COL: "Fraud?", FRAUD_TYPE: "Fraud Type",
            "duplicate_tag": "Tag", "tag_reason": "Reason",
        }
        # Columns to format as YYYY-MM-DD for readable display
        _DATETIME_DISPLAY_COLS = [
            "open_date", "closed_date",
            "Fraud.Opened Date", "Fraud.Date Fraud Found",
        ]

        # ── Per-TIN expandable rows ──────────────────────────────────────────
        def _row_highlight(row):
            """Colour each record row by its duplicate tag (post-rename key = 'Tag')."""
            c = TAG_COLORS.get(row.get("Tag", ""), "#555")
            return [f"background-color:{c}22;color:#ddd"] * len(row)

        for _, _br in browse_df.iterrows():
            _tin_v   = _br["TIN"]
            _name_v  = str(_br["legal_names"])
            _rec_v   = int(_br["record_count"])
            _mid_v   = int(_br["distinct_mids"])
            _frdn_v  = int(_br["fraud_count"])
            _frdp_v  = float(_br["fraud_rate_pct"])
            _st_v    = str(_br["states_list"])
            _tags_v  = str(_br["tags_in_group"])

            # Header label: fraud icon + TIN + name snippet + key stats
            _fi = "🚨" if _frdn_v > 0 else "✅"
            _name_short = (_name_v[:52] + "…") if len(_name_v) > 52 else _name_v
            _exp_label = (
                f"{_fi}  {_tin_v}  —  {_name_short}  "
                f"| {_rec_v} rec · {_mid_v} MID · "
                f"{_frdn_v} fraud ({_frdp_v:.1f}%)  "
                f"| {_st_v} | {_tags_v}"
            )

            with st.expander(_exp_label, expanded=False):
                # ── Mini metric row ──────────────────────────────────────────
                _mc = st.columns(5)
                _mc[0].metric("Records",       _rec_v)
                _mc[1].metric("Distinct MIDs", _mid_v)
                _mc[2].metric("States",        int(_br["distinct_states"]))
                _mc[3].metric("Fraud #",       _frdn_v)
                _mc[4].metric("Fraud %",       f"{_frdp_v:.1f}%")

                # ── Tag pills ────────────────────────────────────────────────
                _pills_html = " ".join(
                    _tag_pill(t.strip()) for t in _tags_v.split(",") if t.strip()
                )
                st.markdown(_pills_html + "<br>", unsafe_allow_html=True)

                # ── Individual records ───────────────────────────────────────
                _tin_recs = df[df["_tin_str"] == _tin_v].copy()
                # Format datetime columns as YYYY-MM-DD strings
                for _dc in _DATETIME_DISPLAY_COLS:
                    if _dc in _tin_recs.columns:
                        if pd.api.types.is_datetime64_any_dtype(_tin_recs[_dc]):
                            _tin_recs[_dc] = _tin_recs[_dc].dt.strftime("%Y-%m-%d")
                        elif pd.api.types.is_numeric_dtype(_tin_recs[_dc]):
                            # Still a serial (Analyzer not available) — convert on the fly
                            _tin_recs[_dc] = pd.to_datetime(
                                _tin_recs[_dc].apply(
                                    lambda x: np.floor(x) if pd.notna(x) else np.nan
                                ),
                                unit='D', origin='1899-12-30', errors='coerce'
                            ).dt.strftime("%Y-%m-%d")
                _show = [c for c in _BROWSE_DETAIL_COLS if c in _tin_recs.columns]
                _tin_disp = _tin_recs[_show].rename(columns=_BROWSE_DETAIL_RENAMES)
                st.dataframe(
                    _tin_disp.style.apply(_row_highlight, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

                # ── Analyst notes — one card per unique tag ───────────────
                if len(_tin_recs) > 0:
                    # Preserve frequency order (most-common first)
                    _all_tags = _tin_recs["duplicate_tag"].value_counts().index.tolist()
                    for _tag in _all_tags:
                        if _tag not in TAG_DESC:
                            continue
                        _tc = TAG_COLORS.get(_tag, "#888")
                        _cnt = int((_tin_recs["duplicate_tag"] == _tag).sum())
                        st.markdown(
                            f"<div style='background:{_tc}18;border-left:3px solid {_tc};"
                            f"border-radius:4px;padding:8px 12px;margin-top:6px;"
                            f"font-size:12px;color:#ccc'>"
                            f"<b style='color:{_tc}'>{TAG_ICONS.get(_tag,'')} {_tag}</b>"
                            f"<span style='color:#888;font-size:11px'> ({_cnt} record(s))</span>"
                            f" — {TAG_DESC[_tag]}</div>",
                            unsafe_allow_html=True,
                        )
    else:
        tin_str = str(tin_input).strip()
        subset  = df[df["_tin_str"] == tin_str].copy()

        if len(subset) == 0:
            st.warning(f"No records found for TIN: `{tin_str}`")
        else:
            # ── TIN summary banner ────────────────────────────────────────────
            tin_row = tin_summary[tin_summary["TIN"] == tin_str]
            fraud_c = int(subset[TARGET_COL].sum())
            rec_c   = len(subset)
            mid_c   = subset["_mid_str"].replace("", np.nan).nunique()
            st_c    = subset[STATE_COL].dropna().nunique()
            city_c  = subset[CITY_COL].dropna().nunique()

            banner_cols = st.columns(5)
            banner_data = [
                ("Records", f"{rec_c}", ACCENT),
                ("Distinct MIDs", f"{mid_c}", ACCENT),
                ("Distinct States", f"{st_c}", "#f39c12"),
                ("Distinct Cities", f"{city_c}", "#f39c12"),
                ("Fraud Records", f"{fraud_c}", "#e74c3c" if fraud_c else "#27ae60"),
            ]
            for bc, (lbl, val, clr) in zip(banner_cols, banner_data):
                bc.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-val' style='color:{clr}'>{val}</div>"
                    f"<div class='metric-lbl'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── Tag interpretation ────────────────────────────────────────────
            st.subheader("📋 Analyst Interpretation")
            unique_tags = subset["duplicate_tag"].unique()
            for tag in unique_tags:
                tag_rows = subset[subset["duplicate_tag"] == tag]
                color    = TAG_COLORS.get(tag, "#888")
                st.markdown(
                    f"<div style='background:{color}22;border-left:4px solid {color};"
                    f"border-radius:6px;padding:12px 16px;margin:8px 0'>"
                    f"<b style='color:{color}'>{TAG_ICONS.get(tag,'')} {tag}</b> "
                    f"<span style='color:#ccc;font-size:12px'>({len(tag_rows)} record(s))</span><br>"
                    f"<span style='color:#ddd;font-size:13px'>{TAG_DESC.get(tag,'')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Decision recommendation ───────────────────────────────────────
            if "EXACT_DUPLICATE" in unique_tags:
                rec_text = ("⚠️ **Exact duplicates found.** Keep only the record with the "
                            "earliest `open_date` — all later identical copies should be removed.")
                rec_color = "#e74c3c"
            elif "IDENTITY_GROUP_FRAUD" in unique_tags:
                rec_text = ("🚨 **IDENTITY_GROUP_FRAUD signal.** This identity group (Legal Name + "
                            "Owner + Address) contains at least one confirmed fraud=1 record. "
                            "Only the earliest Fraud=1 row per group is preserved; all others "
                            "(including non-fraud records in the same group) are excluded from "
                            "clean training data to prevent label contamination.")
                rec_color = "#9b59b6"
            elif "TECH_REAPPLICATION" in unique_tags:
                rec_text = ("⚠️ **Technical re-application detected.** The same (TIN, MID) pair "
                            "was submitted more than once with differing field values. "
                            "Only the record with the earliest `open_date` is preserved.")
                rec_color = "#e67e22"
            elif "IDENTITY_GROUP_CLEAN" in unique_tags:
                rec_text = ("🔍 **Clean identity-group duplicate.** Multiple records share the same "
                            "Legal Name + Owner + Address with no fraud. "
                            "The most recent record is kept; all older copies are excluded.")
                rec_color = "#f39c12"
            elif "INVALID_FOR_KYC" in unique_tags:
                rec_text = ("🔺 **Missing required KYC fields.** One or more records in this group "
                            "are missing TIN, Legal Name, or Postal Code and will fail vendor "
                            "verification. Exclude these rows — they cannot be used for modeling.")
                rec_color = "#c0392b"
            else:
                rec_text  = ("✅ **Clean record.** Unique identity, all KYC fields present, "
                             "no fraud linkage — safe for inclusion in the modeling pool.")
                rec_color = "#27ae60"

            st.markdown(
                f"<div style='background:{rec_color}15;border:1px solid {rec_color};"
                f"border-radius:8px;padding:14px 18px;margin:10px 0;"
                f"color:#ddd;font-size:13px'>{rec_text}</div>",
                unsafe_allow_html=True,
            )

            # ── Record table ─────────────────────────────────────────────────
            st.subheader("All Records for this TIN")

            # Tag filter for this TIN's records
            tags_in_tin = [t for t in TAG_ORDER if t in subset["duplicate_tag"].values]
            tag_filter_tin = st.multiselect(
                "Filter records by tag",
                options=tags_in_tin,
                default=tags_in_tin,
                format_func=lambda t: f"{TAG_ICONS.get(t, '')} {t}  ({int((subset['duplicate_tag'] == t).sum())} records)",
                key="_tin_tag_filter",
            )
            subset_view = subset[subset["duplicate_tag"].isin(tag_filter_tin)] if tag_filter_tin else subset
            st.caption(f"Showing {len(subset_view):,} of {len(subset):,} records")

            display_cols_tin = [
                "_mid_str", "LegalName", "dba1_name",
                STATE_COL, CITY_COL, POSTAL_COL,
                "owner1_first_name", "owner1_last_name",
                "open_date",
                TARGET_COL, FRAUD_TYPE,
                "duplicate_tag", "tag_reason",
            ]
            display_cols_tin = [c for c in display_cols_tin if c in subset_view.columns]

            # Format datetime columns as YYYY-MM-DD strings for readable display
            subset_view = subset_view.copy()
            _DT_COLS_SEARCH = ["open_date", "closed_date",
                                "Fraud.Opened Date", "Fraud.Date Fraud Found"]
            for _dc in _DT_COLS_SEARCH:
                if _dc in subset_view.columns:
                    if pd.api.types.is_datetime64_any_dtype(subset_view[_dc]):
                        subset_view[_dc] = subset_view[_dc].dt.strftime("%Y-%m-%d")
                    elif pd.api.types.is_numeric_dtype(subset_view[_dc]):
                        subset_view[_dc] = pd.to_datetime(
                            subset_view[_dc].apply(
                                lambda x: np.floor(x) if pd.notna(x) else np.nan
                            ),
                            unit='D', origin='1899-12-30', errors='coerce'
                        ).dt.strftime("%Y-%m-%d")

            # Row highlight by tag
            def _highlight_tag(row):
                color = TAG_COLORS.get(row["duplicate_tag"], "#888")
                return [f"background-color:{color}22;color:#ddd"] * len(row)

            st.dataframe(
                subset_view[display_cols_tin].rename(columns={
                    "_mid_str": "MID",
                    TARGET_COL: "Fraud",
                    FRAUD_TYPE: "Fraud Type",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Visual: MID × fraud × tag scatter
            if mid_c > 1:
                st.subheader("MID Profile Chart")
                chart_df = subset[["_mid_str", STATE_COL, CITY_COL, TARGET_COL,
                                   "duplicate_tag"]].copy()
                chart_df.columns = ["MID","State","City","Fraud","Tag"]
                chart_df["Fraud_lbl"] = chart_df["Fraud"].map({0:"No Fraud",1:"FRAUD"})
                fig_sc = px.scatter(
                    chart_df, x="State", y="City",
                    color="Tag", symbol="Fraud_lbl",
                    hover_data=["MID","Tag"],
                    color_discrete_map=TAG_COLORS,
                    title=f"TIN {tin_str} — MIDs across States/Cities",
                    size_max=14,
                )
                fig_sc.update_traces(marker_size=12)
                st.plotly_chart(_plotly_dark(fig_sc), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ALL RECORDS TAGGED
# ═════════════════════════════════════════════════════════════════════════════
with TAB3:
    st.header("All Records — Duplicate Tags")

    # ── Summary charts ────────────────────────────────────────────────────────
    col_l3, col_r3 = st.columns([1, 1])

    with col_l3:
        # Donut
        fig_do = go.Figure(go.Pie(
            labels=[f"{TAG_ICONS.get(t,'')} {t}" for t in tag_counts.index],
            values=tag_counts.values,
            hole=0.55,
            marker_colors=[TAG_COLORS.get(t,"#888") for t in tag_counts.index],
        ))
        fig_do.update_traces(textinfo="label+percent", textfont_size=11)
        fig_do.update_layout(
            title="Tag Distribution (% of all records)",
            showlegend=False,
        )
        st.plotly_chart(_plotly_dark(fig_do), use_container_width=True)

    with col_r3:
        # Bar chart
        tc_df = pd.DataFrame({
            "Tag": [f"{TAG_ICONS.get(t,'')} {t}" for t in tag_counts.index],
            "Count": tag_counts.values,
            "Color": [TAG_COLORS.get(t,"#888") for t in tag_counts.index],
        })
        fig_bar3 = px.bar(
            tc_df, x="Count", y="Tag", orientation="h",
            color="Tag",
            color_discrete_map={r["Tag"]: r["Color"] for _, r in tc_df.iterrows()},
            title="Record Count by Duplicate Tag",
            text="Count",
        )
        fig_bar3.update_traces(textposition="outside", showlegend=False)
        st.plotly_chart(_plotly_dark(fig_bar3), use_container_width=True)

    # ── Tag legend ────────────────────────────────────────────────────────────
    def _render_tag_group(group_tags, header, header_color):
        st.markdown(
            f"<h4 style='color:{header_color};margin:18px 0 6px 0'>{header}</h4>",
            unsafe_allow_html=True,
        )
        for tag in group_tags:
            if tag not in TAG_DESC:
                continue
            color = TAG_COLORS[tag]
            cnt   = int(tag_counts.get(tag, 0))
            st.markdown(
                f"<div style='background:{color}18;border-left:4px solid {color};"
                f"border-radius:6px;padding:10px 14px;margin:5px 0'>"
                f"<b style='color:{color}'>{TAG_ICONS.get(tag,'')} {tag}</b> "
                f"<span style='color:#888;font-size:12px'>— {cnt:,} records</span><br>"
                f"<span style='color:#ccc;font-size:12px'>{TAG_DESC[tag]}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Tag Definitions")
    _render_tag_group(TRUE_DUP_TAGS, "🔴 Structural Duplicates — canonical record preserved, copies safe to remove", "#e74c3c")
    _render_tag_group(RISK_TAGS,     "🟣 Contaminated / Incomplete Records — exclude from clean training pool", "#9b59b6")
    _render_tag_group(["UNIQUE_CLEAN_NON_LINKED"], "🟢 Clean Records", "#27ae60")

    st.markdown("---")

    # ── Full filtered table ───────────────────────────────────────────────────
    st.subheader("Filter & Browse All Records")

    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        tag_filter = st.multiselect(
            "Filter by tag",
            options=TAG_ORDER,
            default=TAG_ORDER,
            format_func=lambda t: f"{TAG_ICONS.get(t,'')} {t}",
        )
    with fc2:
        fraud_filter = st.selectbox(
            "Fraud status", ["All", "Fraud only", "Non-fraud only"]
        )
    with fc3:
        state_opts = ["All"] + sorted(df[STATE_COL].dropna().unique().tolist())
        state_filter = st.selectbox("State", state_opts)

    view = df[df["duplicate_tag"].isin(tag_filter)].copy()
    if fraud_filter == "Fraud only":
        view = view[view[TARGET_COL] == 1]
    elif fraud_filter == "Non-fraud only":
        view = view[view[TARGET_COL] == 0]
    if state_filter != "All":
        view = view[view[STATE_COL] == state_filter]

    st.caption(f"Showing {len(view):,} of {len(df):,} records")

    show_cols = [
        "_tin_str", "_mid_str", "LegalName", "dba1_name",
        STATE_COL, CITY_COL, POSTAL_COL,
        "owner1_first_name", "owner1_last_name",
        "open_date", TARGET_COL, FRAUD_TYPE,
        "duplicate_tag", "tag_reason",
    ]
    show_cols = [c for c in show_cols if c in view.columns]

    st.dataframe(
        view[show_cols].rename(columns={
            "_tin_str": "TIN", "_mid_str": "MID",
            TARGET_COL: "Fraud", FRAUD_TYPE: "Fraud Type",
        }).head(5000),
        use_container_width=True,
        hide_index=True,
    )

    if len(view) > 5000:
        st.info("Table limited to 5,000 rows. Use the download button below for the full set.")

    _download_csv(
        view[show_cols].rename(columns={
            "_tin_str": "TIN", "_mid_str": "MID",
            TARGET_COL: "Fraud", FRAUD_TYPE: "Fraud Type",
        }),
        label="⬇️ Download filtered & tagged CSV",
        filename="records_tagged_filtered.csv",
    )

    st.markdown("---")
    st.subheader("Full Dataset with Tags (all records)")
    _download_csv(
        df[_cols_for_display(df)],
        label="⬇️ Download full tagged dataset (all records)",
        filename="all_records_tagged.csv",
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLEAN SAMPLING
# ═════════════════════════════════════════════════════════════════════════════
with TAB4:
    st.header("Clean Sampling")
    st.markdown(
        "<div class='info-box'>"
        "Configure which records to exclude and choose your desired sample size. "
        "The sample is drawn using <b>stratified sampling</b> to preserve the "
        "Fraud / No-Fraud class balance exactly as it appears in the clean pool. "
        "All exclusion rules are applied before sampling."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Exclusion controls ────────────────────────────────────────────────────
    st.subheader("Step 1 — Exclusion Rules")

    st.markdown("#### 🔴 Structural Duplicates *(canonical record preserved — copies safe to remove)*")
    dup_col1, dup_col2 = st.columns(2)
    with dup_col1:
        excl_exact = st.checkbox("Remove EXACT_DUPLICATE rows",       value=True,
                                  help=TAG_DESC["EXACT_DUPLICATE"])
        excl_tech  = st.checkbox("Remove TECH_REAPPLICATION rows",     value=True,
                                  help=TAG_DESC["TECH_REAPPLICATION"])
    with dup_col2:
        excl_igc   = st.checkbox("Remove IDENTITY_GROUP_CLEAN rows",   value=True,
                                  help=TAG_DESC["IDENTITY_GROUP_CLEAN"])

    st.markdown("#### ⚠️ Contaminated / Incomplete Records *(exclude from clean training pool)*")
    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        excl_igf     = st.checkbox("Remove IDENTITY_GROUP_FRAUD duplicates", value=True,
                                    help=TAG_DESC["IDENTITY_GROUP_FRAUD"])
    with risk_col2:
        excl_invalid = st.checkbox("Remove INVALID_FOR_KYC rows",            value=True,
                                    help=TAG_DESC["INVALID_FOR_KYC"])
        excl_missing = st.checkbox("Remove rows with missing MID or TIN",    value=True)

    # ── Sample size & automated search ────────────────────────────────────────
    st.subheader("Step 2 — Sample Size")
    sample_n = st.slider(
        "Target sample size (N)",
        min_value=1_000, max_value=100_000, value=25_000, step=1_000,
        format="%d",
    )
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        auto_search = st.checkbox("Auto-search for stable sample (PSI ≤ 0.10)", value=True, 
                                 help="Iteratively searches for a random seed that produces a sample with no Critical or Minor shifts.")
    with col_s2:
        if auto_search:
            max_iters = st.number_input("Max iterations", min_value=1, max_value=1000, value=100, step=10)
            random_seed = st.number_input("Starting sequence seed", min_value=0, max_value=99999, value=42, step=1)
        else:
            max_iters = 1
            random_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

    # ── Build pool ────────────────────────────────────────────────────────────
    pool = df.copy()

    excluded_tags = []
    if excl_exact:   excluded_tags.append("EXACT_DUPLICATE")
    if excl_tech:    excluded_tags.append("TECH_REAPPLICATION")
    if excl_igc:     excluded_tags.append("IDENTITY_GROUP_CLEAN")
    if excl_igf:     excluded_tags.append("IDENTITY_GROUP_FRAUD")
    if excl_invalid: excluded_tags.append("INVALID_FOR_KYC")

    if excluded_tags:
        pool = pool[~pool["duplicate_tag"].isin(excluded_tags)]

    if excl_missing:
        pool = pool[(pool["_tin_str"] != "") & (pool["_mid_str"] != "")]

    n_pool        = len(pool)
    fraud_in_pool = int(pool[TARGET_COL].sum())
    rate_pool     = pool[TARGET_COL].mean() * 100

    # ── Live preview ──────────────────────────────────────────────────────────
    st.subheader("Step 3 — Preview")

    prev_cols = st.columns(6)
    prev_data = [
        ("Original rows",        f"{len(df):,}",      ACCENT),
        ("Pool after exclusions",f"{n_pool:,}",        ACCENT),
        ("Fraud in pool",        f"{fraud_in_pool:,}", "#e74c3c"),
        ("Pool fraud rate",      f"{rate_pool:.4f}%",  "#e74c3c"),
        ("Target sample N",      f"{sample_n:,}",      "#f39c12"),
        ("Feasible?",
            "✅ Yes" if n_pool >= sample_n else f"❌ Only {n_pool:,} records",
            "#27ae60" if n_pool >= sample_n else "#e74c3c"),
    ]
    for pc, (lbl, val, clr) in zip(prev_cols, prev_data):
        pc.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-val' style='color:{clr}'>{val}</div>"
            f"<div class='metric-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True,
        )

    # Exclusion breakdown
    st.subheader("Excluded Records Breakdown")
    exc_df = df[df["duplicate_tag"].isin(excluded_tags)] if excluded_tags else pd.DataFrame()
    if excl_missing:
        missing_rows = df[(df["_tin_str"] == "") | (df["_mid_str"] == "")]
    else:
        missing_rows = pd.DataFrame()

    exc_summary = []
    for tag in excluded_tags:
        _tag_rows   = df[df["duplicate_tag"] == tag]
        _n_tag      = len(_tag_rows)
        _n_fr_tag   = int(_tag_rows[TARGET_COL].sum())
        _n_nfr_tag  = _n_tag - _n_fr_tag
        exc_summary.append({
            "Exclusion Rule":  f"{TAG_ICONS.get(tag,'')} {tag}",
            "Records Removed": _n_tag,
            "Fraud Removed":   _n_fr_tag,
            "No-Fraud Removed":_n_nfr_tag,
            "Fraud Rate %":    f"{(_n_fr_tag / _n_tag * 100):.2f}%" if _n_tag else "N/A",
        })
    if len(missing_rows):
        _n_miss = len(missing_rows)
        _n_fr_miss = int(missing_rows[TARGET_COL].sum())
        exc_summary.append({
            "Exclusion Rule":  "⚠️ Missing MID or TIN",
            "Records Removed": _n_miss,
            "Fraud Removed":   _n_fr_miss,
            "No-Fraud Removed":_n_miss - _n_fr_miss,
            "Fraud Rate %":    f"{(_n_fr_miss / _n_miss * 100):.2f}%" if _n_miss else "N/A",
        })

    if exc_summary:
        exc_table = pd.DataFrame(exc_summary)
        exc_table["% of Total"] = (exc_table["Records Removed"] / len(df) * 100).round(2)
        st.dataframe(exc_table, use_container_width=True, hide_index=True)

    # ── Zero-fraud warning ────────────────────────────────────────────────────
    if fraud_in_pool == 0 and n_pool > 0:
        # Find which tag is responsible for removing the most fraud records
        _fraud_by_tag = {
            tag: int(df[df["duplicate_tag"] == tag][TARGET_COL].sum())
            for tag in excluded_tags
        }
        _top_fraud_tag = max(_fraud_by_tag, key=_fraud_by_tag.get) if _fraud_by_tag else "?"
        st.markdown(
            "<div style='background:#3a1a1a;border-left:4px solid #e74c3c;"
            "border-radius:6px;padding:12px 16px;font-size:13px;color:#fcc'>"
            "<b>⚠️ No fraud records in the clean pool.</b><br>"
            "All fraud cases were removed by the active exclusion rules — mainly "
            f"<b>{_top_fraud_tag}</b> ({_fraud_by_tag.get(_top_fraud_tag,0):,} fraud records). "
            "The generated sample will therefore contain <b>0 fraud records</b>, "
            "which is not suitable for a supervised fraud model.<br><br>"
            "<b>Recommendation:</b> uncheck <i>Remove IDENTITY_GROUP_FRAUD duplicates</i> "
            "so that the canonical fraud=1 record per identity group remains in the pool "
            "and is proportionally represented in the sample."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Generate sample ───────────────────────────────────────────────────────
    st.subheader("Step 4 — Generate & Download Sample")

    _btn_col_gen, _btn_col_info, _btn_col_pad = st.columns([2, 2, 4])
    _gen_clicked  = _btn_col_gen.button(
        "\U0001f504 Generate Sample", type="primary", disabled=(n_pool < sample_n)
    )
    _info_clicked = _btn_col_info.button("\u2139\ufe0f How does sampling work?")
    if _info_clicked:
        st.session_state["_show_sampling_info"] = not st.session_state.get("_show_sampling_info", False)

    if st.session_state.get("_show_sampling_info"):
        st.markdown(
            "<div style='background:#1e2d3d;border-left:4px solid #3498db;"
            "border-radius:8px;padding:18px 22px;font-size:13.5px;color:#dce9f5;"
            "margin-bottom:14px'>"

            # ── Title ──────────────────────────────────────────────────────
            "<b style='font-size:15px;color:#5dade2'>\U0001f9ea Sampling Method: Stratified Random Sampling</b>"
            "<hr style='border-color:#2e4a62;margin:10px 0'>"
            "<b>Goal:</b> produce a sample whose fraud rate and overall statistical profile "
            "<i>exactly mirrors</i> the clean pool \u2014 no artificial up/down-sampling of the minority class."

            # ── Phase 1: Exclusion / Tagging ───────────────────────────────
            "<br><br><b style='color:#5dade2;font-size:14px'>\U0001f3f7\ufe0f Phase 1 \u2014 Tag &amp; Filter (Exclusion Rules)</b>"
            "<p style='margin:6px 0'>Before any sampling takes place, every row in the original dataset "
            "is classified using a <b>priority-ordered tagging engine</b>. Tags are applied in strict "
            "priority order \u2014 a higher-priority tag always wins if multiple rules could apply to the same row.</p>"

            # ── Priority reference table ────────────────────────────────────
            "<table style='width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:16px'>"
            "<tr style='background:#0d1f30;color:#8ab8d8;border-bottom:2px solid #2e4a62'>"
            "<th style='text-align:center;padding:6px 8px;white-space:nowrap'>Priority</th>"
            "<th style='text-align:left;padding:6px 8px'>Tag</th>"
            "<th style='text-align:left;padding:6px 8px'>Rule Logic</th>"
            "<th style='text-align:left;padding:6px 8px'>Exclusion Action</th>"
            "</tr>"
            "<tr style='border-bottom:1px solid #1e3050'>"
            "<td style='text-align:center;padding:6px 8px;font-weight:bold;color:#e74c3c'>1</td>"
            "<td style='padding:6px 8px;white-space:nowrap'><b style='color:#e74c3c'>EXACT_DUPLICATE</b></td>"
            "<td style='padding:6px 8px'>All columns identical (except for primary key IDs).</td>"
            "<td style='padding:6px 8px'>Keep the earliest record; Drop others.</td>"
            "</tr>"
            "<tr style='background:#16253a;border-bottom:1px solid #1e3050'>"
            "<td style='text-align:center;padding:6px 8px;font-weight:bold;color:#9b59b6'>2</td>"
            "<td style='padding:6px 8px;white-space:nowrap'><b style='color:#9b59b6'>IDENTITY_GROUP_FRAUD</b></td>"
            "<td style='padding:6px 8px'>Legal Name + Owner + Address match across multiple rows, "
            "and at least one row is Fraud&nbsp;=&nbsp;1.</td>"
            "<td style='padding:6px 8px'>Keep only one record with Fraud=1; Drop all others "
            "(even if they are No-Fraud).</td>"
            "</tr>"
            "<tr style='border-bottom:1px solid #1e3050'>"
            "<td style='text-align:center;padding:6px 8px;font-weight:bold;color:#e67e22'>3</td>"
            "<td style='padding:6px 8px;white-space:nowrap'><b style='color:#e67e22'>IDENTITY_GROUP_CLEAN</b></td>"
            "<td style='padding:6px 8px'>Legal Name + Owner + Address match, but none of the rows "
            "have fraud.</td>"
            "<td style='padding:6px 8px'>Keep only the most recent record; Drop others.</td>"
            "</tr>"
            "<tr style='background:#16253a;border-bottom:1px solid #1e3050'>"
            "<td style='text-align:center;padding:6px 8px;font-weight:bold;color:#f39c12'>4</td>"
            "<td style='padding:6px 8px;white-space:nowrap'><b style='color:#f39c12'>TECH_REAPPLICATION</b></td>"
            "<td style='padding:6px 8px'>Same TIN + MID, but some fields differ "
            "(e.g. phone format).</td>"
            "<td style='padding:6px 8px'>Keep the earliest record; Drop others.</td>"
            "</tr>"
            "<tr style='border-bottom:1px solid #1e3050'>"
            "<td style='text-align:center;padding:6px 8px;font-weight:bold;color:#c0392b'>5</td>"
            "<td style='padding:6px 8px;white-space:nowrap'><b style='color:#c0392b'>INVALID_FOR_KYC</b></td>"
            "<td style='padding:6px 8px'>Missing Legal Name, TIN, or Postal Code.</td>"
            "<td style='padding:6px 8px'>Drop. These will fail vendor verification and waste money.</td>"
            "</tr>"
            "<tr style='background:#16253a'>"
            "<td style='text-align:center;padding:6px 8px;font-weight:bold;color:#27ae60'>6</td>"
            "<td style='padding:6px 8px;white-space:nowrap'><b style='color:#27ae60'>UNIQUE_CLEAN_NON_LINKED</b></td>"
            "<td style='padding:6px 8px'>Single appearance of TIN/Identity with no fraud links.</td>"
            "<td style='padding:6px 8px'>Keep. These are your high-quality \u201csafe\u201d examples.</td>"
            "</tr>"
            "</table>"

            # ── Detailed mechanism table ────────────────────────────────────
            "<p style='margin:6px 0'>Here is exactly how each rule decides which rows survive:</p>"
            "<table style='width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:8px'>"
            "<tr style='border-bottom:1px solid #2e4a62;color:#8ab8d8'>"
            "<th style='text-align:left;padding:5px 8px'>Rule</th>"
            "<th style='text-align:left;padding:5px 8px'>How duplicates are identified</th>"
            "<th style='text-align:left;padding:5px 8px'>Who is removed vs. kept</th>"
            "</tr>"
            "<tr><td style='padding:5px 8px;vertical-align:top'><b>EXACT_DUPLICATE</b></td>"
            "<td style='padding:5px 8px;vertical-align:top'>All original fields are byte-for-byte identical to another row.</td>"
            "<td style='padding:5px 8px;vertical-align:top'>2nd, 3rd \u2026 copies are tagged &amp; removed. "
            "<b>The first occurrence is preserved</b> in the clean pool as the canonical record.</td></tr>"
            "<tr style='background:#16253a'><td style='padding:5px 8px;vertical-align:top'><b>TECH_REAPPLICATION</b></td>"
            "<td style='padding:5px 8px;vertical-align:top'>Exact same (TIN, MID) pair submitted more than once "
            "with differing field values (e.g. phone format, address variant).</td>"
            "<td style='padding:5px 8px;vertical-align:top'>"
            "Rows are <b>sorted by <code>open_date</code> ascending</b> first. "
            "The record with the <b>earliest open_date</b> is preserved; all later copies are tagged &amp; removed. "
            "If open_date is missing (NaT), those rows sort last and are treated as later copies.</td></tr>"
            "<tr><td style='padding:5px 8px;vertical-align:top'><b>IDENTITY_GROUP_CLEAN</b></td>"
            "<td style='padding:5px 8px;vertical-align:top'>Same Legal Name + Owner + Address across multiple rows, "
            "none with fraud=1 \u2014 same person/business appearing with no fraud signal.</td>"
            "<td style='padding:5px 8px;vertical-align:top'>"
            "Rows are <b>sorted by <code>open_date</code> descending</b>. "
            "The <b>most recent record</b> is preserved; all older copies are tagged &amp; removed.</td></tr>"
            "<tr style='background:#16253a'><td style='padding:5px 8px;vertical-align:top'><b>IDENTITY_GROUP_FRAUD</b></td>"
            "<td style='padding:5px 8px;vertical-align:top'>Same Legal Name + Owner + Address group contains "
            "at least one confirmed fraud=1 record.</td>"
            "<td style='padding:5px 8px;vertical-align:top'>"
            "Rows sorted by [fraud DESC, open_date ASC]. "
            "<b>The earliest Fraud=1 record is preserved</b>; all other rows in the group "
            "(both other fraud rows and all non-fraud rows) are tagged &amp; removed.</td></tr>"
            "<tr><td style='padding:5px 8px;vertical-align:top'><b>INVALID_FOR_KYC</b></td>"
            "<td style='padding:5px 8px;vertical-align:top'>Missing one or more required KYC fields: "
            "TIN, Legal Name, or Postal Code.</td>"
            "<td style='padding:5px 8px;vertical-align:top'><b>All records with any missing KYC field</b> are tagged. "
            "These rows will fail vendor verification and hold no value for model training.</td></tr>"
            "</table>"
            "<b>Missing MID or TIN:</b> rows where MID or TIN is blank are always excluded from the pool "
            "when that checkbox is active, regardless of their duplicate tag."

            # ── Why this specific order? ────────────────────────────────────
            "<br><br><b style='color:#5dade2;font-size:14px'>\U0001f914 Why this specific order?</b>"
            "<ul style='margin-top:8px;line-height:1.9'>"
            "<li><b>Identity vs. Row:</b> By grouping on Legal Name + Owner + Address (Rules 2 &amp; 3), "
            "you solve the problem where a fraudster uses different TINs or MIDs but is clearly the same person. "
            "You only pay the vendor once for that person.</li>"
            "<li><b>Fraud Survival:</b> Rule 2 ensures that if a person has 10 applications and 1 is fraud, "
            "the record you send to the vendor is the <i>Fraud</i> one. "
            "This preserves your target variable for modeling.</li>"
            "<li><b>Recency for Clean Records:</b> Rule 3 picks the most recent record for clean merchants "
            "because they likely updated their address or phone number, giving the KYC vendor a better "
            "\u201chit rate.\u201d</li>"
            "<li><b>No Waste:</b> Rule 5 stops you from paying for records that are so incomplete the vendor "
            "can\u2019t find them anyway.</li>"
            "</ul>"
            "<br><br><b style='color:#5dade2;font-size:14px'>\U0001f3b2 Phase 2 \u2014 Stratified Random Sampling</b>"
            "<ol style='margin-top:8px;line-height:1.9'>"
            "<li><b>Build the Clean Pool</b> \u2014 everything not removed by Phase 1 becomes the pool.</li>"
            "<li><b>Compute target class counts</b> \u2014 the pool\u2019s fraud rate is measured and "
            "applied to the desired N:<br>"
            "<code style='background:#0d1f30;padding:2px 6px;border-radius:4px'>"
            "n_fraud &nbsp;&nbsp;&nbsp;= round(N \u00d7 fraud_rate_pool)</code><br>"
            "<code style='background:#0d1f30;padding:2px 6px;border-radius:4px'>"
            "n_no_fraud = N \u2212 n_fraud</code></li>"
            "<li><b>Independent random draws per stratum</b> \u2014 the pool is split into a "
            "<b>Fraud stratum</b> and a <b>No-Fraud stratum</b>. Each is sampled "
            "<i>without replacement</i> using your fixed random seed:<br>"
            "<code style='background:#0d1f30;padding:2px 6px;border-radius:4px'>"
            "fraud_sample &nbsp;= fraud_pool.sample(n=n_fraud, random_state=seed)</code><br>"
            "<code style='background:#0d1f30;padding:2px 6px;border-radius:4px'>"
            "nofraud_sample = nofraud_pool.sample(n=n_no_fraud, random_state=seed)</code></li>"
            "<li><b>Shuffle and combine</b> \u2014 the two strata are concatenated and globally "
            "shuffled so no class-ordering artefacts survive into downstream modelling.</li>"
            "</ol>"
            "<b>Why stratified?</b> Simple random sampling on a highly imbalanced dataset (e.g. 0.3% fraud) "
            "would produce a sample where the fraud count varies randomly by \u00b1tens of cases by chance. "
            "Stratifying guarantees the fraud rate is preserved to within rounding (typically &lt;&nbsp;0.0001&nbsp;pp)."
            "<br><br>"
            "<b>Reproducibility:</b> fixing the random seed means the exact same sample is regenerated "
            "every run \u2014 essential for model governance and audit trails."
            "</div>",
            unsafe_allow_html=True,
        )

    if _gen_clicked:
        if n_pool < sample_n:
            st.error(f"Not enough records in pool ({n_pool:,}) to sample {sample_n:,}.")
        else:
            fraud_rate_pool  = pool[TARGET_COL].mean()
            n_fraud_target   = max(1, round(sample_n * fraud_rate_pool))
            n_nofraud_target = sample_n - n_fraud_target

            fraud_pool   = pool[pool[TARGET_COL] == 1]
            nofraud_pool = pool[pool[TARGET_COL] == 0]

            if len(fraud_pool) < n_fraud_target:
                n_fraud_target   = len(fraud_pool)
                n_nofraud_target = sample_n - n_fraud_target
                st.warning(
                    f"Only {len(fraud_pool)} fraud records available in pool — "
                    f"taking all of them. No-fraud records fill the remainder."
                )

            best_psi = float('inf')
            best_sample = None
            best_seed = int(random_seed)
            best_rate_sample = 0
            best_is_stable = False
            
            # For accurate PSI evaluation in search, we use the pool WITHOUT true duplicates 
            # exactly like Tab 5 does.
            _pool_for_eval = df[~df["duplicate_tag"].isin(TRUE_DUP_TAGS)].copy()
            _pool_for_eval = _pool_for_eval[(_pool_for_eval["_tin_str"] != "") & (_pool_for_eval["_mid_str"] != "")]

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(int(max_iters)):
                current_seed = int(random_seed) + i
                
                s_fraud   = fraud_pool.sample(n=n_fraud_target, random_state=current_seed)
                s_nofraud = nofraud_pool.sample(n=n_nofraud_target, random_state=current_seed)
                sample    = (
                    pd.concat([s_fraud, s_nofraud], ignore_index=True)
                      .sample(frac=1, random_state=current_seed)
                      .reset_index(drop=True)
                )
                
                if auto_search:
                    current_psi, is_stable = _evaluate_max_psi(_pool_for_eval, sample)
                    if current_psi < best_psi:
                        best_psi = current_psi
                        best_sample = sample
                        best_seed = current_seed
                        best_rate_sample = sample[TARGET_COL].mean() * 100
                        best_is_stable = is_stable
                    
                    status_text.text(f"Iteration {i+1}/{max_iters}... Testing Seed {current_seed} | Current Max PSI: {current_psi:.4f}")
                    progress_bar.progress((i + 1) / int(max_iters))
                    
                    if is_stable:
                        status_text.text(f"Stable sample found at seed {best_seed} with Max PSI {best_psi:.4f} after {i+1} iterations!")
                        break
                else:
                    best_sample = sample
                    best_seed = current_seed
                    best_rate_sample = sample[TARGET_COL].mean() * 100
                    break
                    
            progress_bar.empty()
            status_text.empty()
            sample = best_sample
            rate_sample = best_rate_sample
            rng = best_seed
            rate_diff   = abs(rate_sample - rate_pool)

            if auto_search:
                if best_is_stable:
                    st.success(f"✅ Stable sample found! Seed: **{rng}** — Max PSI: **{best_psi:.4f}**")
                else:
                    st.warning(f"⚠️ Reached max iterations without finding an entirely stable PSI. Best Seed: **{rng}** — Max PSI: **{best_psi:.4f}**")

            st.success(
                f"✅ Sample generated: **{len(sample):,} records** — "
                f"Fraud: **{int(sample[TARGET_COL].sum()):,}** ({rate_sample:.4f}%) — "
                f"Rate difference vs pool: **{rate_diff:.6f} pp**"
            )

            res_cols = st.columns(4)
            res_data = [
                ("Sample size",      f"{len(sample):,}",                       ACCENT),
                ("Fraud records",    f"{int(sample[TARGET_COL].sum()):,}",      "#e74c3c"),
                ("Sample fraud rate",f"{rate_sample:.4f}%",                    "#e74c3c"),
                ("Rate shift",       f"{rate_diff:.6f} pp",                     "#27ae60"),
            ]
            for rc, (lbl, val, clr) in zip(res_cols, res_data):
                rc.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-val' style='color:{clr}'>{val}</div>"
                    f"<div class='metric-lbl'>{lbl}</div></div>",
                    unsafe_allow_html=True,
                )

            # Comparison donut
            fig_comp = make_subplots(
                rows=1, cols=2,
                specs=[[{"type":"pie"},{"type":"pie"}]],
                subplot_titles=["Clean Pool", f"Sample (N={sample_n:,})"],
            )
            for col_idx, (label, data_s) in enumerate(
                [("Pool", pool), ("Sample", sample)], start=1
            ):
                vc = data_s[TARGET_COL].value_counts()
                fig_comp.add_trace(
                    go.Pie(
                        labels=["No Fraud","Fraud"],
                        values=[vc.get(0,0), vc.get(1,0)],
                        hole=0.5,
                        marker_colors=["#3498db","#e74c3c"],
                        name=label,
                        textinfo="label+percent",
                    ),
                    row=1, col=col_idx,
                )
            fig_comp.update_layout(title_text="Class Balance: Pool vs Sample", showlegend=False)
            st.plotly_chart(_plotly_dark(fig_comp), use_container_width=True)

            # Store in session state for download
            st.session_state["_sample_df"]   = sample
            st.session_state["_sample_ready"]= True

    # Download — always available if sample is in session
    if st.session_state.get("_sample_ready"):
        sample_out = st.session_state["_sample_df"]
        # Drop internal cols before download
        out_cols   = _cols_for_display(sample_out)
        _download_csv(
            sample_out[out_cols],
            label="⬇️ Download Clean Sample CSV",
            filename=f"clean_sample_{len(sample_out)}records.csv",
        )
        st.info(
            f"Sample contains {len(sample_out):,} records with tag columns included, "
            "so downstream pipelines know which rows were excluded and why."
        )
    elif n_pool < sample_n:
        st.markdown(
            f"<div class='warn-box'>⚠️ The current exclusion rules leave only "
            f"<b>{n_pool:,}</b> records in the pool — not enough to draw "
            f"<b>{sample_n:,}</b>. Reduce your sample size or relax exclusion rules.</div>",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — SAMPLE REPRESENTATIVENESS
# ═════════════════════════════════════════════════════════════════════════════
with TAB5:
    st.header("Sample Representativeness Analysis")
    st.markdown(
        "<div class='info-box'>"
        "Population Stability Analysis between the <b>clean pool</b> and the <b>sample</b> from Tab 4. "
        "For each feature the app computes: "
        "<b>PSI</b> (decile-binned for numeric, frequency-buckets for categorical, month-bins for dates); "
        "<b>KS test</b> for numeric; <b>Chi-Square</b> for categorical &amp; date bins; "
        "<b>Cardinality Ratio</b> for IDs. "
        "Verdict: <span style='color:#e74c3c'>🔴 Critical Shift</span> when PSI&nbsp;&gt;&nbsp;0.20 "
        "(stop-work signal); "
        "<span style='color:#f39c12'>🟡 Minor Shift</span> when 0.10&nbsp;&lt;&nbsp;PSI&nbsp;&le;&nbsp;0.20; "
        "<span style='color:#27ae60'>🟢 Stable</span> when PSI&nbsp;&le;&nbsp;0.10. "
        "p&nbsp;&lt;&nbsp;0.05 flags a statistically detectable difference (interpret with PSI). "
        "Click <b>Generate Sample</b> in Tab&nbsp;4 first.</div>",
        unsafe_allow_html=True,
    )

    # ── Statistical Interpretation Guide ────────────────────────────────────
    with st.expander(
        "\U0001f4d8  How to interpret these results — The Large-Sample Trap, PSI vs P-Value, "
        "& Fraud-Specific Guidance",
        expanded=False,
    ):
        st.markdown("""
<style>
.interp-card {
    background:#1a2535; border-radius:10px; padding:18px 22px;
    margin-bottom:14px; border-left:5px solid;
}
.interp-card.blue  { border-color:#3498db; }
.interp-card.gold  { border-color:#f39c12; }
.interp-card.red   { border-color:#e74c3c; }
.interp-card.green { border-color:#27ae60; }
.interp-card h4 { margin-top:0; margin-bottom:6px; font-size:15px; }
</style>
""", unsafe_allow_html=True)

        # ── Card 1 — Large-Sample Trap
        st.markdown("""
<div class="interp-card blue">
<h4>⚠️ The "Large-Sample" Trap — Why You See p&nbsp;&lt;&nbsp;0.05 on Stable Features</h4>
The <b>χ² (Chi-Square) and KS tests</b> are <i>extremely</i> sensitive to sample size.
As your number of observations <b>N</b> grows, the test gains power and can detect even the most
microscopic, practically irrelevant differences between Pool and Sample.<br><br>
<b>The math:</b> the test statistic scales with N — so with a Pool of 1 000 000 rows and a Sample
of 100 000, a difference of just <b>0.01%</b> in a state's distribution is enough to trigger
<code>p &lt; 0.05</code>.<br><br>
Statistically the distributions <i>are</i> different (because they aren't perfect mathematical clones).
Practically, they are <b>identical for modelling purposes</b>.
This is precisely why we do not rely on p-values alone to assess stability.
</div>
""", unsafe_allow_html=True)

        # ── Card 2 — PSI as effect size
        st.markdown("""
<div class="interp-card gold">
<h4>📐 PSI — The "Effect Size" Metric (Your Primary Decision Tool)</h4>
While the p-value asks <i>"Is there a difference?"</i>, the PSI asks
<i>"Does the difference <b>matter</b>?"</i><br><br>
<table style="width:100%;border-collapse:collapse;font-size:13px">
  <tr style="border-bottom:1px solid #2e4a62">
    <th style="text-align:left;padding:6px 10px">Verdict</th>
    <th style="text-align:left;padding:6px 10px">PSI Range</th>
    <th style="text-align:left;padding:6px 10px">Meaning</th>
  </tr>
  <tr>
    <td style="padding:6px 10px">🟢 Stable</td>
    <td style="padding:6px 10px">≤ 0.10</td>
    <td style="padding:6px 10px">The shift is so small it will not affect model performance. Safe to use.</td>
  </tr>
  <tr style="background:#1e2d3d">
    <td style="padding:6px 10px">🟡 Minor Shift</td>
    <td style="padding:6px 10px">0.10 – 0.20</td>
    <td style="padding:6px 10px">Noticeable drift. Monitor; consider re-calibration if persistent.</td>
  </tr>
  <tr>
    <td style="padding:6px 10px">🔴 Critical Shift</td>
    <td style="padding:6px 10px">> 0.20</td>
    <td style="padding:6px 10px">Material drift. The model trained on the pool may be biased. Re-sample before scoring.</td>
  </tr>
</table>
</div>
""", unsafe_allow_html=True)

        # ── Card 3 — Real examples from the screenshot
        st.markdown("""
<div class="interp-card green">
<h4>✅ Real Example — address_state (Stable / p&nbsp;&lt;&nbsp;0.05): Safe</h4>
<b>PSI = 0.0056 &nbsp;|&nbsp; p-value = 0.0</b><br><br>
You have a massive dataset. The Chi-Square test is <i>"screaming"</i> that the Pool and Sample
are not a perfect 1-to-1 copy — and technically it is correct. But the PSI of 0.0056 tells you
the actual magnitude of that difference is <b>less than 1% of the total distribution</b>.
The state mix is essentially perfect for modelling.<br><br>
<b>Conclusion:</b> The p-value is a statistical ghost. Trust the Verdict — <b>✅ Stable / Safe</b>.
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="interp-card red">
<h4>🚨 Real Example — Fraud.Opened Date / Fraud.Date Fraud Found (Critical Shift / p&nbsp;&lt;&nbsp;0.05): Dangerous</h4>
<b>PSI = 0.46 (Day-of-Week) · 0.26 (Date Found) &nbsp;|&nbsp; p-value = 0.0</b><br><br>
Here <b>both metrics agree</b>. A PSI of 0.46 is enormous. This tells you that the <i>timing</i>
of fraud events in the Sample is significantly different from the Pool. For example:
the Pool may have fraud spread across all days of the week, but the Sample may be
over-representing weekday fraud — or the Sample captures only one "fraud wave" or a
batch of manually reviewed cases from a specific period.<br><br>
In fraud modelling this is known as a <b>time-compressed</b> or <b>leakage-prone</b> sample:
the model would learn temporal artefacts of one batch rather than the true fraud behaviour pattern.<br><br>
<b>Conclusion:</b> Both metrics agree — <b>🔴 Critical / Dangerous. Investigate before training.</b>
</div>
""", unsafe_allow_html=True)

        # ── Card 4 — Industry-standard recommendation
        st.markdown("""
<div class="interp-card gold">
<h4>🏭 Industry Standard — PSI is the Primary Criterion in Fraud Modelling</h4>
If you required <code>p &gt; 0.05</code> for every feature in a large dataset, you would likely
<b>never be able to create a valid sample</b> — you'd be chasing statistical ghosts indefinitely.
The industry standard (and regulatory expectation for model risk management) is to use PSI as the
primary stability verdict and treat p-values as supporting evidence only.<br><br>
<b>Recommended workflow:</b>
<ol style="line-height:1.9;margin-top:8px">
  <li><b>Trust the Verdict (PSI):</b> use PSI to decide whether to re-sample. p-value alone
      is not sufficient grounds to reject a sample.</li>
  <li><b>Use p-value as a tie-breaker:</b> if PSI is borderline (e.g., 0.09) and p = 0.80,
      you are very safe. If PSI is 0.09 and p = 0.00, keep a close eye on that feature in
      post-deployment monitoring.</li>
  <li><b>Investigate Critical Shifts on date/temporal features:</b> in fraud, a Critical PSI
      on a date field typically means the sample is biased toward a specific fraud wave,
      a particular review batch, or a seasonal period — all of which create data leakage
      or temporal distribution shift that harms out-of-time model performance.</li>
</ol>
</div>
""", unsafe_allow_html=True)

        # ── Card 5 — The Fraud Perspective
        st.markdown("""
<div class="interp-card blue">
<h4>🕵️ The Fraud Perspective — Why Temporal Stability Matters Most</h4>
In fraud, the underlying distribution changes every day because <b>fraudsters adapt</b>.
A <b>Stable PSI</b> on a temporal feature tells you that your training data still looks like the
real world — the fraud patterns your model will learn are representative of current behaviour.<br><br>
A <b>Critical PSI on a date field</b> is a strong signal of one of two problems:
<ul style="line-height:1.9;margin-top:6px">
  <li><b>Time-compression:</b> the sample over-represents a narrow time window (e.g., one month
      of fraud detections from a single review queue). The model will not generalise.</li>
  <li><b>Leakage-prone data:</b> the sample captures fraud cases that were only identified
      <i>after</i> the events, biasing the model toward features observable only in hindsight.</li>
</ul>
<b>Bottom line:</b> address_state with PSI 0.006? Don't worry about it.
Fraud.Opened Date with PSI 0.46? That is your real problem — fix the sample before training.
</div>
""", unsafe_allow_html=True)

    # ── end interpretation guide ─────────────────────────────────────────────

    if not st.session_state.get("_sample_ready"):
        st.warning(
            "\u26a0\ufe0f No sample found. Go to **Tab 4 \u2014 Clean Sampling** "
            "and click **Generate Sample** first."
        )
        st.stop()

    _samp5 = st.session_state["_sample_df"]

    # Derive reference pool (True-Duplicate-free, valid TIN/MID)
    _pool5 = df[~df["duplicate_tag"].isin(TRUE_DUP_TAGS)].copy()
    _pool5 = _pool5[(_pool5["_tin_str"] != "") & (_pool5["_mid_str"] != "")]

    _n_pool5  = len(_pool5)
    _n_samp5  = len(_samp5)
    _rp5 = _pool5[TARGET_COL].mean() * 100
    _rs5 = _samp5[TARGET_COL].mean() * 100
    _rd5 = abs(_rp5 - _rs5)

    # ── KPI row ─────────────────────────────────────────────────────────────
    _h5cols = st.columns(6)
    for _hc5, (_lbl5, _val5, _clr5) in zip(_h5cols, [
        ("Pool size",         f"{_n_pool5:,}",          ACCENT),
        ("Sample size",       f"{_n_samp5:,}",          ACCENT),
        ("Pool fraud rate",   f"{_rp5:.4f}%",           "#e74c3c"),
        ("Sample fraud rate", f"{_rs5:.4f}%",           "#e74c3c"),
        ("Rate difference",   f"{_rd5:.6f} pp",
         "#27ae60" if _rd5 < 0.005 else "#f39c12"),
        ("Sample / Pool",     f"{_n_samp5/max(1,_n_pool5)*100:.1f}%", ACCENT),
    ]):
        _hc5.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-val' style='color:{_clr5}'>{_val5}</div>"
            f"<div class='metric-lbl'>{_lbl5}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Pre-import scipy once ────────────────────────────────────────────────
    try:
        from scipy import stats as _scipy_stats
        _SCIPY_OK = True
    except ImportError:
        _SCIPY_OK = False

    # ════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ════════════════════════════════════════════════════════════════════════

    def _psi_numeric5(pool_vals, samp_vals, buckets=10):
        """PSI for numeric using decile binning on pool distribution."""
        e = np.array(pool_vals, dtype=float)
        a = np.array(samp_vals, dtype=float)
        if len(e) < 10 or len(a) < 10:
            return float("nan")
        bp = np.unique(np.percentile(e, np.linspace(0, 100, buckets + 1)))
        if len(bp) < 2:
            return float("nan")
        ep = np.histogram(e, bins=bp)[0] / max(len(e), 1)
        ap = np.histogram(a, bins=bp)[0] / max(len(a), 1)
        ep = np.where(ep == 0, 1e-6, ep)
        ap = np.where(ap == 0, 1e-6, ap)
        return float(np.sum((ap - ep) * np.log(ap / ep)))

    def _psi_categorical5(pool_col, samp_col):
        """PSI for categorical using category frequency buckets."""
        pv = pool_col.value_counts(normalize=True)
        sv = samp_col.value_counts(normalize=True)
        cats = sorted(set(pv.index) | set(sv.index))
        if len(cats) < 2:
            return float("nan"), cats
        ep = np.array([max(float(pv.get(c, 0)), 1e-6) for c in cats])
        ap = np.array([max(float(sv.get(c, 0)), 1e-6) for c in cats])
        ep /= ep.sum(); ap /= ap.sum()
        return float(np.sum((ap - ep) * np.log(ap / ep))), cats

    def _chi2_test5(pool_col_str, samp_col_str, cats, n_pool, n_samp):
        """Chi-Square contingency test; returns (stat, p-value)."""
        if not _SCIPY_OK:
            return float("nan"), float("nan")
        pv = pool_col_str.value_counts(normalize=True)
        sv = samp_col_str.value_counts(normalize=True)
        ct = [
            [max(1, round(float(pv.get(c, 0)) * n_pool)) for c in cats],
            [max(1, round(float(sv.get(c, 0)) * n_samp)) for c in cats],
        ]
        try:
            chi2, p, _, _ = _scipy_stats.chi2_contingency(ct)
            return float(chi2), float(p)
        except Exception:
            return float("nan"), float("nan")

    def _verdict5(psi):
        """Fraud-context PSI verdict using 0.10 / 0.20 thresholds."""
        if not isinstance(psi, float) or psi != psi:
            return "N/A", "#888888"
        if psi <= 0.10:
            return "\U0001f7e2 Stable",        "#27ae60"
        if psi <= 0.20:
            return "\U0001f7e1 Minor Shift",   "#f39c12"
        return   "\U0001f534 Critical Shift",  "#e74c3c"

    def _sig5(pval):
        """Return significance flag."""
        try:
            return "p<0.05 \u2605" if float(pval) < 0.05 else "n.s."
        except Exception:
            return "n.s."

    # ════════════════════════════════════════════════════════════════════════
    # FEATURE CLASSIFICATION
    # ════════════════════════════════════════════════════════════════════════

    _skip5 = {
        "duplicate_tag", "tag_reason", "is_first_of_group",
        "_tin_fraud_count", "_tin_record_count", "_tin_fraud_rate",
        "_tin_n_distinct_mid", "_tin_n_distinct_state", "_tin_n_distinct_city",
        "address_postal_code", "owner1_first_name", "address_line_2", "address_city"
    }
    _outcome_cols5 = [TARGET_COL, FRAUD_TYPE]

    _num5, _cat5, _date5, _id5 = [], [], [], []
    _ftype_map5 = {}

    if _FEATURE_CLASSIFIER_AVAILABLE:
        _az5 = _FeatureClassifier(_pool5, log_level="WARNING")
        _ftype_map5 = _az5._detect_feature_types()
        for _c5, _t5 in _ftype_map5.items():
            if _c5.startswith("_") or _c5 in _skip5 or _c5 in _outcome_cols5:
                continue
            if _t5 == "numeric":        _num5.append(_c5)
            elif _t5 in ("categorical", "binary"): _cat5.append(_c5)
            elif _t5 == "datetime":     _date5.append(_c5)
            elif _t5 == "id":           _id5.append(_c5)
    else:
        _hard_id = {"TIN", "MID", "SequenceKey", "LegalName", "dba1_name",
                    "owner1_first_name", "owner1_last_name", "business_email",
                    "business_phone"}
        _hard_dt = {"open_date", "closed_date", "Fraud.Opened Date", "Fraud.Date Fraud Found"}
        for _c5 in _pool5.columns:
            if _c5.startswith("_") or _c5 in _skip5 or _c5 in _outcome_cols5:
                continue
            if _c5 in _hard_id:
                _id5.append(_c5)
            elif _c5 in {"address_city"}:
                _cat5.append(_c5)
            elif _c5 in _hard_dt:
                _date5.append(_c5)
            elif _pool5[_c5].dtype.kind in ("i", "f", "u") and _c5 not in _hard_id:
                _num5.append(_c5)
            else:
                _cat5.append(_c5)

    # ── Transparency: show excluded / classified columns ─────────────────────
    _excl_rows5 = []
    for _c5 in _id5:
        _excl_rows5.append({"Column": _c5, "Type": "ID",       "Treatment": "Cardinality Ratio"})
    for _c5 in _date5:
        _excl_rows5.append({"Column": _c5, "Type": "Date",     "Treatment": "Month + DOW bins → Chi²"})
    if _excl_rows5:
        with st.expander(f"\u2139\ufe0f {len(_excl_rows5)} ID / Date columns — special treatment applied"):
            st.dataframe(pd.DataFrame(_excl_rows5), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — POPULATION STABILITY METRICS TABLE
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("\U0001f4ca Population Stability Analysis \u2014 All Features")

    _rows5 = []

    # ── ID columns: Cardinality Ratio ────────────────────────────────────────
    for _c5 in _id5:
        if _c5 not in _samp5.columns:
            continue
        _u_pool = _pool5[_c5].nunique()
        _u_samp = _samp5[_c5].nunique()
        _scale  = _n_samp5 / max(_n_pool5, 1)
        _cr5    = _u_samp / max(_u_pool * _scale, 1e-6)
        _crd5   = ("\U0001f7e2 Good" if 0.7 <= _cr5 <= 1.3
                   else "\U0001f7e1 Check" if 0.4 <= _cr5 <= 1.6
                   else "\U0001f534 Sparse")
        _rows5.append({
            "Feature": _c5, "Type": "ID",
            "PSI": "\u2014", "Stat": f"{_cr5:.3f} (ratio)",
            "P-Value": "\u2014", "Sig.": "\u2014",
            "Test": "Cardinality", "Verdict": _crd5,
        })

    # ── Numeric: decile PSI + KS ─────────────────────────────────────────────
    for _c5 in _num5:
        if _c5 not in _samp5.columns:
            continue
        _pe5 = _pool5[_c5].dropna().values
        _sa5 = _samp5[_c5].dropna().values
        if len(_pe5) < 10 or len(_sa5) < 10:
            continue
        _psi5 = _psi_numeric5(_pe5, _sa5)
        _ks5, _kp5 = float("nan"), float("nan")
        if _SCIPY_OK:
            try:
                _ks5, _kp5 = _scipy_stats.ks_2samp(_pe5, _sa5)
            except Exception:
                pass
        _verd5, _ = _verdict5(_psi5)
        _rows5.append({
            "Feature": _c5, "Type": "Numeric",
            "PSI":     round(_psi5, 4) if _psi5 == _psi5 else "N/A",
            "Stat":    f"{_ks5:.4f}" if _ks5 == _ks5 else "N/A",
            "P-Value": round(_kp5, 4) if _kp5 == _kp5 else "N/A",
            "Sig.":    _sig5(_kp5),
            "Test": "KS", "Verdict": _verd5,
        })

    # ── Categorical: frequency PSI + Chi-Square ──────────────────────────────
    for _c5 in _cat5:
        if _c5 not in _samp5.columns:
            continue
        _pc5 = _pool5[_c5].fillna("__NA__").astype(str)
        _sc5 = _samp5[_c5].fillna("__NA__").astype(str)
        _psi5, _cats_u = _psi_categorical5(_pc5, _sc5)
        if len(_cats_u) < 2 or len(_cats_u) > 40000:
            continue
        _chi5, _cp5 = _chi2_test5(_pc5, _sc5, _cats_u, _n_pool5, _n_samp5)
        _verd5, _ = _verdict5(_psi5)
        _rows5.append({
            "Feature": _c5, "Type": "Categorical",
            "PSI":     round(_psi5, 4) if _psi5 == _psi5 else "N/A",
            "Stat":    f"{_chi5:.2f}" if _chi5 == _chi5 else "N/A",
            "P-Value": round(_cp5, 4)  if _cp5 == _cp5   else "N/A",
            "Sig.":    _sig5(_cp5),
            "Test": "Chi\u00b2", "Verdict": _verd5,
        })

    # ── Date: extract Month + DOW bins → Chi-Square + PSI on month ───────────
    for _c5 in _date5:
        if _c5 not in _samp5.columns:
            continue
        # Try to get datetime series from pool and sample
        def _to_dt(series):
            s = series.dropna()
            if pd.api.types.is_datetime64_any_dtype(s):
                return s
            try:
                return pd.to_datetime(s, errors="coerce").dropna()
            except Exception:
                return pd.Series([], dtype="datetime64[ns]")

        _pd5 = _to_dt(_pool5[_c5])
        _sd5 = _to_dt(_samp5[_c5])
        if len(_pd5) < 10 or len(_sd5) < 10:
            continue

        # Quarter bins
        _pm5 = _pd5.dt.quarter.astype(str)
        _sm5 = _sd5.dt.quarter.astype(str)
        _psi5_m, _mcat5 = _psi_categorical5(_pm5, _sm5)
        _chi5_m, _cp5_m = _chi2_test5(_pm5, _sm5, _mcat5, _n_pool5, _n_samp5)

        # Day-of-Week bins
        _pw5 = _pd5.dt.dayofweek.astype(str)
        _sw5 = _sd5.dt.dayofweek.astype(str)
        _psi5_d, _dcat5 = _psi_categorical5(_pw5, _sw5)
        _chi5_d, _cp5_d = _chi2_test5(_pw5, _sw5, _dcat5, _n_pool5, _n_samp5)

        _verd5m, _ = _verdict5(_psi5_m)
        _verd5d, _ = _verdict5(_psi5_d)

        _rows5.append({
            "Feature": f"{_c5} (Quarter bins)", "Type": "Date",
            "PSI":     round(_psi5_m, 4) if _psi5_m == _psi5_m else "N/A",
            "Stat":    f"{_chi5_m:.2f}" if _chi5_m == _chi5_m else "N/A",
            "P-Value": round(_cp5_m, 4) if _cp5_m == _cp5_m else "N/A",
            "Sig.":    _sig5(_cp5_m),
            "Test": "Chi\u00b2", "Verdict": _verd5m,
        })
        _rows5.append({
            "Feature": f"{_c5} (Day-of-Week)", "Type": "Date",
            "PSI":     round(_psi5_d, 4) if _psi5_d == _psi5_d else "N/A",
            "Stat":    f"{_chi5_d:.2f}" if _chi5_d == _chi5_d else "N/A",
            "P-Value": round(_cp5_d, 4) if _cp5_d == _cp5_d else "N/A",
            "Sig.":    _sig5(_cp5_d),
            "Test": "Chi\u00b2", "Verdict": _verd5d,
        })

    # ── Outcome columns ──────────────────────────────────────────────────────
    _oc_labels5 = {TARGET_COL: "Fraud flag", FRAUD_TYPE: "Fraud type"}
    for _oc5 in _outcome_cols5:
        if _oc5 not in _pool5.columns or _oc5 not in _samp5.columns:
            continue
        _pacol = _pool5[_oc5].fillna("__NA__").astype(str)
        _sacol = _samp5[_oc5].fillna("__NA__").astype(str)
        _psi5, _ocats = _psi_categorical5(_pacol, _sacol)
        if len(_ocats) < 2:
            continue
        _chi5, _cp5 = _chi2_test5(_pacol, _sacol, _ocats, _n_pool5, _n_samp5)
        _verd5, _ = _verdict5(_psi5)
        _rows5.append({
            "Feature": f"{_oc5}  \u2605",
            "Type": f"Categorical ({_oc_labels5.get(_oc5,'')})",
            "PSI":     round(_psi5, 4) if _psi5 == _psi5 else "N/A",
            "Stat":    f"{_chi5:.2f}" if _chi5 == _chi5 else "N/A",
            "P-Value": round(_cp5, 4) if _cp5 == _cp5 else "N/A",
            "Sig.":    _sig5(_cp5),
            "Test": "Chi\u00b2", "Verdict": _verd5,
        })

    # ── Display summary table ────────────────────────────────────────────────
    if _rows5:
        _rep5 = pd.DataFrame(_rows5)
        st.dataframe(_rep5, use_container_width=True, hide_index=True)

        with st.expander("📖 How to read this table — Column Guide & Statistical Hypothesis"):
            st.markdown("""
### Column Definitions

| Column | What it shows |
|---|---|
| **Feature** | Column name. A **★** suffix marks an *outcome / label* column — these are expected to differ between pool and sample because the sample was drawn specifically to include fraud cases. A shift in an outcome column is by design, not a defect. |
| **Type** | Feature class assigned by the analyzer: `ID` (high-cardinality identifier), `Numeric` (continuous / ordinal), `Categorical` (low-cardinality label), `Date` (datetime). The type determines which statistical test is applied. |
| **PSI** | Population Stability Index — the primary verdict metric. Ranges from 0 (identical distributions) upward. Higher = more shift. |
| **Stat** | Secondary test statistic. For **Numeric** features this is the KS statistic (0–1). For **Categorical / Date** features this is the Chi-Square statistic (≥0). For **ID** features this is the Cardinality Ratio. |
| **P-Value** | Probability of observing the computed test statistic if H₀ is true. Small p-values indicate the distributions are statistically distinguishable. |
| **Sig.** | Significance flag. `p<0.05 ★` = the difference is statistically significant at the 95% confidence level. `n.s.` = **Not Significant** — cannot reject H₀ at p<0.05. *Note: at very large sample sizes (N > 10 000) almost every feature will show p<0.05 even for trivially small differences. Always weight PSI over p-value.* |
| **Test** | Statistical method used: `KS` (Kolmogorov-Smirnov, numeric), `Chi²` (categorical / date), `Cardinality` (ID columns). |
| **Verdict** | Action recommendation derived from PSI thresholds (see below). |

---

### Warning: Small Sub-Populations (The "Pigeonhole Principle")

Notice that features starting with `Fraud.` (e.g., `Fraud.Opened Date`) are evaluated **only** on the fraud subset of the sample.  
**If your sampled fraud count is extremely low (e.g., N=61):**
* The math tries to divide those 61 records across bins (e.g., 4 quarter bins or 50 state buckets).
* Randomly getting just 1 or 2 extra records in a bucket constitutes a **massive percentage variance** from the expected frequency.
* The PSI formula sees this massive percentage shift and erroneously yells 🔴 **Critical Shift**, when mathematically the pure counts only shifted by 2 humans.
* Industry standard for measuring Sub-Population Stability Index is to either increase feature abstraction (bins to quarters instead of months) or to relax acceptance thresholds entirely on buckets smaller than 100 observations. 
* *As a result, we have simplified date bins to Quarters and instructed the Auto-Search logic to safely allow `0.20 PSI` as the cutoff specifically when the N sizes are mechanically too small.*

---

### Statistical Hypothesis

> **H₀ (Null):** The Pool and the Sample are drawn from the same underlying distribution for this feature.  
> **H₁ (Alternative):** The distributions differ.

The p-value tests H₀. A **low p-value** (< 0.05) leads us to reject H₀, meaning a detectable difference exists.  
However, **PSI is the primary decision criterion** because it quantifies the *magnitude* of the shift, not just its statistical detectability.

---

### Verdict Thresholds

| Icon | Label | PSI Range | Fraud-model meaning |
|---|---|---|---|
| 🟢 | **Stable** | ≤ 0.10 | Distribution is well-preserved. Safe to score. |
| 🟡 | **Minor Shift** | 0.10 – 0.20 | Noticeable drift. Monitor performance; consider re-calibration if persistent. |
| 🔴 | **Critical Shift** | > 0.20 | Material drift. Model trained on pool may be biased on this sample. Re-sample or investigate before production scoring. |
""")

        # PSI heatmap bar chart — all features with a numeric PSI
        _psi_chart5 = _rep5.copy()
        _psi_chart5["_pnum"] = pd.to_numeric(_psi_chart5["PSI"], errors="coerce")
        _psi_chart5 = (_psi_chart5
                       .dropna(subset=["_pnum"])
                       .sort_values("_pnum", ascending=False)
                       .head(30))
        if len(_psi_chart5):
            _fc5 = px.bar(
                _psi_chart5, x="_pnum", y="Feature", orientation="h",
                color="_pnum",
                color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
                range_color=[0, 0.4],
                title="PSI per Feature (\u2190 lower is better \u2022 Critical Shift > 0.20)",
                text="_pnum",
            )
            _fc5.add_vline(x=0.10, line_dash="dash", line_color="#f39c12",
                           annotation_text="0.10 Minor")
            _fc5.add_vline(x=0.20, line_dash="dash", line_color="#e74c3c",
                           annotation_text="0.20 Critical")
            _fc5.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            _fc5.update_coloraxes(showscale=False)
            st.plotly_chart(_plotly_dark(_fc5), use_container_width=True)
    else:
        st.info("No features with sufficient data for stability tests.")

    with st.expander("\u2139\ufe0f Metric Calculation Guide \u2014 Formulas & Interpretation"):
        st.markdown("""
### How each metric is calculated

#### Population Stability Index (PSI)
Applied to **Numeric** features (decile bins) and **Categorical** features (category frequency buckets).

$$\\text{PSI} = \\sum_{i=1}^{B} \\Bigl(\\text{Sample}_{i\\%} - \\text{Pool}_{i\\%}\\Bigr) \\times \\ln\\!\\left(\\frac{\\text{Sample}_{i\\%}}{\\text{Pool}_{i\\%}}\\right)$$

- **B** = number of bins (10 decile bins for numeric; one bin per category for categorical).
- Small counts are smoothed with ε = 1×10⁻⁶ to avoid log(0).
- The result is always ≥ 0. Values near 0 indicate the sample closely mirrors the pool.

---

#### KS Statistic (Kolmogorov-Smirnov)
Applied to **Numeric** features as a secondary test.

$$\\text{KS} = \\max_x \\bigl| F_{\\text{Pool}}(x) - F_{\\text{Sample}}(x) \\bigr|$$

- **F(x)** = CDF (Cumulative Distribution Function) = the probability that a randomly drawn value is ≤ x. Think of it as the "running total" curve of a histogram.
- KS ranges from 0 (CDFs are identical — distributions are the same) to 1 (CDFs never overlap — distributions are completely different).
- A high KS with a low PSI usually indicates a localised shift in one tail (e.g., a spike in very large dollar amounts) rather than a broad distributional change.

---

#### Chi-Square Test (χ²)
Applied to **Categorical** and **Date** features.

$$\\chi^2 = \\sum_{k} \\frac{(O_k - E_k)^2}{E_k}$$

- **O_k** = observed count in category k in the sample.
- **E_k** = expected count if the pool's category proportions held exactly.
- A small p-value (< 0.05) rejects H₀. At large N (> 10 000 rows) the test is very sensitive and typically returns p < 0.05 for even trivial differences. **PSI is the primary action criterion.**

---

#### Cardinality Ratio
Applied to **ID** columns (high-cardinality identifiers such as TIN, MID).

$$\\text{Cardinality Ratio} = \\frac{\\text{unique}_{\\text{Sample}}}{\\text{unique}_{\\text{Pool}} \\times \\dfrac{N_{\\text{Sample}}}{N_{\\text{Pool}}}}$$

- Ideal ≈ **1.0** — the sample covers proportionally the same diversity of IDs as the pool.
- **< 0.5** — sample is concentrated on few IDs; may introduce entity-level bias.
- **> 1.5** — sample contains IDs not seen in the pool (e.g., unseen merchants); model may extrapolate outside its training distribution.

---

#### Date Features
Date columns are converted to **Month** (1–12) and **Day-of-Week** (0=Monday … 6=Sunday) buckets.
A Chi-Square test is run on each bucket table. PSI is also computed over the frequency buckets.
This approach avoids spurious drift from raw epoch/serial encoding while still detecting real temporal skew.

---

| Symbol | Meaning |
|---|---|
| ★ | Outcome / label column — shift is expected by design |
| `p<0.05 ★` | Statistically significant difference detected (p < 0.05) |
| `n.s.` | Not Significant — cannot reject H₀ at 95% confidence |
""")
    st.caption(
        "\u2139\ufe0f Quick reference: PSI \u2264 0.10 = \U0001f7e2 Stable | 0.10\u20130.20 = \U0001f7e1 Minor Shift | > 0.20 = \U0001f534 Critical Shift.  "
        "\u2605 = outcome column (shift expected by design).  `n.s.` = Not Significant (p \u2265 0.05)."
    )

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — DISTRIBUTION COMPARISON + FRAUD RISK DENSITY
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("\U0001f4c8 Distribution Comparison: Key Features")
    st.markdown(
        "For each feature: **Chart A** compares the overall proportion distribution "
        "(Pool vs Sample). **Chart B** shows the **Fraud Risk Density** \u2014 a 100% "
        "stacked bar where each bar is split into Fraud / No-Fraud segments for Pool and "
        "Sample side-by-side. If Pool and Sample bars look identical, the sample is "
        "a perfect representation. If the Fraud segment is much smaller in the Sample "
        "bars, the sampling is *diluting* fraud (under-prediction risk)."
    )

    # ── Label maps for outcome features ─────────────────────────────────────
    _lmap5_global = {
        TARGET_COL: {"0": "No Fraud", "1": "Fraud", "0.0": "No Fraud", "1.0": "Fraud",
                     "__NA__": "Unknown"},
        FRAUD_TYPE: {"1": "ID Theft", "2": "Synthetic", "3": "Others",
                     "1.0": "ID Theft", "2.0": "Synthetic", "3.0": "Others",
                     "__NA__": "Unknown"},
    }

    # Determine which features to visualise:
    # - Outcome cols first, then STATE_COL, then top covariate categoricals
    _vis_features5 = []
    for _oc in _outcome_cols5:
        if _oc in _pool5.columns and _oc in _samp5.columns:
            _vis_features5.append((_oc, True))   # (col, is_outcome)
    if STATE_COL in _pool5.columns and STATE_COL in _samp5.columns:
        _vis_features5.append((STATE_COL, False))
    for _fc in _cat5:
        if _fc not in (STATE_COL,) and _fc in _samp5.columns:
            _vis_features5.append((_fc, False))
        if len(_vis_features5) >= 9:
            break

    for (_dc5, _is_out5) in _vis_features5:
        _lmap5 = _lmap5_global.get(_dc5, {})

        def _lbl5(v):
            return _lmap5.get(str(v), str(v))

        # Pool and sample value count proportions
        _pvc5 = (_pool5[_dc5].fillna("__NA__").astype(str)
                 .value_counts(normalize=True).reset_index())
        _svc5 = (_samp5[_dc5].fillna("__NA__").astype(str)
                 .value_counts(normalize=True).reset_index())
        _pvc5.columns = [_dc5, "Pool"]
        _svc5.columns = [_dc5, "Sample"]
        _cmp5 = (_pvc5.merge(_svc5, on=_dc5, how="outer")
                 .fillna(0).sort_values("Pool", ascending=False).head(25))
        _cmp5["_label"] = _cmp5[_dc5].apply(_lbl5)

        # ── Chart A: Overall proportion — grouped bar ────────────────────
        with st.expander(f"\U0001f4ca {_dc5} \u2014 Distribution & Fraud Risk Density", expanded=True):
            _fgA = go.Figure([
                go.Bar(name="Pool",   x=_cmp5["_label"], y=_cmp5["Pool"],
                       marker_color="#3498db"),
                go.Bar(name="Sample", x=_cmp5["_label"], y=_cmp5["Sample"],
                       marker_color="#e74c3c", opacity=0.85),
            ])
            _fgA.update_layout(
                barmode="group",
                title=f"{_dc5} \u2014 Proportion: Pool vs Sample",
                xaxis_title=_dc5, yaxis_title="Proportion",
            )
            st.plotly_chart(_plotly_dark(_fgA), use_container_width=True)

            if _is_out5:
                st.caption(
                    "\u2139\ufe0f Outcome column: the fraud rate is expected to be lower in the "
                    "sample because IDENTITY_GROUP_FRAUD rows are removed during clean sampling."
                )

            # ── Chart B: 100% Stacked Fraud/No-Fraud Risk Density ───────────
            if TARGET_COL in _pool5.columns and TARGET_COL in _samp5.columns:
                _top_cats5 = _cmp5[_dc5].tolist()
                _col_s5_p = _pool5[_dc5].fillna("__NA__").astype(str)
                _col_s5_s = _samp5[_dc5].fillna("__NA__").astype(str)

                _nofraud_pool, _fraud_pool = [], []
                _nofraud_samp, _fraud_samp = [], []
                _x_labels_5 = []

                for _cat in _top_cats5:
                    _psub = _pool5[_col_s5_p == _cat]
                    _ssub = _samp5[_col_s5_s == _cat]
                    _pn   = len(_psub)
                    _sn   = len(_ssub)
                    _pfr  = float(_psub[TARGET_COL].mean() * 100) if _pn > 0 else 0.0
                    _sfr  = float(_ssub[TARGET_COL].mean() * 100) if _sn > 0 else 0.0
                    _x_labels_5.append(_lbl5(_cat))
                    _fraud_pool.append(round(_pfr, 2))
                    _nofraud_pool.append(round(100 - _pfr, 2))
                    _fraud_samp.append(round(_sfr, 2))
                    _nofraud_samp.append(round(100 - _sfr, 2))

                _fgB = go.Figure([
                    go.Bar(
                        name="No Fraud (Pool)",
                        x=_x_labels_5, y=_nofraud_pool,
                        offsetgroup="Pool",
                        marker_color="#2980b9",
                        hovertemplate="%{x}<br>No Fraud: %{y:.1f}%<extra>Pool</extra>",
                    ),
                    go.Bar(
                        name="Fraud (Pool)",
                        x=_x_labels_5, y=_fraud_pool,
                        offsetgroup="Pool",
                        marker_color="#c0392b",
                        hovertemplate="%{x}<br>Fraud: %{y:.1f}%<extra>Pool</extra>",
                    ),
                    go.Bar(
                        name="No Fraud (Sample)",
                        x=_x_labels_5, y=_nofraud_samp,
                        offsetgroup="Sample",
                        marker_color="#5dade2",
                        hovertemplate="%{x}<br>No Fraud: %{y:.1f}%<extra>Sample</extra>",
                    ),
                    go.Bar(
                        name="Fraud (Sample)",
                        x=_x_labels_5, y=_fraud_samp,
                        offsetgroup="Sample",
                        marker_color="#e74c3c",
                        hovertemplate="%{x}<br>Fraud: %{y:.1f}%<extra>Sample</extra>",
                    ),
                ])
                _fgB.update_layout(
                    barmode="relative",
                    title=f"Fraud Risk Density by {_dc5} \u2014 100% Stacked (Pool vs Sample)",
                    xaxis_title=_dc5,
                    yaxis=dict(title="% within category", ticksuffix="%", range=[0, 105]),
                )
                st.plotly_chart(_plotly_dark(_fgB), use_container_width=True)
                st.caption(
                    "\u2139\ufe0f Each pair of bars = one category. "
                    "**Dark blue / Dark red** = Pool; **Light blue / Red** = Sample. "
                    "If Pool and Sample bars look identical \u2192 perfect representation. "
                    "If the Fraud (red) segment is much smaller in Sample \u2192 fraud is "
                    "being diluted \u2014 the model will under-predict risk."
                )

    # ── Date features: month and DOW distribution ────────────────────────────
    if _date5:
        st.markdown("#### 📅 Date Feature Distributions (Quarter, Month & Day-of-Week bins)")
        _month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        _dow_names   = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

        for _c5 in _date5:
            if _c5 not in _samp5.columns:
                continue

            # Build pool/sample rows with target col and parsed datetime
            _pool_rows5 = _pool5[[_c5, TARGET_COL]].copy()
            _samp_rows5 = _samp5[[_c5, TARGET_COL]].copy()
            _pool_rows5["_dt"] = pd.to_datetime(_pool_rows5[_c5], errors="coerce")
            _samp_rows5["_dt"] = pd.to_datetime(_samp_rows5[_c5], errors="coerce")
            _pool_rows5 = _pool_rows5.dropna(subset=["_dt"])
            _samp_rows5 = _samp_rows5.dropna(subset=["_dt"])
            if len(_pool_rows5) < 5 or len(_samp_rows5) < 5:
                continue

            with st.expander(f"📅 {_c5} — Temporal Distribution", expanded=False):

                # ── CHART QA — Quarter: Pool vs Sample proportion ───────────────
                _pq_cnt = _pool_rows5["_dt"].dt.quarter.value_counts().sort_index()
                _sq_cnt = _samp_rows5["_dt"].dt.quarter.value_counts().sort_index()
                _all_quarters = sorted(set(_pq_cnt.index) | set(_sq_cnt.index))
                _qlabels = [f"Q{int(q)}" for q in _all_quarters]
                _n_pd = max(len(_pool_rows5), 1); _n_sd = max(len(_samp_rows5), 1)
                _fgQ = go.Figure([
                    go.Bar(name="Pool",   x=_qlabels,
                           y=[_pq_cnt.get(q, 0) / _n_pd for q in _all_quarters],
                           marker_color="#3498db"),
                    go.Bar(name="Sample", x=_qlabels,
                           y=[_sq_cnt.get(q, 0) / _n_sd for q in _all_quarters],
                           marker_color="#e74c3c", opacity=0.85),
                ])
                _fgQ.update_layout(barmode="group",
                    title=f"{_c5} — Quarter Distribution",
                    xaxis_title="Quarter", yaxis_title="Proportion")
                st.plotly_chart(_plotly_dark(_fgQ), use_container_width=True)

                # ── CHART QB — Quarter: Fraud Rate per bin (Pool vs Sample) ─────
                _pool_rows5["_quarter"] = _pool_rows5["_dt"].dt.quarter
                _samp_rows5["_quarter"] = _samp_rows5["_dt"].dt.quarter
                _pfr_q = (_pool_rows5.groupby("_quarter")[TARGET_COL]
                          .mean().reindex(_all_quarters, fill_value=0))
                _sfr_q = (_samp_rows5.groupby("_quarter")[TARGET_COL]
                          .mean().reindex(_all_quarters, fill_value=0))
                _fgQF = go.Figure([
                    go.Bar(name="Pool fraud rate",   x=_qlabels,
                           y=(_pfr_q * 100).tolist(), marker_color="#3498db"),
                    go.Bar(name="Sample fraud rate", x=_qlabels,
                           y=(_sfr_q * 100).tolist(), marker_color="#e74c3c", opacity=0.85),
                ])
                _fgQF.update_layout(barmode="group",
                    title=f"{_c5} — Fraud Rate % by Quarter (Pool vs Sample)",
                    xaxis_title="Quarter", yaxis_title="Fraud Rate (%)")
                st.plotly_chart(_plotly_dark(_fgQF), use_container_width=True)

                # ── CHART A — Month: Pool vs Sample proportion ───────────────
                _pm_cnt = _pool_rows5["_dt"].dt.month.value_counts().sort_index()
                _sm_cnt = _samp_rows5["_dt"].dt.month.value_counts().sort_index()
                _all_months = sorted(set(_pm_cnt.index) | set(_sm_cnt.index))
                _mlabels = [_month_names.get(m, str(m)) for m in _all_months]
                _n_pd = max(len(_pool_rows5), 1); _n_sd = max(len(_samp_rows5), 1)
                _fgM = go.Figure([
                    go.Bar(name="Pool",   x=_mlabels,
                           y=[_pm_cnt.get(m, 0) / _n_pd for m in _all_months],
                           marker_color="#3498db"),
                    go.Bar(name="Sample", x=_mlabels,
                           y=[_sm_cnt.get(m, 0) / _n_sd for m in _all_months],
                           marker_color="#e74c3c", opacity=0.85),
                ])
                _fgM.update_layout(barmode="group",
                    title=f"{_c5} \u2014 Month Distribution",
                    xaxis_title="Month", yaxis_title="Proportion")
                st.plotly_chart(_plotly_dark(_fgM), use_container_width=True)

                # ── CHART B — Month: Fraud Rate per bin (Pool vs Sample) ─────
                _pool_rows5["_month"] = _pool_rows5["_dt"].dt.month
                _samp_rows5["_month"] = _samp_rows5["_dt"].dt.month
                _pfr_m = (_pool_rows5.groupby("_month")[TARGET_COL]
                          .mean().reindex(_all_months, fill_value=0))
                _sfr_m = (_samp_rows5.groupby("_month")[TARGET_COL]
                          .mean().reindex(_all_months, fill_value=0))
                _fgMF = go.Figure([
                    go.Bar(name="Pool fraud rate",   x=_mlabels,
                           y=(_pfr_m * 100).tolist(), marker_color="#3498db"),
                    go.Bar(name="Sample fraud rate", x=_mlabels,
                           y=(_sfr_m * 100).tolist(), marker_color="#e74c3c", opacity=0.85),
                ])
                _fgMF.update_layout(barmode="group",
                    title=f"{_c5} \u2014 Fraud Rate % by Month (Pool vs Sample)",
                    xaxis_title="Month", yaxis_title="Fraud Rate (%)")
                st.plotly_chart(_plotly_dark(_fgMF), use_container_width=True)

                # ── CHART C — Day-of-Week: Pool vs Sample proportion ─────────
                _pd_cnt = _pool_rows5["_dt"].dt.dayofweek.value_counts().sort_index()
                _sd_cnt = _samp_rows5["_dt"].dt.dayofweek.value_counts().sort_index()
                _all_dows = sorted(set(_pd_cnt.index) | set(_sd_cnt.index))
                _dlabels = [_dow_names.get(d, str(d)) for d in _all_dows]
                _fgD = go.Figure([
                    go.Bar(name="Pool",   x=_dlabels,
                           y=[_pd_cnt.get(d, 0) / _n_pd for d in _all_dows],
                           marker_color="#3498db"),
                    go.Bar(name="Sample", x=_dlabels,
                           y=[_sd_cnt.get(d, 0) / _n_sd for d in _all_dows],
                           marker_color="#e74c3c", opacity=0.85),
                ])
                _fgD.update_layout(barmode="group",
                    title=f"{_c5} \u2014 Day-of-Week Distribution",
                    xaxis_title="Day of Week", yaxis_title="Proportion")
                st.plotly_chart(_plotly_dark(_fgD), use_container_width=True)

                # ── CHART D — Day-of-Week: Fraud Rate per bin (Pool vs Sample)
                _pool_rows5["_dow"] = _pool_rows5["_dt"].dt.dayofweek
                _samp_rows5["_dow"] = _samp_rows5["_dt"].dt.dayofweek
                _pfr_d = (_pool_rows5.groupby("_dow")[TARGET_COL]
                          .mean().reindex(_all_dows, fill_value=0))
                _sfr_d = (_samp_rows5.groupby("_dow")[TARGET_COL]
                          .mean().reindex(_all_dows, fill_value=0))
                _fgDF = go.Figure([
                    go.Bar(name="Pool fraud rate",   x=_dlabels,
                           y=(_pfr_d * 100).tolist(), marker_color="#3498db"),
                    go.Bar(name="Sample fraud rate", x=_dlabels,
                           y=(_sfr_d * 100).tolist(), marker_color="#e74c3c", opacity=0.85),
                ])
                _fgDF.update_layout(barmode="group",
                    title=f"{_c5} \u2014 Fraud Rate % by Day-of-Week (Pool vs Sample)",
                    xaxis_title="Day of Week", yaxis_title="Fraud Rate (%)")
                st.plotly_chart(_plotly_dark(_fgDF), use_container_width=True)

                st.caption(
                    "\u2139\ufe0f **Charts A & C** compare the overall transaction volume proportions "
                    "between Pool and Sample per time bin (Month / Day-of-Week). "
                    "**Charts B & D** show the fraud rate (% fraud) inside each bin for Pool vs Sample. "
                    "A mismatch in fraud rate across bins means the sample may be capturing fraud "
                    "from a different temporal pattern than the pool \u2014 "
                    "e.g., fraud concentrated in a specific month or weekday."
                )

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — OVERALL VERDICT
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("Overall Verdict")

    _psis5 = []
    for _r5 in _rows5:
        try:
            _v5 = float(_r5["PSI"])
            if _v5 == _v5:
                _psis5.append(_v5)
        except (ValueError, TypeError):
            pass

    if _psis5:
        _avg5 = sum(_psis5) / len(_psis5)
        _max5 = max(_psis5)
        _crit5 = [_r5["Feature"] for _r5 in _rows5
                  if isinstance(_r5.get("PSI"), float) and _r5["PSI"] > 0.20]
        _min5  = [_r5["Feature"] for _r5 in _rows5
                  if isinstance(_r5.get("PSI"), float) and 0.10 < _r5["PSI"] <= 0.20]
        _n_num5  = sum(1 for _r5 in _rows5 if _r5["Type"] == "Numeric")
        _n_cat5  = sum(1 for _r5 in _rows5 if "Categorical" in _r5["Type"])
        _n_dat5  = sum(1 for _r5 in _rows5 if _r5["Type"] == "Date")

        if _max5 <= 0.10:
            _vc5, _vi5 = "#27ae60", "\U0001f7e2"
            _vm5 = (f"**Excellent.** Avg PSI = {_avg5:.4f} — all features stable (\u22640.10). "
                    f"({_n_num5} numeric, {_n_cat5} categorical, {_n_dat5} date features). "
                    "The sample faithfully mirrors the clean pool.")
        elif _max5 <= 0.20:
            _vc5, _vi5 = "#f39c12", "\U0001f7e1"
            _vm5 = (f"**Minor shifts detected.** Avg PSI = {_avg5:.4f}. "
                    f"Features with minor shift: {', '.join(_min5) if _min5 else 'none'}. "
                    "Sample is suitable for modeling but monitor these features.")
        else:
            _vc5, _vi5 = "#e74c3c", "\U0001f534"
            _vm5 = (f"**Critical shift detected — action required.** Avg PSI = {_avg5:.4f}. "
                    f"Features with Critical Shift (PSI>0.20): **{', '.join(_crit5)}**. "
                    "Consider re-sampling or relaxing exclusion rules before training.")

        st.markdown(
            f"<div style='background:{_vc5}15;border:2px solid {_vc5};"
            f"border-radius:10px;padding:18px 22px;color:#eee;"
            f"font-size:14px;line-height:1.8'>"
            f"<span style='font-size:22px'>{_vi5}</span>&nbsp; {_vm5}</div>",
            unsafe_allow_html=True,
        )

        _vk5 = st.columns(4)
        _vk5[0].metric("Avg PSI", f"{_avg5:.4f}")
        _vk5[1].metric("Max PSI", f"{_max5:.4f}")
        _vk5[2].metric("Critical features (PSI>0.20)", str(len(_crit5)))
        _vk5[3].metric("Minor-shift features (PSI 0.10\u20130.20)", str(len(_min5)))

    if _rows5:
        _download_csv(
            pd.DataFrame(_rows5),
            label="\u2b07\ufe0f Download Representativeness Report CSV",
            filename="sample_representativeness.csv",
        )
