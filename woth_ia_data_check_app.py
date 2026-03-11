import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from woth_data_engine import run_data_check_and_cleaning, DQ_TAG_COLORS, DQ_TAG_ICONS, DQ_TAG_DESC

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0f0f1a"
CARD_BG = "#1a1a2e"
BORDER  = "#2c2c4e"
TEXT    = "#e0e0e0"
ACCENT  = "#aad4f5"

st.set_page_config(
    page_title="Woth IA Data Check",
    page_icon="🧹",
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
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _metric_card(label: str, value, color: str = ACCENT) -> str:
    return (
        f"<div class='metric-card'>"
        f"<div class='metric-val' style='color:{color}'>{value}</div>"
        f"<div class='metric-lbl'>{label}</div>"
        f"</div>"
    )

def _tag_pill(tag: str) -> str:
    c = DQ_TAG_COLORS.get(tag, "#888")
    return f"<span class='tag-pill' style='background:{c}'>{DQ_TAG_ICONS.get(tag,'')} {tag}</span>"

def _plotly_dark(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧹 Woth IA Data Check App")
st.markdown("<div class='info-box'>Upload a dataset to run data quality checks, identify format issues/errors, missing values, whitespaces, and standardise records.</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file, low_memory=False)
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            df_raw = None
    else:
        df_raw = None

if df_raw is not None and 'df_clean' not in st.session_state:
    with st.spinner("Analyzing and cleaning dataset automatically..."):
        
        cfg = {}
        for col in df_raw.columns:
            cl = str(col).lower()
            if 'phone' in cl and 'phone' not in cfg: cfg['phone'] = col
            elif 'tin' in cl and 'tin' not in cfg: cfg['tin'] = col
            elif 'address' in cl and 'address' not in cfg: cfg['address'] = col
            elif 'city' in cl and 'city' not in cfg: cfg['city'] = col
            elif 'state' in cl and 'state' not in cfg: cfg['state'] = col
            elif ('zip' in cl or 'postal' in cl) and 'zip' not in cfg: cfg['zip'] = col
            elif ('company' in cl or 'business' in cl) and 'company_name' not in cfg: cfg['company_name'] = col
            elif 'dba' in cl and 'dba' not in cfg: cfg['dba'] = col
            elif 'first' in cl and 'first_name' not in cfg: cfg['first_name'] = col
            elif 'last' in cl and 'last_name' not in cfg: cfg['last_name'] = col
            elif df_raw[col].dtype in ['datetime64[ns]'] and 'datetime' not in cfg: 
                cfg['datetime'] = [col]
        
        st.session_state['cfg'] = cfg
        df_clean, df_issues, stats_summary = run_data_check_and_cleaning(df_raw, cfg)
        
        st.session_state['df_raw'] = df_raw
        st.session_state['df_clean'] = df_clean
        st.session_state['df_issues'] = df_issues
        st.session_state['stats_summary'] = stats_summary
        st.success("Analysis complete!")

# Render results and tabs if data is in session_state
if 'df_clean' in st.session_state:
    df_raw = st.session_state['df_raw']
    df_clean = st.session_state['df_clean']
    df_issues = st.session_state['df_issues']
    stats_dict = st.session_state['stats_summary']
    stats = stats_dict['global']
    feat_qual_orig = stats_dict.get('feature_quality_orig', {})
    feat_qual_clean = stats_dict.get('feature_quality_clean', {})
    
    TAB1, TAB2, TAB3, TAB4, TAB5 = st.tabs([
        "📉 Original Dataset Features (Raw)",
        "⚠️ Detailed Issue Report",
        "✨ Clean vs Original & Export",
        "📖 Standardization Rules",
        "📝 Final and Cleaned Dataset"
    ])
    
    def render_fq_tab(fq_dict, is_clean=False):
        if not fq_dict:
            st.info("No mapped columns to score.")
            return
            
        fq_df = pd.DataFrame.from_dict(fq_dict, orient='index').reset_index()
        fq_df.columns = ['Feature', 'Column', 'Score', 'Missing', 'Invalid', 'Missing %', 'Invalid %']
        
        # Calculate overall score
        overall_score = round(fq_df['Score'].mean(), 1)
        score_color = "#27ae60" if overall_score > 80 else ("#f39c12" if overall_score > 50 else "#e74c3c")
        
        col_chart, col_score = st.columns([3, 1], vertical_alignment="center")
        
        with col_chart:
            # Sort dataframe for better visualization
            fq_df_sorted = fq_df.sort_values(by='Score', ascending=True)
            fig_fq = px.bar(fq_df_sorted, x='Score', y='Column', color='Score', orientation='h',
                            color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
                            range_color=[0, 100], text='Score')
            fig_fq.update_layout(showlegend=False, xaxis_range=[0,105])
            fig_fq.update_coloraxes(showscale=False)
            st.plotly_chart(_plotly_dark(fig_fq), use_container_width=True, key=f"fq_bar_{is_clean}")
            
        with col_score:
            html_content = f"""
            <div style='text-align: center; padding: 2rem; background-color: #1e1e1e; border-radius: 10px; margin-top: 4rem;'>
                <h3 style='margin: 0; color: #ffffff;'>Overall Dataset Score</h3>
                <h1 style='color: {score_color}; font-size: 4rem; margin: 0;'>{overall_score}%</h1>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Feature Assessments")
        cols = st.columns(2)
        
        for idx, row in fq_df.iterrows():
            col = cols[idx % 2]
            score = row['Score']
            color = "#27ae60" if score > 80 else ("#f39c12" if score > 50 else "#e74c3c")
            
            with col:
                with st.expander(f"{row['Column']} — {score}% Score"):
                    st.markdown(f"**Missing Profiles:** {row['Missing']}  |  **Invalid Formats:** {row['Invalid']}")
                    st.markdown(f"**Quality Score:** <span style='color:{color}; font-weight:bold;'>{score}%</span>", unsafe_allow_html=True)
                    if (row['Missing'] == 0 and row['Invalid'] == 0):
                        st.success(f"No formatting or missing value issues detected for `{row['Column']}`!")
                    elif row['Missing'] > 0 or row['Invalid'] > 0:
                        st.markdown(f"**Sample of {'Unresolved' if is_clean else 'Failed'} Records:**")
                        c = row['Column']
                        sample_local = []
                        
                        if is_clean:
                            # CLEAN DATASET SAMPLES
                            if "tin" in st.session_state.get('cfg', {}) and c == st.session_state['cfg']["tin"]:
                                t_mask = df_clean[c].dropna().astype(str).str.match(r"^\d{2}-\d{7}$").eq(False)
                                if t_mask.any():
                                    unres_idx = df_raw.loc[t_mask.index[t_mask]].head(10).index
                                    sample_local = [{"Original Value": str(df_raw.at[i, c]), "Clean Attempt": str(df_clean.at[i, c]), "Failure Reason": "TIN structure incorrect (Length or Delimiter)"} for i in unres_idx]
                            elif "phone" in st.session_state.get('cfg', {}) and c == st.session_state['cfg']["phone"]:
                                p_mask = df_clean[c].dropna().astype(str).str.match(r"^(\+1 )?\(\d{3}\) \d{3}-\d{4}$").eq(False)
                                if p_mask.any():
                                    unres_idx = df_raw.loc[p_mask.index[p_mask]].head(10).index
                                    sample_local = [{"Original Value": str(df_raw.at[i, c]), "Clean Attempt": str(df_clean.at[i, c]), "Failure Reason": "Phone length incorrect"} for i in unres_idx]
                            elif "zip" in st.session_state.get('cfg', {}) and c == st.session_state['cfg']["zip"]:
                                z_mask = df_clean[c].dropna().astype(str).str.match(r"^\d{5}$").eq(False)
                                if z_mask.any():
                                    unres_idx = df_raw.loc[z_mask.index[z_mask]].head(10).index
                                    sample_local = [{"Original Value": str(df_raw.at[i, c]), "Clean Attempt": str(df_clean.at[i, c]), "Failure Reason": "Postal Code not 5-digits"} for i in unres_idx]
                            else:
                                mask = pd.Series(False, index=df_raw.index)
                                if (c + '_is_missing') in df_issues: mask |= df_issues[c + '_is_missing']
                                if (c + '_has_ws') in df_issues: mask |= df_issues[c + '_has_ws']
                                if (c + '_has_sc') in df_issues: mask |= df_issues[c + '_has_sc']
                                if (c + '_has_format') in df_issues: mask |= df_issues[c + '_has_format']
                                
                                changed_df = df_raw[mask]
                                changed_clean = df_clean[mask]
                                for i in changed_df.head(20).index:
                                    orig_v = str(changed_df.at[i, c])
                                    clean_v = str(changed_clean.at[i, c])
                                    if orig_v == clean_v and orig_v != "nan" and orig_v != "None":
                                        sample_local.append({"Original Value": orig_v, "Clean Attempt": clean_v, "Failure Reason": "Unfixable Generic Format / Missing Info."})
                                        if len(sample_local) >= 10: break
                        else:
                            # RAW DATASET SAMPLES
                            mask = pd.Series(False, index=df_raw.index)
                            has_ws = (c + '_has_ws') in df_issues
                            has_sc = (c + '_has_sc') in df_issues
                            has_fmt = (c + '_has_format') in df_issues
                            has_mis = (c + '_is_missing') in df_issues
                            
                            if has_ws: mask |= df_issues[c + '_has_ws']
                            if has_sc: mask |= df_issues[c + '_has_sc']
                            if has_fmt: mask |= df_issues[c + '_has_format']
                            if has_mis: mask |= df_issues[c + '_is_missing']
                            
                            bad_raw = df_raw[mask].head(20)
                            for i in bad_raw.index:
                                val = str(df_raw.at[i, c])
                                reasons = []
                                if has_mis and df_issues.at[i, c + '_is_missing']: reasons.append("Missing Value")
                                if has_fmt and df_issues.at[i, c + '_has_format']: reasons.append("Invalid Format Schema")
                                if has_ws and df_issues.at[i, c + '_has_ws']: reasons.append("Whitespace (Leading/Trailing/Multiple)")
                                if has_sc and df_issues.at[i, c + '_has_sc']: reasons.append("Special Characters Not Allowed")
                                
                                sample_local.append({
                                    "Invalid Value": val,
                                    "Identified Issue": " | ".join(reasons)
                                })
                                if len(sample_local) >= 10: break
                                
                        if sample_local:
                            st.dataframe(pd.DataFrame(sample_local), use_container_width=True)
                        else:
                            st.info("Remaining issues are Missing Values from the raw source.")
                            
    with TAB1:
        st.header("Overall Data Quality Metrics (Raw)")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(_metric_card("Total Rows", f"{stats['total_rows']:,}"), unsafe_allow_html=True)
        c2.markdown(_metric_card("Good Rows", f"{stats['good_rows']:,}", "#27ae60"), unsafe_allow_html=True)
        c3.markdown(_metric_card("Rows w/ Issues", f"{stats['rows_with_issues']:,}", "#e74c3c"), unsafe_allow_html=True)
        c4.markdown(_metric_card("Missing Values", f"{stats['missing_values_count']:,}", "#f39c12"), unsafe_allow_html=True)
        st.markdown("---")
        
        st.subheader("Feature Quality Score")
        st.caption("Quality breakdown per column based on missing and invalid values")
        render_fq_tab(feat_qual_orig, is_clean=False)
        
        st.markdown("---")
        st.subheader("Original Duplicate & Structural Tags")
        if '_duplicate_tag' in df_issues.columns:
            tag_counts = df_issues['_duplicate_tag'].value_counts().reset_index()
            tag_counts.columns = ['Tag', 'Count']
        else:
            tag_counts = pd.DataFrame(columns=['Tag', 'Count'])
            
        col1, col2 = st.columns(2)
        with col1:
            if not tag_counts.empty:
                fig_pie = px.pie(tag_counts, names='Tag', values='Count', color='Tag', 
                                 color_discrete_map=DQ_TAG_COLORS, hole=0.5,
                                 title="Phase 1 Structure (% of records)")
                fig_pie.update_traces(textposition='outside', textinfo='percent+label')
                fig_pie.update_layout(showlegend=False)
                st.plotly_chart(_plotly_dark(fig_pie), use_container_width=True)
        with col2:
            if not tag_counts.empty:
                tag_counts_sorted = tag_counts.sort_values(by='Count', ascending=True)
                fig_bar = px.bar(tag_counts_sorted, x='Count', y='Tag', color='Tag', 
                                 color_discrete_map=DQ_TAG_COLORS, orientation='h',
                                 title="Record Count by Duplicate Tag", text='Count')
                fig_bar.update_layout(showlegend=False)
                st.plotly_chart(_plotly_dark(fig_bar), use_container_width=True)
                
        st.markdown("---")
        st.subheader("Raw Data Sample by Standardization Rules")
        # Build new tables by rule
        rule_sel = st.selectbox("Select Rule Type to View Failing Records:", ["FORMAT_ISSUE", "SPECIAL_CHARS", "WHITESPACE_ISSUE", "MISSING_VALUE"])
        # Find rows mapped to this issue
        rule_df = df_raw.copy()
        df_issues['issue_count'] = df_issues['_issue_tags'].apply(lambda x: len(x.split(',')) if x != "GOOD_DATA" and x else 0)
        rule_df['Total Issues'] = df_issues['issue_count']
        rule_df['Tags'] = df_issues['_issue_tags'].apply(lambda x: str(x).replace("NaN", "").strip(","))
        
        matched_df = rule_df[rule_df['Tags'].str.contains(rule_sel, na=False)]
        if matched_df.empty:
            st.success(f"No records failed the {rule_sel} rule!")
        else:
            st.dataframe(matched_df.head(200), use_container_width=True, hide_index=True)
            
    with TAB2:
        st.subheader("Detailed Issue Report")
        
        # Melt df_issues to find specific column errors
        issue_cols = [c for c in df_issues.columns if c.endswith("_is_missing") or c.endswith("_has_ws") or c.endswith("_has_sc")]
        
        if issue_cols:
            melt_issues = df_issues[issue_cols].reset_index().melt(id_vars='index')
            melt_issues = melt_issues[melt_issues['value'] == True].copy()
            
            # Map severity
            def get_sev(var_name):
                if "_is_missing" in var_name: return "High"
                elif "_has_sc" in var_name: return "Medium"
                return "Low"
                
            def get_clean_col(var_name):
                return var_name.replace("_is_missing", "").replace("_has_ws", "").replace("_has_sc", "")
                
            def get_issue_type(var_name):
                if "_is_missing" in var_name: return "MISSING"
                elif "_has_sc" in var_name: return "SPECIAL CHARACTERS"
                return "FORMAT WHITESPACE"
                
            def get_strategy(var_name):
                if "_has_ws" in var_name: return "Trim and single-space"
                elif "_has_sc" in var_name: return "Strip valid chars"
                return "Flag/Null representation"
            
            melt_issues['Severity'] = melt_issues['variable'].apply(get_sev)
            melt_issues['Feature'] = melt_issues['variable'].apply(get_clean_col)
            melt_issues['Issue Type'] = melt_issues['variable'].apply(get_issue_type)
            melt_issues['Resolution Strategy'] = melt_issues['variable'].apply(get_strategy)
            
            # Fetch Original Values efficiently
            # Vectorized assignment of original values
            orig_vals = []
            for _, row in melt_issues.iterrows():
                try: orig_vals.append(str(df_raw.at[row['index'], row['Feature']]))
                except: orig_vals.append("")
            melt_issues['Original Bad Value'] = orig_vals
            
            total_high = len(melt_issues[melt_issues['Severity'] == 'High'])
            total_med = len(melt_issues[melt_issues['Severity'] == 'Medium'])
            total_low = len(melt_issues[melt_issues['Severity'] == 'Low'])
            
            total_issues = len(melt_issues)
            w_high = int((total_high / max(total_issues, 1)) * 100)
            w_med = int((total_med / max(total_issues, 1)) * 100)
            w_low = int((total_low / max(total_issues, 1)) * 100)
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                sev_df = pd.DataFrame([
                    {'Severity': 'High', 'Count': total_high},
                    {'Severity': 'Medium', 'Count': total_med},
                    {'Severity': 'Low', 'Count': total_low}
                ])
                fig_sev = px.bar(sev_df, x='Severity', y='Count', color='Severity',
                                 color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#3498db'},
                                 title="Issues by Severity", text='Count')
                fig_sev.update_layout(showlegend=False, xaxis_title="", yaxis_title="")
                st.plotly_chart(_plotly_dark(fig_sev), use_container_width=True, key="sev_bar")
                
            with col_chart2:
                type_counts = melt_issues['Issue Type'].value_counts().reset_index()
                type_counts.columns = ['Issue Type', 'Count']
                fig_type = px.pie(type_counts, names='Issue Type', values='Count', color='Issue Type',
                                  color_discrete_sequence=px.colors.qualitative.Pastel,
                                  title="Proportion of Issue Types", hole=0.4)
                fig_type.update_layout(showlegend=False)
                fig_type.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(_plotly_dark(fig_type), use_container_width=True, key="type_pie")
                
            st.markdown("---")
            st.markdown("##### Filter Extracted Issue Database")
            f1, f2 = st.columns(2)
            sev_filter = f1.selectbox("Severity Level:", ["All", "High", "Medium", "Low"])
            feat_filter = f2.selectbox("Feature Name:", ["All"] + sorted(list(melt_issues['Feature'].unique())))
            
            filtered_issues = melt_issues.copy()
            if sev_filter != "All": filtered_issues = filtered_issues[filtered_issues['Severity'] == sev_filter]
            if feat_filter != "All": filtered_issues = filtered_issues[filtered_issues['Feature'] == feat_filter]
            
            display_cols = ['index', 'Feature', 'Severity', 'Issue Type', 'Original Bad Value', 'Resolution Strategy']
            filtered_issues = filtered_issues.rename(columns={'index': 'Record ID'})
            display_cols[0] = 'Record ID'
            
            if len(filtered_issues) == 0:
                st.info(f"No corresponding issues found.")
            else:
                st.dataframe(filtered_issues[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No detailed column-level issues detected.")
            
    with TAB3:
        st.subheader("Cleaned Dataset Feature Score Quality")
        render_fq_tab(feat_qual_clean, is_clean=True)
        st.markdown("---")
        st.subheader("Data Quality Comparison Metrics")
        
        # Calculate metrics
        orig_missing = df_raw.isna().sum().sum()
        clean_missing = df_clean.isna().sum().sum()
        
        format_flags = sum(df_issues[c].sum() for c in df_issues.columns if "_has_format" in c)
        ws_flags = sum(df_issues[c].sum() for c in df_issues.columns if "_has_ws" in c)
        sc_flags = sum(df_issues[c].sum() for c in df_issues.columns if "_has_sc" in c)
        total_orig_anomalies = format_flags + ws_flags + sc_flags
        
        # Calculate pseudo-unfixable
        unfixable = 0
        unfixable_records = pd.DataFrame()
        
        cfg = st.session_state.get("cfg", {})
        if "tin" in cfg and cfg["tin"] in df_clean.columns:
            t_col = cfg["tin"]
            t_mask = df_clean[t_col].dropna().astype(str).str.match(r"^\d{2}-\d{7}$").eq(False)
            if t_mask.any():
                unfixable += t_mask.sum()
                bad_tins = df_raw.loc[t_mask.index[t_mask]].copy()
                bad_tins['Unfixable Feature'] = 'TIN'
                bad_tins['Failure Reason'] = 'Does not match exactly 9 digits after stripping'
                unfixable_records = pd.concat([unfixable_records, bad_tins])
                
        if "phone" in cfg and cfg["phone"] in df_clean.columns:
            p_col = cfg["phone"]
            p_mask = df_clean[p_col].dropna().astype(str).str.match(r"^(\+1 )?\(\d{3}\) \d{3}-\d{4}$").eq(False)
            if p_mask.any():
                unfixable += p_mask.sum()
                bad_phones = df_raw.loc[p_mask.index[p_mask]].copy()
                bad_phones['Unfixable Feature'] = 'Phone'
                bad_phones['Failure Reason'] = 'Length not exactly 10 or 11 integers'
                unfixable_records = pd.concat([unfixable_records, bad_phones])
        
        if "zip" in cfg and cfg["zip"] in df_clean.columns:
            z_col = cfg["zip"]
            z_mask = df_clean[z_col].dropna().astype(str).str.match(r"^\d{5}$").eq(False)
            if z_mask.any():
                unfixable += z_mask.sum()
                bad_zips = df_raw.loc[z_mask.index[z_mask]].copy()
                bad_zips['Unfixable Feature'] = 'Zip'
                bad_zips['Failure Reason'] = 'Invalid Postal Code format'
                unfixable_records = pd.concat([unfixable_records, bad_zips])
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.markdown(_metric_card("Total Rows", f"{len(df_raw):,}", "#3498db"), unsafe_allow_html=True)
        m_col2.markdown(_metric_card("Original Missing Stats", f"{orig_missing:,}", "#e74c3c"), unsafe_allow_html=True)
        m_col3.markdown(_metric_card("Total Broken Formats", f"{total_orig_anomalies:,}", "#f39c12"), unsafe_allow_html=True)
        m_col4.markdown(_metric_card("Unresolved (Unfixable)", f"{unfixable:,}", unfixable > 0 and "#e74c3c" or "#27ae60"), unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Unified Data Transformation Comparison")
        st.markdown("<div class='info-box'>Compare ALL modified and unfixable original entries with standardized output. Filter by Resolution Status to find exact failures.</div>", unsafe_allow_html=True)
        
        diff_mask = df_issues["_has_issue"] == True
        changed_df = df_raw[diff_mask]
        changed_clean = df_clean[diff_mask]
        
        if changed_df.empty:
            st.info("No modifications were required for any column.")
        else:
            # Build merged row-based dataframe for comparison
            comp_rows = []
            display_cols = list(st.session_state.get('cfg', {}).values())
            if not display_cols:
                display_cols = [c for c in df_raw.select_dtypes(include=['object', 'string']).columns.tolist() if c in changed_df.columns]
                
            for col in display_cols:
                if col in changed_df.columns:
                    # Find any row where THIS SPECIFIC COL had a flagged issue
                    col_has_issue_mask = pd.Series(False, index=changed_df.index)
                    if col + "_is_missing" in df_issues.columns: col_has_issue_mask |= df_issues[col + "_is_missing"]
                    if col + "_has_ws" in df_issues.columns: col_has_issue_mask |= df_issues[col + "_has_ws"]
                    if col + "_has_sc" in df_issues.columns: col_has_issue_mask |= df_issues[col + "_has_sc"]
                    if col + "_has_format" in df_issues.columns: col_has_issue_mask |= df_issues[col + "_has_format"]
                    
                    idx = changed_df[col_has_issue_mask].index
                    for i in idx:
                        f_tags = []
                        if df_issues.get(col + "_is_missing", pd.Series({i:False})).at[i]: f_tags.append("MISSING_VALUE")
                        if df_issues.get(col + "_has_ws", pd.Series({i:False})).at[i]: f_tags.append("WHITESPACE_ISSUE")
                        if df_issues.get(col + "_has_sc", pd.Series({i:False})).at[i]: f_tags.append("SPECIAL_CHARS")
                        if df_issues.get(col + "_has_format", pd.Series({i:False})).at[i]: f_tags.append("FORMAT_ISSUE")
                        
                        if '_duplicate_tag' in df_issues.columns and pd.notna(df_issues.at[i, '_duplicate_tag']):
                            f_tags.append(str(df_issues.at[i, '_duplicate_tag']))
                            
                        orig_v = str(changed_df.at[i, col])
                        clean_v = str(changed_clean.at[i, col])
                        failure_reason = ""
                        
                        is_unfixable = False
                        if "FORMAT_ISSUE" in f_tags:
                            if "tin" in cfg and col == cfg["tin"]:
                                if not pd.isna(changed_clean.at[i, col]) and not str(changed_clean.at[i, col]).startswith("nan") and not __import__('re').match(r"^\d{2}-\d{7}$", clean_v):
                                    is_unfixable = True
                                    failure_reason = "TIN Length Error (Not 9 digits)"
                            elif "phone" in cfg and col == cfg["phone"]:
                                if not pd.isna(changed_clean.at[i, col]) and not str(changed_clean.at[i, col]).startswith("nan") and not __import__('re').match(r"^(\+1 )?\(\d{3}\) \d{3}-\d{4}$", clean_v):
                                    is_unfixable = True
                                    failure_reason = "Phone Length Error (Not 10/11 digits)"
                            elif "zip" in cfg and col == cfg["zip"]:
                                if not pd.isna(changed_clean.at[i, col]) and not str(changed_clean.at[i, col]).startswith("nan") and not __import__('re').match(r"^\d{5}$", clean_v):
                                    is_unfixable = True
                                    failure_reason = "Invalid Postal Code format"
                            else:
                                if orig_v == clean_v:
                                    is_unfixable = True
                                    failure_reason = "Could not parse formatting"
                                    
                        if is_unfixable:
                            f_tags.append("UNFIXABLE_BY_ENGINE")
                            
                        comp_rows.append({
                            'Row ID': i,
                            'Feature': col,
                            'Original Value': orig_v,
                            'Standardization': clean_v,
                            'Triggered Issue': ",".join(f_tags) if f_tags else "STANDARDIZED",
                            'Failure Reason': failure_reason
                        })
                        
            if not comp_rows:
                st.info("No string-based formatting anomalies were found among the checked records.")
            else:
                comp_df = pd.DataFrame(comp_rows)
                comp_df['Resolution'] = comp_df['Failure Reason'].apply(lambda x: 'Unresolved' if x else 'Resolved/Standardized')
                
                # --- Draw Unresolved Summary Chart ---
                unres_df = comp_df[comp_df['Resolution'] == 'Unresolved']
                if not unres_df.empty:
                    st.markdown("##### 🚨 Unresolved Engine Format Failures")
                    chart_df = unres_df.groupby(['Feature', 'Failure Reason']).size().reset_index(name='Count')
                    fig_unres = px.bar(chart_df, x='Count', y='Feature', color='Failure Reason', orientation='h', 
                                       title="Distribution of Unfixable Records by Feature & Reason", text='Count',
                                       color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(_plotly_dark(fig_unres), use_container_width=True)
                    st.markdown("---")
                
                # Interactive Display Setup
                f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                with f_col1:
                    filter_feat = st.selectbox("Feature:", ["All"] + sorted(list(comp_df['Feature'].unique())))
                with f_col2:
                    filter_tag = st.selectbox("Tag:", [
                        "All", 
                        "FORMAT_ISSUE", "SPECIAL_CHARS", "WHITESPACE_ISSUE", "MISSING_VALUE",
                        "EXACT_DUPLICATE", "IDENTITY_GROUP_FRAUD", "IDENTITY_GROUP_CLEAN", 
                        "TECH_REAPPLICATION", "INVALID_FOR_KYC"
                    ])
                with f_col3:
                    res_filter = st.selectbox("Resolution:", ["All", "Resolved/Standardized", "Unresolved"])
                with f_col4:
                    search_str = st.text_input("Search Engine:")
                    
                # Applying Filter Trigers
                filtered_comp = comp_df.copy()
                if filter_feat != "All":
                    filtered_comp = filtered_comp[filtered_comp['Feature'] == filter_feat]
                if filter_tag != "All":
                    filtered_comp = filtered_comp[filtered_comp['Triggered Issue'].str.contains(filter_tag, na=False)]
                if res_filter != "All":
                    filtered_comp = filtered_comp[filtered_comp['Resolution'] == res_filter]
                if search_str:
                    filtered_comp = filtered_comp[filtered_comp.astype(str).apply(lambda x: x.str.contains(search_str, case=False)).any(axis=1)]
                
                st.markdown(f"**Found {len(filtered_comp)} modified records matching criteria:**")
                st.dataframe(
                    filtered_comp,
                    use_container_width=False,
                    hide_index=True,
                    column_config={
                        'Row ID': st.column_config.NumberColumn("Row #", format="%d", width="small"),
                        'Feature': st.column_config.TextColumn("Standardized Feature", width="small"),
                        'Original Value': st.column_config.TextColumn("Raw Original", width="medium"),
                        'Standardization': st.column_config.TextColumn("Clean Value", width="medium"),
                        'Triggered Issue': st.column_config.TextColumn("Issue Categories")
                    }
                )
            

            
    with TAB4:
        st.markdown("<h3 style='margin-bottom:2px;'>📖 Standardization Rules & Data Checks Definition</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='background:#1b2533; border-left:4px solid #3498db; padding:8px 16px; margin-bottom:16px; color:#dcdcdc; font-size:15px; border-radius:4px;'>
                <b style='color:#3498db;'>Feature-Level Quality Checks & Standardization Engine</b><br>
                Before calculating statistics or evaluating duplicates, the Woth IA Engine scans every feature cell-by-cell. It categorizes bad records using predefined rules, scores their quality, and applies aggressive standardization where possible.
            </div>
            
            <style>
            .rules-table { width: 100%; border-collapse: collapse; margin-bottom: 24px; font-size: 13px; font-family: sans-serif; }
            .rules-table th { background: #131a24; color: #acc2d9; padding: 10px; border-bottom: 1px solid #233346; text-align: left; }
            .rules-table td { padding: 10px; border-bottom: 1px solid #1a2533; color: #ccc; }
            .rules-table tr:hover { background: #1a2533; }
            .t-pri { color: #e74c3c; font-weight: bold; text-align: center; }
            .t-tag { font-weight: bold; }
            .err-tag { background: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; }
            .warn-tag { background: #f39c12; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; }
            </style>
            
            <div style='font-size:14px; font-weight:bold; color:#ddd; margin-bottom:12px;'>How individual features are reviewed, flagged, and standardized:</div>
            
            <table class='rules-table'>
                <tr><th>Feature Analyzed</th><th>Issue Identification Rules (Flagging)</th><th>Standardization Strategy (Fixing)</th></tr>
                <tr>
                    <td class='t-tag'>Legal Company Name & DBA</td>
                    <td><span class='warn-tag'>SPECIAL CHARS</span> Flagged if name contains `#`, `$`, `%`, `*`, `^`, etc.<br><span class='warn-tag'>WHITESPACE</span> Flagged if leading/trailing/double spaces exist.</td>
                    <td>All special characters are deleted. Consecutive spaces are shrunk to a single space. Name is forced into <code>UPPERCASE</code> to ensure exact matching across variants.</td>
                </tr>
                <tr>
                    <td class='t-tag'>Business Address</td>
                    <td><span class='err-tag'>MISSING</span> Flagged if address is completely blank.<br><span class='warn-tag'>FORMAT</span> Flagged if formatting violates standard US postal structure.</td>
                    <td>Determines Type <b>Commercial</b> vs <b>Residential</b> by searching for keywords (<i>suite, bldg, floor, plaza, center</i>). Then strips irregular characters and spaces.</td>
                </tr>
                <tr>
                    <td class='t-tag'>TIN (Tax ID)</td>
                    <td><span class='err-tag'>MISSING</span> Missing TIN invalidates KYC processes.<br><span class='warn-tag'>FORMAT ISSUE</span> Flagged if the count of numbers inside the string is not exactly 9 digits.</td>
                    <td>Safely trims floating decimals (e.g. <code>.0</code>). Strips alphabetical characters and hyphens. Only if it yields exactly 9 digits is it formatted natively as <code>XX-XXXXXXX</code>.</td>
                </tr>
                <tr>
                    <td class='t-tag'>Telephone Numbers</td>
                    <td><span class='warn-tag'>FORMAT ISSUE</span> Flagged if the raw value contains less than 10 or more than 11 true integer digits.</td>
                    <td>Trims floats (<code>.0</code>). Extracts only numeric digits. 10 digits become <code>(XXX) XXX-XXXX</code>. 11 digits starting with 1 become <code>+1 (XXX) XXX-XXXX</code>.</td>
                </tr>
                <tr>
                    <td class='t-tag'>Datetime Columns</td>
                    <td><span class='warn-tag'>FORMAT ISSUE</span> Flagged if string fails ISO parser expectations.</td>
                    <td>String coercion via pandas. Valid structures are mapped to <code>YYYY-MM-DD</code>. Unparseable trash becomes <code>NaT</code> rather than corrupting aggregations.</td>
                </tr>
            </table>
            
            <div style='background:#1b2533; border-left:4px solid #e67e22; padding:8px 16px; margin-bottom:16px; color:#dcdcdc; font-size:15px; border-radius:4px;'>
                <b style='color:#e67e22;'>Entity Resolution (Duplicate Tags)</b><br>
                Once features are standardized using the rules above, the system applies structural tags in priority order to accurately categorize exact clones versus complex Identity groups.
            </div>

            <table class='rules-table'>
                <tr><th>Priority</th><th>Tag</th><th>Rule Logic</th><th>Exclusion Action</th></tr>
                <tr><td class='t-pri'>1</td><td class='t-tag' style='color:#c0392b;'>EXACT_DUPLICATE</td><td>All original fields are byte-for-byte identical to another row.</td><td>The first occurrence is preserved; all later copies are dropped.</td></tr>
                <tr><td class='t-pri'>2</td><td class='t-tag' style='color:#8e44ad;'>IDENTITY_GROUP_FRAUD</td><td>Same TIN + Address group contains at least one confirmed fraud=1 record.</td><td>Preserve the earliest fraud record; exclude all others to prevent signal leakage.</td></tr>
                <tr><td class='t-pri'>3</td><td class='t-tag' style='color:#e67e22;'>IDENTITY_GROUP_CLEAN</td><td>Same TIN + Address group, none are fraud.</td><td>Keep only the most recent updated record; Drop repetitive older ones.</td></tr>
                <tr><td class='t-pri'>4</td><td class='t-tag' style='color:#f1c40f;'>TECH_REAPPLICATION</td><td>Same TIN exactly, differing minor field values (resubmissions).</td><td>Preserve earliest attempt chronologically; drop others.</td></tr>
                <tr><td class='t-pri'>5</td><td class='t-tag' style='color:#c0392b;'>INVALID_FOR_KYC</td><td>Missing Legal Name, TIN, or Postal Code entirely.</td><td>Dropped immediately. These will fail Vendor KYC verification and waste budget.</td></tr>
                
            </table>
            """
        , unsafe_allow_html=True)

    with TAB5:
        st.subheader("✅ Final Cleaned Model Dataset")
        st.markdown("<div class='info-box'>This dataset represents the final pristine records with all engine standardizations applied. Any columns with unresolvable anomalies or missing values should be handled upstream or via imputation.</div>", unsafe_allow_html=True)
        
        # Export logic
        col1, col2 = st.columns([1, 4])
        with col1:
            @st.cache_data
            def convert_csv_final(df):
                return df.to_csv(index=False).encode('utf-8')
            csv_final = convert_csv_final(df_clean)
            st.download_button("⬇️ Download Final CSV", csv_final, "woth_final_cleaned_data.csv", "text/csv", use_container_width=True)
            
        st.markdown("---")
        
        # Visual representation:
        # Instead of just showing df_clean normally, we want to highlight changes.
        # But for 25k records, styling row by row is slow. We can style a paginated slice.
        
        def styling_logic(val, row_idx, col_name):
            if col_name not in df_raw.columns: return ''
            orig = str(df_raw.at[row_idx, col_name])
            clean = str(val)
            if orig != clean and orig != 'nan' and clean != 'nan':
                return 'background-color: #1a4f2c; color: #a8e6cf' # Green for fixed
            return ''
            
        # Select features to show
        cfg = st.session_state.get('cfg', {})
        mapping_cols = list(cfg.values())
        if not mapping_cols:
            mapping_cols = df_clean.columns.tolist()
            
        display_df = df_clean.copy()
        
        # We'll just show the first 100 rows styled to avoid memory limits in Streamlit web browser rendering
        st.markdown(f"**Showing the first 100 rows with green highlights indicating engine corrections:**")
        
        styled_slice = display_df.head(100).style.apply(lambda x: [styling_logic(v, x.name, x.index[i]) for i, v in enumerate(x)], axis=1)
        st.dataframe(styled_slice, use_container_width=True)