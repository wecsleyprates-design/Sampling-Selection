import pandas as pd
import plotly.graph_objects as go
import datetime

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>WOTH IA — Enterprise Data Quality Report</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f0f2f5; color: #2c3e50; margin: 0; padding: 20px 10px; font-size: 14px; line-height: 1.6; }}
.container {{ max-width: 1340px; margin: 0 auto; }}
.rpt-header {{ background: linear-gradient(135deg, #1a2a4a 0%, #2c3e6e 100%);
    color: white; border-radius: 12px; padding: 40px 44px; margin-bottom: 22px; text-align: center; }}
.rpt-header h1 {{ font-size: 28px; font-weight: 700; margin: 0 0 8px; letter-spacing: -0.3px; }}
.rpt-header .sub {{ font-size: 14px; opacity: 0.72; margin: 0 0 10px; }}
.rpt-header .ts  {{ font-size: 11px; opacity: 0.5; }}
.exec {{ background: #fff; border: 1px solid #d4e6f1; border-radius: 10px;
    padding: 28px 34px; margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
.exec h2 {{ font-size: 17px; color: #1a2a4a; margin: 0 0 18px;
    padding-bottom: 10px; border-bottom: 2px solid #d4e6f1; }}
.exec-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
.exec-item {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 15px 12px; text-align: center; }}
.exec-lbl  {{ font-size: 10px; text-transform: uppercase; color: #95a5a6; letter-spacing: 0.5px; margin-bottom: 5px; }}
.exec-val  {{ font-size: 24px; font-weight: 800; line-height: 1.15; }}
.exec-note {{ font-size: 11px; color: #95a5a6; margin-top: 3px; }}
.section {{ background: #fff; border-radius: 10px; padding: 30px 36px;
    margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.07); }}
.sec-hdr {{ display: flex; align-items: center; gap: 12px;
    border-bottom: 2px solid #ecf0f1; padding-bottom: 13px; margin-bottom: 22px; }}
.sec-num {{ background: #1a2a4a; color: #fff; padding: 3px 11px;
    border-radius: 20px; font-size: 12px; font-weight: 700; }}
.section h2 {{ font-size: 19px; font-weight: 700; color: #1a2a4a; margin: 0; }}
.section h3 {{ font-size: 15px; font-weight: 600; color: #2c3e50;
    margin: 22px 0 10px; padding-bottom: 5px; border-bottom: 1px solid #ecf0f1; }}
.section h4 {{ font-size: 13px; font-weight: 600; color: #34495e; margin: 16px 0 7px; }}
.subsec {{ background: #f7f9fc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 22px 26px; margin: 18px 0; }}
.subsec-ttl {{ font-size: 13px; font-weight: 700; color: #1a2a4a; text-transform: uppercase;
    letter-spacing: 0.5px; margin: 0 0 14px; padding-bottom: 8px; border-bottom: 1px solid #d5dde6; }}
.metrics-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px; margin: 14px 0 22px; }}
.metric-card {{ background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 17px 14px; text-align: center; }}
.metric-value {{ font-size: 26px; font-weight: 800; color: #2c3e50;
    line-height: 1.1; margin-bottom: 4px; }}
.metric-label {{ font-size: 10px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }}
.chart-grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 14px 0; }}
.chart-full   {{ margin: 14px 0; }}
.chart-box {{ background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 12px; overflow: hidden; min-height: 40px; }}
.tbl-wrap {{ overflow-x: auto; margin: 12px 0; border: 1px solid #e2e8f0; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th {{ background: #1a2a4a; color: #ecf0f1; padding: 9px 11px;
    text-align: left; font-weight: 600; white-space: nowrap; }}
td {{ padding: 8px 11px; border-bottom: 1px solid #ecf0f1; }}
tr:nth-child(even) td {{ background: #f8fafc; }}
tr:hover td {{ background: #eaf3fb; }}
.interp {{ background: #eaf3fb; border-left: 4px solid #2980b9;
    border-radius: 0 6px 6px 0; padding: 13px 16px; margin: 14px 0;
    font-size: 13px; line-height: 1.7; }}
.interp.warn    {{ background: #fef9e7; border-left-color: #f39c12; }}
.interp.danger  {{ background: #fdedec; border-left-color: #e74c3c; }}
.interp.success {{ background: #eafaf1; border-left-color: #27ae60; }}
.interp b {{ display: block; font-size: 13px; font-weight: 700; margin-bottom: 5px; }}
.a-note {{ background: #fdfffe; border: 1px solid #d5e8d4;
    border-left: 5px solid #27ae60; border-radius: 0 6px 6px 0;
    padding: 13px 17px; margin: 18px 0 14px; }}
.a-note h4 {{ margin: 0 0 6px; color: #1d8348; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px; }}
.a-note p  {{ margin: 0; font-size: 13px; line-height: 1.65; }}
.rule-hdr {{ background: #1a2a4a; color: #ecf0f1; padding: 13px 17px; margin: 12px 0;
    border-radius: 5px; border-left: 5px solid #3498db; font-size: 13px; line-height: 1.6; }}
.rule-hdr.orange {{ border-left-color: #e67e22; }}
.rule-hdr.red    {{ border-left-color: #e74c3c; }}
.rtable {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px; }}
.rtable th {{ background: #2c3e50; color: #ecf0f1; padding: 9px 11px; text-align: left; font-weight: 600; }}
.rtable td {{ padding: 9px 11px; border-bottom: 1px solid #455a64; color: #e8eaf6; background: #1e2d3d; }}
.rtable tr:nth-child(even) td {{ background: #253545; color: #e8eaf6; }}
.rtable tr:hover td {{ background: #2e4560; color: #ffffff; }}
code {{ background: rgba(44,62,80,0.75); color: #aed6f1; padding: 2px 5px; border-radius: 3px; font-size: 11px; }}
.badge {{ display: inline-block; padding: 2px 9px; border-radius: 12px;
    font-size: 11px; font-weight: 700; color: white; margin: 1px; }}
.b-red   {{ background: #e74c3c; }}
.b-green {{ background: #27ae60; }}
.b-blue  {{ background: #2980b9; }}
.b-orng  {{ background: #e67e22; }}
.footer {{ text-align: center; padding: 22px; color: #95a5a6;
    font-size: 12px; border-top: 1px solid #ecf0f1; margin-top: 18px; }}
.metrics-container {{
    display: flex; justify-content: space-between; margin-bottom: 30px; gap: 15px;
}}
@media print {{ body {{ background: white; padding: 0; }} .section, .exec {{ box-shadow: none; }} }}
@media (max-width: 768px) {{ .chart-grid-2, .exec-row {{ grid-template-columns: 1fr; }} }}
</style></head>
<body>
<div class="container">
<div class="rpt-header">
    <h1>&#129529; WOTH IA &mdash; Enterprise Data Quality Report</h1>
    <p class="sub">Standardization &middot; Anomaly Detection &middot; Duplicate Identification &middot; Compliance Readiness</p>
    <p class="ts">Generated: {timestamp}</p>
</div>
{content}
<div class="footer">WOTH IA Data Check Engine &nbsp;&middot;&nbsp; Internal Use Only &nbsp;&middot;&nbsp; {timestamp}</div>
</div>
</body>
</html>
"""

def generate_html_report(metrics_html, charts_html, tables_html, analyst_notes, raw_stats=None):
    """
    Assembles the full 6-part HTML report (exec summary + 5 sections) from all app tabs.
    raw_stats: optional dict with keys: total_rows, good_rows, rows_with_issues, missing_values,
               tin_dups, addr_dups, unfixable, overall_fq_raw, overall_fq_clean
    """
    rs            = raw_stats or {}
    total         = rs.get('total_rows', 0)
    good          = rs.get('good_rows', 0)
    iss           = rs.get('rows_with_issues', 0)
    missing       = rs.get('missing_values', 0)
    tin_dups      = rs.get('tin_dups', 0)
    addr_dups     = rs.get('addr_dups', 0)
    unfixable     = rs.get('unfixable', 0)
    fq_raw        = rs.get('overall_fq_raw', 0)
    fq_clean      = rs.get('overall_fq_clean', 0)
    pct_iss       = round(iss  / total * 100, 1) if total else 0
    pct_tin       = round(tin_dups  / total * 100, 1) if total else 0
    pct_addr      = round(addr_dups / total * 100, 1) if total else 0
    pct_good      = round(good / total * 100, 1) if total else 0
    improvement   = round(fq_clean - fq_raw, 1) if fq_clean and fq_raw else 0

    def _ic(v, g, r):
        return "#27ae60" if v <= g else ("#f39c12" if v <= r else "#e74c3c")
    def _pic(v):
        return "#27ae60" if v > 80 else ("#f39c12" if v > 50 else "#e74c3c")

    # ─────────────────────────────── EXECUTIVE SUMMARY ───────────────────────────────
    exec_s = f"""
    <div class="exec">
        <h2>&#128202; Executive Summary &mdash; Dataset Health Dashboard</h2>
        <div class="exec-row">
            <div class="exec-item">
                <div class="exec-lbl">Total Records</div>
                <div class="exec-val" style="color:#2980b9">{total:,}</div>
                <div class="exec-note">Dataset size</div>
            </div>
            <div class="exec-item">
                <div class="exec-lbl">Clean Records</div>
                <div class="exec-val" style="color:#27ae60">{good:,}</div>
                <div class="exec-note">{pct_good}% of total</div>
            </div>
            <div class="exec-item">
                <div class="exec-lbl">Records w/ Issues</div>
                <div class="exec-val" style="color:{_ic(pct_iss,10,30)}">{iss:,}</div>
                <div class="exec-note">{pct_iss}% of total</div>
            </div>
            <div class="exec-item">
                <div class="exec-lbl">Missing Values</div>
                <div class="exec-val" style="color:{'#e74c3c' if missing>0 else '#27ae60'}">{missing:,}</div>
                <div class="exec-note">Across all fields</div>
            </div>
            <div class="exec-item">
                <div class="exec-lbl">TIN Duplicates</div>
                <div class="exec-val" style="color:{'#e74c3c' if tin_dups>0 else '#27ae60'}">{tin_dups:,}</div>
                <div class="exec-note">{pct_tin}% of total</div>
            </div>
            <div class="exec-item">
                <div class="exec-lbl">Address Duplicates</div>
                <div class="exec-val" style="color:{'#e74c3c' if addr_dups>0 else '#27ae60'}">{addr_dups:,}</div>
                <div class="exec-note">{pct_addr}% of total</div>
            </div>
            <div class="exec-item">
                <div class="exec-lbl">Raw DQ Score</div>
                <div class="exec-val" style="color:{_pic(fq_raw)}">{fq_raw}%</div>
                <div class="exec-note">Avg feature quality</div>
            </div>
            <div class="exec-item">
                <div class="exec-lbl">Clean DQ Score</div>
                <div class="exec-val" style="color:{_pic(fq_clean)}">{fq_clean}%</div>
                <div class="exec-note">Post-standardization</div>
            </div>
            <div class="exec-item">
                <div class="exec-lbl">Unresolvable</div>
                <div class="exec-val" style="color:{'#e74c3c' if unfixable>0 else '#27ae60'}">{unfixable:,}</div>
                <div class="exec-note">Cannot be auto-fixed</div>
            </div>
        </div>
    </div>"""

    content = exec_s

    # ─────────────────────────────── SECTION 1 ───────────────────────────────
    health  = "Strong &#10003;" if pct_iss < 20 else ("Moderate &#9888;" if pct_iss < 50 else "Weak &#128308;")
    ic1     = "success" if pct_iss < 20 else ("warn" if pct_iss < 50 else "danger")
    miss_txt = (f"&#9888; {missing:,} blank/null values detected across the schema &mdash; "
                "these create gaps in downstream analytics and model features.")  if missing > 0 else \
               "&#10003; No blank/null values detected across the full schema."
    fq_txt = (f"Average raw feature quality score of {fq_raw}% indicates "
              + ("strong structural conformance." if fq_raw > 80
                 else ("moderate conformance &mdash; some fields require attention." if fq_raw > 50
                       else "poor structural quality &mdash; significant remediation required."))) if fq_raw > 0 else ""

    fq_raw_chart = ""
    if charts_html.get('fq_bar'):
        fq_raw_chart = f"""
        <h3>Feature Quality Scores &mdash; Raw Dataset</h3>
        <div class="chart-full"><div class="chart-box">{charts_html['fq_bar']}</div></div>"""

    content += f"""
    <div class="section">
        <div class="sec-hdr"><span class="sec-num">SECTION 1</span><h2>Global Dataset Quality Overview (Raw)</h2></div>
        <div class="metrics-row">{metrics_html.get('tab1_metrics', '')}</div>
        {fq_raw_chart}
        <div class="interp {ic1}">
            <b>Automatic Interpretation &mdash; Raw Data Quality</b>
            The engine ingested <strong>{total:,}</strong> total records.
            <strong>{iss:,}</strong> records ({pct_iss}%) exhibit at least one format error, whitespace anomaly, or missing-value flag.
            Overall raw data health is classified as: <strong>{health}</strong>.<br>
            {miss_txt}<br>{fq_txt}
        </div>
        {_note_html('Raw Data &amp; Quality Metrics', analyst_notes.get('note_tab1', ''))}
    </div>"""
    
    # ─────────────────────────────── SECTION 2 ───────────────────────────────
    # 2A: Anomaly Analysis
    anomaly_charts = ""
    if charts_html.get('sev_bar') or charts_html.get('type_pie'):
        anomaly_charts = f"""
        <div class="chart-grid-2">
            <div class="chart-box">{charts_html.get('sev_bar', '')}</div>
            <div class="chart-box">{charts_html.get('type_pie', '')}</div>
        </div>"""

    issues_table = ""
    if tables_html.get('filtered_issues'):
        issues_table = f"""
        <h4>Top Anomalies by Feature &amp; Severity</h4>
        <div class="tbl-wrap">{tables_html['filtered_issues']}</div>"""

    anomaly_interp = ('<div class="interp warn"><b>Anomalies Detected</b>'
                      'Format errors, whitespace issues, or missing values were identified. '
                      'Review the charts above for severity distribution and the table for specific field-level failures.</div>'
                      if charts_html.get('sev_bar') else
                      '<div class="interp success"><b>No Anomalies Found</b>'
                      'No format, whitespace, or missing-value anomalies were detected across the mapped columns.</div>')

    # 2B: Duplicate Detection
    dup_charts = ""
    if charts_html.get('tin_dup_bar') or charts_html.get('addr_dup_bar'):
        dup_charts = f"""
        <div class="chart-grid-2">
            <div class="chart-box">{charts_html.get('tin_dup_bar', '')}</div>
            <div class="chart-box">{charts_html.get('addr_dup_bar', '')}</div>
        </div>"""

    dup_tin_tbl = ""
    if tables_html.get('dup_tin_table'):
        dup_tin_tbl = f"""
        <h4>&#128308; Records with Duplicate TIN (Tax ID appears more than once in dataset)</h4>
        <div class="tbl-wrap">{tables_html['dup_tin_table']}</div>"""

    dup_addr_tbl = ""
    if tables_html.get('dup_addr_table'):
        dup_addr_tbl = f"""
        <h4>&#128308; Records with Duplicate Address Combination (address_1 + zip + city appears more than once)</h4>
        <div class="tbl-wrap">{tables_html['dup_addr_table']}</div>"""

    tin_line  = (f"<span class='badge b-red'>TIN Duplicates: {tin_dups:,} ({pct_tin}%)</span> &nbsp;"
                 f"{tin_dups:,} records share a Tax ID (EIN) with at least one other record. "
                 "Indicates potential duplicate business entity registrations, data entry overlap, or multiple accounts for the same entity. "
                 "These records require manual review before inclusion in regulatory or analytics datasets.") \
                if tin_dups > 0 else \
                "<span class='badge b-green'>TIN: All Unique &#10003;</span> &nbsp;No TIN duplicates detected &mdash; all Tax IDs appear exactly once in this dataset."

    addr_line = (f"<span class='badge b-red'>Address Duplicates: {addr_dups:,} ({pct_addr}%)</span> &nbsp;"
                 f"{addr_dups:,} records share an identical combination of address_1 + zip_code + city with at least one other record. "
                 "Co-located businesses or inadvertent data entry duplication are the most likely causes. "
                 "Cross-reference TIN values for these records to determine whether they represent distinct legal entities.") \
                if addr_dups > 0 else \
                "<span class='badge b-green'>Address: All Unique &#10003;</span> &nbsp;No address-combination duplicates &mdash; all physical locations are distinct."

    dup_title = "Duplicate Assessment &mdash; PASS" if (tin_dups + addr_dups) == 0 else "Duplicate Assessment &mdash; Action Required"
    dup_class = "success" if (tin_dups + addr_dups) == 0 else "danger"

    action_box = ""
    if (tin_dups + addr_dups) > 0:
        action_box = f"""
        <div class="interp warn" style="margin-top:16px;">
            <b>Recommended Actions</b>
            1. Cross-reference duplicate TIN records with your source CRM/ERP to confirm whether they represent the same or different legal entities.<br>
            2. For address duplicates, verify whether co-location is expected (e.g., registered agents) or inadvertent (data entry errors).<br>
            3. Use <code>Duplicate_TIN_CHECK</code> and <code>Duplicate_Address_CHECK</code> flags in the final export to filter and triage duplicates efficiently.<br>
            4. Consider deduplication or entity-resolution workflows before loading this dataset into production systems.
        </div>"""

    content += f"""
    <div class="section">
        <div class="sec-hdr"><span class="sec-num">SECTION 2</span><h2>Detailed Issue Report &amp; Duplicate Detection</h2></div>

        <div class="subsec">
            <div class="subsec-ttl">&#9888; A &mdash; Anomaly &amp; Format Issue Analysis</div>
            {anomaly_charts}
            {issues_table}
            {anomaly_interp}
        </div>

        <div class="subsec">
            <div class="subsec-ttl">&#128269; B &mdash; Duplicate Detection Analysis (TIN &amp; Address)</div>
            <div class="metrics-row">{metrics_html.get('dup_metrics', '')}</div>
            {dup_charts}
            <div class="interp {dup_class}">
                <b>{dup_title}</b>
                {tin_line}<br><br>{addr_line}
            </div>
            {dup_tin_tbl}
            {dup_addr_tbl}
            {action_box}
        </div>

        {_note_html('Issue Report &amp; Duplicate Detection', analyst_notes.get('note_tab2', ''))}
    </div>"""

    # ─────────────────────────────── SECTION 3 ───────────────────────────────
    clean_fq_chart = ""
    if charts_html.get('fq_bar_clean'):
        clean_fq_chart = f"""
        <h3>Feature Quality Scores &mdash; Post-Standardization</h3>
        <div class="chart-full"><div class="chart-box">{charts_html['fq_bar_clean']}</div></div>"""

    unres_chart = ""
    if charts_html.get('unres_bar'):
        unres_chart = f"""
        <h3>&#128680; Unresolved Engine Failures by Feature</h3>
        <div class="chart-full"><div class="chart-box">{charts_html['unres_bar']}</div></div>"""

    comp_tbl = ""
    if tables_html.get('comp_table'):
        comp_tbl = f"""
        <h3>Row-Level Modification Comparison</h3>
        <div class="tbl-wrap">{tables_html['comp_table']}</div>"""

    clean_ic   = "success" if improvement >= 0 else "danger"
    imp_txt    = (f"Post-standardization average feature quality is <strong>{fq_clean}%</strong>"
                  + (f" &mdash; an improvement of <strong>+{improvement}%</strong> over the raw score." if improvement > 0
                     else (" &mdash; same as the raw score; no quality gain from standardization." if improvement == 0
                           else f" &mdash; score decreased by {abs(improvement)}%; review unresolvable records below."))) \
                 if fq_clean > 0 else "Feature quality scoring was not computed for the post-clean dataset."
    unfx_txt   = (f" <strong>{unfixable:,} records</strong> could not be automatically standardized and require manual remediation."
                  if unfixable > 0 else
                  " All flagged records were successfully standardized or confirmed as structurally valid.")

    content += f"""
    <div class="section">
        <div class="sec-hdr"><span class="sec-num">SECTION 3</span><h2>Clean vs Original Modifications</h2></div>
        <div class="metrics-row">{metrics_html.get('tab3_metrics', '')}</div>
        {clean_fq_chart}
        <div class="interp {clean_ic}">
            <b>Automatic Interpretation &mdash; Standardization Results</b>
            {imp_txt}{unfx_txt}
        </div>
        {unres_chart}
        {comp_tbl}
        {_note_html('Clean vs Original &amp; Export', analyst_notes.get('note_tab3', ''))}
    </div>"""

    # ─────────────────────────────── SECTION 4 ───────────────────────────────
    content += f"""
    <div class="section">
        <div class="sec-hdr"><span class="sec-num">SECTION 4</span><h2>Standardization Business Rules &amp; Data Checks</h2></div>

        <div class="rule-hdr">
            <strong>Feature-Level Quality Checks &amp; Standardization Engine</strong><br>
            The Worth IA engine scans every feature cell-by-cell. It categorizes poor-quality records using predefined rules, scores feature quality, and applies aggressive standardization where algorithmically possible.
        </div>
        <table class="rtable">
            <tr><th>Feature Domain</th><th>Issue Identification (Flagging Criteria)</th><th>Standardization Strategy</th></tr>
            <tr>
                <td><strong>Legal Name &amp; DBA</strong></td>
                <td>Special characters (<code>#$%*^</code>); leading/trailing/double whitespace.</td>
                <td>Special characters removed. Spaces collapsed. Forced to <code>UPPERCASE</code>.</td>
            </tr>
            <tr>
                <td><strong>Business Address (Piped)</strong></td>
                <td>Missing value; irregular <code>|</code> delimiter structure.</td>
                <td>Splits on <code>|</code> into discrete <code>_worth</code> columns. Removes company-name prefix. Joins into clean <code>full_address_worth</code>.</td>
            </tr>
            <tr>
                <td><strong>Tax ID (TIN / EIN)</strong></td>
                <td>Missing value (KYC invalidation); digit count &ne;9 after stripping non-numerics.</td>
                <td>Strips <code>.0</code> float artifacts. Removes non-digit chars. Auto-prepends <code>0</code> to 8-digit TINs. Outputs <code>XX-XXXXXXX</code>.</td>
            </tr>
            <tr>
                <td><strong>Telephone Numbers</strong></td>
                <td>Raw integer count &lt;10 or &gt;11 digits.</td>
                <td>Extracts digits only. 10-digit &rarr; <code>(XXX) XXX-XXXX</code>. 11-digit starting with 1 &rarr; <code>+1 (XXX) XXX-XXXX</code>.</td>
            </tr>
            <tr>
                <td><strong>Datetime Fields</strong></td>
                <td>String fails ISO 8601 parser.</td>
                <td>Coerced via pandas. Valid &rarr; <code>YYYY-MM-DD</code>. Unparseable &rarr; <code>NaT</code>.</td>
            </tr>
            <tr>
                <td><strong>ZIP / Postal Code</strong></td>
                <td>Digit count &ne;5 after stripping non-numerics.</td>
                <td>Validates 5-digit US ZIP. Invalid formats flagged as <code>FORMAT_ISSUE</code>.</td>
            </tr>
        </table>

        <div class="rule-hdr orange" style="margin-top:24px;">
            <strong>Duplicate Detection Rules</strong><br>
            After all standardization is applied, the engine identifies two classes of duplicates based on key business identifiers and surfaces them as binary flags in the final export.
        </div>
        <table class="rtable">
            <tr><th>Detection Type</th><th>Matching Criteria</th><th>Output Flag</th><th>Values</th><th>Business Impact</th></tr>
            <tr>
                <td><strong>TIN Duplicate</strong></td>
                <td>Standardized Tax ID (EIN) appears &gt;1 time. Only non-empty TINs considered. Matching is format-insensitive after normalization.</td>
                <td><code>Duplicate_TIN_CHECK</code></td>
                <td><code>Yes</code> &mdash; Duplicate<br><code>No</code> &mdash; Unique</td>
                <td>Indicates potential duplicate business entities. Critical for KYC/AML compliance.</td>
            </tr>
            <tr>
                <td><strong>Address Duplicate</strong></td>
                <td>Exact match on <code>address_1_worth</code> + <code>zip_code_worth</code> + <code>city_worth</code> appearing &gt;1 time. All three must be non-empty.</td>
                <td><code>Duplicate_Address_CHECK</code></td>
                <td><code>Yes</code> &mdash; Duplicate<br><code>No</code> &mdash; Unique</td>
                <td>Identifies co-located or physically identical entities. May represent legitimate branches or data defects.</td>
            </tr>
        </table>

        {_note_html('Standardization Rules', analyst_notes.get('note_tab4', ''))}
    </div>"""

    # ─────────────────────────────── SECTION 5 ───────────────────────────────
    final_tbl = ""
    if tables_html.get('final_table'):
        final_tbl = f"""
        <h3>Final Export Data Preview (Top 50 Records)</h3>
        <div class="tbl-wrap">{tables_html['final_table']}</div>"""

    fin_tin_txt  = (f"&#128308; <strong>{tin_dups:,}</strong> records carry <code>Duplicate_TIN_CHECK = Yes</code> &mdash; shared Tax ID with at least one other record."
                    if tin_dups > 0 else
                    "&#10003; <strong>Duplicate_TIN_CHECK</strong>: All records are unique by TIN.")
    fin_addr_txt = (f"&#128308; <strong>{addr_dups:,}</strong> records carry <code>Duplicate_Address_CHECK = Yes</code> &mdash; shared address combination with at least one other record."
                    if addr_dups > 0 else
                    "&#10003; <strong>Duplicate_Address_CHECK</strong>: All records are unique by address combination.")
    fin_action   = ("&#9888; Review and resolve duplicate records before loading into production systems, regulatory filings, or ML pipelines."
                    if (tin_dups + addr_dups) > 0 else
                    "&#10003; This dataset is free of TIN and address-combination duplicates and is ready for downstream consumption.")

    content += f"""
    <div class="section">
        <div class="sec-hdr"><span class="sec-num">SECTION 5</span><h2>Final Pristine Dataset with Duplicate Flags</h2></div>

        <div class="rule-hdr red">
            <strong>Output Schema &amp; Duplicate Detection Flags</strong><br>
            The final export contains three column groups:<br>
            <strong>(1)</strong> <code>_received</code> &mdash; original raw values exactly as ingested;<br>
            <strong>(2)</strong> <code>_worth</code> &mdash; standardized, cleaned values produced by the Worth IA engine;<br>
            <strong>(3)</strong> Duplicate flags &mdash; <code>Duplicate_TIN_CHECK</code> and <code>Duplicate_Address_CHECK</code> (Yes/No).<br><br>
            Excluded from export: <code>Tag_worth</code>, <code>DuplicateRuleKey_worth</code>, and internal engine tracking columns.
        </div>

        <div class="metrics-row">{metrics_html.get('dup_metrics', '')}</div>

        <div class="interp {'danger' if (tin_dups + addr_dups) > 0 else 'success'}">
            <b>Dataset Completeness &amp; Duplicate Summary</b>
            Total records in final export: <strong>{total:,}</strong>.<br>
            {fin_tin_txt}<br>
            {fin_addr_txt}<br>
            {fin_action}
        </div>

        {final_tbl}
        {_note_html('Final Dataset &amp; Export Schema', analyst_notes.get('note_tab5', ''))}
    </div>"""

    return HTML_TEMPLATE.format(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        content=content
    )


def _note_html(section_name, note_text):
    """Build the analyst-notes HTML block; returns empty string if no note."""
    if not note_text or str(note_text).strip() == "":
        return ""
    formatted = str(note_text).replace('\n', '<br>')
    return (f'<div class="a-note"><h4>Analyst Notes &mdash; {section_name}</h4>'
            f'<p>{formatted}</p></div>')


# Backward-compat alias used by older callers
_build_analyst_note_html = _note_html


def extract_plotly_html(fig) -> str:
    """Converts a plotly figure to an embedded HTML string without full page wrappers."""
    if fig is None:
        return ""
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def dict_to_metric_html(label, value, color="#2980b9"):
    return (
        f'<div class="metric-card" style="border-top:3px solid {color};">'
        f'<div class="metric-value" style="color:{color};">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>'
    )



