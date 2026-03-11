import pandas as pd
import plotly.graph_objects as go
import datetime

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>WOTH IA - Comprehensive Business Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .section {{
            margin-bottom: 60px;
            padding-bottom: 40px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .metrics-container {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 30px;
            gap: 15px;
        }}
        .metric-card {{
            flex: 1;
            background: #fdfdfd;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2980b9;
        }}
        .analyst-notes {{
            background-color: #e8f4f8;
            border-left: 5px solid #3498db;
            padding: 15px;
            margin-top: 20px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
        .analyst-notes h4 {{
            margin-top: 0;
            color: #2980b9;
        }}
        .chart-container {{
            margin: 30px 0;
            overflow-x: auto;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
            color: #333;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            text-align: right;
            color: #7f8c8d;
            font-size: 14px;
        }}
        .rule-box {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            border-left: 5px solid #3498db;
        }}
        .rule-table th {{
            background: #34495e;
            color: #ecf0f1;
        }}
        .rule-table td, .rule-table th {{
            border-bottom: 1px solid #455a64;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Comprehensive Data Quality & Standardization Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>

        {content}

    </div>
</body>
</html>
"""

def generate_html_report(metrics_html, charts_html, tables_html, analyst_notes):
    """
    Assembles the 5-section HTML report from all app tabs.
    """
    content = ""
    
    # ---------------------------------------------------------
    # SECTION 1: Raw Dataset Metrics & Identity Structures
    # ---------------------------------------------------------
    s1_charts = ""
    if charts_html.get('tag_pie') or charts_html.get('tag_bar'):
        s1_charts = f"""
        <div class="chart-container">
            <div style="flex:1;">{charts_html.get('tag_pie', '')}</div>
            <div style="flex:1;">{charts_html.get('tag_bar', '')}</div>
        </div>
        """
        
    content += f"""
    <div class="section">
        <h2>1. Global Dataset Quality Overview (Raw)</h2>
        {_build_analyst_note_html('Raw Data & Identity Resolution Structure', analyst_notes.get('note_tab1', ''))}
        
        <div class="metrics-container">
            {metrics_html.get('tab1_metrics', '')}
        </div>
        
        {s1_charts}
    </div>
    """
    
    # ---------------------------------------------------------
    # SECTION 2: Detailed Issue Report & Filtered Data
    # ---------------------------------------------------------
    s2_charts = ""
    if charts_html.get('sev_bar') or charts_html.get('type_pie'):
        s2_charts = f"""
        <div class="chart-container">
            <div style="flex:1;">{charts_html.get('sev_bar', '')}</div>
            <div style="flex:1;">{charts_html.get('type_pie', '')}</div>
        </div>
        """
        
    s2_tables = ""
    if tables_html.get('filtered_issues'):
        s2_tables = f"""
        <h3>Top Anomalies by Feature & Severity</h3>
        {tables_html['filtered_issues']}
        """

    content += f"""
    <div class="section">
        <h2>2. Detailed Issue Report (Anomalies)</h2>
        {_build_analyst_note_html('Engine Anomaly Flagging', analyst_notes.get('note_tab2', ''))}
        
        {s2_charts}
        {s2_tables}
    </div>
    """
    
    # ---------------------------------------------------------
    # SECTION 3: Clean vs Original Output Comparison
    # ---------------------------------------------------------
    s3_charts = ""
    if charts_html.get('fq_bar_clean'):
        s3_charts += f"""
        <h3>Cleaned Feature Quality Profile</h3>
        <div class="chart-container">
            {charts_html['fq_bar_clean']}
        </div>
        """
        
    s3_unres = ""
    if charts_html.get('unres_bar'):
        s3_unres += f"""
        <div class="chart-container">
            {charts_html['unres_bar']}
        </div>
        """
        
    s3_comp_table = ""
    if tables_html.get('comp_table'):
        s3_comp_table = f"""
        <h3>Row-Level Modification Diffs</h3>
        {tables_html['comp_table']}
        """
        
    content += f"""
    <div class="section">
        <h2>3. Clean vs Original Modifications</h2>
        {_build_analyst_note_html('Cleaning Rules & Unfixable Output', analyst_notes.get('note_tab3', ''))}
        
        <div class="metrics-container">
            {metrics_html.get('tab3_metrics', '')}
        </div>
        
        {s3_charts}
        {s3_unres}
        {s3_comp_table}
    </div>
    """
    
    # ---------------------------------------------------------
    # SECTION 4: Standardization Rules Dictionary
    # ---------------------------------------------------------
    # We statically inject a generic version of the rules table from Tab 4
    content += f"""
    <div class="section">
        <h2>4. Standardization Business Rules & Data Checks</h2>
        {_build_analyst_note_html('Applied Rule Matrices', analyst_notes.get('note_tab4', ''))}
        
        <div class="rule-box">
            <b>Feature-Level Quality Checks & Engine Standardization</b><br>
            The engine scans every explicit feature before calculating statistics. It categorizes invalid syntax and trims/nullifies non-conforming data to enforce standard schemas.
        </div>
        
        <table class="rule-table">
            <tr><th>Feature Domain</th><th>Flag Triggers</th><th>Standardization Resolution</th></tr>
            <tr>
                <td><b>Company/Legal Names</b></td>
                <td>Special Chars (%, ^, $), Whitespaces</td>
                <td>Stripping special characters, shrinking repeated whitespaces, converting to UPPERCASE.</td>
            </tr>
            <tr>
                <td><b>Piped Addresses</b></td>
                <td>Missing / Irregular Pipeline formats</td>
                <td>Aggressively splits by pipe `|`. Cross-references <code>lgl_nm</code> / <code>dba_nm</code> and drops identified company prefixes before joining remaining data into clean arrays.</td>
            </tr>
            <tr>
                <td><b>Tax ID (TIN)</b></td>
                <td>Format Length Error (Not 9 digits)</td>
                <td>Trims trailing `.0` floats. Strips syntaxes. Prepends `0` to exactly 8-digit TINs. Exposes unformatted matrix.</td>
            </tr>
            <tr>
                <td><b>Telephony</b></td>
                <td>Format Length Error (<10 or >11)</td>
                <td>Pulls raw numbers out of strings. Checks length bounding constraint, assigns `+1 (XXX)` or `(XXX)` patterns dynamically.</td>
            </tr>
        </table>
        
        <div class="rule-box" style="border-left: 5px solid #e67e22;">
            <b>Entity Resolution (Duplicative Clustering)</b><br>
            Identifies overlapping entities structurally by cross-collating Name, Address, and TIN. 
        </div>
        <table class="rule-table">
            <tr><th>Tag Priority</th><th>Categorization Logic</th><th>Pipeline Resolution Action</th></tr>
            <tr><td><b>EXACT_DUPLICATE</b></td><td>Row is perfectly byte-for-byte identical.</td><td>Keeps the first row appearing in index; silently drops the rest.</td></tr>
            <tr><td><b>IDENTITY_GROUP_FRAUD</b></td><td>Differing info but same structural entity, with at least 1 fraud flag.</td><td>Prioritizes keeping existing fraud entity to prevent leakage.</td></tr>
            <tr><td><b>IDENTITY_GROUP_CLEAN</b></td><td>Differing info but same structural entity, safely clean.</td><td>Prioritizes preserving newest update entry; drops historical noise.</td></tr>
            <tr><td><b>TECH_REAPPLICATION</b></td><td>TIN is mathematically identical, address shifts.</td><td>Keeps earliest chronology attempt, drops later churn.</td></tr>
            <tr><td><b>INVALID_FOR_KYC</b></td><td>Total omission of name/TIN/zip metadata points.</td><td>Strict drop rule; Vendor API will reject this inherently.</td></tr>
        </table>
    </div>
    """
    
    # ---------------------------------------------------------
    # SECTION 5: Final Cleaned Dataset Export
    # ---------------------------------------------------------
    s5_table = ""
    if tables_html.get('final_table'):
        s5_table = tables_html['final_table']
        
    content += f"""
    <div class="section">
        <h2>5. Final Pristine Model Dataset Overview</h2>
        {_build_analyst_note_html('Dataset Assembly Schema', analyst_notes.get('note_tab5', ''))}
        
        <h3>Dataset Export Schema Preview (Top 50)</h3>
        {s5_table}
    </div>
    """
    
    # Assembly
    return HTML_TEMPLATE.format(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        content=content
    )

def _build_analyst_note_html(section_name, note_text):
    if not note_text or note_text.strip() == "":
        return ""
    
    # Convert newlines to breaks
    formatted_text = str(note_text).replace(chr(10), '<br>')
    
    return f"""
    <div class="analyst-notes">
        <h4>Analyst Notes - {section_name}</h4>
        <p style="margin-bottom:0px; line-height: 1.5;">{formatted_text}</p>
    </div>
    """

def extract_plotly_html(fig: go.Figure) -> str:
    """Converts a plotly figure to an embedded HTML string without full page wrappers."""
    if fig is None: return ""
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
def dict_to_metric_html(label, value, color="#2980b9"):
    return f"""
    <div class="metric-card">
        <div style="font-size: 13px; color: #7f8c8d; text-transform: uppercase; margin-bottom: 5px;">{label}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
    </div>
    """
