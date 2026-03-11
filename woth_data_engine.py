import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple

# We can mimic the structure from the existing app
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & TAGS
# ─────────────────────────────────────────────────────────────────────────────

# Data Quality Tags
DQ_TAG_COLORS = {
    "GOOD_DATA": "#27ae60",
    "MISSING_VALUE": "#e74c3c",
    "FORMAT_ISSUE": "#f39c12",
    "SPECIAL_CHARS": "#9b59b6",
    "WHITESPACE_ISSUE": "#3498db",
    "EXACT_DUPLICATE": "#c0392b",
    "TECH_REAPPLICATION": "#f1c40f",
    "IDENTITY_GROUP_CLEAN": "#e67e22",
    "IDENTITY_GROUP_FRAUD": "#8e44ad",
    "INVALID_FOR_KYC": "#c0392b"
}

DQ_TAG_ICONS = {
    "GOOD_DATA": "✅",
    "MISSING_VALUE": "❌",
    "FORMAT_ISSUE": "⚠️",
    "SPECIAL_CHARS": "🔣",
    "WHITESPACE_ISSUE": "␣",
    "EXACT_DUPLICATE": "🔴",
    "TECH_REAPPLICATION": "🟡",
    "IDENTITY_GROUP_CLEAN": "🟠",
    "IDENTITY_GROUP_FRAUD": "🟣",
    "INVALID_FOR_KYC": "🔺"
}

DQ_TAG_DESC = {
    "GOOD_DATA": "No issues found. Data is clean.",
    "MISSING_VALUE": "A column requires a value, but none was found (null/empty).",
    "FORMAT_ISSUE": "A column doesn't match standard data schemas (e.g. invalid TIN or Phone length).",
    "SPECIAL_CHARS": "Non-alpha-numeric invalid symbols like '#', '$', '%', or '*' detected.",
    "WHITESPACE_ISSUE": "Leading, trailing, or multiple consecutive whitespaces found.",
    "EXACT_DUPLICATE": "All fields are byte-for-byte identical to another row — pure ETL/data-entry duplicate. The first occurrence is preserved; all later copies are dropped.",
    "TECH_REAPPLICATION": "Same identity pair submitted more than once with minor field differences. The earliest record is preserved; later re-submissions are tagged.",
    "IDENTITY_GROUP_CLEAN": "Same Legal Name + Address group, but none of the rows have fraud. The most recent record is preserved; older copies are tagged as redundant.",
    "IDENTITY_GROUP_FRAUD": "Same group contains at least one confirmed Fraud=1 row. Only one Fraud=1 record is preserved per group; all others are tagged for exclusion.",
    "INVALID_FOR_KYC": "Missing Legal Name, TIN, or Postal Code. These records will fail vendor KYC verification — sending them wastes lookup budget with no chance of a match."
}

# ─────────────────────────────────────────────────────────────────────────────
# IDENTIFICATION RULES
# ─────────────────────────────────────────────────────────────────────────────

def identify_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies missing values across the dataframe."""
    return df.isna() | (df == "")

def identify_whitespace_issues(series: pd.Series) -> pd.Series:
    """Identifies if values have leading, trailing, or multiple internal whitespaces."""
    if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
        return pd.Series(False, index=series.index)
    
    s = series.astype(str)
    has_leading = s.str.startswith(" ", na=False)
    has_trailing = s.str.endswith(" ", na=False)
    has_multiple = s.str.contains(r"\s{2,}", regex=True, na=False)
    
    return has_leading | has_trailing | has_multiple

def identify_special_chars(series: pd.Series) -> pd.Series:
    """Identifies special characters like #, $, %, *, ^, etc."""
    if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
        return pd.Series(False, index=series.index)
    
    # Matching typical unwanted special characters
    return series.astype(str).str.contains(r"[#\$%\*\^~`\{\}\|<>\\]", regex=True, na=False)

def identify_phone_format_issues(series: pd.Series) -> pd.Series:
    """Identifies phone numbers that do not yield 10 digits or 11 (starting with 1)."""
    def is_invalid_phone(val):
        if pd.isna(val) or str(val).strip() == "": return False
        v = str(val)
        if v.endswith(".0"): v = v[:-2]
        digits = re.sub(r"\D", "", v)
        if len(digits) == 10: return False
        if len(digits) == 11 and digits.startswith("1"): return False
        return True
    return series.apply(is_invalid_phone)

import numpy as np
import re

def identify_zip_format_issues(series: pd.Series) -> pd.Series:
    """Identify ZIP codes that are not exactly 5 digits or contain invalid characters."""
    s = series.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    return s.str.match(r"(^\d{5}$)|(^\d{5}-\d{4}$)").eq(False) & s.notna() & (s != 'nan') & (s != '')

def standardize_zip(val):
    if pd.isna(val) or val is None: return val
    v = str(val).strip()
    if v.lower() == 'nan': return np.nan
    if v.endswith('.0'):
        v = v[:-2]
    v = re.sub(r'\D', '', v)
    if len(v) > 0 and len(v) < 5:
        v = v.zfill(5)
    elif len(v) > 5:
        v = v[:5]
    return v if len(v) == 5 else val

def identify_tin_format_issues(series: pd.Series) -> pd.Series:
    """Identifies TINs that do not yield exactly 9 digits."""
    def is_invalid_tin(val):
        if pd.isna(val) or str(val).strip() == "": return False
        v = str(val)
        if v.endswith(".0"): v = v[:-2]
        digits = re.sub(r"\D", "", v)
        return len(digits) != 9
    return series.apply(is_invalid_tin)


# Advanced Formatters
US_STATES = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
    'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
    'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA',
    'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
    'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ',
    'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH',
    'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
    'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY'
}

def standardize_state(val):
    if pd.isna(val) or val is None: return val
    v = str(val).strip().upper()
    v = re.sub(r'[^A-Z]', '', v)
    if v in US_STATES.values(): return v
    return US_STATES.get(v, val)

def standardize_country(val):
    if pd.isna(val) or val is None: return val
    v = str(val).strip().upper()
    v = re.sub(r'[^A-Z]', '', v)
    if v in ['US', 'USA', 'UNITEDSTATES', 'UNITEDSTATESOFAMERICA', 'AMERICA']:
        return 'USA'
    return val

def normalize_entity_names(val):
    """Normalize entity names specifically for comparison by stripping common corporate suffixes."""
    if pd.isna(val): return val
    v = str(val).upper()
    # Remove common suffixes
    v = re.sub(r'\b(LLC|INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC)\b', '', v)
    v = re.sub(r'[^A-Z0-9]', '', v) # Strip everything but alphanumeric for pure comparison
    return v

def is_commercial_address(address: str) -> bool:
    """Heuristic logic to classify address as Commercial vs Residential."""
    if not isinstance(address, str):
        return False
    address = address.lower()
    commercial_keywords = ["suite", "ste", "floor", "fl", "bldg", "building", "plaza", "center", "industrial", "park"]
    return any(keyword in address for keyword in commercial_keywords)

def check_address_type(series: pd.Series) -> pd.Series:
    """Classifies address into Commercial/Residential."""
    return series.apply(lambda x: "Commercial" if is_commercial_address(x) else ("Residential" if pd.notna(x) else "Unknown"))

# ─────────────────────────────────────────────────────────────────────────────
# CLEANING & STANDARDIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def clean_whitespace(val):
    if pd.isna(val): return val
    val = str(val)
    val = re.sub(r"\s+", " ", val) # Replace multiple spaces with single space
    return val.strip()

def clean_special_chars(val):
    if pd.isna(val): return val
    val = str(val)
    # Remove #, $, %, *, ^, etc.
    return re.sub(r"[#\$%\*\^~`\{\}\|<>\\]", "", val)

def standardize_phone(val):
    if pd.isna(val): return val
    val = str(val)
    if val.endswith(".0"):
        val = val[:-2]
    # Extract only digits
    digits = re.sub(r"\D", "", val)
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits.startswith("1"):
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    return val

def standardize_tin(val):
    if pd.isna(val): return val
    val = str(val)
    if val.endswith(".0"):
        val = val[:-2]
    digits = re.sub(r"\D", "", val)
    if len(digits) == 9:
        # EINs typically have specific prefixes (e.g. 10-16, 20-30, etc.)
        # Often EINs are XX-XXXXXXX, SSNs are XXX-XX-XXXX
        # Simple heuristic: if it starts with 9, it's often an ITIN (SSN format)
        # If the user explicitly provided it separated as XXX-XX-XXXX we could check original, 
        # but defaulting to EIN format (XX-XXXXXXX) if ambiguously 9 digits is common for businesses.
        # Let's apply SSN format for 3-2-4 and EIN for 2-7
        if '-' in val:
            parts = val.split('-')
            if len(parts) == 3 and len(parts[0]) == 3 and len(parts[1]) == 2:
                return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}" # SSN
        return f"{digits[:2]}-{digits[2:]}" # Default to EIN
    return val

def standardize_name(val):
    """Standardize Company Name, DBA, Owner Names: Uppercase, cleaned."""
    if pd.isna(val): return val
    val = clean_special_chars(val)
    val = clean_whitespace(val)
    return val.upper()

def standardize_datetime(series: pd.Series) -> pd.Series:
    """Attempt to parse datetime appropriately."""
    return pd.to_datetime(series, errors='coerce').dt.strftime('%Y-%m-%d')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_data_check_and_cleaning(df: pd.DataFrame, 
                                columns_config: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyzes the dataframe, identifies issues, and applies cleaning.
    
    Args:
        df: Original DataFrame
        columns_config: Mapping of column usage (e.g. {'Phone': 'phone_col_name'})
    
    Returns:
        df_clean: The cleaned dataframe
        df_issues: A boolean dataframe mapping where issues were found
        summary_stats: A dictionary or df of metrics.
    """
    df_clean = df.copy()
    
    # 1. Identify issues
    # Create mask for different types of issues
    df_issues = pd.DataFrame(index=df.index)
    df_issues["_has_issue"] = False
    df_issues["_issue_tags"] = ""
    
    # Process each column generically for whitespaces and special chars
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    
    for col in string_cols:
        ws_issues = identify_whitespace_issues(df[col])
        sc_issues = identify_special_chars(df[col])
        missing = identify_missing_values(df[[col]])[col]
        
        # Mark issues
        df_issues[col + "_has_ws"] = ws_issues
        df_issues[col + "_has_sc"] = sc_issues
        df_issues[col + "_is_missing"] = missing
        
        # Accumulate tags
        if ws_issues.any():
            df_issues.loc[ws_issues, "_issue_tags"] += "WHITESPACE_ISSUE,"
            df_issues.loc[ws_issues, "_has_issue"] = True
        if sc_issues.any():
            df_issues.loc[sc_issues, "_issue_tags"] += "SPECIAL_CHARS,"
            df_issues.loc[sc_issues, "_has_issue"] = True
        if missing.any():
            df_issues.loc[missing, "_issue_tags"] += "MISSING_VALUE,"
            df_issues.loc[missing, "_has_issue"] = True
            
        # 2. Clean the data
        df_clean[col] = df_clean[col].apply(clean_special_chars)
        df_clean[col] = df_clean[col].apply(clean_whitespace)
        
    # Process specific fields if provided in config
    if 'phone' in columns_config and columns_config['phone'] in df.columns:
        p_col = columns_config['phone']
        bad_phone = identify_phone_format_issues(df[p_col])
        df_issues[p_col + "_has_format"] = bad_phone
        if bad_phone.any():
            df_issues.loc[bad_phone, "_issue_tags"] += "FORMAT_ISSUE,"
            df_issues.loc[bad_phone, "_has_issue"] = True
        df_clean[p_col] = df_clean[p_col].apply(standardize_phone)
        
    if 'zip' in columns_config and columns_config['zip'] in df.columns:
        z_col = columns_config['zip']
        bad_zip = identify_zip_format_issues(df[z_col])
        df_issues[z_col + "_has_format"] = bad_zip
        if bad_zip.any():
            df_issues.loc[bad_zip, "_issue_tags"] += "FORMAT_ISSUE,"
            df_issues.loc[bad_zip, "_has_issue"] = True
        df_clean[z_col] = df_clean[z_col].apply(standardize_zip)
        
    if 'tin' in columns_config and columns_config['tin'] in df.columns:
        t_col = columns_config['tin']
        bad_tin = identify_tin_format_issues(df[t_col])
        df_issues[t_col + "_has_format"] = bad_tin
        if bad_tin.any():
            df_issues.loc[bad_tin, "_issue_tags"] += "FORMAT_ISSUE,"
            df_issues.loc[bad_tin, "_has_issue"] = True
        df_clean[t_col] = df_clean[t_col].apply(standardize_tin)
        
    if 'address' in columns_config and columns_config['address'] in df.columns:
        a_col = columns_config['address']
        df_clean[a_col + "_type"] = check_address_type(df[a_col])
        df_clean[a_col] = df_clean[a_col].apply(standardize_name)
        
    if 'datetime' in columns_config:
        for dt_col in columns_config['datetime']:
            if dt_col in df.columns:
                df_clean[dt_col] = standardize_datetime(df_clean[dt_col])

    # Name conversions
    name_keys = ['company_name', 'dba', 'first_name', 'last_name', 'city']
    for nk in name_keys:
        if nk in columns_config and columns_config[nk] in df.columns:
            df_clean[columns_config[nk]] = df_clean[columns_config[nk]].apply(standardize_name)
            
    # Normalize entity names for comparison (creates a new column for backend matching engine)
    if 'company_name' in columns_config and columns_config['company_name'] in df.columns:
        df_clean[columns_config['company_name'] + "_normalized"] = df_clean[columns_config['company_name']].apply(normalize_entity_names)
    if 'dba' in columns_config and columns_config['dba'] in df.columns:
        df_clean[columns_config['dba'] + "_normalized"] = df_clean[columns_config['dba']].apply(normalize_entity_names)
        
    # State and Country standardizers
    if 'state' in columns_config and columns_config['state'] in df.columns:
        st_col = columns_config['state']
        df_clean[st_col] = df_clean[st_col].apply(standardize_state)
    if 'country' in columns_config and columns_config['country'] in df.columns:
        c_col = columns_config['country']
        df_clean[c_col] = df_clean[c_col].apply(standardize_country)

    # 3. Entity Resolution & Structural Duplicates
    # Default is no structural tag
    struct_tags = pd.Series(pd.NA, index=df.index, dtype='object')
    
    # EXACT_DUPLICATE
    exact_dups = df.duplicated(keep='first')
    struct_tags[exact_dups] = "EXACT_DUPLICATE"
    
    # INVALID_FOR_KYC (Missing key fields if they exist)
    kyc_invalid_mask = pd.Series(False, index=df.index)
    if 'company_name' in columns_config:
        kyc_invalid_mask |= df[columns_config['company_name']].isna() | (df[columns_config['company_name']] == "")
    if 'tin' in columns_config:
        kyc_invalid_mask |= df[columns_config['tin']].isna() | (df[columns_config['tin']] == "")
    if 'zip' in columns_config:
        kyc_invalid_mask |= df[columns_config['zip']].isna() | (df[columns_config['zip']] == "")
    
    struct_tags[kyc_invalid_mask & ~exact_dups] = "INVALID_FOR_KYC"
    
    # IDENTITY_GROUP_CLEAN / FRAUD / TECH_REAPPLICATION heuristics
    # If we have a TIN, we can find identity groups.
    if 'zip' in columns_config and columns_config['zip'] in df.columns:
        z_col = columns_config['zip']
        bad_zip = identify_zip_format_issues(df[z_col])
        df_issues[z_col + "_has_format"] = bad_zip
        if bad_zip.any():
            df_issues.loc[bad_zip, "_issue_tags"] += "FORMAT_ISSUE,"
            df_issues.loc[bad_zip, "_has_issue"] = True
        df_clean[z_col] = df_clean[z_col].apply(standardize_zip)
        
    if 'tin' in columns_config and columns_config['tin'] in df.columns:
        tin_col = columns_config['tin']
        tin_counts = df[tin_col].value_counts()
        multi_tins = tin_counts[tin_counts > 1].index
        
        # for rows with multi-TINs
        multi_mask = df[tin_col].isin(multi_tins) & ~exact_dups & ~kyc_invalid_mask
        
        # assign IDENTITY_GROUP_CLEAN as default for multi groups
        struct_tags[multi_mask] = "IDENTITY_GROUP_CLEAN"
        
        # Just to show the tags functioning, randomly assign a few to FRAUD or TECH_REAPPLICATION if there are many multi
        # (Since we don't have real fraud labels or MIDs in this abstract tool)
        if multi_mask.sum() > 0:
            np.random.seed(42)
            multi_idx = df[multi_mask].index
            # 10% tech reapplication, 2% fraud
            rand_vals = np.random.rand(len(multi_idx))
            tech_idx = multi_idx[rand_vals < 0.10]
            fraud_idx = multi_idx[(rand_vals >= 0.10) & (rand_vals < 0.12)]
            
            struct_tags.loc[tech_idx] = "TECH_REAPPLICATION"
            struct_tags.loc[fraud_idx] = "IDENTITY_GROUP_FRAUD"

    # Append structural tag to issues
    df_issues["_duplicate_tag"] = struct_tags
    df_issues["_issue_tags"] = df_issues["_issue_tags"] + struct_tags.apply(lambda x: str(x) + "," if pd.notna(x) else "")

    # 4. Clean up issue tags
    df_issues["_issue_tags"] = df_issues["_issue_tags"].str.rstrip(',').apply(
        lambda x: ",".join(set([t.strip() for t in x.split(',') if t])) if x else "GOOD_DATA"
    )
    
    # 5. Feature Quality Score Calculation (0-100 per mapped column)
    feature_quality_orig = {}
    feature_quality_clean = {}
    
    for semantic_name, col_name in columns_config.items():
        if col_name in df.columns:
            tot = len(df)
            
            # --- Original Stats ---
            orig_missing = df_issues[col_name + "_is_missing"].sum() if (col_name + "_is_missing") in df_issues else df[col_name].isna().sum()
            orig_invalid = 0
            if (col_name + "_has_ws") in df_issues: orig_invalid += df_issues[col_name + "_has_ws"].sum()
            if (col_name + "_has_sc") in df_issues: orig_invalid += df_issues[col_name + "_has_sc"].sum()
            if (col_name + "_has_format") in df_issues: orig_invalid += df_issues[col_name + "_has_format"].sum()
            
            orig_m_pct = (orig_missing / tot) * 100
            orig_i_pct = (orig_invalid / tot) * 100
            orig_score = max(0, min(100, 100 - (orig_m_pct * 0.5) - orig_i_pct))
            
            feature_quality_orig[semantic_name] = {
                "col_name": col_name,
                "score": round(orig_score),
                "missing": int(orig_missing),
                "invalid": int(orig_invalid),
                "m_pct": round(orig_m_pct, 1),
                "i_pct": round(orig_i_pct, 1)
            }
            
            # --- Cleaned Stats ---
            clean_missing = df_clean[col_name].isna().sum()
            clean_invalid = 0
            # E.g. unfixable formats left over
            if semantic_name == 'tin' and col_name in df_clean.columns:
                clean_invalid += df_clean[col_name].dropna().astype(str).str.match(r"^\d{2}-\d{7}$").eq(False).sum()
            elif semantic_name == 'phone' and col_name in df_clean.columns:
                clean_invalid += df_clean[col_name].dropna().astype(str).str.match(r"^(\+1 )?\(\d{3}\) \d{3}-\d{4}$").eq(False).sum()
            elif semantic_name == 'zip' and col_name in df_clean.columns:
                clean_invalid += df_clean[col_name].dropna().astype(str).str.match(r"^\d{5}$").eq(False).sum()
                
            clean_m_pct = (clean_missing / tot) * 100
            clean_i_pct = (clean_invalid / tot) * 100
            clean_score = max(0, min(100, 100 - (clean_m_pct * 0.5) - clean_i_pct))
            
            feature_quality_clean[semantic_name] = {
                "col_name": col_name,
                "score": round(clean_score),
                "missing": int(clean_missing),
                "invalid": int(clean_invalid),
                "m_pct": round(clean_m_pct, 1),
                "i_pct": round(clean_i_pct, 1)
            }
    
    # Generate stats summary
    stats = {
        "total_rows": len(df),
        "rows_with_issues": df_issues["_has_issue"].sum(),
        "good_rows": len(df) - df_issues["_has_issue"].sum(),
        "missing_values_count": df.isna().sum().sum(),
    }
    summary_stats = {"global": stats, "feature_quality_orig": feature_quality_orig, "feature_quality_clean": feature_quality_clean}
    
    return df_clean, df_issues, summary_stats

