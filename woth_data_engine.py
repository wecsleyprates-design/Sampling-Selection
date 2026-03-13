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
    "INVALID_FOR_KYC": "#c0392b",
    "UNIQUE_CLEAN_NON_LINKED": "#27ae60"
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
    "INVALID_FOR_KYC": "🔺",
    "UNIQUE_CLEAN_NON_LINKED": "🟢"
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
    "INVALID_FOR_KYC": "Missing Legal Name/DBA fallback, TIN, or Postal Code. These records will fail vendor KYC verification — sending them wastes lookup budget with no chance of a match.",
    "UNIQUE_CLEAN_NON_LINKED": "Record does not match duplicate rules and has required KYC fields."
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
        v = str(val).strip()
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

def tin_to_9_digits_no_format(val):
    """Convert TIN-like value to exactly 9 digits when possible, left-padding 8-digit values."""
    if pd.isna(val):
        return ""
    digits = re.sub(r"\D", "", str(val))
    if len(digits) == 8:
        digits = "0" + digits
    return digits if len(digits) == 9 else ""

def normalize_for_key(val):
    """Uppercase, trim, and collapse non-alphanumeric to support stable matching keys."""
    if pd.isna(val):
        return ""
    v = str(val).strip().upper()
    v = re.sub(r"[^A-Z0-9]", "", v)
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

def parse_piped_address(row, addr_col, lgl_col, dba_col):
    addr = str(row.get(addr_col, ''))
    if pd.isna(row.get(addr_col)) or addr.strip() == '' or addr.lower() == 'nan':
        return pd.Series(['', '', '', '', '', '', ''])
    
    parts = [p.strip() for p in addr.split('|')]
    if len(parts) < 4:
        # Fallback if there aren't enough pipes
        a1 = parts[0]
        return pd.Series([a1, '', '', '', '', '', a1])
        
    # The last 4 sections of the pipe structure are always City, Region, Zip, Country
    country = parts[-1]
    zipc = parts[-2]
    region = parts[-3]
    city = parts[-4]
    
    prefix_parts = parts[:-4]
    if not prefix_parts:
        return pd.Series(['', '', city, region, zipc, country, addr])
        
    # We now filter prefix_parts to drop anything that is clearly just the company name.
    # If it is NOT the company name, or if it has strong physical address heuristics, we keep it.
    import re
    import difflib
    valid_address_parts = []
    
    lgl = str(row.get(lgl_col, '')).strip().upper() if lgl_col else ''
    dba = str(row.get(dba_col, '')).strip().upper() if dba_col else ''
    def clean_for_cmp(v): return re.sub(r'[^A-Z0-9]', '', v)
    c_lgl = clean_for_cmp(lgl)
    c_dba = clean_for_cmp(dba)
    
    for p in prefix_parts:
        if not p.strip(): continue
        p_upper = p.upper()
        
        is_valid_heuristic = False
        # Strong heuristics for a valid physical address
        if re.search(r'(^\d+)|(P\.?O\.?\s*BOX)|(PO BOX)|(STE\s+\d+)|(UNIT\s+\w+)|(DEPT\s+\d+)|(ROUTE\s+\d+)|(PMB\s+\d+)|(^[A-Z]\d+)', p_upper):
            is_valid_heuristic = True
        elif re.search(r'\b(AVE|BLVD|DR|ST|RD|LN|WAY|PKWY|CIR|PL|TR|CT|HWY|HIGHWAY|ROAD|COURT|STREET|URB|CALLE)\b', p_upper):
            is_valid_heuristic = True
            
        if is_valid_heuristic:
            valid_address_parts.append(p)
        else:
            # Check if this part is just the company name
            c_p = clean_for_cmp(p_upper)
            is_company = False
            if len(c_p) > 3:
                r_lgl = difflib.SequenceMatcher(None, c_p, c_lgl).ratio() if c_lgl else 0
                r_dba = difflib.SequenceMatcher(None, c_p, c_dba).ratio() if c_dba else 0
                
                if c_p == c_lgl or c_p == c_dba or r_lgl > 0.85 or r_dba > 0.85:
                    is_company = True
                elif not re.match(r'^\d+', p.strip()):
                    if (c_lgl and c_lgl in c_p and len(c_lgl) > 5) or (c_dba and c_dba in c_p and len(c_dba) > 5):
                        if max(r_lgl, r_dba) > 0.65:
                            is_company = True
                            
            # Keep the part if it's NOT the company name
            if not is_company:
                valid_address_parts.append(p)
                
    if not valid_address_parts and prefix_parts:
        # Fallback to prevent total data loss if everything looked like a company name
        valid_address_parts = [prefix_parts[-1]]
            
    a1 = valid_address_parts[0] if len(valid_address_parts) > 0 else ''
    a2 = valid_address_parts[1] if len(valid_address_parts) > 1 else ''
    if len(valid_address_parts) > 2:
        a2 = ", ".join(valid_address_parts[1:])
    
    valid_parts = [p for p in [a1, a2, city, region, zipc, country] if p]
    full_addr = ", ".join(valid_parts)
    
    return pd.Series([a1, a2, city, region, zipc, country, full_addr])

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
    val = str(val).strip()
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
    
    if len(digits) == 8:
        digits = "0" + digits
        
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
    for semantic_name, col_name in columns_config.items():
        if col_name not in df.columns: continue
        
        if semantic_name.startswith('phone'):
            bad_phone = identify_phone_format_issues(df[col_name])
            df_issues[col_name + "_has_format"] = bad_phone
            if bad_phone.any():
                df_issues.loc[bad_phone, "_issue_tags"] += "FORMAT_ISSUE,"
                df_issues.loc[bad_phone, "_has_issue"] = True
            df_clean[col_name] = df_clean[col_name].apply(standardize_phone)
            
        elif semantic_name.startswith('zip'):
            bad_zip = identify_zip_format_issues(df[col_name])
            df_issues[col_name + "_has_format"] = bad_zip
            if bad_zip.any():
                df_issues.loc[bad_zip, "_issue_tags"] += "FORMAT_ISSUE,"
                df_issues.loc[bad_zip, "_has_issue"] = True
            df_clean[col_name] = df_clean[col_name].apply(standardize_zip)
            
        elif semantic_name.startswith('tin'):
            bad_tin = identify_tin_format_issues(df[col_name])
            df_issues[col_name + "_has_format"] = bad_tin
            if bad_tin.any():
                df_issues.loc[bad_tin, "_issue_tags"] += "FORMAT_ISSUE,"
                df_issues.loc[bad_tin, "_has_issue"] = True
            df_clean[col_name] = df_clean[col_name].apply(standardize_tin)
            
        elif semantic_name.startswith('address'):
            df_clean[col_name + "_type"] = check_address_type(df[col_name])
            df_clean[col_name] = df_clean[col_name].apply(standardize_name)
            
        elif semantic_name.startswith('datetime'):
            df_clean[col_name] = standardize_datetime(df_clean[col_name])
            
        elif semantic_name.startswith('company_name') or semantic_name.startswith('dba') or semantic_name.startswith('first_name') or semantic_name.startswith('last_name') or semantic_name.startswith('city'):
            df_clean[col_name] = df_clean[col_name].apply(standardize_name)
            if semantic_name.startswith('company_name') or semantic_name.startswith('dba'):
                df_clean[col_name + "_normalized"] = df_clean[col_name].apply(normalize_entity_names)
                
        elif semantic_name.startswith('state'):
            df_clean[col_name] = df_clean[col_name].apply(standardize_state)
            
        elif semantic_name.startswith('country'):
            df_clean[col_name] = df_clean[col_name].apply(standardize_country)
            
    # NEW: Business Address Parsing for Pipes
    bus_addr_col = next((c for c in df.columns if 'business_address' in str(c).lower()), None)
    if bus_addr_col:
        lgl_col = next((c for c in df.columns if 'lgl_nm' in str(c).lower()), None)
        dba_col = next((c for c in df.columns if 'dba_nm' in str(c).lower() or 'dba' in str(c).lower() and 'lgl' not in str(c).lower()), None)
        
        parsed_cols = df.apply(lambda row: parse_piped_address(row, bus_addr_col, lgl_col, dba_col), axis=1)
        parsed_cols.columns = ['address_1_worth', 'address_2_worth', 'city_worth', 'region_worth', 'zip_code_worth', 'country_worth', 'full_address_worth']

        parsed_cols['address_1_worth'] = parsed_cols['address_1_worth'].apply(standardize_name)
        parsed_cols['address_2_worth'] = parsed_cols['address_2_worth'].apply(standardize_name)
        parsed_cols['city_worth'] = parsed_cols['city_worth'].apply(standardize_name)
        parsed_cols['region_worth'] = parsed_cols['region_worth'].apply(standardize_state)
        parsed_cols['zip_code_worth'] = parsed_cols['zip_code_worth'].apply(standardize_zip)
        parsed_cols['country_worth'] = parsed_cols['country_worth'].apply(standardize_country)
        parsed_cols['full_address_worth'] = parsed_cols['full_address_worth'].apply(standardize_name)
        
        for c in parsed_cols.columns:
            df_clean[c] = parsed_cols[c]

    # Build core _worth features required for entity-resolution rules.
    tin_src_col = next((c for c in df_clean.columns if str(c).lower() == 'tin'), None)
    lgl_src_col = next((c for c in df_clean.columns if str(c).lower() == 'lgl_nm'), None)
    dba_src_col = next((c for c in df_clean.columns if str(c).lower() == 'dba_nm'), None)

    if tin_src_col:
        df_clean['TIN_worth_no_format'] = df_clean[tin_src_col].apply(tin_to_9_digits_no_format)
    else:
        df_clean['TIN_worth_no_format'] = ""

    if lgl_src_col:
        df_clean['lgl_nm_worth'] = df_clean[lgl_src_col].apply(standardize_name)
    else:
        df_clean['lgl_nm_worth'] = ""

    if dba_src_col:
        df_clean['dba_nm_worth'] = df_clean[dba_src_col].apply(standardize_name)
    else:
        df_clean['dba_nm_worth'] = ""

    # 3. Entity Resolution & Structural Duplicates
    # Deterministic, priority-based rules using only _worth fields.
    struct_tags = pd.Series("UNIQUE_CLEAN_NON_LINKED", index=df.index, dtype='object')
    duplicate_rule_key_used = pd.Series("UNIQUE_CLEAN_NON_LINKED", index=df.index, dtype='object')
    
    # NEW: Duplicate detection flags for TIN and Address combinations
    tin_duplicate_flag = pd.Series("No", index=df.index, dtype='object')
    address_duplicate_flag = pd.Series("No", index=df.index, dtype='object')

    tin9 = df_clean['TIN_worth_no_format'].apply(normalize_for_key)
    lgl_w = df_clean['lgl_nm_worth'].apply(normalize_for_key)
    dba_w = df_clean['dba_nm_worth'].apply(normalize_for_key)
    name_key = lgl_w.where(lgl_w != "", dba_w)
    a1_w = df_clean.get('address_1_worth', pd.Series("", index=df.index)).apply(normalize_for_key)
    a2_w = df_clean.get('address_2_worth', pd.Series("", index=df.index)).apply(normalize_for_key)
    city_w = df_clean.get('city_worth', pd.Series("", index=df.index)).apply(normalize_for_key)
    region_w = df_clean.get('region_worth', pd.Series("", index=df.index)).apply(normalize_for_key)
    zip5_w = df_clean.get('zip_code_worth', pd.Series("", index=df.index)).apply(standardize_zip).fillna("").astype(str).apply(normalize_for_key)
    country_w = df_clean.get('country_worth', pd.Series("", index=df.index)).apply(normalize_for_key)
    
    # NEW: Identify TIN duplicates (TINs that appear more than once, even if not empty)
    if (tin9 != "").any():
        tin_counts = tin9[tin9 != ""].value_counts()
        duplicate_tins = tin_counts[tin_counts > 1].index.tolist()
        tin_duplicate_mask = tin9.isin(duplicate_tins) & (tin9 != "")
        tin_duplicate_flag[tin_duplicate_mask] = "Yes"
    
    # NEW: Identify address combination duplicates (address_1_worth + zip_code_worth + city_worth)
    address_combo_df = pd.DataFrame({
        'a1': a1_w,
        'zip': zip5_w,
        'city': city_w
    })
    address_combo_key = address_combo_df.astype(str).apply(lambda row: '|'.join(row), axis=1)
    # Only consider non-empty address combos as duplicates
    non_empty_mask = (address_combo_df['a1'] != '') & (address_combo_df['zip'] != '') & (address_combo_df['city'] != '')
    if non_empty_mask.any():
        addr_counts = address_combo_key[non_empty_mask].value_counts()
        duplicate_addrs = addr_counts[addr_counts > 1].index.tolist()
        address_duplicate_mask = address_combo_key.isin(duplicate_addrs) & non_empty_mask
        address_duplicate_flag[address_duplicate_mask] = "Yes"

    # Simplified: All records marked as UNIQUE_CLEAN_NON_LINKED
    # Complex structural deduplication rules are excluded from the pipeline
    # Only simple TIN and Address duplicate flags are retained (set above)

    # Keep original format checks if a zip column is configured.
    if 'zip' in columns_config and columns_config['zip'] in df.columns:
        z_col = columns_config['zip']
        bad_zip = identify_zip_format_issues(df[z_col])
        df_issues[z_col + "_has_format"] = bad_zip
        if bad_zip.any():
            df_issues.loc[bad_zip, "_issue_tags"] += "FORMAT_ISSUE,"
            df_issues.loc[bad_zip, "_has_issue"] = True
        df_clean[z_col] = df_clean[z_col].apply(standardize_zip)

    # Keep duplicate outputs in cleaned dataset for downstream export/audit.
    df_clean["Duplicate_TIN_CHECK"] = tin_duplicate_flag
    df_clean["Duplicate_Address_CHECK"] = address_duplicate_flag
    
    # Append duplicate flags to issues (simplified - no structural tags)
    df_issues["_tin_duplicate_flag"] = tin_duplicate_flag
    df_issues["_address_duplicate_flag"] = address_duplicate_flag
    df_issues["_issue_tags"] = df_issues["_issue_tags"].str.rstrip(',').fillna("GOOD_DATA")
    
    # 4. Clean up issue tags
    df_issues["_issue_tags"] = df_issues["_issue_tags"].str.rstrip(',').apply(
        lambda x: ",".join(set([t.strip() for t in x.split(',') if t])) if x else "GOOD_DATA"
    )
    
    # 5. Feature Quality Score Calculation (0-100 per mapped column)
    feature_quality_orig = {}
    feature_quality_clean = {}
    
    for col_name in df.columns:
        tot = max(1, len(df))
        
        # Determine semantic role from columns_config mapping dynamically
        semantic_name = 'general'
        for k, v in columns_config.items():
            if v == col_name:
                semantic_name = k
                break
                
        # --- Original Stats ---
        orig_missing = df_issues[col_name + "_is_missing"].sum() if (col_name + "_is_missing") in df_issues else df[col_name].isna().sum()
        orig_invalid = 0
        if (col_name + "_has_ws") in df_issues: orig_invalid += df_issues[col_name + "_has_ws"].sum()
        if (col_name + "_has_sc") in df_issues: orig_invalid += df_issues[col_name + "_has_sc"].sum()
        if (col_name + "_has_format") in df_issues: orig_invalid += df_issues[col_name + "_has_format"].sum()
        
        orig_m_pct = (orig_missing / tot) * 100
        orig_i_pct = (orig_invalid / tot) * 100
        orig_penalty = (orig_m_pct * 0.5) + orig_i_pct
        orig_score = max(0, min(100, 100 - orig_penalty))
        
        if (orig_missing > 0 or orig_invalid > 0) and orig_score > 99:
            orig_score = 99.0
            
        feature_quality_orig[col_name] = {
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
        
        # Unfixable formats left over logic
        if semantic_name.startswith('tin') and col_name in df_clean.columns:
            clean_invalid += df_clean[col_name].dropna().astype(str).str.match(r"^\d{2}-\d{7}$").eq(False).sum()
        elif semantic_name.startswith('phone') and col_name in df_clean.columns:
            clean_invalid += df_clean[col_name].dropna().astype(str).str.match(r"^(\+1 )?\(\d{3}\) \d{3}-\d{4}$").eq(False).sum()
        elif semantic_name.startswith('zip') and col_name in df_clean.columns:
            clean_invalid += df_clean[col_name].dropna().astype(str).str.match(r"^\d{5}$").eq(False).sum()
            
        clean_m_pct = (clean_missing / tot) * 100
        clean_i_pct = (clean_invalid / tot) * 100
        clean_penalty = (clean_m_pct * 0.5) + clean_i_pct
        clean_score = max(0, min(100, 100 - clean_penalty))
        
        if (clean_missing > 0 or clean_invalid > 0) and clean_score > 99:
            clean_score = 99.0
            
        feature_quality_clean[col_name] = {
            "col_name": col_name,
            "score": round(clean_score),
            "missing": int(clean_missing),
            "invalid": int(clean_invalid),
            "m_pct": round(clean_m_pct, 1),
            "i_pct": round(clean_i_pct, 1)
        }
        
    # Restrict final output to strictly what the user requires, hiding engine arrays
    raw_col_order = df.columns.tolist()
    required_worth_cols = [
        'TIN_worth_no_format',
        'lgl_nm_worth',
        'dba_nm_worth',
        'address_1_worth',
        'address_2_worth',
        'city_worth',
        'region_worth',
        'zip_code_worth',
        'country_worth',
    ]
    worth_cols = [c for c in df_clean.columns if c.endswith('_worth') and c not in raw_col_order]
    extra_required = [c for c in required_worth_cols if c in df_clean.columns and c not in raw_col_order]
    worth_cols = list(dict.fromkeys(worth_cols + extra_required))
    output_audit_cols = [
        'Duplicate_TIN_CHECK',
        'Duplicate_Address_CHECK',
    ]
    output_audit_cols = [c for c in output_audit_cols if c in df_clean.columns and c not in raw_col_order]
    final_output_cols = [c for c in raw_col_order if c in df_clean.columns] + worth_cols + output_audit_cols
    df_clean = df_clean[final_output_cols]
    
    # Generate stats summary
    stats = {
        "total_rows": len(df),
        "rows_with_issues": df_issues["_has_issue"].sum(),
        "good_rows": len(df) - df_issues["_has_issue"].sum(),
        "missing_values_count": df.isna().sum().sum(),
    }
    summary_stats = {"global": stats, "feature_quality_orig": feature_quality_orig, "feature_quality_clean": feature_quality_clean}
    
    return df_clean, df_issues, summary_stats

