# Worth IA Data Check App

A robust, full-featured Python application built with Streamlit and Pandas for automated data quality assessment, feature standardization, and entity resolution (deduplication).

## 🚀 Overview

The **Worth IA Data Check App** acts as a data engineering pipeline with a beautiful frontend. It is designed to ingest raw tabular data (CSV or Excel) and immediately run an automated anomaly detection suite across all columns. It identifies missing values, whitespace inconsistencies, special character violations, and schema formatting errors. 

Rather than just flagging errors, the engine attempts to **intelligently standardize** the data using heavily defined business rules, generating a pristine output datatset ready for modeling or KYC API consumption.

## 🛠 Features & Capabilities

1. **Automated Feature Scoring**: 
   The pipeline computes a "Quality Score" (0-100%) for every analyzed column based on the volume of missing data and unfixable format violations. This allows analysts to instantly identify which fields are most at risk.

2. **Intelligent Feature Standardization**:
   The `woth_data_engine.py` script applies custom logic arrays based on semantic column inference (detecting columns by keywords like 'name', 'tin', 'address'):
   - **Legal Names & DBAs**: Aggressively strips special characters (`,`, `$`, `%`, `*`), shrinks multiple whitespaces into a single space, and forcefully uppercases the string to allow for exact match grouping.
   - **TIN (Tax ID)**: Prevents numeric cast failures by trimming trailing `.0` floats. Strips alphabetical characters and hyphens. Most importantly: Detects exactly 8-digit TIN anomalies and automatically prefixes them with a `0` to satisfy the 9-digit US standard. Outputs `XX-XXXXXXX` or `XXX-XX-XXXX`.
   - **Telephony**: Extracts only functional numeric digits. Dynamically assigns `(XXX) XXX-XXXX` formatting to 10-digit arrays, and `+1 (XXX) XXX-XXXX` to 11-digit arrays.
   - **Datetime Types**: Coerces varied string layouts into a universal ISO `YYYY-MM-DD` format utilizing pandas natively.
   - **Business Address Parsing**: Capable of reading addresses concatenated by pipe `|` delimiters. It parses the array, checks if the first item redundantly matches the company's legal name (and drops it if true), and assigns the remaining pieces to distinct `address_1`, `address_2`, `city`, `state`, `zip` and `country` fields.

3. **Entity Resolution (Duplicate Handling)**:
   The app goes beyond basic record matching by using grouped behavioral logic. It applies priority tags to handle identical or conflicting records:
   - **`[1] EXACT_DUPLICATE`**: Drops byte-for-byte identical rows (preserves the first).
   - **`[2] IDENTITY_GROUP_FRAUD`**: Identifies conflicting updates to the same entity, prioritizing the preservation of known fraud flags to prevent data leakage.
   - **`[3] IDENTITY_GROUP_CLEAN`**: Keeps the most recent update of a clean entity and drops older historical noise.
   - **`[4] TECH_REAPPLICATION`**: Identifies attempts to re-apply with a shifting address but identical TIN.
   - **`[5] INVALID_FOR_KYC`**: Immediately purges garbage lines missing all foundational attributes (Name, TIN, and Zip).

4. **Dynamic Comprehensive HTML Reporting**:
   Includes an embedded, 5-chapter HTML generation script (`html_report_template.py`) that executes securely offline. 
   - It captures live Plotly pie charts and bar charts rendered directly in the Streamlit frontend.
   - It automatically generates 5 custom AI-like "Analyst Interpretation" paragraphs synthesizing the calculated statistics of your specific data.
   - It outputs an executive-ready standalone HTML file.

## 🗂 Final Output Schema

When downloading the final cleaned `.csv` matrix via Tab 5:
- **`_received` Fields**: The app preserves your raw input data columns, appending `_received` to the column header.
- **`_worth` Fields**: Contains the algorithmically modified, clean standardizations.
- **Analytical Overlays**: Appends a hashed Universal Identifier, unformatted numeric arrays, and explicitly includes the final evaluated Duplicate Label tags.
