
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import warnings
import logging
from scipy import stats

warnings.filterwarnings('ignore')

class Analyzer:
    """
    Modular dataset analyzer with fully independent, self-contained analysis methods.

    KEY FEATURES:
    - Each analysis is INDEPENDENT - no need to run others first
    - Uses proper logging (not print statements)
    - Stores all results in dictionaries for programmatic access
    - Each method displays tables, plots, and recommendations
    """

    def __init__(self, data: pd.DataFrame, outcome_feature: Optional[str] = None,
                 log_level: str = 'INFO'):
        """
        Initialize the analyzer.

        Parameters:
        -----------
        data : pd.DataFrame
            Dataset to analyze
        outcome_feature : str, optional
            Name of the binary outcome/target variable
        log_level : str, default='INFO'
            Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        self.data = data.copy()
        self.outcome_feature = outcome_feature

        # Setup logging
        self.logger = self._setup_logger(log_level)

        # Store all analysis results here
        self.results = {
            'feature_types': {},
            'missing_values': {},
            'low_variance': {},
            'sparse_categorical': {},
            'correlations': {},
            'outliers': {},
            'outcome_correlations': {},
            'leakage_risk': {},
            'feature_quality': {},
            'safe_mode': {}
        }

        self.logger.info("=" * 80)
        self.logger.info(" Analyzer Initialized")
        rows, cols = self.data.shape
        self.logger.info(f" . Dataset shape: {rows:,} rows x {cols:,} columns")
        if self.outcome_feature:
            self.logger.info(f" . Outcome feature: '{self.outcome_feature}'")
        self.logger.info("=" * 80)

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup logger with custom formatting."""
        logger = logging.getLogger(f'DatasetAnalyzer_{id(self)}')
        logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))

        # Simple formatter
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.propagate = False

        return logger

    def set_log_level(self, level: str):
        """
        Change logging level dynamically.

        Parameters:
        -----------
        level : str
            'INFO', 'DEBUG', 'WARNING', 'ERROR'
        """
        self.logger.setLevel(getattr(logging, level.upper()))
        for handler in self.logger.handlers:
            handler.setLevel(getattr(logging, level.upper()))

    # =========================================================================
    # HELPER METHODS FOR CONSISTENT FEATURE TYPE RETRIEVAL
    # =========================================================================

    def _get_features_by_type(self, feature_types: List[str], use_cached: bool = True) -> List[str]:
        """
        Get list of features by type(s), using cached feature type analysis if available.

        This ensures consistency across all analysis methods by using the same
        feature type classification from analyze_feature_types().

        Parameters:
        -----------
        feature_types : List[str]
            Types to retrieve: 'numeric', 'categorical', 'binary', 'datetime', 'id', 'outcome'
        use_cached : bool, default=True
            If True and feature_types analysis has been run, use those results.
            If False or no cache, fall back to pandas dtype detection.

        Returns:
        --------
        list : Features matching the specified types
        """
        # Check if we have cached feature type analysis
        if use_cached and 'feature_type_map' in self.results.get('feature_types', {}):
            feature_type_map = self.results['feature_types']['feature_type_map']
            features = [
                feat for feat, ftype in feature_type_map.items()
                if ftype in feature_types
            ]
            return features

        # Fall back to pandas dtype detection if no cache
        features = []

        if 'numeric' in feature_types:
            # Get numeric excluding binary (which have 2 unique values)
            num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            for col in num_cols:
                if col != self.outcome_feature and self.data[col].nunique() > 2:
                    features.append(col)

        if 'binary' in feature_types:
            # Binary can be numeric or categorical with exactly 2 unique values
            for col in self.data.columns:
                if col != self.outcome_feature and self.data[col].nunique() == 2:
                    features.append(col)

        if 'categorical' in feature_types:
            # Get categorical excluding binary
            cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in cat_cols:
                if col != self.outcome_feature and self.data[col].nunique() > 2:
                    features.append(col)

        if 'datetime' in feature_types:
            dt_cols = self.data.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
            features.extend([col for col in dt_cols if col != self.outcome_feature])

        if 'outcome' in feature_types and self.outcome_feature:
            features.append(self.outcome_feature)

        return list(set(features)) # Remove duplicates

    def _get_numeric_features(self, exclude_binary: bool = True, use_cached: bool = True) -> List[str]:
        """
        Get numeric features, optionally excluding binary.

        Parameters:
        -----------
        exclude_binary : bool, default=True
            If True, exclude features with only 2 unique values
        use_cached : bool, default=True
            Use cached feature type analysis if available

        Returns:
        --------
        list : Numeric feature names
        """
        if exclude_binary:
            return self._get_features_by_type(['numeric'], use_cached=use_cached)
        else:
            return self._get_features_by_type(['numeric', 'binary'], use_cached=use_cached)

    def _get_categorical_features(self, exclude_binary: bool = True, use_cached: bool = True) -> List[str]:
        """
        Get categorical features, optionally excluding binary.

        Parameters:
        -----------
        exclude_binary : bool, default=True
            If True, exclude features with only 2 unique values
        use_cached : bool, default=True
            Use cached feature type analysis if available

        Returns:
        --------
        list : Categorical feature names
        """
        if exclude_binary:
            return self._get_features_by_type(['categorical'], use_cached=use_cached)
        else:
            return self._get_features_by_type(['categorical', 'binary'], use_cached=use_cached)

    # =========================================================================
    # FEATURE TYPE ANALYSIS - COMPLETELY INDEPENDENT
    # =========================================================================

    def analyze_feature_types(self, show_plots: bool = True, show_details: bool = True,
                             show_samples: bool = True, sample_size: int = 10,
                             show_all_features: bool = False):
        """
        ⭐ INDEPENDENT ANALYSIS - No prerequisites needed!

        Analyzes feature type distribution with data samples and validation:
        - Detects: numeric, categorical, binary, datetime, ID columns
        - Shows: Distribution table, bar chart, detailed feature lists, data samples
        - Validates: Shows actual data values to verify type detection
        - Stores: All results in self.results['feature_types']

        Parameters:
        -----------
        show_plots : bool, default=True
            Whether to display bar chart
        show_details : bool, default=True
            Whether to show detailed feature lists
        show_samples : bool, default=True
            Whether to show data samples for each feature type
        sample_size : int, default=10
            Number of features to show samples for (per type)
        show_all_features : bool, default=False
            If True, show samples for ALL features (ignores sample_size)

        Results stored in self.results['feature_types']:
        -----------------------------------------------
        - type_counts: {type: count}
        - grouped_features: {type: [feature_list]}
        - total_features: Total number of features
        - sample_data: {feature_name: sample_values}
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 FEATURE TYPE ANALYSIS (Independent - No Prerequisites)")
        self.logger.info("=" * 80)

        # Detect feature types (completely self-contained)
        self.logger.debug("Detecting feature types from scratch...")
        feature_type_map = self._detect_feature_types()

        # Group features by type
        grouped_features = {}
        for feature, ftype in feature_type_map.items():
            if ftype not in grouped_features:
                grouped_features[ftype] = []
            grouped_features[ftype].append(feature)

        # Count by type
        type_counts = {ftype: len(features) for ftype, features in grouped_features.items()}

        # Store results
        self.results['feature_types'] = {
            'feature_type_map': feature_type_map,
            'grouped_features': grouped_features,
            'type_counts': type_counts,
            'total_features': len(feature_type_map),
            'sample_data': {},
            'analysis_timestamp': pd.Timestamp.now()
        }

        # Display summary table
        self.logger.info("\n🔢 FEATURE TYPE DISTRIBUTION")
        self.logger.info("-" * 60)

        type_df = pd.DataFrame(list(type_counts.items()), columns=['Feature Type', 'Count'])
        type_df = type_df.sort_values('Count', ascending=False)
        total = type_df['Count'].sum()
        type_df['Percentage'] = (type_df['Count'] / total * 100).round(1)

        self.logger.info("\n" + type_df.to_string(index=False))

        # Display plot
        if show_plots:
            self._plot_feature_type_distribution(type_counts)

        # Display detailed lists
        if show_details:
            self.logger.info("\n📂 DETAILED FEATURE LISTS BY TYPE")
            self.logger.info("-" * 60)

            for ftype in ['numeric', 'categorical', 'binary', 'datetime', 'id', 'outcome', 'unknown']:
                if ftype in grouped_features and grouped_features[ftype]:
                    features = grouped_features[ftype]
                    self.logger.info(f"\n▶ {ftype.upper()} ({len(features)} features):")

                    # Display in columns for readability
                    for i in range(0, len(features), 3):
                        row = features[i:i+3]
                        self.logger.info("    " + " | ".join(f"{f:<25}" for f in row))

        # Display sample data if requested
        if show_samples:
            self._display_feature_samples(grouped_features, sample_size, show_all_features)

        # Recommendations
        self.logger.info("\n💡 RECOMMENDATIONS:")
        if 'id' in type_counts and type_counts['id'] > 0:
            self.logger.info(f"  • Remove {type_counts['id']} ID columns before modeling")
        if 'datetime' in type_counts and type_counts['datetime'] > 0:
            self.logger.info(f"  • Consider feature engineering for {type_counts['datetime']} datetime columns")
        if 'unknown' in type_counts and type_counts['unknown'] > 0:
            self.logger.warning(f"  • ⚠️ {type_counts['unknown']} features have unknown type - investigate!")

        self.logger.info("\n✅ Feature type analysis complete!")
        self.logger.info(f"  • Results stored in: analyzer.results['feature_types']")

    def _detect_feature_types(self) -> Dict[str, str]:
        """Detect the type of each feature (internal method)."""
        feature_types = {}

        for col in self.data.columns:
            if col == self.outcome_feature:
                feature_types[col] = 'outcome'
                continue

            series = self.data[col]

            # CRITICAL: Check if feature is completely empty or has no usable values
            # This catches features that are 100% NaN, blank, or special missing codes
            non_null_count = series.notna().sum()

            if non_null_count == 0:
                # All values are NaN/None - cannot determine type
                feature_types[col] = 'unknown'
                self.logger.debug(f"Feature '{col}' classified as 'unknown': all values are NaN/None (100%)")
                continue

            # For string/object columns, check if ALL non-null values are just whitespace
            if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    # Check if ALL non-null values are empty strings or whitespace
                    non_empty_count = non_null_series.astype(str).str.strip().str.len().gt(0).sum()
                    if non_empty_count == 0:
                        # ALL non-null values are blank/whitespace - cannot determine type
                        feature_types[col] = 'unknown'
                        self.logger.debug(f"Feature '{col}' classified as 'unknown': all non-null values are blank/whitespace (100%)")
                        continue

            # IMPORTANT: Check for datetime BEFORE constant check
            # A constant datetime value (e.g., all 'oct15_25') is still a valid datetime feature
            # Check for datetime
            if self._is_datetime_column(series):
                feature_types[col] = 'datetime'
                self.logger.debug(f"Feature '{col}' classified as 'datetime'")
                continue

            # Check if feature has only ONE unique value (constant column)
            # Only mark as 'unknown' if it's an ACTUAL special missing code
            # IMPORTANT: Numeric constants (all 0s, all 1s) should still be classified as 'numeric'
            # IMPORTANT: Datetime constants (all 'oct15_25') were already handled above
            # IMPORTANT: Regular constant values like 'US', 'MATCHED', 'active' should be classified as categorical/binary
            unique_count = series.nunique()
            if unique_count == 1:
                # Check if it's a numeric constant (all 0s, all 1s, etc.)
                if pd.api.types.is_numeric_dtype(series):
                    # Numeric constant - will be classified as 'numeric' later
                    self.logger.debug(f"Feature '{col}' has constant numeric value - will classify as 'numeric'")
                else:
                    # Non-numeric, non-datetime constant - check if it's a special missing code
                    unique_val = series.dropna().iloc[0] if non_null_count > 0 else None
                    
                    # List of special missing codes that should be marked as 'unknown'
                    special_missing_codes = {
                        'unknown', 'unk', 'n/a', 'na', 'nan', 'null', 'none', 'missing', 
                        'not available', 'not applicable', 'undefined', 'empty', ''
                    }
                    
                    # Check if the constant value is a special missing code
                    is_special_code = False
                    if unique_val is not None:
                        str_val = str(unique_val).strip().lower()
                        is_special_code = str_val in special_missing_codes or len(str_val) == 0
                    
                    if is_special_code:
                        # It's a special missing code - mark as unknown
                        feature_types[col] = 'unknown'
                        self.logger.debug(f"Feature '{col}' classified as 'unknown': constant special missing code '{unique_val}'")
                        continue
                    else:
                        # It's a regular constant value (like 'US', 'MATCHED', 'active')
                        # Let it be classified normally as categorical/binary below
                        self.logger.debug(f"Feature '{col}' has constant value '{unique_val}' - will classify by type")

            # Now proceed with normal type detection for features with sufficient variation

            # Check for ID columns
            if self._is_id_column(col, series):
                feature_types[col] = 'id'
            # Check for binary
            elif series.nunique() == 2:
                feature_types[col] = 'binary'
            # Low-cardinality numeric → categorical code
            # Catches columns like fraud_type (1=ID Theft, 2=Synthetic, 3=Others),
            # risk_tier (1-5), product_category (1-8), etc.
            # Criteria: native numeric dtype + 3‒10 distinct values +
            #            all integer-valued + max absolute value ≤ 100
            # (max ≤ 100 prevents mis-classifying year columns such as 2020/2021)
            elif (
                pd.api.types.is_numeric_dtype(series)
                and 3 <= unique_count <= 10
                and (series.dropna() % 1 == 0).all()
                and float(series.dropna().abs().max()) <= 100
            ):
                feature_types[col] = 'categorical'
                self.logger.debug(
                    f"Feature '{col}' classified as 'categorical' (low-cardinality integer code: "
                    f"{unique_count} distinct values in "
                    f"{sorted(series.dropna().unique().astype(int).tolist())})"
                )
            # Check for numeric (including numeric stored as strings)
            elif pd.api.types.is_numeric_dtype(series):
                feature_types[col] = 'numeric'
            # Check if object/string column contains numeric values
            elif self._is_numeric_string_column(series):
                feature_types[col] = 'numeric'
            # Check for categorical
            elif pd.api.types.is_object_dtype(series) or str(series.dtype) == 'category':
                feature_types[col] = 'categorical'
            else:
                feature_types[col] = 'unknown'

        return feature_types

    def _is_id_column(self, col_name: str, series: pd.Series) -> bool:
        """
        Check if a column is an identifier (ID) column.

        Detection layers (applied in order, first match wins):

        1. Substring-based name heuristics  — e.g. '_id', 'account_number'
        2. Exact / word-boundary name match — e.g. 'MID', 'TIN', 'SSN'
        3. Domain-specific prefix/suffix    — e.g. 'mid_', '_mid', 'tin_'
        4. Statistical fingerprint for numeric columns:
             a. High-uniqueness integers (≥ 80% unique, all int-valued)
             b. Large-digit IDs: median ≥ 7 digits (covers 9-digit TIN / 13-digit MID)
             c. Moderate-uniqueness + very large magnitude (≥ 1 billion)
             d. Constant digit-length pattern (e.g. all values are exactly 9 digits)
        """
        name_lower = col_name.lower().strip()
        n = len(series)
        if n == 0:
            return False
        n_unique = series.nunique()
        unique_ratio = n_unique / n

        # ── 1. Substring keywords (original list, expanded) ──────────────────
        _substr_keywords = [
            'id_', '_id', '_key', 'key_',
            '_code', 'code_',
            '_number', 'number_', '_num', 'num_', '_nbr', 'nbr_',
            '_ref', 'ref_',
            '_token', 'token_',
            '_acct', 'acct_', '_account', 'account_',
            '_seq', 'seq_', '_serial', 'serial_',
            '_uuid', 'uuid_', '_guid', 'guid_',
        ]
        if any(kw in name_lower for kw in _substr_keywords):
            if unique_ratio > 0.80:
                return True

        # ── 2. Exact-name match for known short ID abbreviations ──────────────
        # Split on common delimiters so e.g. "customer_mid" and "mid" both match.
        _name_parts = set(name_lower.replace('.', '_').replace(' ', '_').split('_'))
        _exact_ids = {
            # Generic
            'id', 'uid', 'pid', 'cid', 'rid', 'fid', 'bid', 'nid', 'sid',
            'key', 'ref', 'uuid', 'guid', 'token', 'seq', 'serial',
            # Payment / acquiring
            'mid',   # Merchant ID
            'tid',   # Terminal ID
            'arn',   # Acquirer Reference Number
            'stan',  # System Trace Audit Number
            'rrn',   # Retrieval Reference Number
            'ica',   # Interbank Card Association number
            'bin',   # Bank Identification Number (first 6 digits of PAN)
            'pan',   # Primary Account Number (full card number)
            # Tax / identity
            'tin',   # Tax Identification Number
            'ein',   # Employer Identification Number
            'ssn',   # Social Security Number
            'itin',  # Individual Taxpayer Identification Number
            'fein',  # Federal EIN
            'npi',   # National Provider Identifier
            'duns',  # D-U-N-S number
            # Accounts / banking
            'acct', 'account',
            'routing', 'aba',
            # Other common domain IDs
            'msisdn', 'imei', 'imsi',
        }
        # Lower threshold for exact known-ID names: even 5% uniqueness is enough
        # because the name alone is strong evidence (e.g. MID repeats per merchant)
        if _name_parts & _exact_ids:
            if unique_ratio > 0.05:
                return True

        # ── 3. Domain-specific prefix/suffix (loose boundary match) ──────────
        _prefix_suffix = [
            'mid_', '_mid', 'tid_', '_tid',
            'tin_', '_tin', 'ssn_', '_ssn', 'ein_', '_ein',
            'bin_', '_bin', 'pan_', '_pan',
            'arn_', '_arn', 'rrn_', '_rrn',
        ]
        if any(name_lower.startswith(p.rstrip('_')) or
               name_lower.endswith(p.lstrip('_')) or
               p in name_lower
               for p in _prefix_suffix):
            if unique_ratio > 0.05:
                return True

        # ── 4. Statistical fingerprint for numeric/float columns ──────────────
        if pd.api.types.is_numeric_dtype(series) and n_unique > 10:
            sample = series.dropna()
            if len(sample) == 0:
                return False

            # 4a. High-uniqueness integers -----------------------------------
            # All values must be integer-valued (no meaningful decimal part)
            try:
                is_int_valued = (sample % 1 == 0).all()
            except Exception:
                is_int_valued = False

            if is_int_valued:
                median_val = float(sample.abs().median())

                # 4b. Large-digit IDs: ≥ 7 digits (covers 9-digit TIN and
                # 13-digit MID).  Use a very low uniqueness floor (2%) because
                # identifiers like MID repeat heavily across many transactions.
                if median_val >= 1_000_000 and unique_ratio > 0.02:
                    return True

                # 4c. Very-large-magnitude: ≥ 10 digits (≥ 1 billion).
                # Even 1% uniqueness is enough — values this large are
                # overwhelmingly IDs (card numbers, MIDs, account numbers, etc.)
                if median_val >= 1_000_000_000 and unique_ratio > 0.01:
                    return True

            # 4d. Constant digit-length pattern --------------------------------
            # IDs like SSN (always 9 digits) or MID (always 13 digits) have
            # a very tight digit-count distribution.  Require ≥ 7 digits and
            # ≥ 90% of values sharing that exact length.  Uniqueness floor is
            # low (2%) to handle repeated MID/TIN in transaction-level data.
            try:
                digit_lengths = sample.abs().apply(
                    lambda x: len(str(int(x))) if x > 0 else 1
                )
                mode_len = int(digit_lengths.mode()[0])
                if mode_len >= 7 and (digit_lengths == mode_len).mean() >= 0.90:
                    if unique_ratio > 0.02:
                        return True
            except Exception:
                pass

        return False

    def _is_datetime_column(self, series: pd.Series) -> bool:
        """
        Check if column contains datetime data in various formats.
        When an Excel serial date is detected, the column is **auto-converted**
        in-place on self.data and the issue is registered via _register_type_issue().

        Detects & fixes:
        - Native datetime types
        - Excel serial dates (e.g. 44927.54, 45681.0) — stored as numeric floats
          → AUTO-CONVERTED to datetime using Excel epoch (1899-12-30)
        - YYYY format (1992, 2001, 2024) - including floats like 1969.0
        - YYYYMM format (202212, 202301, 202405)
        - MMYYYY format (122021, 012022, 072023)
        - Standard string datetimes (parseable by pd.to_datetime)
        - Text-based date patterns:
          * monthDD_YY: 'oct15_25', 'jan01_26', 'dec31_24'
          * month_YY: 'oct_25', 'jan_26', 'dec_2024'
          * monthYY: 'oct25', 'jan26', 'dec2024'
        """
        # Native datetime type
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        # For numeric columns, check for date patterns
        if pd.api.types.is_numeric_dtype(series):
            sample = series.dropna().head(1000)
            if len(sample) == 0:
                return False

            # ── Excel serial date detection & auto-fix ──────────────────────
            # Excel stores dates as days since 1899-12-30.
            # Typical modern dates map to ~40000–50000 (2009–2036).
            # We use a wider range 20000–60000 to cover ~1954–2064.
            min_serial, max_serial = 20000, 60000
            if sample.between(min_serial, max_serial).mean() > 0.8:
                # Convert using pd.to_datetime with Excel epoch (1899-12-30) and unit='D'.
                # We floor to integer days first to avoid Timedelta overflow from fractional
                # intra-day values stored in some Excel files.
                raw_col = self.data[series.name]
                try:
                    converted_col = pd.to_datetime(
                        raw_col.where(raw_col.notna(), other=np.nan).astype(float).apply(
                            lambda x: np.floor(x) if pd.notna(x) else np.nan
                        ),
                        unit='D',
                        origin='1899-12-30',
                        errors='coerce'
                    )
                except Exception:
                    converted_col = pd.to_datetime(raw_col, errors='coerce')
                converted_sample = converted_col.dropna().dt.date[:5].tolist()

                self._register_type_issue(
                    feature=series.name,
                    detected_type='datetime',
                    actual_type='numeric',
                    issue='Excel serial date detected — values are numeric day-counts (Excel epoch 1899-12-30). Column auto-converted to datetime.',
                    sample=sample.tolist()[:5]
                )
                self.logger.warning(
                    f"⚠️  Feature '{series.name}': Excel serial date detected. "
                    f"Raw numeric sample: {sample.tolist()[:5]}  "
                    f"→ Auto-converted to datetime. "
                    f"Converted sample: {converted_sample}"
                )

                # ── Apply fix in-place on analyzer.data ──────────────────────
                self.data[series.name] = converted_col
                self.logger.warning(
                    f"   ✅ '{series.name}' fixed in analyzer.data — dtype is now: {self.data[series.name].dtype}"
                )
                return True

            # Convert to integers (handles both int and float)
            sample_int = sample.astype(int)

            # Convert to string for pattern matching (now without decimal points)
            sample_str = sample_int.astype(str)

            # Check YYYY format (4 digits: 1992, 2001, 2024, 1969.0 -> 1969)
            # Only if values look like years (reasonable range)
            yyyy_pattern = sample_str.str.match(r'^\d{4}$')
            if yyyy_pattern.sum() / len(sample) > 0.8:
                # Should be in reasonable year range (1900-2100)
                years = sample_int
                if ((years >= 1900) & (years <= 2100)).sum() / len(sample) > 0.8:
                    # Check if it has temporal characteristics
                    year_range = years.max() - years.min()
                    if 0 < year_range <= 150:  # Reasonable span for years
                        return True

            # Check YYYYMM format (6 digits: 202212, 199901)
            yyyymm_pattern = sample_str.str.match(r'^\d{6}$')
            if yyyymm_pattern.sum() / len(sample) > 0.8:
                # Validate: first 4 digits should be reasonable year (1900-2100)
                years = sample_str.str[:4].astype(int)
                if ((years >= 1900) & (years <= 2100)).sum() / len(sample) > 0.8:
                    # Validate: last 2 digits should be valid month (01-12)
                    months = sample_str.str[-2:].astype(int)
                    if ((months >= 1) & (months <= 12)).sum() / len(sample) > 0.8:
                        return True

            # Check MMYYYY format (6 digits: 122021, 012022)
            # First 2 digits = month (01-12), Last 4 = year
            if (sample_int >= 11900).all() and (sample_int <= 122100).all():
                months = (sample_int // 10000).astype(int)
                years = (sample_int % 10000).astype(int)
                valid_months = ((months >= 1) & (months <= 12)).sum() / len(sample)
                valid_years = ((years >= 1900) & (years <= 2100)).sum() / len(sample)
                if valid_months > 0.8 and valid_years > 0.8:
                    return True

        # Check for string datetime columns
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            sample = series.dropna().head(100)
            if len(sample) == 0:
                return False

            # First, try standard datetime parsing
            try:
                pd.to_datetime(sample, errors='raise')
                return True
            except Exception:
                pass

            # Check for text-based date patterns like 'oct15_25', 'jan_26', 'dec_2024'
            sample_lower = sample.astype(str).str.lower().str.strip()
            month_abbrevs = [
                'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
            ]
            # Pattern 1: monthDD_YY (e.g., 'oct15_25', 'jan01_26')
            pattern1 = r'^(' + '|'.join(month_abbrevs) + r')\d{1,2}_\d{2,4}$'
            match_count1 = sample_lower.str.match(pattern1).sum()
            # Pattern 2: month_YY or month_YYYY (e.g., 'oct_25', 'jan_2026')
            pattern2 = r'^(' + '|'.join(month_abbrevs) + r')_\d{2,4}$'
            match_count2 = sample_lower.str.match(pattern2).sum()
            # Pattern 3: monthYY (no separator, e.g., 'oct25', 'jan26')
            pattern3 = r'^(' + '|'.join(month_abbrevs) + r')\d{2,4}$'
            match_count3 = sample_lower.str.match(pattern3).sum()
            total_matches = match_count1 + match_count2 + match_count3
            if total_matches / len(sample) > 0.8:
                return True

        return False

    def _register_type_issue(self, feature, detected_type, actual_type, issue, sample=None):
        """
        Register a data type identification issue for later reporting via report_type_issues().

        Parameters
        ----------
        feature      : column name
        detected_type: what the analyzer tagged it as (e.g. 'datetime')
        actual_type  : what the raw data dtype actually is (e.g. 'numeric')
        issue        : human-readable description of the problem
        sample       : list of example raw values
        """
        if not hasattr(self, '_type_issues'):
            self._type_issues = []
        self._type_issues.append({
            'feature': feature,
            'detected_type': detected_type,
            'actual_type': actual_type,
            'issue': issue,
            'sample': sample
        })

    def report_type_issues(self) -> pd.DataFrame:
        """
        Return a DataFrame with all data type identification issues found and fixed.

        Each row represents one feature where the raw storage type differs from
        the logical / intended type (e.g. Excel serial dates stored as numeric).

        Columns
        -------
        feature        : column name
        detected_type  : type the analyzer assigned (datetime, numeric, …)
        actual_type    : raw dtype reported by pandas
        issue          : plain-English description of the discrepancy
        sample         : sample raw values that triggered the detection

        Example
        -------
        >>> analyzer.report_type_issues()
        """
        if not hasattr(self, '_type_issues') or not self._type_issues:
            self.logger.info("✅ No data type identification issues found.")
            return pd.DataFrame(columns=['feature', 'detected_type', 'actual_type', 'issue', 'sample'])

        df = pd.DataFrame(self._type_issues)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("⚠️  DATA TYPE IDENTIFICATION ISSUES REPORT")
        self.logger.info("=" * 80)
        self.logger.info(f"  Total issues found: {len(df)}")
        self.logger.info("-" * 80)

        for _, row in df.iterrows():
            self.logger.info(
                f"  Feature      : {row['feature']}\n"
                f"  Detected as  : {row['detected_type']}\n"
                f"  Raw dtype    : {row['actual_type']}\n"
                f"  Issue        : {row['issue']}\n"
                f"  Sample values: {row['sample']}\n"
                + "-" * 80
            )

        self.results['type_issues'] = {'issues': df.to_dict(orient='records'), 'total': len(df)}
        return df

    def _is_numeric_string_column(self, series: pd.Series) -> bool:
        """
        Check if object/string column contains PURE numeric values.

        This detects columns where numeric data is stored as strings:
        - '30', '32', '45' -> numeric
        - '9', '0', '1' -> numeric
        - 'apple', 'banana' -> not numeric

        Columns that look numeric but carry semantic meaning as codes are
        intentionally excluded:
        - Leading-zero strings: '07702', '00123' -> ZIP / postal codes, NOT numeric
          (leading zeros are meaningless for arithmetic but meaningful for codes)
        - Fixed-length digit strings where leading zeros appear in >5% of values
          are classified as categorical codes, not numerics.

        Uses sampling for efficiency on large datasets.
        """
        # Only check object/string columns
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return False

        # Get non-null sample (up to 1000 values for efficiency)
        sample = series.dropna().head(1000)

        if len(sample) == 0:
            return False

        # ── Leading-zero guard ────────────────────────────────────────────────
        # Strings like '07702' or '00123' represent coded values (postal codes,
        # account codes, etc.).  A real numeric column would never have leading
        # zeros.  If >5% of non-null string values start with '0' followed by
        # at least one more digit, treat the column as a code (not numeric).
        try:
            str_sample = sample.astype(str).str.strip()
            leading_zero_ratio = str_sample.str.match(r'^0\d+').sum() / len(str_sample)
            if leading_zero_ratio > 0.05:
                self.logger.debug(
                    f"  _is_numeric_string_column: '{series.name}' has {leading_zero_ratio*100:.1f}% "
                    f"leading-zero values — classified as categorical code, not numeric."
                )
                return False
        except Exception:
            pass

        # Try to convert to numeric
        try:
            # Attempt conversion - if most values convert successfully, it's numeric
            converted = pd.to_numeric(sample, errors='coerce')
            if isinstance(converted, pd.Series):
                non_null_converted = converted.notna()

                # If >80% of values can be converted to numeric, consider it numeric
                conversion_rate = non_null_converted.sum() / len(sample)

                if conversion_rate > 0.8:
                    # Additional check: ensure reasonable numeric range
                    # This helps distinguish '0001', '0002' (IDs) from '30', '45' (numeric)
                    numeric_values = converted.dropna()

                    if len(numeric_values) > 0:
                        # Check if values have reasonable variance (not all identical)
                        # and are not sequential IDs
                        unique_count = series.nunique()
                        total_count = len(series)

                        # If almost all values are unique, might be an ID
                        if unique_count / total_count > 0.95:
                            return False
                        return True
            else:
                return False

            return False

        except Exception:
            return False

    def _plot_feature_type_distribution(self, type_counts: Dict[str, int]):
        """Create bar chart of feature type distribution."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort by count
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        types = [t.upper() for t, _ in sorted_types]
        counts = [c for _, c in sorted_types]

        # Create bars
        cmap = plt.get_cmap('Set3')
        colors = [cmap(i) for i in np.linspace(0, 1, len(types))]
        bars = ax.bar(types, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_title('Feature Type Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Feature Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()

        self.logger.info("\n📊 Plot displayed above ⬆️")

    def _display_feature_samples(self, grouped_features: Dict[str, List[str]],
                               sample_size: int, show_all: bool):
        """Display sample data for each feature type to validate detection."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📋 DATA SAMPLES FOR TYPE VALIDATION")
        self.logger.info("=" * 80)

        for ftype in ['numeric', 'categorical', 'binary', 'datetime', 'id', 'outcome', 'unknown']:
            if ftype not in grouped_features or not grouped_features[ftype]:
                continue

            features = grouped_features[ftype]
            features_to_show = features if show_all else features[:sample_size]

            if not features_to_show:
                continue

            self.logger.info(f"\n▶ {ftype.upper()} Features ({len(features)} total, showing {len(features_to_show)}):")
            self.logger.info("-" * 80)

            for feature in features_to_show:
                if feature not in self.data.columns:
                    continue

                # Get non-null sample values
                sample_values = self.data[feature].dropna().head(10).tolist()

                # Store in results
                self.results['feature_types']['sample_data'][feature] = {
                    'type': ftype,
                    'sample_values': sample_values,
                    'unique_count': self.data[feature].nunique(),
                    'missing_count': self.data[feature].isnull().sum(),
                    'dtype': str(self.data[feature].dtype)
                }

                # Format display based on type
                if ftype == 'numeric':
                    # Convert to numeric if stored as string
                    numeric_data = pd.to_numeric(self.data[feature], errors='coerce')
                    stats = numeric_data.describe()
                    self.logger.info(f" • {feature}:")
                    self.logger.info(f"   Sample: {sample_values[:5]}")
                    self.logger.info(f"   Range: [{stats['min']:.2f}, {stats['max']:.2f}] | " +
                                     f"Mean: {stats['mean']:.2f} | Std: {stats['std']:.2f}")

                elif ftype == 'categorical':
                    value_counts = self.data[feature].value_counts()
                    self.logger.info(f" • {feature}:")
                    self.logger.info(f"   Sample: {sample_values[:5]}")
                    self.logger.info(f"   Unique: {len(value_counts)} | " +
                                     f"Top value: '{value_counts.index[0]}' ({value_counts.iloc[0]} occurrences)")

                elif ftype == 'binary':
                    value_counts = self.data[feature].value_counts()
                    self.logger.info(f" • {feature}:")
                    self.logger.info(f"   Values: {list(value_counts.index)}")
                    self.logger.info(f"   Distribution: {dict(value_counts)}")

                elif ftype == 'datetime':
                    self.logger.info(f" • {feature}:")
                    self.logger.info(f"   Sample: {sample_values[:3]}")

                    # Get statistics for datetime (works for both datetime64 and numeric dates)
                    data_clean = self.data[feature].dropna()
                    if len(data_clean) > 0:
                        # Calculate percentiles and stats
                        if pd.api.types.is_datetime64_any_dtype(self.data[feature]):
                            # Native datetime type
                            min_val = self.data[feature].min()
                            max_val = self.data[feature].max()
                            self.logger.info(f"   Range: {min_val} to {max_val}")
                        elif pd.api.types.is_numeric_dtype(self.data[feature]):
                            # For numeric datetime formats (YYYY, YYYYMM, etc.)
                            stats = data_clean.describe(percentiles=[0.25, 0.5, 0.75, 0.95])
                            mode_val = data_clean.mode().iloc[0] if len(data_clean.mode()) > 0 else 'N/A'

                            self.logger.info(f"   Min: {stats['min']:.0f} | 25%: {stats['25%']:.0f} | " +
                                             f"Median: {stats['50%']:.0f} | 75%: {stats['75%']:.0f} | " +
                                             f"95%: {stats['95%']:.0f} | Max: {stats['max']:.0f}")
                            self.logger.info(f"   Most Frequent: {mode_val}")
                        else:
                            # For text-based datetime formats (oct15_25, jan_26, etc.)
                            # Show unique count and most frequent value
                            unique_count = data_clean.nunique()
                            value_counts = data_clean.value_counts()
                            most_common = value_counts.index[0] if len(value_counts) > 0 else 'N/A'
                            most_common_count = value_counts.iloc[0] if len(value_counts) > 0 else 0

                            self.logger.info(f"   Unique: {unique_count} | " +
                                             f"Most Frequent: '{most_common}' ({most_common_count} occurrences)")

                elif ftype == 'id':
                    self.logger.info(f" • {feature}:")
                    self.logger.info(f"   Sample: {sample_values[:3]}")
                    self.logger.info(f"   Unique: {self.data[feature].nunique()} values")

                elif ftype == 'outcome':
                    value_counts = self.data[feature].value_counts()
                    self.logger.info(f" • {feature}:")
                    self.logger.info(f"   Sample: {sample_values[:5]}")
                    self.logger.info(f"   Distribution: {dict(value_counts)}")

                elif ftype == 'unknown':
                    # Unknown features - only actual special missing codes or completely empty features
                    series = self.data[feature]
                    non_null_count = series.notna().sum()
                    total_count = len(series)
                    missing_pct = (1 - non_null_count / total_count) * 100
                    unique_count = series.nunique()

                    self.logger.info(f" • {feature}:")
                    self.logger.info(f"   Sample: {sample_values[:5] if sample_values else '(all NaN/blank)'}")

                    # Determine the specific type of "unknown" and show appropriate status
                    if non_null_count == 0:
                        # All NaN/None
                        self.logger.info(f"   Status: 100.0% missing values | {unique_count} unique value(s)")
                        self.logger.info(f"   Reason: All values are NaN/None - cannot determine type")

                    elif unique_count == 1 and non_null_count > 0:
                        # Constant special missing code
                        unique_val = series.dropna().iloc[0]
                        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                            str_val = str(unique_val)
                            if len(str_val.strip()) == 0:
                                # It's whitespace or empty string
                                self.logger.info(f"   Status: 100.0% blank/whitespace | {unique_count} unique value(s)")
                                self.logger.info(f"   Reason: All values are blank/whitespace - no usable data")
                            else:
                                # It's a special missing code like 'UNKNOWN', 'N/A', 'NULL'
                                self.logger.info(f"   Status: 100.0% special missing code | {unique_count} unique value(s)")
                                self.logger.info(f"   Reason: All values are special missing code ('{unique_val}')")
                        else:
                            # Shouldn't happen but log it anyway
                            self.logger.info(f"   Status: {missing_pct:.1f}% missing | {unique_count} unique value(s)")
                            self.logger.info(f"   Reason: Cannot determine type ('{unique_val}')")
                    else:
                        # All non-null values are whitespace/blanks
                        self.logger.info(f"   Status: 100.0% blank/whitespace | {unique_count} unique value(s)")
                        self.logger.info(f"   Reason: All non-null values are blank/whitespace - no usable data")

            if not show_all and len(features) > sample_size:
                remaining = len(features) - sample_size
                self.logger.info(f"\n   ... {remaining} more {ftype} features (use show_all_features=True to see all)")

        self.logger.info("\n💡 TIP: Use analyzer.inspect_feature('feature_name') for detailed individual analysis")
        self.logger.info("   Or inspect multiple: analyzer.inspect_feature(['feature1', 'feature2'])")

    def inspect_feature(self, feature_names, show_plot: bool = True):
        """
        Inspect one or more features with sample data, statistics, and plots.

        Parameters:
        -----------
        feature_names : str or list of str
            Name(s) of the feature(s) to inspect
            - Single feature: 'age'
            - Multiple features: ['age', 'income', 'education']
        show_plot : bool, default=True
            Whether to show visualizations
        """
        """
        Results stored in self.results['feature_inspection'][feature_name]:
        -----------------------------------------------------------------
        - feature_type: Detected type
        - sample_values: First 20 non-null values
        - statistics: Detailed stats (mean, std, unique, missing, etc.)
        - value_distribution: Value counts for categorical/binary

        Examples:
        ---------
        >>> # Inspect single feature
        >>> analyzer.inspect_feature('age')

        >>> # Inspect multiple features
        >>> analyzer.inspect_feature(['age', 'income', 'education'])

        >>> # Without plots
        >>> analyzer.inspect_feature(['status', 'type'], show_plot=False)
        """
        # Convert single feature to list for uniform processing
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        # Validate all features exist
        missing_features = [f for f in feature_names if f not in self.data.columns]
        if missing_features:
            self.logger.error(f"❌ Feature(s) not found in dataset: {missing_features}")
            return

        # Inspect each feature
        for i, feature_name in enumerate(feature_names):
            if i > 0:
                self.logger.info("\n" + "-" * 80)

            self._inspect_single_feature(feature_name, show_plot)

        # Summary if multiple features
        if len(feature_names) > 1:
            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"✅ Inspection complete for {len(feature_names)} features!")
            self.logger.info(f" • Inspected: {feature_names}")
            self.logger.info(f" • Results stored in: analyzer.results['feature_inspection']")

    def _inspect_single_feature(self, feature_name: str, show_plot: bool = True):
        """Internal method to inspect a single feature."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"🔍 INSPECTING FEATURE: '{feature_name}'")
        self.logger.info("=" * 80)

        # Detect feature type
        feature_type = self._detect_single_feature_type(feature_name)

        # Get sample data
        sample_values = self.data[feature_name].dropna().head(20).tolist()
        missing_count = self.data[feature_name].isnull().sum()
        missing_pct = (missing_count / len(self.data)) * 100

        # Basic info
        self.logger.info(f"\n📋 Basic Information:")
        self.logger.info(f" • Type: {feature_type}")
        self.logger.info(f" • Data Type: {self.data[feature_name].dtype}")
        self.logger.info(f" • Total Values: {len(self.data)}")
        self.logger.info(f" • Missing: {missing_count} ({missing_pct:.2f}%)")
        self.logger.info(f" • Unique: {self.data[feature_name].nunique()}")

        # Type-specific analysis
        inspection_result = {
            'feature_type': feature_type,
            'sample_values': sample_values,
            'dtype': str(self.data[feature_name].dtype),
            'total_count': len(self.data),
            'missing_count': missing_count,
            'missing_percentage': missing_pct,
            'unique_count': self.data[feature_name].nunique()
        }

        if feature_type == 'numeric':
            # Convert to numeric if stored as string
            numeric_data = pd.to_numeric(self.data[feature_name], errors='coerce')
            stats = numeric_data.describe()
            self.logger.info(f"\n📈 Numeric Statistics:")
            self.logger.info(f" • Mean: {stats['mean']:.4f}")
            self.logger.info(f" • Std: {stats['std']:.4f}")
            self.logger.info(f" • Min: {stats['min']:.4f}")
            self.logger.info(f" • 25%: {stats['25%']:.4f}")
            self.logger.info(f" • Median: {stats['50%']:.4f}")
            self.logger.info(f" • 75%: {stats['75%']:.4f}")
            self.logger.info(f" • Max: {stats['max']:.4f}")

            inspection_result['statistics'] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'q25': stats['25%'],
                'median': stats['50%'],
                'q75': stats['75%'],
                'max': stats['max']
            }

            if show_plot:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Histogram
                axes[0].hist(numeric_data.dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[0].set_title(f'Distribution: {feature_name}')
                axes[0].set_xlabel('Value')
                axes[0].set_ylabel('Frequency')
                axes[0].grid(alpha=0.3)

                # Box plot
                axes[1].boxplot(numeric_data.dropna(), vert=True)
                axes[1].set_title(f'Box Plot: {feature_name}')
                axes[1].set_ylabel('Value')
                axes[1].grid(alpha=0.3)

                plt.tight_layout()
                plt.show()
                self.logger.info("\n📊 Plots displayed above ⬆️")

        elif feature_type in ['categorical', 'binary', 'outcome']:
            value_counts = self.data[feature_name].value_counts()
            self.logger.info(f"\n📊 Value Distribution:")

            display_counts = value_counts.head(10)
            for value, count in display_counts.items():
                pct = (count / len(self.data)) * 100
                self.logger.info(f" • '{value}': {count} ({pct:.2f}%)")

            if len(value_counts) > 10:
                self.logger.info(f"   ... {len(value_counts) - 10} more unique values")

            inspection_result['value_distribution'] = dict(value_counts)

            if show_plot and len(value_counts) <= 20:
                plt.figure(figsize=(12, 6))
                value_counts.head(20).plot(kind='bar', edgecolor='black', alpha=0.7)
                plt.title(f'Value Distribution: {feature_name}')
                plt.xlabel('Value')
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.show()
                self.logger.info("\n📊 Plot displayed above ⬆️")

        elif feature_type == 'datetime':
            self.logger.info(f"\n📅 Datetime Information:")

            data_clean = self.data[feature_name].dropna()

            if pd.api.types.is_datetime64_any_dtype(self.data[feature_name]):
                # Native datetime type
                min_date = self.data[feature_name].min()
                max_date = self.data[feature_name].max()
                date_range = max_date - min_date
                self.logger.info(f" • Earliest: {min_date}")
                self.logger.info(f" • Latest: {max_date}")
                self.logger.info(f" • Range: {date_range}")

                inspection_result['datetime_info'] = {
                    'min_date': str(min_date),
                    'max_date': str(max_date),
                    'range': str(date_range)
                }

                if show_plot:
                    # Timeline count plot
                    date_counts = self.data[feature_name].value_counts().sort_index()
                    plt.figure(figsize=(14, 5))
                    plt.plot(date_counts.index, date_counts.values, marker='o', alpha=0.7)
                    plt.title(f'Timeline: {feature_name}')
                    plt.xlabel('Date')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.show()

            else:
                # Numeric datetime format (YYYY, YYYYMM, etc.)
                stats = data_clean.describe(percentiles=[0.25, 0.5, 0.75, 0.95])
                mode_val = data_clean.mode().iloc[0] if len(data_clean.mode()) > 0 else 'N/A'

                self.logger.info(f" • Min: {stats['min']:.0f}")
                self.logger.info(f" • 25th Percentile: {stats['25%']:.0f}")
                self.logger.info(f" • Median (50th): {stats['50%']:.0f}")
                self.logger.info(f" • 75th Percentile: {stats['75%']:.0f}")
                self.logger.info(f" • 95th Percentile: {stats['95%']:.0f}")
                self.logger.info(f" • Max: {stats['max']:.0f}")
                self.logger.info(f" • Most Frequent Value: {mode_val}")
                self.logger.info(f" • Range: {stats['max'] - stats['min']:.0f}")

                inspection_result['datetime_info'] = {
                    'min': float(stats['min']),
                    'q25': float(stats['25%']),
                    'median': float(stats['50%']),
                    'q75': float(stats['75%']),
                    'q95': float(stats['95%']),
                    'max': float(stats['max']),
                    'mode': float(mode_val) if mode_val != 'N/A' else None,
                    'range': float(stats['max'] - stats['min'])
                }

                if show_plot:
                    # Create distribution plots for numeric datetime
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                    # Histogram
                    axes[0].hist(data_clean, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
                    axes[0].set_title(f'Distribution: {feature_name}')
                    axes[0].set_xlabel('Value')
                    axes[0].set_ylabel('Frequency')
                    axes[0].grid(alpha=0.3)
                    axes[0].axvline(stats['50%'], color='red', linestyle='--', linewidth=2, label=f"Median: {stats['50%']:.0f}")
                    axes[0].legend()

                    # Value counts over time (sorted)
                    value_counts = self.data[feature_name].value_counts().sort_index()
                    axes[1].plot(value_counts.index, value_counts.values, marker='o', alpha=0.7, color='green')
                    axes[1].set_title(f'Timeline: {feature_name}')
                    axes[1].set_xlabel('Year/Period')
                    axes[1].set_ylabel('Count')
                    axes[1].grid(alpha=0.3)
                    axes[1].tick_params(axis='x', rotation=45)

                    plt.tight_layout()
                    plt.show()
                    self.logger.info("\n📊 Plots displayed above ⬆️")

        # Sample values
        self.logger.info(f"\n📋 Sample Values (first 20 non-null):")
        self.logger.info(f"   {sample_values}")

        # Store results
        if 'feature_inspection' not in self.results:
            self.results['feature_inspection'] = {}
        self.results['feature_inspection'][feature_name] = inspection_result

        self.logger.info(f"\n✅ Inspection complete for '{feature_name}'!")
        self.logger.info(f" • Results stored in: analyzer.results['feature_inspection']['{feature_name}']")

    def _detect_single_feature_type(self, feature_name: str) -> str:
        """Detect type of a single feature (helper for inspect_feature)."""
        col = feature_name
        series = self.data[col]

        if col == self.outcome_feature:
            return 'outcome'

        # CRITICAL: Check if feature is completely empty or has no usable values
        # Only mark as unknown if 100% of values are missing/blank/special codes
        non_null_count = series.notna().sum()

        if non_null_count == 0:
            # All values are NaN/None - cannot determine type
            return 'unknown'

        # For string/object columns, check if ALL non-null values are just whitespace
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                # Check if ALL non-null values are empty strings or whitespace
                non_empty_count = non_null_series.astype(str).str.strip().str.len().gt(0).sum()
                if non_empty_count == 0:
                    # ALL non-null values are blank/whitespace
                    return 'unknown'

        # IMPORTANT: Check for datetime BEFORE constant check
        # A constant datetime value (e.g., all 'oct15_25') is still a valid datetime feature
        if self._is_datetime_column(series):
            return 'datetime'

        # Check if feature has only ONE unique value (constant column)
        # This catches 100% identical non-numeric, non-datetime values (special codes like 'UNKNOWN', 'NA', 'NULL')
        # IMPORTANT: Numeric constants (all 0s, all 1s) should still be classified as 'numeric'
        # IMPORTANT: Datetime constants (all 'oct15_25') were already handled above
        unique_count = series.nunique()
        if unique_count == 1:
            # Check if it's a numeric constant (all 0s, all 1s, etc.)
            if pd.api.types.is_numeric_dtype(series):
                # Numeric constant - will be classified as 'numeric' later
                pass # Continue with normal classification
            else:
                # Non-numeric, non-datetime constant - likely a special code like 'UNKNOWN', 'NA', empty string
                return 'unknown'

        # Now proceed with normal type detection

        # Check if ID column
        if self._is_id_column(col, series):
            return 'id'

        # Check if binary
        if series.nunique() == 2:
            return 'binary'

        # Check if numeric (native numeric type)
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'

        # Check if numeric stored as string
        if self._is_numeric_string_column(series):
            return 'numeric'

        # Categorical (remaining object/string types)
        if pd.api.types.is_object_dtype(series) or str(series.dtype) == 'category':
            return 'categorical'

        return 'unknown'

    # =========================================================================
    # MISSING VALUES ANALYSIS - COMPLETELY INDEPENDENT
    # =========================================================================

    def analyze_missing_values(self, threshold: float = 0.5, show_plots: bool = True,
                              show_feature_types: bool = True,
                              detect_special_codes: bool = True,
                              convert_special_to_nan: bool = True,
                              strip_whitespace: bool = True,
                              show_all_features: bool = True) -> Dict[str, Any]:
        """
        ⭐ INDEPENDENT ANALYSIS - No prerequisites needed!

        Analyzes missing values across ALL features (numeric, categorical, datetime, etc.):
        - Automatically cleans whitespace from legitimate values
        - Detects and converts special missing codes to NaN
        - Shows: Two-table report (above threshold + optional complete list)
        - Stores: Complete missing info in self.results['missing_values']

        Parameters:
        -----------
        threshold : float, default=0.5
            Flag features with missing > threshold (0.5 = 50%)
        show_plots : bool, default=True
            Whether to display bar chart
        show_feature_types : bool, default=True
            Whether to show feature type alongside missing values
        detect_special_codes : bool, default=True
            Whether to detect special missing codes (blanks, 'UNKNOWN', '999', etc.)
            Detects 50+ patterns including: NA, NULL, UNKNOWN, 999, -1, pure whitespace, etc.
        convert_special_to_nan : bool, default=True
            Whether to convert detected special codes to NaN automatically.
            This standardizes missing value representation across the dataset.
        strip_whitespace : bool, default=True
            Whether to automatically strip leading/trailing whitespace from string values.
            Cleans: 'INDIVIDUAL     ' -> 'INDIVIDUAL' while preserving pure whitespace as missing.
        show_all_features : bool, default=True
            Whether to show the complete table with ALL features that have missing values.
            - True: Shows both (1) features above threshold AND (2) complete list
            - False: Shows only features above threshold
        """
        """
        Returns:
        --------
        dict : Summary with total_missing, overall_percentage, high_missing_features

        Example:
        --------
        >>> analyzer = Analyzer(df)
        >>> # Show only critical features (above threshold)
        >>> summary = analyzer.analyze_missing_values(threshold=0.5, show_all_features=False)
        >>>
        >>> # Show both critical AND complete list (default)
        >>> summary = analyzer.analyze_missing_values(threshold=0.5, show_all_features=True)
        >>> print(f"Total missing: {summary['total_missing']:,}")
        >>> print(f"High missing features: {summary['high_missing_features']}")
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🔍 MISSING VALUES ANALYSIS (Independent - No Prerequisites)")
        self.logger.info("=" * 80)

        # Strip whitespace from string columns if requested (data cleaning)
        if strip_whitespace:
            self.logger.info("\n🧹 Stripping leading/trailing whitespace from string columns...")
            cleaned_count = self._strip_whitespace_from_strings()
            if cleaned_count > 0:
                self.logger.info(f" ✅ Cleaned {cleaned_count} features")
            else:
                self.logger.info(" ℹ️ No string features needed cleaning")

        # Detect and optionally convert special missing codes
        special_codes_detected = {}
        if detect_special_codes:
            self.logger.debug("Detecting special missing codes...")
            special_codes_detected = self._detect_special_missing_codes()

            if convert_special_to_nan and special_codes_detected:
                self.logger.debug("Converting special codes to NaN...")
                self._convert_special_codes_to_nan(special_codes_detected)

        # Calculate missing values (completely self-contained - ALL FEATURES)
        self.logger.debug("Analyzing missing values from scratch (all feature types)...")
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100

        # Detect feature types if requested (for display purposes)
        feature_types_map = {}
        if show_feature_types:
            feature_types_map = self._detect_feature_types()

        # Create detailed dictionary for ALL features with missing values
        missing_by_feature = {}
        for col in self.data.columns:
            if missing_counts[col] > 0:
                missing_by_feature[col] = {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_percentages[col]),
                    'severity': self._get_missing_severity(missing_percentages[col]),
                    'feature_type': feature_types_map.get(col, 'unknown') if show_feature_types else None
                }

        # Overall statistics
        total_cells = int(self.data.shape[0] * self.data.shape[1])
        total_missing = int(missing_counts.sum())
        overall_percentage = (total_missing / total_cells) * 100

        # Features above threshold
        features_above_threshold = [
            col for col in missing_by_feature.keys()
            if missing_by_feature[col]['percentage'] > threshold * 100
        ]

        # Store results
        self.results['missing_values'] = {
            'total_missing': total_missing,
            'total_cells': total_cells,
            'overall_percentage': overall_percentage,
            'features_with_missing': len(missing_by_feature),
            'missing_by_feature': missing_by_feature,
            'features_above_threshold': features_above_threshold,
            'threshold_used': threshold,
            'special_codes_detected': special_codes_detected,
            'special_codes_converted': convert_special_to_nan,
            'analysis_timestamp': pd.Timestamp.now()
        }

        # Display summary
        self.logger.info("\n📊 OVERALL STATISTICS:")
        self.logger.info("-" * 60)
        self.logger.info(f" • Total Features Analyzed: {len(self.data.columns)} (all types)")
        self.logger.info(f" • Total Missing Values: {total_missing:,}")
        self.logger.info(f" • Overall Missing Rate: {overall_percentage:.2f}%")
        self.logger.info(f" • Features with Missing: {len(missing_by_feature)}")
        self.logger.info(f" • Features > {threshold*100}% missing: {len(features_above_threshold)}")

        sorted_features = sorted(
            missing_by_feature.items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        ) if missing_by_feature else []

        if missing_by_feature:
            # FIRST TABLE: Features ABOVE threshold (critical features)
            if features_above_threshold:
                if show_feature_types:
                    self.logger.info(f"\n🚨 FEATURES ABOVE THRESHOLD (>{threshold*100}% missing) - {len(features_above_threshold)} features:")
                    self.logger.info("-" * 95)
                    self.logger.info(f"{'Feature':<30} {'Type':<12} {'Missing Count':<15} {'Percentage':<12} {'Severity'}")
                    self.logger.info("-" * 95)
                else:
                    self.logger.info(f"\n🚨 FEATURES ABOVE THRESHOLD (>{threshold*100}% missing) - {len(features_above_threshold)} features:")
                    self.logger.info("-" * 80)
                    self.logger.info(f"{'Feature':<30} {'Missing Count':<15} {'Percentage':<12} {'Severity'}")
                    self.logger.info("-" * 80)

                # Show only features above threshold
                for feature, info in sorted_features:
                    if info['percentage'] > threshold * 100:
                        if show_feature_types:
                            self.logger.info(
                                f"{feature:<30} {info['feature_type']:<12} {info['count']:<15,} "
                                f"{info['percentage']:<11.2f}% {info['severity']}"
                            )
                        else:
                            self.logger.info(
                                f"{feature:<30} {info['count']:<15,} "
                                f"{info['percentage']:<11.2f}% {info['severity']}"
                            )

                # Show breakdown by feature type for ABOVE THRESHOLD features
                if show_feature_types and features_above_threshold:
                    self.logger.info("\n📊 MISSING VALUES BY FEATURE TYPE (Above Threshold):")
                    self.logger.info("-" * 60)

                    # Group ABOVE THRESHOLD features by type
                    above_threshold_by_type = {}
                    for feature, info in sorted_features:
                        if info['percentage'] > threshold * 100:
                            ftype = info['feature_type']
                            if ftype not in above_threshold_by_type:
                                above_threshold_by_type[ftype] = {
                                    'count': 0,
                                    'features': []
                                }
                            above_threshold_by_type[ftype]['count'] += 1
                            above_threshold_by_type[ftype]['features'].append((feature, info['percentage']))

                    for ftype, data in sorted(above_threshold_by_type.items(), key=lambda x: x[1]['count'], reverse=True):
                        self.logger.info(f" • {ftype.upper()}: {data['count']} features")

                        # Sort features by percentage (descending) and show ALL
                        sorted_type_features = sorted(data['features'], key=lambda x: x, reverse=True)

                        for feat, pct in sorted_type_features: # Show ALL features (no limit)
                            self.logger.info(f"     - {feat}: {pct:.2f}%")

            # SECOND TABLE: ALL features with missing values (comprehensive view)
            # Only show if show_all_features=True
            if show_all_features:
                if show_feature_types:
                    self.logger.info(f"\n📋 ALL FEATURES WITH MISSING VALUES (Complete List) - {len(missing_by_feature)} features:")
                    self.logger.info("-" * 95)
                    self.logger.info(f"{'Feature':<30} {'Type':<12} {'Missing Count':<15} {'Percentage':<12} {'Severity'}")
                    self.logger.info("-" * 95)
                else:
                    self.logger.info(f"\n📋 ALL FEATURES WITH MISSING VALUES (Complete List) - {len(missing_by_feature)} features:")
                    self.logger.info("-" * 80)
                    self.logger.info(f"{'Feature':<30} {'Missing Count':<15} {'Percentage':<12} {'Severity'}")
                    self.logger.info("-" * 80)

                # Show ALL features (no limit)
                for feature, info in sorted_features:
                    if show_feature_types:
                        self.logger.info(
                            f"{feature:<30} {info['feature_type']:<12} {info['count']:<15,} "
                            f"{info['percentage']:<11.2f}% {info['severity']}"
                        )
                    else:
                        self.logger.info(
                            f"{feature:<30} {info['count']:<15,} "
                            f"{info['percentage']:<11.2f}% {info['severity']}"
                        )

                # Show breakdown by feature type for ALL features
                if show_feature_types:
                    self.logger.info("\n📊 MISSING VALUES BY FEATURE TYPE (All Features):")
                    self.logger.info("-" * 60)

                    # Group ALL features by type
                    missing_by_type = {}
                    for feature, info in missing_by_feature.items():
                        ftype = info['feature_type']
                        if ftype not in missing_by_type:
                            missing_by_type[ftype] = {
                                'count': 0,
                                'features': []
                            }
                        missing_by_type[ftype]['count'] += 1
                        # Store feature with its percentage for sorting
                        missing_by_type[ftype]['features'].append((feature, info['percentage']))

                    for ftype, data in sorted(missing_by_type.items(), key=lambda x: x[1]['count'], reverse=True):
                        self.logger.info(f" • {ftype.upper()}: {data['count']} features")

                        # Sort features by percentage (descending) and show ALL
                        sorted_type_features = sorted(data['features'], key=lambda x: x, reverse=True)

                        for feat, pct in sorted_type_features: # Show ALL features (no limit)
                            self.logger.info(f"     - {feat}: {pct:.2f}%")

        # Display special codes for ABOVE THRESHOLD features
        if detect_special_codes and special_codes_detected:
            # Filter special codes to only show above-threshold features
            above_threshold_special = {
                feat: info for feat, info in special_codes_detected.items()
                if feat in [f for f, i in sorted_features if i['percentage'] > threshold * 100]
            }

            if above_threshold_special:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("🔍 SPECIAL MISSING CODES DETECTED (Above Threshold)")
                self.logger.info("=" * 80)
                self.logger.info("\nSpecial codes like blanks, 'UNKNOWN', '999', etc. that represent missing values:")
                self.logger.info("\nLegend:")
                self.logger.info("  🔴 Pure Whitespace - Empty cells with only spaces (true missing)")
                self.logger.info("  🟡 Known Indicator - Standard missing codes (NA, UNKNOWN, NULL, etc.)")
                self.logger.info("  🔵 Numeric Code - Special numbers (999, -1, -999, etc.)")

                for feature in sorted(above_threshold_special.keys(),
                                     key=lambda x: above_threshold_special[x]['total_count'],
                                     reverse=True):
                    info = above_threshold_special[feature]
                    breakdown = info.get('breakdown', {})

                    self.logger.info(f"\n▶ {feature}:")
                    self.logger.info(f" • Total special codes: {info['total_count']:,}")

                    # Show breakdown by type
                    if breakdown.get('pure_whitespace', 0) > 0:
                        self.logger.info(f"   🔴 Pure whitespace: {breakdown['pure_whitespace']:,} occurrences")
                    if breakdown.get('known_indicator', 0) > 0:
                        self.logger.info(f"   🟡 Known indicators: {breakdown['known_indicator']:,} occurrences")
                    if breakdown.get('numeric_code', 0) > 0:
                        self.logger.info(f"   🔵 Numeric codes: {breakdown['numeric_code']:,} occurrences")

                    # Show top 5 specific codes with their types
                    self.logger.info(" • Top codes detected:")
                    for code, code_info in list(info['codes'].items())[:5]:
                        if isinstance(code_info, dict):
                            display = code_info.get('display', str(code))
                            count = code_info['count']
                            code_type = code_info['type']

                            # Choose icon based on type
                            icon = '🔴' if code_type == 'pure_whitespace' else '🟡' if code_type == 'known_indicator' else '🔵'
                            self.logger.info(f"     {icon} {display}: {count:,} occurrences")
                        else:
                            # Fallback for old format
                            self.logger.info(f"     - '{code}': {code_info:,} occurrences")

                    if len(info['codes']) > 5:
                        self.logger.info(f"     ... and {len(info['codes']) - 5} more codes")

                if convert_special_to_nan:
                    self.logger.info("\n✅ Special codes converted to NaN and included in missing values above")
                else:
                    self.logger.info("\n💡 TIP: Use convert_special_to_nan=True to convert these to NaN")

            # Display special codes for ALL features
            if show_all_features:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("🔍 SPECIAL MISSING CODES DETECTED (All Features)")
                self.logger.info("=" * 80)
                self.logger.info("\nSpecial codes like blanks, 'UNKNOWN', '999', etc. that represent missing values:")
                self.logger.info("\nLegend:")
                self.logger.info("  🔴 Pure Whitespace - Empty cells with only spaces (true missing)")
                self.logger.info("  🟡 Known Indicator - Standard missing codes (NA, UNKNOWN, NULL, etc.)")
                self.logger.info("  🔵 Numeric Code - Special numbers (999, -1, -999, etc.)")

                for feature in sorted(special_codes_detected.keys(),
                                     key=lambda x: special_codes_detected[x]['total_count'],
                                     reverse=True):
                    info = special_codes_detected[feature]
                    breakdown = info.get('breakdown', {})

                    self.logger.info(f"\n▶ {feature}:")
                    self.logger.info(f" • Total special codes: {info['total_count']:,}")

                    # Show breakdown by type
                    if breakdown.get('pure_whitespace', 0) > 0:
                        self.logger.info(f"   🔴 Pure whitespace: {breakdown['pure_whitespace']:,} occurrences")
                    if breakdown.get('known_indicator', 0) > 0:
                        self.logger.info(f"   🟡 Known indicators: {breakdown['known_indicator']:,} occurrences")
                    if breakdown.get('numeric_code', 0) > 0:
                        self.logger.info(f"   🔵 Numeric codes: {breakdown['numeric_code']:,} occurrences")

                    # Show top 5 specific codes with their types
                    self.logger.info(" • Top codes detected:")
                    for code, code_info in list(info['codes'].items())[:5]:
                        if isinstance(code_info, dict):
                            display = code_info.get('display', str(code))
                            count = code_info['count']
                            code_type = code_info['type']

                            # Choose icon based on type
                            icon = '🔴' if code_type == 'pure_whitespace' else '🟡' if code_type == 'known_indicator' else '🔵'
                            self.logger.info(f"     {icon} {display}: {count:,} occurrences")
                        else:
                            # Fallback for old format
                            self.logger.info(f"     - '{code}': {code_info:,} occurrences")

                    if len(info['codes']) > 5:
                        self.logger.info(f"     ... and {len(info['codes']) - 5} more codes")

                if convert_special_to_nan:
                    self.logger.info("\n✅ Special codes converted to NaN and included in missing values above")
                else:
                    self.logger.info("\n💡 TIP: Use convert_special_to_nan=True to convert these to NaN")

        # Display plot
        if show_plots:
            self._plot_missing_values(missing_by_feature, threshold)
        else:
            if not missing_by_feature:
                self.logger.info("\n✅ No missing values found in the dataset!")

        # Recommendations
        self.logger.info("\n💡 RECOMMENDATIONS:")
        if len(features_above_threshold) > 0:
            self.logger.warning(f"  • ⚠️ {len(features_above_threshold)} features have >{threshold*100}% missing - consider removal:")
            for feat in features_above_threshold[:5]:
                pct = missing_by_feature[feat]['percentage']
                self.logger.warning(f"      - {feat}: {pct:.1f}% missing")
            if len(features_above_threshold) > 5:
                self.logger.warning(f"      ... and {len(features_above_threshold) - 5} more")

        critical_features = [f for f, info in missing_by_feature.items() if info['percentage'] > 90]
        if critical_features:
            self.logger.error(f"  • 🔴 {len(critical_features)} features are >90% missing - likely unusable")

        self.logger.info("\n✅ Missing values analysis complete!")
        self.logger.info(f"  • Results stored in: analyzer.results['missing_values']")

        # Return summary for programmatic access
        return {
            'total_missing': total_missing,
            'overall_percentage': overall_percentage,
            'features_with_missing': list(missing_by_feature.keys()),
            'high_missing_features': features_above_threshold,
            'critical_features': critical_features,
            'special_codes_detected': len(special_codes_detected) if special_codes_detected else 0
        }

    def _get_missing_severity(self, percentage: float) -> str:
        """Determine severity of missing values."""
        if percentage < 5:
            return "🟢 Low"
        elif percentage < 20:
            return "🟡 Moderate"
        elif percentage < 50:
            return "🟠 High"
        else:
            return "🔴 Critical"

    def _strip_whitespace_from_strings(self) -> int:
        """
        Strip leading and trailing whitespace from all string/object columns.

        This cleans legitimate values like:
        - 'INDIVIDUAL     ' -> 'INDIVIDUAL'
        - '    LLC    ' -> 'LLC'

        Does NOT remove values that are ONLY whitespace (those remain for detection).
        """
        cleaned_count = 0

        for col in self.data.columns:
            # Only process string/object columns
            if pd.api.types.is_object_dtype(self.data[col]) or pd.api.types.is_string_dtype(self.data[col]):
                # Check if any values have leading/trailing whitespace (excluding pure whitespace)
                non_null = self.data[col].dropna()

                if len(non_null) > 0:
                    # Check if there are any values that need stripping
                    # (have content and differ from their stripped version)
                    needs_cleaning = False
                    for value in non_null.head(100).unique(): # Sample first 100 unique values
                        if isinstance(value, str) and value != value.strip() and value.strip() != '':
                            needs_cleaning = True
                            break

                    if needs_cleaning:
                        # Apply stripping while preserving NaN
                        # Only strip values that have content (not pure whitespace)
                        def strip_if_content(x):
                            if pd.isna(x):
                                return x
                            if isinstance(x, str):
                                stripped = x.strip()
                                # If stripping results in empty string, keep original (it's pure whitespace)
                                # This preserves them for special code detection
                                if stripped == '':
                                    return x
                                else:
                                    return stripped
                            return x

                        self.data[col] = self.data[col].apply(strip_if_content)
                        cleaned_count += 1
                        self.logger.debug(f"      Stripped whitespace from: {col}")

        return cleaned_count

    def _detect_special_missing_codes(self) -> Dict[str, Dict]:
        """
        Detect special missing code patterns in the dataset.

        Detects ONLY pure special codes (cells with no actual content):
        - Blank strings with ONLY whitespace: ' ', '  ', '   ' (after stripping becomes empty)
        - Common missing indicators: 'UNKNOWN', 'NA', 'N/A', 'NULL', 'NONE', 'MISSING', etc.
        - Special numeric codes: 999, 9999, -1, -99, -999, -9999, etc.
        - Various formats: '<NA>', '<NULL>', 'NaN', 'UNDEF', 'VOID', 'BLANK', 'EMPTY', etc.
        - Placeholder symbols: '-', '--', '---', '?', '.'

        Does NOT flag legitimate values with trailing spaces:
        - 'INDIVIDUAL     '        -> NOT flagged (has content after strip)
        - '               '        -> Flagged (only whitespace, no content)
        """
        """
        Returns:
        --------
        dict : {feature_name: {'total_count': int, 'codes': {code: count, type: str}}}
               where type is: 'pure_whitespace', 'known_indicator', 'numeric_code'
        """
        # Define comprehensive list of special missing codes for string columns
        STRING_MISSING_CODES = {
            # NA variations
            'NA', 'N/A', 'NA.', 'N.A.', '<NA>',
            # NaN variations
            'NAN', '<NAN>',
            # NULL variations
            'NULL', '<NULL>',
            # None variations
            'NONE', '<NONE>',
            # Missing variations
            'MISSING', '<MISSING>', 'MISS',
            # Unknown variations
            'UNKNOWN', 'UNK', 'UNKN', 'UNDEF', 'UNDEFINED',
            # Empty/blank variations
            'BLANK', 'EMPTY', 'VOID', 'NIL',
            # No data variations
            'NODATA', 'NO DATA', 'NOT AVAILABLE', 'NOTAVAILABLE',
            # Placeholder symbols
            '-', '--', '---', '?', '.',
        }

        # Define numeric missing codes (as integers for numeric columns)
        NUMERIC_MISSING_CODES = {
            -1, -9, -99, -999, -9999, -99999,
            999, 9999, 99999, 999999, 9999999
        }

        # Define string representations of numeric codes (for string/object columns)
        STRING_NUMERIC_CODES = {
            '-1', '-9', '-99', '-999', '-9999', '-99999',
            '999', '9999', '99999', '999999', '9999999'
        }

        special_codes_by_feature = {}

        for col in self.data.columns:
            # Skip if all values are already NaN
            if self.data[col].isnull().all():
                continue

            # Check for special codes with type classification
            special_codes = {}

            # For string/object columns
            if pd.api.types.is_object_dtype(self.data[col]) or pd.api.types.is_string_dtype(self.data[col]):
                non_null = self.data[col].dropna()

                for value in non_null.unique():
                    if isinstance(value, str):
                        stripped_value = value.strip()
                        stripped_upper = stripped_value.upper()

                        # 1. Check if it's ONLY whitespace (pure blank)
                        # This catches: ' ', '  ', '   ', but NOT 'INDIVIDUAL  '
                        if stripped_value == '':
                            if value not in special_codes:
                                count = int((non_null == value).sum())
                                special_codes[value] = {
                                    'count': count,
                                    'type': 'pure_whitespace',
                                    'display': f"'{value}' (pure whitespace - {len(value)} spaces)"
                                }

                        # 2. Check if stripped value is a known string missing indicator
                        elif stripped_upper in STRING_MISSING_CODES:
                            if value not in special_codes:
                                count = int((non_null == value).sum())
                                special_codes[value] = {
                                    'count': count,
                                    'type': 'known_indicator',
                                    'display': f"'{value}' (known missing indicator)"
                                }

                        # 3. Check if it's a numeric code stored as string
                        elif stripped_value in STRING_NUMERIC_CODES:
                            if value not in special_codes:
                                count = int((non_null == value).sum())
                                special_codes[value] = {
                                    'count': count,
                                    'type': 'numeric_code',
                                    'display': f"'{value}' (special numeric code)"
                                }

            # For numeric columns, check special numeric codes
            elif pd.api.types.is_numeric_dtype(self.data[col]):
                non_null = self.data[col].dropna()

                for value in non_null.unique():
                    # Check if the numeric value is a known missing indicator
                    if value in NUMERIC_MISSING_CODES:
                        if value not in special_codes:
                            count = int((non_null == value).sum())
                            special_codes[value] = {
                                'count': count,
                                'type': 'numeric_code',
                                'display': f"{value} (special numeric code)"
                            }

            # Store if special codes found
            if special_codes:
                total_count = sum(code_info['count'] for code_info in special_codes.values())

                # Categorize by type
                whitespace_count = sum(info['count'] for info in special_codes.values() if info['type'] == 'pure_whitespace')
                indicator_count = sum(info['count'] for info in special_codes.values() if info['type'] == 'known_indicator')
                numeric_count = sum(info['count'] for info in special_codes.values() if info['type'] == 'numeric_code')

                special_codes_by_feature[col] = {
                    'total_count': total_count,
                    'codes': special_codes,
                    'breakdown': {
                        'pure_whitespace': whitespace_count,
                        'known_indicator': indicator_count,
                        'numeric_code': numeric_count
                    }
                }

        return special_codes_by_feature

    def _convert_special_codes_to_nan(self, special_codes_detected: Dict):

        """
        Convert detected special codes to NaN in the dataset.
        Handles both old format (code: count) and new format (code: {count, type, display}).
        """
        for feature, info in special_codes_detected.items():
            for code, code_info in info['codes'].items():
                # Handle both old and new format
                if isinstance(code_info, dict):
                    # New format with type information
                    pass # Just need the code key
                else:
                    # Old format (code: count)
                    pass # Just need the code key

                # Convert to NaN
                self.data.loc[self.data[feature] == code, feature] = np.nan

        self.logger.debug(f"Converted special codes to NaN in {len(special_codes_detected)} features")

    def _plot_missing_values(self, missing_by_feature: Dict, threshold: float):
        """Create missing values bar chart (sorted by percentage descending)."""
        # Sort features by percentage (descending) and take top 20
        sorted_features = sorted(
            missing_by_feature.items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )[:20]

        features = [f[0] for f in sorted_features]  # Get feature names
        percentages = [f[1]['percentage'] for f in sorted_features]  # Get percentages

        # Plotagem de valores ausentes
        fig, ax = plt.subplots(figsize=(12, 8))

        # Color by severity
        colors = []
        for pct in percentages:
            if pct < 5:
                colors.append('#2ecc71')
            elif pct < 20:
                colors.append('#f39c12')
            elif pct < 50:
                colors.append('#e67e22')
            else:
                colors.append('#e74c3c')

        # Plot bars (reverse order so highest is at top)
        bars = ax.barh(features[::-1], percentages[::-1], color=colors[::-1],
                       alpha=0.8, edgecolor='black', linewidth=1.2)

        # Add threshold line
        ax.axvline(threshold * 100, color='darkred', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold*100}%)', zorder=10)

        # Add value labels (also reversed)
        for bar, pct in zip(bars, percentages[::-1]):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', va='center', fontweight='bold', fontsize=9)

        ax.set_xlabel('Missing Percentage (%)', fontsize=12)
        ax.set_title('Missing Values by Feature (Top 20)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()

        self.logger.info("\n📊 Plot displayed above ⬆️")

    # ==============================================================================
    # LOW VARIANCE ANALYSIS - COMPLETELY INDEPENDENT
    # ==============================================================================

    def analyze_low_variance(self, variance_threshold: float = 0.01, show_plots: bool = True) -> Dict[str, Any]:
        """
        ⭐ INDEPENDENT ANALYSIS - No prerequisites needed!
        Identifies numeric features with low variance (near-constant):
        - Detects: Low variance and constant features
        - Shows: Summary table, bar chart, recommendations
        - Stores: Variance info in self.results['low_variance']
        
        Returns:
        --------
        dict : Summary with variance analysis results
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📉 LOW VARIANCE ANALYSIS (Independent - No Prerequisites)")
        self.logger.info("=" * 80)

        # Get numeric features using the analyzer's own feature_type_map so that
        # ID columns (TIN, MID, SequenceKey) and low-cardinality integer codes
        # (Fraud.Type 1/2/3) that are now classified as 'id'/'categorical' are
        # never included in the variance analysis.
        self.logger.debug("Analyzing variance from scratch...")
        numeric_features = self._get_numeric_features(exclude_binary=True, use_cached=True)
        if not numeric_features:
            # Fallback if no feature type analysis has been run
            numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if self.outcome_feature and self.outcome_feature in numeric_features:
                numeric_features.remove(self.outcome_feature)

        # Safety filter: keep only columns that
        #   (a) have a numeric pandas dtype, AND
        #   (b) are NOT classified as id / categorical / datetime / outcome / unknown
        #       in the feature_type_map (avoids TIN, MID, fraud_type, etc.)
        _ft_map = self.results.get('feature_types', {}).get('feature_type_map', {})
        _NON_NUMERIC_TYPES = {'id', 'categorical', 'datetime', 'outcome', 'unknown'}
        numeric_features = [
            col for col in numeric_features
            if pd.api.types.is_numeric_dtype(self.data[col])
            and _ft_map.get(col, 'numeric') not in _NON_NUMERIC_TYPES
        ]

        self.logger.debug(f"Found {len(numeric_features)} numeric features for variance analysis")
        
        if len(numeric_features) == 0:
            self.logger.warning("⚠️ No numeric features found for variance analysis")
            self.results['low_variance'] = {
                'total_numeric': 0,
                'variance_threshold': variance_threshold,
                'variance_by_feature': {},
                'low_variance_features': [],
                'constant_features': [],
                'low_variance_count': 0,
                'analysis_timestamp': pd.Timestamp.now()
            }
            return {
                'total_numeric': 0,
                'low_variance_count': 0,
                'constant_count': 0,
                'variance_threshold': variance_threshold
            }

        # Cálculo de variância
        variance_by_feature = {}
        low_variance_features = []
        constant_features = []

        for col in numeric_features:
            var = self.data[col].var()
            std = self.data[col].std()
            data_range = self.data[col].max() - self.data[col].min()

            variance_by_feature[col] = {
                'variance': float(var) if not pd.isna(var) else 0.0,
                'std_dev': float(std) if not pd.isna(std) else 0.0,
                'range': float(data_range) if not pd.isna(data_range) else 0.0
            }

            if var == 0:
                constant_features.append(col)
            elif var < variance_threshold:
                low_variance_features.append(col)

        # Store results
        self.results['low_variance'] = {
            'total_numeric': len(numeric_features),
            'variance_threshold': variance_threshold,
            'variance_by_feature': variance_by_feature,
            'low_variance_features': low_variance_features,
            'constant_features': constant_features,
            'low_variance_count': len(low_variance_features),
            'analysis_timestamp': pd.Timestamp.now()
        }

        # Display summary
        self.logger.info("\n📊 SUMMARY:")
        self.logger.info("-" * 60)
        self.logger.info(f" • Total Numeric Features: {len(numeric_features)}")
        self.logger.info(f" • Variance Threshold: {variance_threshold}")
        self.logger.info(f" • Low Variance Features: {len(low_variance_features)}")
        self.logger.info(f" • Constant Features (var=0): {len(constant_features)}")

        if constant_features:
            self.logger.warning("\n⚠️ CONSTANT FEATURES (variance = 0):")
            for feat in constant_features:
                unique_val = self.data[feat].dropna().iloc[0] if len(self.data[feat].dropna()) > 0 else 'N/A'
                self.logger.warning(f"  • {feat} = {unique_val}")

        # Exibição de resultados de variância
        if low_variance_features or constant_features:
            all_features_to_display = []

            # Add constant features
            for feat in constant_features:
                all_features_to_display.append((feat, 0.0, 0.0, '🔴 Critical'))

            # Add low variance features with severity
            for feat in low_variance_features:
                var = variance_by_feature[feat]['variance']
                std = variance_by_feature[feat]['std_dev']
                severity = self._get_variance_severity(var, variance_threshold)
                all_features_to_display.append((feat, var, std, severity))

            # Sort by variance descending
            all_features_to_display.sort(key=lambda x: x, reverse=True)

            self.logger.info("\n📉 LOW VARIANCE FEATURES (sorted by variance descending):")
            self.logger.info("-" * 95)
            self.logger.info(f"{'Feature':<40} {'Variance':<16} {'Std Dev':<16} {'Severity'}")
            self.logger.info("-" * 95)

            for feat, var, std, severity in all_features_to_display[:20]:
                self.logger.info(f"{feat:<40} {var:<16.8f} {std:<16.8f} {severity}")

            if len(all_features_to_display) > 20:
                self.logger.info(f"\n ... and {len(all_features_to_display) - 20} more features")

            if show_plots:
                # Combine constant and low variance features for plotting
                all_problem_features = constant_features + low_variance_features
                self.logger.info(f"\n📊 Preparing plot with {len(all_problem_features)} features ({len(constant_features)} constant + {len(low_variance_features)} low variance)")
                if all_problem_features:
                    self._plot_low_variance(variance_by_feature, all_problem_features, variance_threshold)
                else:
                    self.logger.warning("⚠️ No features to plot (this shouldn't happen - please report)")
        else:
            self.logger.info("\n✅ No low or constant variance features found!")
            if show_plots:
                self.logger.info("   (No plot to display - all numeric features have sufficient variance)")

        # Recommendations
        self.logger.info("\n💡 RECOMMENDATIONS:")
        if constant_features:
            self.logger.warning(f"  🔴 Remove {len(constant_features)} constant features - they provide no information")
        if low_variance_features:
            self.logger.warning(f"  🟡 Consider removing {len(low_variance_features)} low variance features")
        if not constant_features and not low_variance_features:
            self.logger.info("  ✅ All numeric features have sufficient variance")

        self.logger.info("\n✅ Low variance analysis complete!")
        self.logger.info(" • Results stored in: analyzer.results['low_variance']")
        
        return {
            'total_numeric': len(numeric_features),
            'low_variance_count': len(low_variance_features),
            'constant_count': len(constant_features),
            'variance_threshold': variance_threshold
        }

    # Métodos auxiliares de variância
    def _get_variance_severity(self, variance: float, threshold: float) -> str:
        """Determine severity of low variance."""
        if variance == 0:
            return "🔴 Critical"
        elif variance < threshold * 0.3:
            return "🔴 Critical"
        elif variance < threshold * 0.6:
            return "🟡 High"
        else:
            return "🟢 Moderate"

    def _plot_low_variance(self, variance_by_feature, low_variance_features, threshold):
        """Plot low variance features (sorted by variance descending - highest at top)."""
        if not low_variance_features:
            self.logger.info("\n✅ No low variance features to plot - all features have sufficient variance!")
            return
            
        sorted_features = sorted(
            low_variance_features,
            key=lambda x: variance_by_feature[x]['variance'],
            reverse=True
        )[:15]

        features = sorted_features
        variances = [variance_by_feature[f]['variance'] for f in features]

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(features[::-1], variances[::-1], color='coral', alpha=0.8, edgecolor='black', linewidth=1.2)

        ax.axvline(threshold, color='darkred', linestyle='--', linewidth=2, label=f'Threshold ({threshold})', zorder=10)

        for bar, var in zip(bars, variances[::-1]):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{var:.6f}', va='center', fontsize=8)

        ax.set_xlabel('Variance', fontsize=12)
        ax.set_title('Low Variance Features', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        #
        plt.tight_layout()
        plt.show()
        self.logger.info("\n📊 Plot displayed above ⬆️")

    # ==============================================================================
    # SPARSE CATEGORICAL ANALYSIS - COMPLETELY INDEPENDENT
    # ==============================================================================

    def analyze_sparse_categorical(self, sparse_threshold: float = 0.01, show_plots: bool = True) -> Dict[str, Any]:
        """⭐ INDEPENDENT ANALYSIS - No prerequisites needed!"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🏷️ SPARSE CATEGORICAL ANALYSIS (Independent - No Prerequisites)")
        self.logger.info("=" * 80)

        self.logger.debug("Analyzing sparse categories from scratch...")
        
        # Use feature type classification instead of raw dtypes
        # This ensures only properly classified categorical features are analyzed,
        # including numeric-dtype columns (e.g. fraud_type=1/2/3) that the analyzer
        # has explicitly labelled as 'categorical'.
        categorical_features = self._get_categorical_features(exclude_binary=True, use_cached=True)
        if not categorical_features:
            # Fallback to dtype if feature type analysis hasn't been run.
            # Include object/category dtype columns AND any numeric-dtype columns
            # that look like categorical codes (low cardinality, max value ≤ 100).
            self.logger.debug("Feature type analysis not available, using dtypes as fallback")
            _fallback_cats = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            # Add numeric columns that behave like categorical codes
            for _col in self.data.select_dtypes(include=[np.number]).columns:
                if _col == self.outcome_feature:
                    continue
                _s = self.data[_col].dropna()
                if (
                    3 <= int(_s.nunique()) <= 10
                    and (_s % 1 == 0).all()
                    and float(_s.abs().max()) <= 100
                    and _col not in _fallback_cats
                ):
                    _fallback_cats.append(_col)
            categorical_features = [
                c for c in _fallback_cats if c != self.outcome_feature
            ]

        sparse_by_feature = {}
        features_with_sparse = []
        dominance_by_feature = {}

        # Análise de cada feature categórica
        for col in categorical_features:
            value_counts = self.data[col].value_counts(normalize=True)
            sparse_categories = value_counts[value_counts <= sparse_threshold]

            dominant_value = value_counts.index[0] if len(value_counts) > 0 else None
            dominance_pct = float(value_counts.iloc[0] * 100) if len(value_counts) > 0 else 0.0

            dominance_by_feature[col] = {
                'dominance_pct': dominance_pct,
                'dominant_value': dominant_value,
                'unique_count': len(value_counts),
                'value_counts': value_counts.to_dict()
            }

            if len(sparse_categories) > 0:
                sparse_by_feature[col] = {
                    'total_categories': len(value_counts),
                    'sparse_categories': len(sparse_categories),
                    'sparse_percentage': (len(sparse_categories) / len(value_counts)) * 100,
                    'min_frequency': float(sparse_categories.min()),
                    'sparse_category_list': sparse_categories.to_dict(),
                    'dominance_pct': dominance_pct,
                    'dominant_value': dominant_value
                }
                features_with_sparse.append(col)

        # Store results
        self.results['sparse_categorical'] = {
            'total_categorical': len(categorical_features),
            'sparse_threshold': sparse_threshold,
            'categorical_features': categorical_features,
            'sparse_by_feature': sparse_by_feature,
            'features_with_sparse': features_with_sparse,
            'sparse_count': len(features_with_sparse),
            'dominance_by_feature': dominance_by_feature,
            'analysis_timestamp': pd.Timestamp.now()
        }

        self.logger.info("\n📊 SUMMARY:")
        self.logger.info("-" * 60)
        self.logger.info(f" • Total Categorical Features: {len(categorical_features)}")
        self.logger.info(f" • Sparse Threshold: {sparse_threshold} ({sparse_threshold*100}%)")

        # Exibição de sumário categórico
        self.logger.info(f" • Features with Sparse Categories: {len(features_with_sparse)}")

        if features_with_sparse:
            self.logger.info("\n📉 FEATURES WITH SPARSE CATEGORIES:")
            self.logger.info("-" * 90)
            self.logger.info(f"{'Feature':<30} {'Total Categories':<18} {'Sparse Categories':<20} {'Sparse %'}")
            self.logger.info("-" * 90)

            sorted_sparse = sorted(features_with_sparse, key=lambda x: sparse_by_feature[x]['sparse_percentage'], reverse=True)

            for feat in sorted_sparse[:15]:
                info = sparse_by_feature[feat]
                self.logger.info(f"{feat:<30} {info['total_categories']:<18} {info['sparse_categories']:<20} {info['sparse_percentage']:.1f}%")

            if len(sorted_sparse) > 15:
                self.logger.info(f"\n ... and {len(sorted_sparse) - 15} more features with sparse categories")

            self.logger.info("\n👑 SPARSE CATEGORICAL FEATURES (sorted by dominance):")
            self.logger.info("-" * 90)
            self.logger.info(f"{'Feature':<34} {'Dominance':<10} {'Dominant Value':<24} {'Severity'}")
            self.logger.info("-" * 90)

            sorted_by_dominance = sorted(features_with_sparse, key=lambda x: sparse_by_feature[x]['dominance_pct'], reverse=True)

            for feat in sorted_by_dominance[:50]:
                info = sparse_by_feature[feat]
                dom_pct = info['dominance_pct']
                dom_val = str(info['dominant_value'])[:22]

                # Severidade baseada na dominância
                if dom_pct >= 99:
                    severity = "🔴 Critical"
                elif dom_pct >= 95:
                    severity = "🟠 High"
                elif dom_pct >= 90:
                    severity = "🟡 Moderate"
                else:
                    severity = "🟢 Low"

                self.logger.info(f"{feat:<34} {dom_pct:>6.1f}% {dom_val:<24} {severity}")

            if len(sorted_by_dominance) > 50:
                self.logger.info(f"\n ... and {len(sorted_by_dominance) - 50} more features with sparse categories")

            self.logger.info("\n" + "#" * 80)
            self.logger.info("# 🔍 Specific Sparse Categories (threshold <= " + f"{sparse_threshold:.2f})")
            self.logger.info("#" * 80)

            sorted_by_sparse_count = sorted(features_with_sparse, key=lambda x: sparse_by_feature[x]['sparse_categories'], reverse=True)

            for feat in sorted_by_sparse_count[:10]:
                info = sparse_by_feature[feat]
                sparse_list = info['sparse_category_list']
                sparse_count = info['sparse_categories']
                total_count = info['total_categories']
                sparse_pct_of_total = (sparse_count / total_count) * 100

                self.logger.info(f"\n🏷️ {feat} ({sparse_count} sparse categories, {sparse_pct_of_total:.1f}% of the total {total_count} categories)")

                sorted_sparse_cats = sorted(sparse_list.items(), key=lambda x: x[1], reverse=True)

                # Top 5 categorias esparsas
                for i, (cat_val, freq) in enumerate(sorted_sparse_cats[:5]):
                    cat_str = str(cat_val) if cat_val != '' else "''"
                    self.logger.info(f"  • {cat_str}: {freq:.6f} ({freq*100:.4f}%)")

                if len(sorted_sparse_cats) > 5:
                    self.logger.info(f"  ... and {len(sorted_sparse_cats) - 5} more sparse categories")

            critical_features = [f for f in sorted_by_dominance if sparse_by_feature[f]['dominance_pct'] >= 99]

            if critical_features:
                self.logger.info("\n💡 DETAILED ANALYSIS & RECOMMENDATIONS:")
                self.logger.info("-" * 90)

                for feat in critical_features[:5]:
                    info = sparse_by_feature[feat]
                    dom_pct = info['dominance_pct']
                    dom_val = info['dominant_value']
                    unique_count = info['total_categories']

                    severity = "🔴 Critical" if dom_pct >= 99 else "🟠 High" if dom_pct >= 95 else "🟡 Moderate"

                    self.logger.info(f"\n♦️ {feat}")
                    self.logger.info(f" • Dominant category: '{dom_val}' ({dom_pct:.2f}%)")
                    self.logger.info(f" • Unique categories: {unique_count}")
                    self.logger.info(f" • Severity: {severity}")

                    if dom_pct >= 99:
                        self.logger.info(" • 💡 Recommendation: Strong candidate for removal - extremely dominated by single category (>99%)")
                    elif dom_pct >= 95:
                        self.logger.info(" • 💡 Recommendation: Consider removal or binary encoding - highly dominated (>95%)")
                    else:
                        self.logger.info(" • 💡 Recommendation: Group rare categories or use frequency/target encoding")

            # Display plot if requested
            if show_plots:
                self._plot_sparse_categorical(sparse_by_feature, sorted_sparse)
        else:
            self.logger.info("\n✅ No sparse categories found!")

        self.logger.info("\n💡 RECOMMENDATIONS:")
        if features_with_sparse:
            self.logger.warning(f"  🟡 {len(features_with_sparse)} features have sparse categories:")
            self.logger.info("    • Group rare categories into 'Other'")
            self.logger.info("    • Use target encoding for rare categories")
            self.logger.info("    • Consider frequency encoding")

        self.logger.info("\n✅ Sparse categorical analysis complete!")
        self.logger.info(" • Results stored in: analyzer.results['sparse_categorical']")

        return {
            'sparse_count': len(features_with_sparse),
            'features_with_sparse': features_with_sparse,
            'sparse_threshold': sparse_threshold
        }

    def _plot_sparse_categorical(self, sparse_by_feature, sorted_features):
        """Plot sparse categorical features in descending order."""
        features = sorted_features[:15]
        sparse_pcts = [sparse_by_feature[f]['sparse_percentage'] for f in features]

        fig, ax = plt.subplots(figsize=(12, 8))
        features_reversed = features[::-1]
        sparse_pcts_reversed = sparse_pcts[::-1]

        bars = ax.barh(features_reversed, sparse_pcts_reversed, color='orange', alpha=0.8, edgecolor='black', linewidth=1.2)

        for bar, pct in zip(bars, sparse_pcts_reversed):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{pct:.1f}%', va='center', fontweight='bold', fontsize=9)

        #
        ax.set_xlabel('Sparse Categories (%)', fontsize=12)
        ax.set_title('Features with Sparse Categories (Descending Order)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.show()
        self.logger.info("\n📊 Plot displayed above ⬆️")

    # ==============================================================================
    # PAIRWISE CORRELATION ANALYSIS - COMPLETELY INDEPENDENT
    # ==============================================================================

    def analyze_correlations(self, correlation_threshold: float = 0.8, show_plots: bool = True, max_features_plot: int = 30) -> Dict[str, Any]:
        """⭐ INDEPENDENT ANALYSIS - No prerequisites needed!"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🔗 PAIRWISE CORRELATION ANALYSIS (Independent - No Prerequisites)")
        self.logger.info("=" * 80)

        self.logger.debug("Calculating correlations from scratch...")

        # Use feature type classification instead of raw dtypes so that ID columns
        # (TIN, MID, SequenceKey) and low-cardinality integer codes (Fraud.Type 1/2/3)
        # are never included in the correlation matrix.
        numeric_features = self._get_numeric_features(exclude_binary=True, use_cached=True)
        if not numeric_features:
            # Fallback to dtype if feature type analysis hasn't been run
            self.logger.debug("Feature type analysis not available, using dtypes as fallback")
            numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if self.outcome_feature and self.outcome_feature in numeric_features:
                numeric_features.remove(self.outcome_feature)

        # Secondary guard: strip any feature the feature_type_map classifies as
        # id / categorical / datetime / outcome / unknown — these should never
        # appear in a Pearson correlation matrix regardless of their pandas dtype.
        _ft_map = self.results.get('feature_types', {}).get('feature_type_map', {})
        _NON_NUMERIC_TYPES = {'id', 'categorical', 'datetime', 'outcome', 'unknown'}
        numeric_features = [
            col for col in numeric_features
            if _ft_map.get(col, 'numeric') not in _NON_NUMERIC_TYPES
        ]

        if len(numeric_features) < 2:
            self.logger.warning("⚠️ Need at least 2 numeric features for correlation analysis")
            return {
                'high_correlation_count': 0,
                'features_involved': 0,
                'correlation_threshold': correlation_threshold
            }

        # Convert all columns to numeric, coercing errors (like '#ERROR!' strings) to NaN
        numeric_data = self.data[numeric_features].apply(pd.to_numeric, errors='coerce')
        
        # Remove columns that became all NaN or have insufficient non-null values
        valid_cols = [col for col in numeric_data.columns if numeric_data[col].notna().sum() > 1]
        
        if len(valid_cols) < 2:
            self.logger.warning("⚠️ After cleaning non-numeric values, insufficient numeric features for correlation analysis")
            return {
                'high_correlation_count': 0,
                'features_involved': 0,
                'correlation_threshold': correlation_threshold
            }
        
        numeric_data = numeric_data[valid_cols]
        corr_matrix = numeric_data.corr(method='pearson')

        high_correlations = []
        features_in_correlations = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= correlation_threshold:
                    feat1 = corr_matrix.columns[i]
                    feat2 = corr_matrix.columns[j]
                    high_correlations.append({
                        'feature_1': feat1,
                        'feature_2': feat2,
                        'correlation': float(corr_value),
                        'abs_correlation': float(abs(corr_value))
                    })
                    features_in_correlations.add(feat1)
                    features_in_correlations.add(feat2)

        high_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)

        feature_max_corr = {}
        for corr_info in high_correlations:
            feat1, feat2 = corr_info['feature_1'], corr_info['feature_2']
            abs_corr = corr_info['abs_correlation']
            feature_max_corr[feat1] = max(feature_max_corr.get(feat1, 0), abs_corr)
            feature_max_corr[feat2] = max(feature_max_corr.get(feat2, 0), abs_corr)

        features_to_remove = sorted(feature_max_corr.items(), key=lambda x: x, reverse=True)

        # Armazenamento de resultados de correlação
        self.results['correlations'] = {
            'total_features': len(numeric_features),
            'correlation_method': 'pearson',
            'correlation_threshold': correlation_threshold,
            'correlation_matrix': corr_matrix,
            'high_correlations': high_correlations,
            'high_correlation_count': len(high_correlations),
            'features_in_correlations': list(features_in_correlations),
            'features_in_correlations_count': len(features_in_correlations),
            'feature_max_corr': feature_max_corr,
            'features_to_remove': features_to_remove,
            'analysis_timestamp': pd.Timestamp.now()
        }

        self.logger.info("\n📊 SUMMARY:")
        self.logger.info("-" * 60)
        self.logger.info(f" • Total Numeric Features: {len(numeric_features)}")
        self.logger.info(f" • Correlation method: pearson")
        self.logger.info(f" • Correlation Threshold: {correlation_threshold}")
        self.logger.info(f" • High Correlations Found: {len(high_correlations)}")
        self.logger.info(f" • Features involved in high correlations: {len(features_in_correlations)}")

        if high_correlations:
            self.logger.info("\n💎 HIGHLY CORRELATED FEATURE PAIRS (sorted by correlation):")
            self.logger.info("-" * 95)
            self.logger.info(f"{'Feature 1':<30} {'Feature 2':<30} {'Correlation':<14} {'Severity'}")
            self.logger.info("-" * 95)

            for corr_info in high_correlations[:15]:
                corr_val = corr_info['correlation']
                abs_corr = corr_info['abs_correlation']
                severity = "🔴 Critical" if abs_corr >= 0.99 else "🟠 High" if abs_corr >= 0.95 else "🟡 Moderate" if abs_corr >= 0.85 else "🟢 Low"

                #
                self.logger.info(f"{corr_info['feature_1']:<30} {corr_info['feature_2']:<30} {corr_val:>11.4f}   {severity}")

            if len(high_correlations) > 15:
                self.logger.info(f"\n ... and {len(high_correlations) - 15} more pairs")

            self._display_multicollinearity_recommendations(high_correlations, features_to_remove, correlation_threshold)

            if show_plots:
                self._plot_correlation_heatmap(corr_matrix, max_features_plot)
        else:
            self.logger.info("\n✅ No high correlations found!")

        self.logger.info("\n✅ Correlation analysis complete!")
        self.logger.info(" • Results stored in: analyzer.results['correlations']")

        return {
            'high_correlation_count': len(high_correlations),
            'features_involved': len(features_in_correlations),
            'correlation_threshold': correlation_threshold
        }

    def _display_multicollinearity_recommendations(self, high_correlations, features_to_remove, threshold):
        """Display detailed multicollinearity recommendations."""
        self.logger.info("\n💡 MULTICOLLINEARITY RECOMMENDATIONS:")
        self.logger.info("-" * 90)

        critical = [c for c in high_correlations if c['abs_correlation'] >= 0.99]
        high = [c for c in high_correlations if 0.95 <= c['abs_correlation'] < 0.99]
        moderate = [c for c in high_correlations if 0.85 <= c['abs_correlation'] < 0.95]

        # Multicolinearidade crítica
        if critical:
            self.logger.info(f"\n🔴 Critical Multicollinearity (|r| >= 0.99):")
            self.logger.info(f"   Found {len(critical)} pair(s) with near-perfect correlation")
            self.logger.info("   💡 Recommendation: Remove one feature from each pair immediately")
            for corr_info in critical[:5]:
                self.logger.info(f"      • {corr_info['feature_1']} ↔ {corr_info['feature_2']}: r={corr_info['correlation']:.4f}")
            if len(critical) > 5:
                self.logger.info(f"      ... and {len(critical) - 5} more critical pairs")

        if high:
            self.logger.info(f"\n🟠 High Multicollinearity (0.95 <= |r| < 0.99):")
            self.logger.info(f"   Found {len(high)} pair(s) with very high correlation")
            self.logger.info("   💡 Recommendation: Consider removing one feature from each pair")
            for corr_info in high[:5]:
                self.logger.info(f"      • {corr_info['feature_1']} ↔ {corr_info['feature_2']}: r={corr_info['correlation']:.4f}")
            if len(high) > 5:
                self.logger.info(f"      ... and {len(high) - 5} more high correlation pairs")

        if moderate:
            self.logger.info(f"\n🟡 Moderate Multicollinearity (0.85 <= |r| < 0.95):")
            self.logger.info(f"   Found {len(moderate)} pair(s) with moderate correlation")
            self.logger.info("   💡 Recommendation: Monitor these pairs during modeling")

        if features_to_remove:
            self.logger.info("\n🎯 SUGGESTED FEATURES FOR REMOVAL:")
            self.logger.info(" • Remove one feature from each pair")
            self.logger.info(" • Use PCA to combine correlated features")
            self.logger.info(" • Apply regularization (L1/L2) in modeling")

        #
            num_to_show = min(len(features_to_remove), 10)
            self.logger.info(f" \n {num_to_show} feature(s) with highest max correlation:")
            for feat, max_corr in features_to_remove[:num_to_show]:
                self.logger.info(f"      • {feat} (max |correlation| = {max_corr:.4f})")

            if len(features_to_remove) > num_to_show:
                self.logger.info(f"      ... and {len(features_to_remove) - num_to_show} more features")

    def _plot_correlation_heatmap(self, corr_matrix, max_features):
        """Plot correlation heatmap."""
        if len(corr_matrix.columns) > max_features:
            avg_corr = corr_matrix.abs().mean().sort_values(ascending=False)
            top_features = avg_corr.head(max_features).index
            corr_matrix = corr_matrix.loc[top_features, top_features]
            self.logger.info(f"\n📊 Showing top {max_features} features (highest avg correlation)")

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(corr_matrix.columns, fontsize=8)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=11)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.show()
        self.logger.info("\n📊 Plot displayed above ⬆️")

    # ==============================================================================
    # OUTCOME CORRELATION ANALYSIS - COMPLETELY INDEPENDENT
    # ==============================================================================

    def analyze_outcome_correlation(self, show_plots: bool = True) -> Dict[str, Any]:
        """⭐ INDEPENDENT ANALYSIS - No prerequisites needed!"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎯 OUTCOME CORRELATION ANALYSIS (Independent - No Prerequisites)")
        self.logger.info("=" * 80)

        if not self.outcome_feature:
            self.logger.warning("⚠️ No outcome feature specified. Cannot perform outcome correlation analysis.")
            return {'error': 'No outcome feature specified'}

        unique_values = self.data[self.outcome_feature].dropna().unique()
        if len(unique_values) != 2:
            self.logger.warning(f"⚠️ Outcome variable '{self.outcome_feature}' is not binary (has {len(unique_values)} unique values).")
            return {'error': f'Outcome not binary: {len(unique_values)} unique values'}

        self.logger.info(f"\n📊 Analyzing correlations with outcome: '{self.outcome_feature}'")
        self.logger.info(f" • Outcome values: {list(unique_values)}")

        # Use feature type classification instead of raw dtypes so that ID columns
        # (TIN, MID, SequenceKey) and low-cardinality integer codes (Fraud.Type 1/2/3)
        # are never fed into point-biserial or Cramér's V under the wrong category.
        numeric_features = self._get_numeric_features(exclude_binary=True, use_cached=True)
        if not numeric_features:
            # Fallback to dtype if feature type analysis hasn't been run
            self.logger.debug("Feature type analysis not available, using dtypes as fallback")
            numeric_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if self.outcome_feature in numeric_features:
                numeric_features.remove(self.outcome_feature)

        # Secondary guard: strip features the feature_type_map classifies as
        # id / categorical / datetime / outcome / unknown regardless of pandas dtype.
        _ft_map = self.results.get('feature_types', {}).get('feature_type_map', {})
        _NON_NUMERIC_TYPES = {'id', 'categorical', 'datetime', 'outcome', 'unknown'}
        numeric_features = [
            col for col in numeric_features
            if _ft_map.get(col, 'numeric') not in _NON_NUMERIC_TYPES
        ]

        self.logger.info(f" • Numeric features found: {len(numeric_features)}")

        # Use feature type classification for categorical features too.
        # _get_categorical_features uses the feature_type_map and therefore
        # correctly includes numeric-dtype columns classified as 'categorical'
        # (e.g. Fraud.Type = {1, 2, 3}).
        categorical_features = self._get_categorical_features(exclude_binary=True, use_cached=True)
        if not categorical_features:
            # Fallback: object/category dtype columns PLUS numeric-dtype low-cardinality codes.
            self.logger.debug("Feature type analysis not available, using dtypes as fallback")
            _fallback_cats = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            for _col in self.data.select_dtypes(include=[np.number]).columns:
                if _col == self.outcome_feature:
                    continue
                _s = self.data[_col].dropna()
                if (
                    3 <= int(_s.nunique()) <= 10
                    and (_s % 1 == 0).all()
                    and float(_s.abs().max()) <= 100
                    and _col not in _fallback_cats
                ):
                    _fallback_cats.append(_col)
            categorical_features = [c for c in _fallback_cats if c != self.outcome_feature]

        self.logger.info(f" • Categorical features found: {len(categorical_features)}")

        pb_results = self._point_biserial_correlation(numeric_features)
        cramers_results = self._cramers_v_analysis(categorical_features)

        self.results['outcome_correlation'] = {
            'outcome_feature': self.outcome_feature,
            'outcome_values': list(unique_values),
            'point_biserial': pb_results,
            'cramers_v': cramers_results,
            'analysis_timestamp': pd.Timestamp.now()
        }

        self._display_outcome_correlation_results(pb_results, cramers_results, show_plots)

        self.logger.info("\n✅ Outcome correlation analysis complete!")
        self.logger.info(" • Results stored in: analyzer.results['outcome_correlation']")

        return {
            'numeric_features_analyzed': pb_results['feature_count'],
            'categorical_features_analyzed': cramers_results['feature_count'],
            'strong_numeric_correlations': pb_results.get('strong_count', 0),
            'strong_categorical_associations': cramers_results.get('strong_count', 0)
        }

    def _point_biserial_correlation(self, numerical_features):
        """Calculate Point-Biserial Correlation between numerical features and binary outcome."""
        if not numerical_features:
            return {'correlations': {}, 'interpretations': {}, 'feature_count': 0, 'strong_count': 0, 'weak_count': 0, 'summary': "No numerical features found..."}

        # Processamento de Point-Biserial
        correlations = {}
        interpretations = {}

        unique_values = self.data[self.outcome_feature].dropna().unique()
        binary_col = self.data[self.outcome_feature].copy()
        
        # Convert target to binary 0/1 if needed
        if set(unique_values) != {0, 1}:
            # Map first unique value to 0, second to 1
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            binary_col = binary_col.map(mapping)
            self.logger.debug(f"Mapped outcome values: {unique_values[0]} -> 0, {unique_values[1]} -> 1")

        for feature in numerical_features:
            try:
                # Convert to numeric, coercing errors (like '#ERROR!' strings) to NaN
                feature_numeric = pd.to_numeric(self.data[feature], errors='coerce')
                
                valid_idx = feature_numeric.notna() & binary_col.notna()
                feature_clean = feature_numeric[valid_idx]
                outcome_clean = binary_col[valid_idx]

                if len(feature_clean) < 10:
                    correlations[feature] = None
                    interpretations[feature] = {'error': f"Insufficient data ({len(feature_clean)} observations, need >=10)"}
                    continue

                # Calculate Point-Biserial correlation (Pearson correlation with binary variable)
                corr_matrix = np.corrcoef(outcome_clean, feature_clean)
                # Extract correlation coefficient from 2x2 matrix (position [0,1] or [1,0])
                corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
                correlations[feature] = corr

                abs_corr = abs(corr)
                if abs_corr >= 0.5:
                    strength, risk_level = "🔴 Very Strong", "critical"
                elif abs_corr >= 0.3:
                    strength, risk_level = "🟠 Strong", "high"
                elif abs_corr >= 0.1:
                    strength, risk_level = "🟡 Moderate", "moderate"
                else:
                    strength, risk_level = "🟢 Weak", "low"

                # Direção e contagem
                direction = "Positive" if corr > 0 else "Negative" if corr < 0 else "None"

                interpretations[feature] = {
                    'correlation': corr, 'abs_correlation': abs_corr,
                    'strength': strength, 'risk_level': risk_level,
                    'direction': direction, 'sample_size': len(feature_clean)
                }
            except Exception as e:
                correlations[feature] = None
                interpretations[feature] = {'error': str(e)}

        strong_count = sum(1 for v in interpretations.values() if 'abs_correlation' in v and v['abs_correlation'] >= 0.3)
        weak_count = sum(1 for v in interpretations.values() if 'abs_correlation' in v and v['abs_correlation'] < 0.1)

        return {
            'correlations': correlations, 'interpretations': interpretations,
            'feature_count': len(numerical_features), 'strong_count': strong_count,
            'weak_count': weak_count, 'summary': f"Point-biserial analysis completed for {len(numerical_features)} features."
        }

    def _cramers_v_analysis(self, categorical_features):
        """Calculate Cramér's V statistic between categorical features and binary outcome."""
        if not categorical_features:
            return {'cramers_v': {}, 'interpretations': {}, 'feature_count': 0, 'strong_count': 0, 'weak_count': 0, 'summary': "No categorical features found..."}

        # Cálculo de Cramér's V
        cramers_v = {}
        chi2_stats = {}
        interpretations = {}

        for feature in categorical_features:
            try:
                feature_data = self.data[feature].dropna()
                outcome_data = self.data[self.outcome_feature].dropna()

                common_idx = feature_data.index.intersection(outcome_data.index)
                if len(common_idx) < 10:
                    cramers_v[feature] = None
                    interpretations[feature] = {'error': f"Insufficient data ({len(common_idx)} obs)"}
                    continue

                contingency_table = pd.crosstab(self.data.loc[common_idx, feature], self.data.loc[common_idx, self.outcome_feature])

                total_obs = contingency_table.sum().sum()
                if contingency_table.size == 0 or total_obs < 10:
                    cramers_v[feature] = None
                    interpretations[feature] = {'error': "Insufficient data for chi-square"}
                    continue
                
                # Check for high cardinality (potential ID-like features)
                n_categories = len(contingency_table)
                cardinality_ratio = n_categories / total_obs
                is_high_cardinality = cardinality_ratio > 0.5  # >50% unique values

                observed = contingency_table.values
                row_totals, col_totals = observed.sum(axis=1), observed.sum(axis=0)
                total = observed.sum()
                expected = np.outer(row_totals, col_totals) / total
                chi2 = ((observed - expected) ** 2 / expected).sum()

                # Finalização do V de Cramér
                n = total
                min_dim = min(contingency_table.shape) - 1
                v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0

                cramers_v[feature], chi2_stats[feature] = v, float(chi2)

                if v >= 0.5: strength, risk_level = "🔴 Very Strong", "critical"
                elif v >= 0.25: strength, risk_level = "🟠 Strong", "high"
                elif v >= 0.15: strength, risk_level = "🟡 Moderate", "moderate"
                elif v >= 0.05: strength, risk_level = "🟢 Weak", "low"
                else: strength, risk_level = "⚪ Very Weak", "very_low"
                
                # Add warning for high-cardinality features with strong association (likely data leakage)
                leakage_warning = None
                if is_high_cardinality and v >= 0.9:
                    leakage_warning = f"⚠️ HIGH CARDINALITY ({n_categories} categories, {cardinality_ratio:.1%} unique) + Perfect Association = Likely DATA LEAKAGE!"

                interpretations[feature] = {
                    'cramers_v': v, 'chi2': float(chi2),
                    'degrees_freedom': (contingency_table.shape[0]-1)*(contingency_table.shape[1]-1),
                    'strength': strength, 'risk_level': risk_level,
                    'sample_size': int(n), 'contingency_shape': contingency_table.shape,
                    'n_categories': n_categories,
                    'cardinality_ratio': cardinality_ratio,
                    'high_cardinality': is_high_cardinality,
                    'leakage_warning': leakage_warning
                }
            except Exception as e:
                cramers_v[feature] = None
                interpretations[feature] = {'error': str(e)}

        # Contagem por força
        strong_count = sum(1 for v in interpretations.values() if 'cramers_v' in v and v['cramers_v'] >= 0.25)
        weak_count = sum(1 for v in interpretations.values() if 'cramers_v' in v and v['cramers_v'] < 0.05)

        return {
            'cramers_v': cramers_v, 'chi2_stats': chi2_stats, 'interpretations': interpretations,
            'feature_count': len(categorical_features), 'strong_count': strong_count,
            'weak_count': weak_count, 'summary': f"Cramér's V analysis completed for {len(categorical_features)} features."
        }

    def _display_outcome_correlation_results(self, pb_results, cramers_results, show_plots):
        """Display comprehensive outcome correlation results."""
        self.logger.info("\n📊 SUMMARY:")
        self.logger.info("-" * 60)
        self.logger.info(f" • Numerical Features Analyzed: {pb_results['feature_count']}")
        self.logger.info(f" • Categorical Features Analyzed: {cramers_results['feature_count']}")
        self.logger.info(f" • Strong Numeric Correlations (|r| >= 0.3): {pb_results['strong_count']}")
        self.logger.info(f" • Strong Categorical Associations (V >= 0.25): {cramers_results['strong_count']}")

        if pb_results['interpretations']:
            sorted_pb = sorted([(f, info) for f, info in pb_results['interpretations'].items() if 'abs_correlation' in info],
                               key=lambda x: x[1]['abs_correlation'], reverse=True)
            if sorted_pb:
                self.logger.info("\n📉 POINT-BISERIAL CORRELATION (Numeric Features):")
                self.logger.info("-" * 95)
                self.logger.info(f"{'Feature':<35} {'Correlation':<14} {'Direction':<12} {'Strength'}")
                self.logger.info("-" * 95)

                #
                for feat, info in sorted_pb[:20]:
                    self.logger.info(f"{feat:<35} {info['correlation']:>11.4f}   {info['direction']:<12} {info['strength']}")

                if len(sorted_pb) > 20:
                    self.logger.info(f"\n ... and {len(sorted_pb) - 20} more numeric features")

        if cramers_results['interpretations']:
            sorted_cramers = sorted([(f, info) for f, info in cramers_results['interpretations'].items() if 'cramers_v' in info],
                                    key=lambda x: x[1].get('cramers_v', 0.0), reverse=True)
            if sorted_cramers:
                self.logger.info("\n📉 CRAMÉR'S V ASSOCIATION (Categorical Features):")
                self.logger.info("-" * 120)
                self.logger.info("{:<35} {:<14} {:<20} {}".format("Feature", "Cramér's V", "Strength", "Warning"))
                self.logger.info("-" * 120)
                for feat, info in sorted_cramers[:20]:
                    warning = info.get('leakage_warning', '')
                    if warning:
                        self.logger.info(f"{feat:<35} {info['cramers_v']:>11.4f}   {info['strength']:<20} {warning}")
                    else:
                        self.logger.info(f"{feat:<35} {info['cramers_v']:>11.4f}   {info['strength']}")

        self._display_outcome_correlation_risks(pb_results, cramers_results)
        if show_plots and (pb_results['interpretations'] or cramers_results['interpretations']):
            self._plot_outcome_correlations(pb_results, cramers_results)

    def _display_outcome_correlation_risks(self, pb_results, cramers_results):
        """Display risk analysis for outcome correlations."""
        self.logger.info("\n⚠️ MODELING RISK ASSESSMENT:")
        self.logger.info("=" * 90)

        # Riscos de Leakage e Baixo Valor
        high_risk_numeric = [(f, info) for f, info in pb_results['interpretations'].items() if 'abs_correlation' in info and info['abs_correlation'] >= 0.5]
        high_risk_categorical = [(f, info) for f, info in cramers_results['interpretations'].items() if 'cramers_v' in info and info['cramers_v'] >= 0.5]

        if high_risk_numeric or high_risk_categorical:
            self.logger.info("\n🔴 HIGH RISK - Very Strong Correlations (Potential Data Leakage):")
            if high_risk_numeric:
                self.logger.info(f"\n   Numeric features ({len(high_risk_numeric)}):")
                for feat, info in sorted(high_risk_numeric, key=lambda x: x[1]['abs_correlation'], reverse=True)[:10]:
                    self.logger.info(f"      • {feat}: r={info['correlation']:.4f} ({info['direction']})")
            if high_risk_categorical:
                self.logger.info(f"\n   Categorical features ({len(high_risk_categorical)}):")
                for feat, info in sorted(high_risk_categorical, key=lambda x: x[1]['cramers_v'], reverse=True)[:10]:
                    self.logger.info(f"      • {feat}: V={info['cramers_v']:.4f}")

        low_value_numeric = [(f, info) for f, info in pb_results['interpretations'].items() if 'abs_correlation' in info and info['abs_correlation'] < 0.05]
        low_value_categorical = [(f, info) for f, info in cramers_results['interpretations'].items() if 'cramers_v' in info and info['cramers_v'] < 0.05]

        if low_value_numeric or low_value_categorical:
            self.logger.info("\n⚪ LOW PREDICTIVE VALUE - Very Weak Correlations:")
            if low_value_numeric:
                self.logger.info(f"\n   Numeric features ({len(low_value_numeric)}):")
                for feat, info in sorted(low_value_numeric, key=lambda x: x[1]['abs_correlation'])[:10]:
                    self.logger.info(f"      • {feat}: r={info['correlation']:.4f}")

        # Faixa Ideal
        if low_value_categorical:
            self.logger.info(f"\n   Categorical features ({len(low_value_categorical)}):")
            for feat, info in sorted(low_value_categorical, key=lambda x: x[1]['cramers_v'])[:10]:
                self.logger.info(f"      • {feat}: V={info['cramers_v']:.4f}")

        optimal_numeric = [(f, info) for f, info in pb_results['interpretations'].items() if 'abs_correlation' in info and 0.1 <= info['abs_correlation'] < 0.5]
        optimal_categorical = [(f, info) for f, info in cramers_results['interpretations'].items() if 'cramers_v' in info and 0.15 <= info['cramers_v'] < 0.5]

        if optimal_numeric or optimal_categorical:
            self.logger.info("\n🟡 OPTIMAL RANGE - Moderate to Strong Correlations:")
            self.logger.info("   ✅ These features likely have good predictive power without data leakage risk")
            self.logger.info(f"   • {len(optimal_numeric)} numeric features in optimal range")
            self.logger.info(f"   • {len(optimal_categorical)} categorical features in optimal range")

    def _plot_outcome_correlations(self, pb_results, cramers_results):
        """Plot outcome correlation results."""
        numeric_data = [(f, info['abs_correlation']) for f, info in pb_results['interpretations'].items() if 'abs_correlation' in info]
        categorical_data = [(f, info['cramers_v']) for f, info in cramers_results['interpretations'].items() if 'cramers_v' in info]
        numeric_data.sort(key=lambda x: x[1], reverse=True)
        categorical_data.sort(key=lambda x: x[1], reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # First subplot: Numeric features (Point-Biserial)
        if numeric_data:
            features_num, values_num = [f for f, _ in numeric_data[:15]], [v for _, v in numeric_data[:15]]
            axes[0].barh(features_num[::-1], values_num[::-1], color='steelblue', alpha=0.8, edgecolor='black')
            axes[0].set_title('Top 15 Numeric Features\n(Point-Biserial Correlation)', fontweight='bold')
            axes[0].set_xlabel('Absolute Correlation')
            for i, v in enumerate(values_num[::-1]):
                axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        else:
            axes[0].text(0.5, 0.5, 'No numeric features analyzed', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Top 15 Numeric Features\n(Point-Biserial Correlation)', fontweight='bold')

        # Second subplot: Categorical features (Cramér's V)
        if categorical_data:
            features_cat, values_cat = [f for f, _ in categorical_data[:15]], [v for _, v in categorical_data[:15]]
            axes[1].barh(features_cat[::-1], values_cat[::-1], color='coral', alpha=0.8, edgecolor='black')
            axes[1].set_title('Top 15 Categorical Features\n(Cramér\'s V Association)', fontweight='bold')
            axes[1].set_xlabel('Cramér\'s V')
            for i, v in enumerate(values_cat[::-1]):
                axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        else:
            axes[1].text(0.5, 0.5, 'No categorical features analyzed', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Top 15 Categorical Features\n(Cramér\'s V Association)', fontweight='bold')

        plt.tight_layout()
        plt.show()
        self.logger.info("\n📊 Plot displayed above ⬆️")

    # ==============================================================================
    # RUN ALL ANALYSES
    # ==============================================================================

    def run_all_analyses(self, show_plots: bool = True, show_samples: bool = True, show_details: bool = True, 
                         missing_threshold: float = 0.5, detect_special_codes: bool = True, 
                         convert_special_to_nan: bool = True, strip_whitespace: bool = True, 
                         show_all_features: bool = True, show_feature_types: bool = True, 
                         variance_threshold: float = 0.01, sparse_threshold: float = 0.01, 
                         correlation_threshold: float = 0.8, max_features_plot: int = 30):
        """Run all available analyses in sequence with customizable parameters.
        
        Parameters:
        -----------
        show_plots : bool, default=True
            Display all visualizations (bar charts, heatmaps, plots)
        show_samples : bool, default=True
            Show data samples for feature type validation
        show_details : bool, default=True
            Show detailed feature lists organized by type
        missing_threshold : float, default=0.5
            Flag features with missing values above this threshold (0.5 = 50%)
        detect_special_codes : bool, default=True
            Detect special missing codes (UNKNOWN, 999, blanks, etc.)
        convert_special_to_nan : bool, default=True
            Convert detected special codes to NaN
        strip_whitespace : bool, default=True
            Clean whitespace from string values
        show_all_features : bool, default=True
            Show complete list of all features with missing values
        show_feature_types : bool, default=True
            Display feature types alongside missing value statistics
        variance_threshold : float, default=0.01
            Flag features with variance below this threshold
        sparse_threshold : float, default=0.01
            Flag rare categories with frequency below this threshold (0.01 = 1%)
        correlation_threshold : float, default=0.8
            Flag feature pairs with correlation above this threshold
        max_features_plot : int, default=30
            Maximum number of features to display in correlation heatmap
        """
        # Logger de parâmetros
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 RUNNING COMPLETE DATASET ANALYSIS")
        self.logger.info("=" * 80)
        self.logger.info("\n📋 Analysis Parameters:")
        self.logger.info(f" • Show plots: {show_plots}")
        self.logger.info(f" • Show details: {show_details}")
        self.logger.info(f" • Show samples: {show_samples}")
        self.logger.info(f" • Missing threshold: {missing_threshold*100}%")
        self.logger.info(f" • Variance threshold: {variance_threshold}")
        self.logger.info(f" • Sparse threshold: {sparse_threshold*100}%")
        self.logger.info(f" • Correlation threshold: {correlation_threshold}")

        # Execução sequencial das análises com TODOS os plots habilitados
        self.analyze_feature_types(
            show_plots=show_plots,
            show_details=show_details,
            show_samples=show_samples,
            show_all_features=show_all_features
        )
        self.analyze_missing_values(
            threshold=missing_threshold,
            show_plots=show_plots,
            show_feature_types=show_feature_types,
            detect_special_codes=detect_special_codes,
            convert_special_to_nan=convert_special_to_nan,
            strip_whitespace=strip_whitespace,
            show_all_features=show_all_features
        )
        self.analyze_low_variance(
            variance_threshold=variance_threshold,
            show_plots=show_plots
        )
        self.analyze_sparse_categorical(
            sparse_threshold=sparse_threshold,
            show_plots=show_plots
        )
        self.analyze_correlations(
            correlation_threshold=correlation_threshold,
            show_plots=show_plots,
            max_features_plot=max_features_plot
        )
        self.analyze_outcome_correlation(
            show_plots=show_plots
        )

        # Data leakage + feature quality
        if self.outcome_feature:
            try:
                self.detect_data_leakage_risk()
            except Exception as e:
                self.logger.warning(f"Leakage detection skipped due to error: {e}")
            try:
                self.score_feature_quality()
            except Exception as e:
                self.logger.warning(f"Feature quality scoring skipped due to error: {e}")

        # Comprehensive alerts (always runs last so it can cross-reference other results)
        try:
            self.analyze_data_alerts()
        except Exception as e:
            self.logger.warning(f"Data alerts scan skipped due to error: {e}")

        self.logger.info("\n" + "=" * 80)
        self.logger.info("✅ COMPLETE ANALYSIS FINISHED")
        self.logger.info("=" * 80)

    # ==============================================================================
    # COMPREHENSIVE DATA ALERTS
    # ==============================================================================

    def analyze_data_alerts(self) -> pd.DataFrame:
        """Scan every column for data quality issues and return a consolidated Alerts report.

        Alert types detected
        --------------------
        MIXED TYPE          : column contains both numeric and string values
        FORMAT INCONSISTENCY: multiple distinct formats in the same column (phone, date strings, etc.)
        HIGH MISSING        : > 20% null values
        CONSTANT            : only one unique non-null value
        NEAR CONSTANT       : one value accounts for > 99% of non-null rows
        ZEROS DOMINANT      : > 90% of numeric values are zero
        HIGH CARDINALITY    : categorical with unique-ratio > 50% (likely an ID)
        SKEWED              : |skewness| > 2 for numeric columns
        OUTLIERS            : values beyond mean ± 3 std deviations
        DUPLICATE ROWS      : identical rows present in the dataset
        IMBALANCED TARGET   : target class ratio < 5% or > 95%
        NEGATIVE VALUES     : unexpected negatives in numeric column
        UNIQUE              : all values are unique (pure ID column)
        """
        import re
        alerts = []

        def _add(col, alert_type, severity, detail):
            alerts.append({
                'feature':    col,
                'alert_type': alert_type,
                'severity':   severity,   # HIGH / MEDIUM / LOW
                'detail':     detail,
            })

        n_rows = len(self.data)

        # ── 1. DUPLICATE ROWS ─────────────────────────────────────────────────
        n_dup = self.data.duplicated().sum()
        if n_dup > 0:
            _add('(dataset)', 'DUPLICATE ROWS', 'HIGH',
                 f'{n_dup:,} duplicate rows found ({n_dup/n_rows*100:.1f}% of dataset)')

        # ── 2. IMBALANCED TARGET ─────────────────────────────────────────────
        if self.outcome_feature and self.outcome_feature in self.data.columns:
            vc = self.data[self.outcome_feature].value_counts(normalize=True)
            min_share = vc.min()
            if min_share < 0.05:
                _add(self.outcome_feature, 'IMBALANCED TARGET', 'HIGH',
                     f'Minority class = {min_share*100:.2f}% — severe class imbalance')
            elif min_share < 0.15:
                _add(self.outcome_feature, 'IMBALANCED TARGET', 'MEDIUM',
                     f'Minority class = {min_share*100:.2f}% — moderate class imbalance')

        # ── Per-column checks ─────────────────────────────────────────────────
        for col in self.data.columns:
            series = self.data[col]
            non_null = series.dropna()
            n_non_null = len(non_null)
            if n_non_null == 0:
                _add(col, 'HIGH MISSING', 'HIGH', '100% missing — column is empty')
                continue

            missing_pct = series.isna().mean()

            # ── 3. HIGH MISSING ──────────────────────────────────────────────
            if missing_pct >= 0.5:
                _add(col, 'HIGH MISSING', 'HIGH',
                     f'{missing_pct*100:.1f}% missing values')
            elif missing_pct >= 0.2:
                _add(col, 'HIGH MISSING', 'MEDIUM',
                     f'{missing_pct*100:.1f}% missing values')

            # ── 4. MIXED TYPE ─────────────────────────────────────────────────
            raw_types = set(type(v).__name__ for v in non_null)
            if len(raw_types) > 1:
                _add(col, 'MIXED TYPE', 'HIGH',
                     f'Column contains multiple Python types: {raw_types}. '
                     f'Must be standardised before modelling.')

            # ── numeric column checks ─────────────────────────────────────────
            if pd.api.types.is_numeric_dtype(series):
                num = non_null.astype(float)

                # 5. CONSTANT
                if num.nunique() == 1:
                    _add(col, 'CONSTANT', 'HIGH',
                         f'All non-null values are identical ({num.iloc[0]})')
                    continue

                # 6. NEAR CONSTANT
                top_share = num.value_counts(normalize=True).iloc[0]
                if top_share >= 0.99:
                    _add(col, 'NEAR CONSTANT', 'MEDIUM',
                         f'Top value ({num.value_counts().index[0]}) = {top_share*100:.1f}% of rows')

                # 7. ZEROS DOMINANT
                zero_share = (num == 0).mean()
                if zero_share >= 0.9:
                    _add(col, 'ZEROS DOMINANT', 'MEDIUM',
                         f'{zero_share*100:.1f}% of values are zero')

                # 8. SKEWED
                try:
                    skew = float(num.skew())
                    if abs(skew) > 5:
                        _add(col, 'SKEWED', 'HIGH',
                             f'Skewness = {skew:.2f} (|skew| > 5 — highly skewed)')
                    elif abs(skew) > 2:
                        _add(col, 'SKEWED', 'MEDIUM',
                             f'Skewness = {skew:.2f} (|skew| > 2)')
                except Exception:
                    pass

                # 9. OUTLIERS (3-sigma rule)
                try:
                    mean, std = float(num.mean()), float(num.std())
                    if std > 0:
                        outlier_mask = (num < mean - 3 * std) | (num > mean + 3 * std)
                        n_out = int(outlier_mask.sum())
                        if n_out > 0:
                            out_pct = n_out / n_non_null * 100
                            sev = 'HIGH' if out_pct > 5 else 'MEDIUM' if out_pct > 1 else 'LOW'
                            _add(col, 'OUTLIERS', sev,
                                 f'{n_out:,} outlier(s) detected ({out_pct:.2f}% of non-null) '
                                 f'beyond mean ± 3σ  [mean={mean:.2f}, std={std:.2f}]')
                except Exception:
                    pass

                # 10. NEGATIVE VALUES
                n_neg = int((num < 0).sum())
                if n_neg > 0:
                    _add(col, 'NEGATIVE VALUES', 'LOW',
                         f'{n_neg:,} negative value(s) — verify if valid for this feature')

                # 11. UNIQUE (pure ID)
                if num.nunique() == n_non_null and n_non_null > 50:
                    _add(col, 'UNIQUE', 'MEDIUM',
                         f'All {n_non_null:,} non-null values are unique — likely an ID/key column')

            # ── string / object column checks ─────────────────────────────────
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                str_series = non_null.astype(str)

                # 5. CONSTANT
                if str_series.nunique() == 1:
                    _add(col, 'CONSTANT', 'HIGH',
                         f'All non-null values are identical ("{str_series.iloc[0]}")')
                    continue

                # 6. NEAR CONSTANT
                top_share = str_series.value_counts(normalize=True).iloc[0]
                if top_share >= 0.99:
                    _add(col, 'NEAR CONSTANT', 'MEDIUM',
                         f'Top value ("{str_series.value_counts().index[0]}") = {top_share*100:.1f}% of rows')

                # 11. UNIQUE (pure ID / free text)
                if str_series.nunique() == n_non_null and n_non_null > 50:
                    _add(col, 'UNIQUE', 'MEDIUM',
                         f'All {n_non_null:,} non-null values are unique — possible ID or free-text')

                # 12. HIGH CARDINALITY
                cardinality_ratio = str_series.nunique() / n_non_null
                if 0.5 <= cardinality_ratio < 1.0 and str_series.nunique() > 50:
                    _add(col, 'HIGH CARDINALITY', 'MEDIUM',
                         f'{str_series.nunique():,} unique values ({cardinality_ratio*100:.1f}% of non-null) — '
                         f'high cardinality; consider encoding or grouping')

                # 13. FORMAT INCONSISTENCY (phone numbers, dates as strings, mixed case, etc.)
                # Phone format check
                if any(kw in col.lower() for kw in ('phone', 'fax', 'mobile', 'tel')):
                    phone_patterns = {
                        'digits_only':    r'^\d{10}$',
                        'dashes':         r'^\d{3}-\d{3}-\d{4}$',
                        'parens':         r'^\(\d{3}\)\s*\d{3}-\d{4}$',
                        'plus_country':   r'^\+1\d{10}$',
                        'dots':           r'^\d{3}\.\d{3}\.\d{4}$',
                        'spaces':         r'^\d{3}\s\d{3}\s\d{4}$',
                    }
                    fmt_counts = {}
                    for name, pat in phone_patterns.items():
                        cnt = int(str_series.str.match(pat).sum())
                        if cnt > 0:
                            fmt_counts[name] = cnt
                    unmatched = n_non_null - sum(fmt_counts.values())
                    if unmatched > 0:
                        fmt_counts['other/unrecognised'] = unmatched
                    if len(fmt_counts) > 1:
                        fmt_summary = ', '.join(f'{k}: {v:,}' for k, v in
                                                sorted(fmt_counts.items(), key=lambda x: -x[1]))
                        _add(col, 'FORMAT INCONSISTENCY', 'HIGH',
                             f'Phone number formats are mixed → {fmt_summary}. '
                             f'Standardise to a single format before modelling.')

                # Date-as-string format check
                elif any(kw in col.lower() for kw in ('date', 'dt', 'time', 'timestamp')):
                    date_patterns = {
                        'YYYY-MM-DD':   r'^\d{4}-\d{2}-\d{2}$',
                        'MM/DD/YYYY':   r'^\d{1,2}/\d{1,2}/\d{4}$',
                        'DD/MM/YYYY':   r'^\d{1,2}/\d{1,2}/\d{4}$',
                        'YYYY/MM/DD':   r'^\d{4}/\d{2}/\d{2}$',
                        'DD-Mon-YYYY':  r'^\d{2}-[A-Za-z]{3}-\d{4}$',
                        'Mon DD YYYY':  r'^[A-Za-z]{3}\s+\d{1,2}\s+\d{4}$',
                    }
                    fmt_counts = {}
                    for name, pat in date_patterns.items():
                        cnt = int(str_series.str.match(pat).sum())
                        if cnt > 0:
                            fmt_counts[name] = cnt
                    if len(fmt_counts) > 1:
                        fmt_summary = ', '.join(f'{k}: {v:,}' for k, v in
                                                sorted(fmt_counts.items(), key=lambda x: -x[1]))
                        _add(col, 'FORMAT INCONSISTENCY', 'MEDIUM',
                             f'Date string formats are mixed → {fmt_summary}. '
                             f'Parse to a single datetime type.')

                # Generic mixed-case inconsistency (e.g. "NEW YORK", "New York", "new york")
                elif str_series.nunique() > 1:
                    lower_unique  = str_series.str.lower().nunique()
                    actual_unique = str_series.nunique()
                    if actual_unique > lower_unique:
                        n_case_dups = actual_unique - lower_unique
                        _add(col, 'FORMAT INCONSISTENCY', 'LOW',
                             f'{n_case_dups} value(s) differ only by case (e.g. "YES" vs "yes" vs "Yes"). '
                             f'Consider normalising to lowercase.')

        # ── Build report DataFrame ────────────────────────────────────────────
        if not alerts:
            self.logger.info("✅ No data quality alerts found — dataset looks clean!")
            df_alerts = pd.DataFrame(columns=['feature','alert_type','severity','detail'])
        else:
            df_alerts = pd.DataFrame(alerts)
            # Sort: HIGH first, then MEDIUM, then LOW; within severity sort by feature
            sev_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            df_alerts['_sev_order'] = df_alerts['severity'].map(sev_order)
            df_alerts = (df_alerts
                         .sort_values(['_sev_order', 'feature'])
                         .drop(columns='_sev_order')
                         .reset_index(drop=True))

        # ── Print formatted report ────────────────────────────────────────────
        sev_icons = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🔵'}
        counts = df_alerts['severity'].value_counts() if len(df_alerts) else {}
        n_high   = int(counts.get('HIGH',   0))
        n_medium = int(counts.get('MEDIUM', 0))
        n_low    = int(counts.get('LOW',    0))

        self.logger.info("\n" + "=" * 80)
        self.logger.info("⚠️   DATA QUALITY ALERTS REPORT")
        self.logger.info("=" * 80)
        self.logger.info(f"  Total alerts : {len(df_alerts)}")
        self.logger.info(f"  🔴 HIGH      : {n_high}")
        self.logger.info(f"  🟡 MEDIUM    : {n_medium}")
        self.logger.info(f"  🔵 LOW       : {n_low}")
        self.logger.info("-" * 80)

        if len(df_alerts) > 0:
            current_sev = None
            for _, row in df_alerts.iterrows():
                if row['severity'] != current_sev:
                    current_sev = row['severity']
                    icon = sev_icons.get(current_sev, '')
                    self.logger.info(f"\n{icon} {current_sev} ALERTS")
                    self.logger.info("-" * 80)
                self.logger.info(f"  [{row['alert_type']:25s}]  {row['feature']}")
                self.logger.info(f"   └─ {row['detail']}")

        self.logger.info("=" * 80)

        # Cache in results
        self.results['alerts'] = {
            'total':  len(df_alerts),
            'high':   n_high,
            'medium': n_medium,
            'low':    n_low,
            'table':  df_alerts,
        }
        return df_alerts

    # ==============================================================================
    # UTILITY METHODS
    # ==============================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all completed analyses."""
        summary = {'dataset_shape': self.data.shape, 'outcome_feature': self.outcome_feature, 'analyses_available': list(self.results.keys())}

        if 'feature_types' in self.results: summary['feature_types'] = self.results['feature_types'].get('type_counts', {})
        if 'missing_values' in self.results:
            summary['missing_values'] = {'total_missing': self.results['missing_values'].get('total_missing', 0), 'overall_percentage': self.results['missing_values'].get('overall_percentage', 0)}
        if 'low_variance' in self.results: summary['low_variance'] = {'low_variance_count': self.results['low_variance'].get('low_variance_count', 0)}
        if 'sparse_categorical' in self.results: summary['sparse_categorical'] = {'sparse_count': self.results['sparse_categorical'].get('sparse_count', 0)}
        if 'correlations' in self.results: summary['correlations'] = {'high_correlation_count': self.results['correlations'].get('high_correlation_count', 0)}

        return summary

    # Exportação de resultados
    def export_results(self, filepath: str, format: str = 'json'):
        """Export analysis results to file."""
        import json, pickle, numpy as np, math

        def clean_for_json(obj):
            """Recursively clean object for JSON serialization."""
            if isinstance(obj, dict): return {key: clean_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list): return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                if np.isnan(obj) or np.isinf(obj): return None
                return obj.item()
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj): return None
                return obj
            elif hasattr(obj, 'to_dict'): return clean_for_json(obj.to_dict())
            #
            elif hasattr(obj, '__dict__'): return str(obj)
            else: return obj

        if format == 'json':
            export_data = {}
            for key, value in self.results.items():
                if value:
                    if key == 'correlations' and 'correlation_matrix' in value:
                        value_copy = value.copy()
                        value_copy['correlation_matrix'] = value_copy['correlation_matrix'].to_dict()
                        export_data[key] = clean_for_json(value_copy)
                    else: export_data[key] = clean_for_json(value)
            with open(filepath, 'w') as f: json.dump(export_data, f, indent=2)
        elif format == 'pickle':
            with open(filepath, 'wb') as f: pickle.dump(self.results, f)

        self.logger.info(f"✅ Results exported to: {filepath}")

    # ==============================================================================
    # BIVARIATE ANALYSIS - Feature Relationship Exploration
    # ==============================================================================
    
    def analyze_bivariate_relationships(
        self, 
        feature_pairs: Optional[list] = None,
        auto_select_top: int = 5,
        show_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze relationships between pairs of features with comprehensive visualizations.
        
        Parameters:
        -----------
        feature_pairs : list of tuples, optional
            Specific feature pairs to analyze: [('feature1', 'feature2'), ...]
            If None, automatically selects top correlated/associated pairs
        auto_select_top : int, default=5
            Number of top feature pairs to auto-select if feature_pairs is None
        show_plots : bool, default=True
            Whether to display visualizations
            
        Returns:
        --------
        dict : Analysis results for each feature pair
        
        Examples:
        ---------
        >>> # Analyze specific pairs
        >>> analyzer.analyze_bivariate_relationships(
        ...     feature_pairs=[('age', 'income'), ('education', 'occupation')]
        ... )
        
        >>> # Auto-select top 5 correlated pairs
        >>> analyzer.analyze_bivariate_relationships(auto_select_top=5)
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🔍 BIVARIATE RELATIONSHIP ANALYSIS")
        self.logger.info("=" * 80)
        
        # Auto-select feature pairs if not specified
        if feature_pairs is None:
            feature_pairs = self._auto_select_feature_pairs(auto_select_top)
            self.logger.info(f"\n📊 Auto-selected {len(feature_pairs)} feature pairs based on correlations")
        else:
            self.logger.info(f"\n📊 Analyzing {len(feature_pairs)} specified feature pairs")
        
        if not feature_pairs:
            self.logger.warning("⚠️ No feature pairs to analyze")
            return {}
        
        results = {}
        
        for i, (feat1, feat2) in enumerate(feature_pairs):
            self.logger.info(f"\n{'─'*80}")
            self.logger.info(f"[{i+1}/{len(feature_pairs)}] Analyzing: '{feat1}' vs '{feat2}'")
            
            # Validate features exist
            if feat1 not in self.data.columns or feat2 not in self.data.columns:
                self.logger.warning(f"⚠️ One or both features not found in dataset. Skipping...")
                continue
            
            # Detect feature types
            type1 = self._detect_single_feature_type(feat1)
            type2 = self._detect_single_feature_type(feat2)
            
            self.logger.info(f" • '{feat1}' type: {type1}")
            self.logger.info(f" • '{feat2}' type: {type2}")
            
            # Analyze based on feature type combinations
            pair_key = f"{feat1}_vs_{feat2}"
            
            if type1 == 'numeric' and type2 == 'numeric':
                results[pair_key] = self._analyze_numeric_vs_numeric(feat1, feat2, show_plots)
            elif type1 == 'numeric' and type2 in ['categorical', 'binary', 'outcome']:
                results[pair_key] = self._analyze_numeric_vs_categorical(feat1, feat2, show_plots)
            elif type1 in ['categorical', 'binary', 'outcome'] and type2 == 'numeric':
                results[pair_key] = self._analyze_numeric_vs_categorical(feat2, feat1, show_plots)
            elif type1 in ['categorical', 'binary', 'outcome'] and type2 in ['categorical', 'binary', 'outcome']:
                results[pair_key] = self._analyze_categorical_vs_categorical(feat1, feat2, show_plots)
            else:
                self.logger.info(f" • Skipping unsupported type combination: {type1} vs {type2}")
                continue
        
        # Store results
        self.results['bivariate_analysis'] = {
            'pairs_analyzed': len(results),
            'results': results,
            'analysis_timestamp': pd.Timestamp.now()
        }
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"✅ Bivariate analysis complete! Analyzed {len(results)} feature pairs")
        self.logger.info(" • Results stored in: analyzer.results['bivariate_analysis']")
        
        return results
    
    def _auto_select_feature_pairs(self, top_n: int = 5) -> list:
        """Auto-select top correlated/associated feature pairs."""
        pairs = []
        
        # Try to use existing correlation analysis results
        if 'correlations' in self.results and self.results['correlations']:
            high_corr = self.results['correlations'].get('high_correlations', [])
            for item in high_corr[:top_n]:
                pairs.append((item['feature_1'], item['feature_2']))
        
        # If we need more pairs, add outcome correlations
        if len(pairs) < top_n and self.outcome_feature and 'outcome_correlation' in self.results:
            outcome_corr = self.results['outcome_correlation']
            
            # Add top numeric features
            if 'point_biserial' in outcome_corr and outcome_corr['point_biserial']:
                pb_results = outcome_corr['point_biserial'].get('correlations', {})
                sorted_pb = sorted(pb_results.items(), key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True)
                for feat, corr in sorted_pb[:min(3, top_n - len(pairs))]:
                    if corr is not None:
                        pairs.append((feat, self.outcome_feature))
            
            # Add top categorical features
            if len(pairs) < top_n and 'cramers_v' in outcome_corr and outcome_corr['cramers_v']:
                cv_results = outcome_corr['cramers_v'].get('associations', {})
                sorted_cv = sorted(cv_results.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
                for feat, assoc in sorted_cv[:min(3, top_n - len(pairs))]:
                    if assoc is not None and assoc > 0.1:
                        pairs.append((feat, self.outcome_feature))
        
        # If still no pairs, select randomly from features
        if not pairs:
            numeric_features = self._get_numeric_features(exclude_binary=True, use_cached=True)
            if len(numeric_features) >= 2:
                pairs.append((numeric_features[0], numeric_features[1]))
        
        return pairs[:top_n]
    
    def _analyze_numeric_vs_numeric(self, feat1: str, feat2: str, show_plots: bool) -> dict:
        """Analyze relationship between two numeric features."""
        # Convert to numeric, handling errors
        data1 = pd.to_numeric(self.data[feat1], errors='coerce')
        data2 = pd.to_numeric(self.data[feat2], errors='coerce')
        
        # Find valid observations
        valid_idx = data1.notna() & data2.notna()
        clean1 = data1[valid_idx]
        clean2 = data2[valid_idx]
        
        if len(clean1) < 10:
            self.logger.warning(f" • Insufficient data ({len(clean1)} valid observations)")
            return {'error': 'Insufficient data'}
        
        # Calculate correlation
        correlation = np.corrcoef(clean1, clean2)[0, 1]
        
        self.logger.info(f" • Pearson Correlation: {correlation:.4f}")
        self.logger.info(f" • Valid observations: {len(clean1):,}")
        
        if show_plots:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scatter plot
            axes[0].scatter(clean1, clean2, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
            axes[0].set_xlabel(feat1)
            axes[0].set_ylabel(feat2)
            axes[0].set_title(f'Scatter Plot\nCorrelation: {correlation:.3f}', fontweight='bold')
            axes[0].grid(alpha=0.3)
            
            # Add trend line
            z = np.polyfit(clean1, clean2, 1)
            p = np.poly1d(z)
            axes[0].plot(clean1, p(clean1), "r--", alpha=0.8, linewidth=2, label='Trend')
            axes[0].legend()
            
            # Hexbin density plot
            hb = axes[1].hexbin(clean1, clean2, gridsize=30, cmap='YlOrRd', mincnt=1)
            axes[1].set_xlabel(feat1)
            axes[1].set_ylabel(feat2)
            axes[1].set_title('Density Plot (Hexbin)', fontweight='bold')
            plt.colorbar(hb, ax=axes[1], label='Count')
            
            plt.tight_layout()
            plt.show()
            self.logger.info(" • 📊 Plots displayed above ⬆️")
        
        return {
            'type': 'numeric_vs_numeric',
            'correlation': float(correlation),
            'valid_observations': int(len(clean1)),
            'feat1_mean': float(clean1.mean()),
            'feat2_mean': float(clean2.mean())
        }
    
    def _analyze_numeric_vs_categorical(self, numeric_feat: str, cat_feat: str, show_plots: bool) -> dict:
        """Analyze relationship between numeric and categorical features."""
        # Convert numeric feature
        numeric_data = pd.to_numeric(self.data[numeric_feat], errors='coerce')
        cat_data = self.data[cat_feat]
        
        # Find valid observations
        valid_idx = numeric_data.notna() & cat_data.notna()
        clean_numeric = numeric_data[valid_idx]
        clean_cat = cat_data[valid_idx]
        
        if len(clean_numeric) < 10:
            self.logger.warning(f" • Insufficient data ({len(clean_numeric)} valid observations)")
            return {'error': 'Insufficient data'}
        
        # Get category statistics
        category_stats = {}
        categories = clean_cat.value_counts().head(10).index.tolist()
        
        for cat in categories:
            cat_values = clean_numeric[clean_cat == cat]
            if len(cat_values) > 0:
                category_stats[str(cat)] = {
                    'mean': float(cat_values.mean()),
                    'median': float(cat_values.median()),
                    'count': int(len(cat_values))
                }
        
        self.logger.info(f" • Categories analyzed: {len(categories)}")
        self.logger.info(f" • Valid observations: {len(clean_numeric):,}")
        
        if show_plots:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Box plot
            plot_data = [clean_numeric[clean_cat == cat].dropna() for cat in categories]
            bp = axes[0].boxplot(plot_data, labels=[str(c)[:20] for c in categories], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            axes[0].set_xlabel(cat_feat)
            axes[0].set_ylabel(numeric_feat)
            axes[0].set_title(f'{numeric_feat} by {cat_feat}\n(Box Plot)', fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', alpha=0.3)
            
            # Violin plot
            axes[1].violinplot(plot_data, positions=range(len(categories)), showmeans=True, showmedians=True)
            axes[1].set_xticks(range(len(categories)))
            axes[1].set_xticklabels([str(c)[:20] for c in categories], rotation=45, ha='right')
            axes[1].set_xlabel(cat_feat)
            axes[1].set_ylabel(numeric_feat)
            axes[1].set_title(f'{numeric_feat} by {cat_feat}\n(Violin Plot)', fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            self.logger.info(" • 📊 Plots displayed above ⬆️")
        
        return {
            'type': 'numeric_vs_categorical',
            'categories_analyzed': len(categories),
            'valid_observations': int(len(clean_numeric)),
            'category_stats': category_stats
        }
    
    def _analyze_categorical_vs_categorical(self, feat1: str, feat2: str, show_plots: bool) -> dict:
        """Analyze relationship between two categorical features."""
        data1 = self.data[feat1]
        data2 = self.data[feat2]
        
        # Find valid observations
        valid_idx = data1.notna() & data2.notna()
        clean1 = data1[valid_idx]
        clean2 = data2[valid_idx]
        
        if len(clean1) < 10:
            self.logger.warning(f" • Insufficient data ({len(clean1)} valid observations)")
            return {'error': 'Insufficient data'}
        
        # Create contingency table
        contingency = pd.crosstab(clean1, clean2)
        
        # Calculate Cramér's V
        chi2 = stats.chi2_contingency(contingency)[0]
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        self.logger.info(f" • Cramér's V: {cramers_v:.4f}")
        self.logger.info(f" • Valid observations: {len(clean1):,}")
        self.logger.info(f" • Feature 1 categories: {len(contingency.index)}")
        self.logger.info(f" • Feature 2 categories: {len(contingency.columns)}")
        
        if show_plots and len(contingency.index) <= 15 and len(contingency.columns) <= 15:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Stacked bar chart
            contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
            contingency_pct.plot(kind='bar', stacked=True, ax=axes[0], alpha=0.8, edgecolor='black', linewidth=0.5)
            axes[0].set_xlabel(feat1)
            axes[0].set_ylabel('Percentage (%)')
            axes[0].set_title(f'Distribution: {feat2} by {feat1}\n(Stacked %)', fontweight='bold')
            axes[0].legend(title=feat2, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(axis='y', alpha=0.3)
            
            # Heatmap
            im = axes[1].imshow(contingency.values, cmap='YlOrRd', aspect='auto')
            axes[1].set_xticks(range(len(contingency.columns)))
            axes[1].set_yticks(range(len(contingency.index)))
            axes[1].set_xticklabels([str(c)[:15] for c in contingency.columns], rotation=45, ha='right', fontsize=8)
            axes[1].set_yticklabels([str(c)[:15] for c in contingency.index], fontsize=8)
            axes[1].set_xlabel(feat2)
            axes[1].set_ylabel(feat1)
            axes[1].set_title(f'Contingency Table\nCramér\'s V: {cramers_v:.3f}', fontweight='bold')
            plt.colorbar(im, ax=axes[1], label='Count')
            
            plt.tight_layout()
            plt.show()
            self.logger.info(" • 📊 Plots displayed above ⬆️")
        
        return {
            'type': 'categorical_vs_categorical',
            'cramers_v': float(cramers_v),
            'valid_observations': int(len(clean1)),
            'feat1_categories': int(len(contingency.index)),
            'feat2_categories': int(len(contingency.columns))
        }

    # ============================================================
    # 1) DATA LEAKAGE RISK DETECTION
    # ============================================================
    def detect_data_leakage_risk(
        self,
        max_unique_ratio_for_id_check: float = 0.95,
        corr_threshold_numeric: float = 0.98,
        cramers_v_threshold: float = 0.90,
        show_top: int = 25
    ) -> Dict[str, Any]:
        """Detect common data leakage patterns relative to the outcome/target feature.

        Heuristics implemented:
        - Exact target duplication (feature == target)
        - Near duplication (feature matches target after simple coercions)
        - High numeric correlation with target (Pearson/point-biserial)
        - High categorical association with target (Cramér's V)
        - ID-like leakage: high uniqueness and deterministic target by ID
        - Name-based suspicion: columns containing target-related keywords
        """

        if not self.outcome_feature:
            raise ValueError("Outcome feature must be provided for leakage detection")

        target = self.data[self.outcome_feature]
        results: Dict[str, Any] = {
            'exact_duplicates': [],
            'near_duplicates': [],
            'high_numeric_correlations': [],
            'high_categorical_associations': [],
            'id_like_determinism': [],
            'name_based_suspects': [],
            'thresholds': {
                'corr_threshold_numeric': corr_threshold_numeric,
                'cramers_v_threshold': cramers_v_threshold,
                'max_unique_ratio_for_id_check': max_unique_ratio_for_id_check
            }
        }

        # Basic keyword suspicion (lightweight, not deterministic)
        target_name = str(self.outcome_feature).lower()
        leak_keywords = {
            target_name,
            'label', 'target', 'outcome', 'fraud', 'chargeback', 'default',
            'approved', 'declined', 'decision', 'result', 'status', 'is_fraud',
            'y', 'class'
        }
        for col in self.data.columns:
            if col == self.outcome_feature:
                continue
            col_l = col.lower()
            if any(k in col_l for k in leak_keywords):
                results['name_based_suspects'].append({'feature': col, 'reason': 'keyword_match'})

        # Try to characterize target type
        target_nonnull = target.dropna()
        is_binary = False
        if len(target_nonnull) > 0:
            uniq = pd.unique(target_nonnull)
            is_binary = len(uniq) == 2

        # Precompute numeric/categorical lists
        numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        cat_cols = [c for c in self.data.columns if c not in numeric_cols and c != self.outcome_feature]

        # Exact / near duplicates
        for col in self.data.columns:
            if col == self.outcome_feature:
                continue
            s = self.data[col]

            # Exact match on overlapping non-null rows
            mask = target.notna() & s.notna()
            if mask.sum() == 0:
                continue
            if (s[mask].values == target[mask].values).all():
                results['exact_duplicates'].append({'feature': col, 'overlap_rows': int(mask.sum())})
                continue

            # Near-duplicate: try simple coercions
            if col in numeric_cols and target.dtype.kind in {'i','u','f'}:
                # numeric target: compare rounded
                try:
                    if np.allclose(s[mask].astype(float).values, target[mask].astype(float).values, equal_nan=True):
                        results['near_duplicates'].append({'feature': col, 'method': 'numeric_allclose', 'overlap_rows': int(mask.sum())})
                except Exception:
                    pass
            else:
                # string coercion / strip
                try:
                    s2 = s.astype(str).str.strip().str.lower()
                    t2 = target.astype(str).str.strip().str.lower()
                    if (s2[mask].values == t2[mask].values).all():
                        results['near_duplicates'].append({'feature': col, 'method': 'string_strip_lower', 'overlap_rows': int(mask.sum())})
                except Exception:
                    pass

        # High numeric correlations with target
        if is_binary:
            # point-biserial is Pearson with binary target coerced to 0/1
            try:
                tbin = target.astype(float)
            except Exception:
                # map two unique values to 0/1
                u = list(pd.unique(target_nonnull))[:2]
                mapping = {u[0]: 0.0, u[1]: 1.0}
                tbin = target.map(mapping).astype(float)

            for col in numeric_cols:
                if col == self.outcome_feature:
                    continue
                x = self.data[col]
                mask = x.notna() & tbin.notna()
                if mask.sum() < 10:
                    continue
                try:
                    r = np.corrcoef(x[mask].astype(float), tbin[mask])[0,1]
                    if np.isfinite(r) and abs(r) >= corr_threshold_numeric:
                        results['high_numeric_correlations'].append({'feature': col, 'correlation': float(r), 'abs_correlation': float(abs(r)), 'n': int(mask.sum())})
                except Exception:
                    continue
        else:
            # numeric target correlation
            if self.outcome_feature in numeric_cols:
                y = target.astype(float)
                for col in numeric_cols:
                    if col == self.outcome_feature:
                        continue
                    x = self.data[col]
                    mask = x.notna() & y.notna()
                    if mask.sum() < 10:
                        continue
                    try:
                        r = np.corrcoef(x[mask].astype(float), y[mask])[0,1]
                        if np.isfinite(r) and abs(r) >= corr_threshold_numeric:
                            results['high_numeric_correlations'].append({'feature': col, 'correlation': float(r), 'abs_correlation': float(abs(r)), 'n': int(mask.sum())})
                    except Exception:
                        continue

        # High categorical association with target (Cramér's V)
        if len(cat_cols) > 0 and len(target_nonnull) > 0:
            for col in cat_cols:
                s = self.data[col]
                mask = s.notna() & target.notna()
                if mask.sum() < 50:
                    continue
                try:
                    contingency = pd.crosstab(s[mask], target[mask])
                    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                        continue
                    chi2 = ((contingency - contingency.mean())**2 / contingency.mean()).sum().sum()
                    n = contingency.sum().sum()
                    r, k = contingency.shape
                    denom = n * (min(k-1, r-1))
                    if denom <= 0:
                        continue
                    v = float(np.sqrt(chi2 / denom))
                    if np.isfinite(v) and v >= cramers_v_threshold:
                        results['high_categorical_associations'].append({'feature': col, 'cramers_v': v, 'n': int(mask.sum())})
                except Exception:
                    continue

        # ID-like determinism check
        for col in self.data.columns:
            if col == self.outcome_feature:
                continue
            s = self.data[col]
            # Heuristic: ID-like if mostly unique and not too missing
            nonnull = s.dropna()
            if len(nonnull) < 100:
                continue
            unique_ratio = nonnull.nunique() / max(len(nonnull), 1)
            if unique_ratio < max_unique_ratio_for_id_check:
                continue
            # Determinism: within-group target variance (or entropy) very low
            tmp = pd.DataFrame({'idcol': s, 'target': target}).dropna()
            if len(tmp) < 200:
                continue
            # If each id appears once, determinism isn't meaningful; require some repeats
            counts = tmp['idcol'].value_counts()
            repeated_ids = counts[counts >= 2].index
            if len(repeated_ids) < 25:
                continue
            sub = tmp[tmp['idcol'].isin(repeated_ids)]
            # For binary, check if each id maps to single target value
            if is_binary:
                by = sub.groupby('idcol')['target'].nunique()
                deterministic_ratio = (by == 1).mean()
                if deterministic_ratio >= 0.98:
                    results['id_like_determinism'].append({'feature': col, 'unique_ratio': float(unique_ratio), 'deterministic_ratio': float(deterministic_ratio), 'repeated_ids': int(len(repeated_ids))})
            else:
                # numeric: check within-id std
                by = sub.groupby('idcol')['target'].std()
                deterministic_ratio = (by.fillna(0.0) <= 1e-9).mean()
                if deterministic_ratio >= 0.98:
                    results['id_like_determinism'].append({'feature': col, 'unique_ratio': float(unique_ratio), 'deterministic_ratio': float(deterministic_ratio), 'repeated_ids': int(len(repeated_ids))})

        # Sort outputs
        results['high_numeric_correlations'] = sorted(results['high_numeric_correlations'], key=lambda d: d.get('abs_correlation', 0.0), reverse=True)
        results['high_categorical_associations'] = sorted(results['high_categorical_associations'], key=lambda d: d.get('cramers_v', 0.0), reverse=True)

        # Log summary
        self.logger.info("\n" + "="*80)
        self.logger.info("🧪 DATA LEAKAGE RISK DETECTION")
        self.logger.info("="*80)
        self.logger.info(f" • Exact duplicates: {len(results['exact_duplicates'])}")
        self.logger.info(f" • Near duplicates:  {len(results['near_duplicates'])}")
        self.logger.info(f" • High numeric corr (≥{corr_threshold_numeric}): {len(results['high_numeric_correlations'])}")
        self.logger.info(f" • High Cramér's V (≥{cramers_v_threshold}): {len(results['high_categorical_associations'])}")
        self.logger.info(f" • ID-like determinism: {len(results['id_like_determinism'])}")
        self.logger.info(f" • Name-based suspects: {len(results['name_based_suspects'])}")

        if results['exact_duplicates']:
            self.logger.info("\n🚨 EXACT DUPLICATES (highest risk):")
            for item in results['exact_duplicates'][:show_top]:
                self.logger.info(f" • {item['feature']} (overlap_rows={item['overlap_rows']:,})")

        if results['high_numeric_correlations']:
            self.logger.info("\n🚨 HIGH NUMERIC CORRELATIONS:")
            for item in results['high_numeric_correlations'][:show_top]:
                self.logger.info(f" • {item['feature']}: r={item['correlation']:.4f} (n={item['n']:,})")

        if results['high_categorical_associations']:
            self.logger.info("\n🚨 HIGH CATEGORICAL ASSOCIATIONS:")
            for item in results['high_categorical_associations'][:show_top]:
                self.logger.info(f" • {item['feature']}: V={item['cramers_v']:.4f} (n={item['n']:,})")

        self.results['leakage_risk'] = results
        return results

    # ============================================================
    # 2) FEATURE QUALITY SCORING
    # ============================================================
    def score_feature_quality(
        self,
        missing_weight: float = 0.35,
        uniqueness_weight: float = 0.20,
        variance_weight: float = 0.20,
        outlier_weight: float = 0.10,
        leakage_penalty_weight: float = 0.15,
        max_features_logged: int = 25
    ) -> pd.DataFrame:
        """Create a simple 0-100 quality score per feature.

        Score components:
        - Completeness (lower missing = higher score)
        - Reasonable uniqueness (penalize near-IDs)
        - Variance / entropy proxy (numeric variance; categorical effective diversity)
        - Outlier rate (numeric only)
        - Leakage penalty (if detect_data_leakage_risk has flagged the feature)
        """
        # Ensure leakage results exist
        leak = self.results.get('leakage_risk') or {}
        leak_feats = set([d['feature'] for d in leak.get('exact_duplicates', [])] +
                         [d['feature'] for d in leak.get('near_duplicates', [])] +
                         [d['feature'] for d in leak.get('high_numeric_correlations', [])] +
                         [d['feature'] for d in leak.get('high_categorical_associations', [])] +
                         [d['feature'] for d in leak.get('id_like_determinism', [])] +
                         [d['feature'] for d in leak.get('name_based_suspects', [])])

        rows = []
        nrows = len(self.data)

        for col in self.data.columns:
            if col == self.outcome_feature:
                continue
            s = self.data[col]
            missing_rate = float(s.isna().mean())
            nonnull = s.dropna()
            unique_ratio = float(nonnull.nunique() / max(len(nonnull), 1)) if len(nonnull) else 0.0

            is_numeric = pd.api.types.is_numeric_dtype(s)
            # Variance / diversity proxy normalized (rough)
            if is_numeric:
                var = float(nonnull.var()) if len(nonnull) > 1 else 0.0
                # robust scale: use IQR as reference
                q1, q3 = (float(nonnull.quantile(0.25)), float(nonnull.quantile(0.75))) if len(nonnull) else (0.0, 0.0)
                iqr = max(q3 - q1, 1e-9)
                var_score = float(np.tanh(var / (iqr**2 + 1e-9)))  # in [0,1)
                # Outlier rate via IQR rule
                if len(nonnull) >= 20:
                    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
                    outlier_rate = float(((nonnull < lower) | (nonnull > upper)).mean())
                else:
                    outlier_rate = 0.0
            else:
                # categorical diversity: effective number of categories
                if len(nonnull) > 0:
                    vc = nonnull.astype(str).value_counts(normalize=True)
                    entropy = float(-(vc * np.log(vc + 1e-12)).sum())
                    # normalize entropy by log(k)
                    k = max(len(vc), 1)
                    var_score = float(entropy / max(np.log(k), 1e-9))
                else:
                    var_score = 0.0
                outlier_rate = 0.0

            # Component scores in [0,1]
            completeness = 1.0 - missing_rate
            # Uniqueness: ideal is neither constant nor ID-like.
            # Use bell-shaped preference centered around ~0.2 unique ratio for typical categoricals, and ~0.9 for continuous numeric.
            if is_numeric:
                uniq_score = 1.0 - min(max((unique_ratio - 0.98) / 0.02, 0.0), 1.0)  # penalize near-IDs
            else:
                # penalize both too low and too high
                uniq_score = float(np.exp(-((unique_ratio - 0.20) / 0.30)**2))

            outlier_score = 1.0 - min(outlier_rate / 0.10, 1.0)  # 10%+ outliers => strong penalty
            leakage_penalty = 1.0 if col in leak_feats else 0.0  # 1 means penalize

            # Weighted sum -> [0,100]
            raw = (
                missing_weight * completeness +
                uniqueness_weight * uniq_score +
                variance_weight * var_score +
                outlier_weight * outlier_score -
                leakage_penalty_weight * leakage_penalty
            )
            score = float(np.clip(raw, 0.0, 1.0) * 100.0)

            rows.append({
                'feature': col,
                'type': 'numeric' if is_numeric else 'categorical',
                'missing_rate': missing_rate,
                'unique_ratio': unique_ratio,
                'diversity_score': var_score,
                'outlier_rate': outlier_rate,
                'leakage_flag': (col in leak_feats),
                'quality_score': score
            })

        df_scores = pd.DataFrame(rows).sort_values('quality_score', ascending=False).reset_index(drop=True)
        self.results['feature_quality'] = {
            'table': df_scores,
            'weights': {
                'missing_weight': missing_weight,
                'uniqueness_weight': uniqueness_weight,
                'variance_weight': variance_weight,
                'outlier_weight': outlier_weight,
                'leakage_penalty_weight': leakage_penalty_weight
            }
        }

        self.logger.info("\n" + "="*80)
        self.logger.info("🏅 FEATURE QUALITY SCORING")
        self.logger.info("="*80)
        self.logger.info(f" • Scored features: {len(df_scores)}")

        self.logger.info("\nTop features by quality score:")
        for _, r in df_scores.head(max_features_logged).iterrows():
            flag = " 🚨leak" if r['leakage_flag'] else ""
            self.logger.info(f" • {r['feature']}: {r['quality_score']:.1f} ({r['type']}){flag}")

        self.logger.info("\nLowest features by quality score:")
        for _, r in df_scores.tail(min(max_features_logged, len(df_scores))).iterrows():
            flag = " 🚨leak" if r['leakage_flag'] else ""
            self.logger.info(f" • {r['feature']}: {r['quality_score']:.1f} ({r['type']}){flag}")

        return df_scores

    # ============================================================
    # 3) LARGE DATASET SAFE MODE (SAMPLING)
    # ============================================================
    @staticmethod
    def safe_sample_dataset(
        data: pd.DataFrame,
        target_feature: Optional[str] = None,
        max_rows: int = 500_000,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Return a safe sampled dataset for expensive EDA steps.

        - If data has <= max_rows, returns original.
        - If target_feature is provided and appears classification-like, uses stratified sampling.
        """
        n = len(data)
        info = {'original_rows': int(n), 'max_rows': int(max_rows), 'sampled': False, 'sample_rows': int(n)}

        if n <= max_rows:
            return {'data': data, 'info': info}

        info['sampled'] = True
        sample_n = max_rows

        if target_feature and target_feature in data.columns:
            y = data[target_feature]
            y_nonnull = y.dropna()
            uniq = pd.unique(y_nonnull) if len(y_nonnull) else []
            is_classification = (len(uniq) > 0 and len(uniq) <= 20)  # heuristic
            if is_classification:
                # stratified sample (approx) using group-wise sampling
                df = data.copy()
                df['_tmp_target_'] = y
                parts = []
                # ensure proportions
                vc = df['_tmp_target_'].value_counts(dropna=False)
                for cls, cnt in vc.items():
                    frac = min(sample_n / n, 1.0)
                    take = int(max(1, round(cnt * frac)))
                    part = df[df['_tmp_target_'] == cls].sample(n=min(take, cnt), random_state=random_state)
                    parts.append(part)
                sampled = pd.concat(parts, axis=0).drop(columns=['_tmp_target_'])
                # If we overshot/undershot, trim or top-up randomly
                if len(sampled) > sample_n:
                    sampled = sampled.sample(n=sample_n, random_state=random_state)
                elif len(sampled) < sample_n:
                    remaining = data.drop(sampled.index, errors='ignore')
                    if len(remaining) > 0:
                        topup = remaining.sample(n=min(sample_n-len(sampled), len(remaining)), random_state=random_state)
                        sampled = pd.concat([sampled, topup], axis=0)
                info['sample_rows'] = int(len(sampled))
                info['strategy'] = 'stratified'
                return {'data': sampled, 'info': info}

        sampled = data.sample(n=sample_n, random_state=random_state)
        info['sample_rows'] = int(len(sampled))
        info['strategy'] = 'random'
        return {'data': sampled, 'info': info}

