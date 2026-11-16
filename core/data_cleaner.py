"""
SmartSight AI - Data Cleaner Module
Provides non-destructive, auditable cleaning operations with before/after previews and metrics.
"""

from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np


class DataCleaner:
    """Safe, auditable dataset transformation engine."""

    @staticmethod
    def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> Tuple[pd.Series, float, float]:
        """Detects outliers using the Interquartile Range (IQR) method."""
        clean_s = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean_s) < 4:
            return pd.Series(False, index=series.index), 0.0, 0.0
        q1 = clean_s.quantile(0.25)
        q3 = clean_s.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)
        is_outlier = (series < lower_bound) | (series > upper_bound)
        return is_outlier, float(lower_bound), float(upper_bound)

    @classmethod
    def clean_dataset(
        cls,
        df: pd.DataFrame,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes non-destructive data cleaning based on configured options.
        Always leaves the original input DataFrame unchanged.
        """
        if options is None:
            options = {
                "remove_duplicates": True,
                "handle_missing_num": "median",  # 'mean', 'median', 'zero', 'drop', 'none'
                "handle_missing_cat": "mode",    # 'mode', 'constant', 'drop', 'none'
                "cat_fill_constant": "Missing",
                "remove_empty_cols": True,
                "remove_empty_rows": True,
                "strip_whitespace": True,
                "convert_numeric_strings": True,
                "convert_dates": True,
                "handle_outliers": "none",       # 'none', 'clip', 'remove'
            }

        # Work on an isolated copy
        cleaned = df.copy(deep=True)
        actions_taken: List[str] = []

        initial_rows, initial_cols = cleaned.shape
        initial_missing = int(cleaned.isna().sum().sum())
        initial_duplicates = int(cleaned.duplicated().sum())

        # 1. Strip Whitespace from headers and string entries
        if options.get("strip_whitespace", True):
            cleaned.columns = [str(c).strip() for c in cleaned.columns]
            for col in cleaned.select_dtypes(include=["object", "string"]).columns:
                cleaned[col] = cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            actions_taken.append("Trimmed whitespace across column names and string cells.")

        # 2. Convert numeric-looking strings (e.g. '$12,500.00' or '15%')
        conversions: Dict[str, str] = {}
        if options.get("convert_numeric_strings", True):
            for col in cleaned.columns:
                if cleaned[col].dtype == "object":
                    sample = cleaned[col].dropna().astype(str).str.strip()
                    if len(sample) > 0:
                        sanitized = (
                            sample.str.replace("$", "", regex=False)
                            .str.replace(",", "", regex=False)
                            .str.replace("%", "", regex=False)
                        )
                        num_s = pd.to_numeric(sanitized, errors="coerce")
                        if num_s.notna().sum() >= max(3, int(0.75 * len(sample))):
                            cleaned[col] = pd.to_numeric(
                                cleaned[col].astype(str)
                                .str.replace("$", "", regex=False)
                                .str.replace(",", "", regex=False)
                                .str.replace("%", "", regex=False)
                                .str.strip(),
                                errors="coerce"
                            )
                            conversions[col] = "numeric"
                            actions_taken.append(f"Converted column '{col}' to numeric format.")

        # 3. Convert date-looking strings
        if options.get("convert_dates", True):
            for col in cleaned.columns:
                if cleaned[col].dtype == "object" and col not in conversions:
                    sample = cleaned[col].dropna().astype(str).str.strip()
                    if len(sample) > 0:
                        try:
                            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                            if parsed.notna().sum() >= max(3, int(0.85 * len(sample))):
                                cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce", format="mixed")
                                conversions[col] = "datetime"
                                actions_taken.append(f"Parsed column '{col}' as datetime.")
                        except Exception:
                            pass

        # 4. Remove completely empty rows & columns
        cols_dropped = 0
        if options.get("remove_empty_cols", True):
            empty_cols = [c for c in cleaned.columns if cleaned[c].isna().all()]
            if empty_cols:
                cleaned.drop(columns=empty_cols, inplace=True)
                cols_dropped += len(empty_cols)
                actions_taken.append(f"Removed {len(empty_cols)} completely empty columns: {empty_cols}.")

        rows_dropped_empty = 0
        if options.get("remove_empty_rows", True):
            before_cnt = len(cleaned)
            cleaned.dropna(how="all", inplace=True)
            rows_dropped_empty = before_cnt - len(cleaned)
            if rows_dropped_empty > 0:
                actions_taken.append(f"Removed {rows_dropped_empty} completely blank rows.")

        # 5. Remove Duplicate Rows
        duplicates_removed = 0
        if options.get("remove_duplicates", True):
            before_cnt = len(cleaned)
            cleaned.drop_duplicates(inplace=True)
            duplicates_removed = before_cnt - len(cleaned)
            if duplicates_removed > 0:
                actions_taken.append(f"Removed {duplicates_removed} duplicate rows.")

        # 6. Handle Missing Values in Numerical Columns
        missing_num_strategy = options.get("handle_missing_num", "median")
        num_filled_count = 0
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_missing = int(cleaned[col].isna().sum())
            if col_missing > 0:
                if missing_num_strategy == "mean":
                    val = cleaned[col].mean()
                    cleaned[col] = cleaned[col].fillna(val)
                    num_filled_count += col_missing
                    actions_taken.append(f"Filled {col_missing} missing in '{col}' with mean ({val:.2f}).")
                elif missing_num_strategy == "median":
                    val = cleaned[col].median()
                    cleaned[col] = cleaned[col].fillna(val)
                    num_filled_count += col_missing
                    actions_taken.append(f"Filled {col_missing} missing in '{col}' with median ({val:.2f}).")
                elif missing_num_strategy == "zero":
                    cleaned[col] = cleaned[col].fillna(0)
                    num_filled_count += col_missing
                    actions_taken.append(f"Filled {col_missing} missing in '{col}' with 0.")
                elif missing_num_strategy == "drop":
                    cleaned.dropna(subset=[col], inplace=True)
                    actions_taken.append(f"Dropped rows with missing values in '{col}'.")

        # 7. Handle Missing Values in Categorical Columns
        missing_cat_strategy = options.get("handle_missing_cat", "mode")
        cat_filled_count = 0
        non_numeric_cols = [c for c in cleaned.columns if c not in numeric_cols]

        for col in non_numeric_cols:
            col_missing = int(cleaned[col].isna().sum())
            if col_missing > 0:
                if missing_cat_strategy == "mode":
                    mode_vals = cleaned[col].mode()
                    if not mode_vals.empty:
                        fill_val = mode_vals.iloc[0]
                        cleaned[col] = cleaned[col].fillna(fill_val)
                        cat_filled_count += col_missing
                        actions_taken.append(f"Filled {col_missing} missing in '{col}' with mode ('{fill_val}').")
                elif missing_cat_strategy == "constant":
                    const_val = options.get("cat_fill_constant", "Missing")
                    cleaned[col] = cleaned[col].fillna(const_val)
                    cat_filled_count += col_missing
                    actions_taken.append(f"Filled {col_missing} missing in '{col}' with constant '{const_val}'.")
                elif missing_cat_strategy == "drop":
                    cleaned.dropna(subset=[col], inplace=True)
                    actions_taken.append(f"Dropped rows with missing values in '{col}'.")

        # 8. Outlier handling
        outlier_strategy = options.get("handle_outliers", "none")
        outliers_handled = 0
        if outlier_strategy in ["clip", "remove"]:
            for col in cleaned.select_dtypes(include=[np.number]).columns:
                is_out, low, high = cls.detect_outliers_iqr(cleaned[col])
                count = int(is_out.sum())
                if count > 0:
                    outliers_handled += count
                    if outlier_strategy == "clip":
                        cleaned[col] = cleaned[col].clip(lower=low, upper=high)
                        actions_taken.append(f"Clipped {count} outliers in '{col}' to range [{low:.2f}, {high:.2f}].")
                    elif outlier_strategy == "remove":
                        cleaned = cleaned[~is_out]
                        actions_taken.append(f"Filtered out {count} outlier rows based on '{col}'.")

        final_rows, final_cols = cleaned.shape
        final_missing = int(cleaned.isna().sum().sum())

        summary = {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "rows_removed": initial_rows - final_rows,
            "initial_cols": initial_cols,
            "final_cols": final_cols,
            "cols_removed": cols_dropped,
            "initial_duplicates": initial_duplicates,
            "duplicates_removed": duplicates_removed,
            "initial_missing": initial_missing,
            "final_missing": final_missing,
            "missing_filled_numerical": num_filled_count,
            "missing_filled_categorical": cat_filled_count,
            "outliers_handled": outliers_handled,
            "type_conversions": conversions,
            "actions_taken": actions_taken,
        }

        return cleaned, summary
