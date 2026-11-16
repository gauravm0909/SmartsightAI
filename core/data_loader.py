"""
SmartSight AI - Data Loader Module
Handles multi-format ingestion, validation, column type detection, and dataset inspection.
"""

import os
import io
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class DataLoader:
    """Enterprise-grade file loader supporting CSV, XLSX, and XLS with automated schema detection."""

    MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB default ceiling
    SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]

    @staticmethod
    def validate_file(file_path_or_bytes: Union[str, bytes, io.BytesIO], filename: Optional[str] = None) -> Tuple[bool, str]:
        """Validates file existence, format, and size."""
        if isinstance(file_path_or_bytes, str):
            if not os.path.exists(file_path_or_bytes):
                return False, f"File not found: {file_path_or_bytes}"
            size = os.path.getsize(file_path_or_bytes)
            if size == 0:
                return False, "Selected file is completely empty (0 bytes)."
            if size > DataLoader.MAX_FILE_SIZE_BYTES:
                return False, f"File exceeds maximum allowed size ({DataLoader.MAX_FILE_SIZE_BYTES / (1024*1024):.1f} MB)."
            ext = os.path.splitext(file_path_or_bytes)[1].lower()
            if ext not in DataLoader.SUPPORTED_EXTENSIONS:
                return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(DataLoader.SUPPORTED_EXTENSIONS)}"
            return True, "Valid file"

        elif isinstance(file_path_or_bytes, (bytes, io.BytesIO)):
            if filename is None:
                return False, "Filename must be provided when passing binary content."
            ext = os.path.splitext(filename)[1].lower()
            if ext not in DataLoader.SUPPORTED_EXTENSIONS:
                return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(DataLoader.SUPPORTED_EXTENSIONS)}"
            size = len(file_path_or_bytes) if isinstance(file_path_or_bytes, bytes) else file_path_or_bytes.getbuffer().nbytes
            if size == 0:
                return False, "Uploaded file is empty (0 bytes)."
            if size > DataLoader.MAX_FILE_SIZE_BYTES:
                return False, f"File exceeds maximum allowed size ({DataLoader.MAX_FILE_SIZE_BYTES / (1024*1024):.1f} MB)."
            return True, "Valid file"

        return False, "Invalid file input format"

    @classmethod
    def load_dataset(cls, file_source: Union[str, bytes, io.BytesIO], filename: Optional[str] = None) -> pd.DataFrame:
        """
        Loads CSV or Excel data into a pandas DataFrame with automatic delimiter & encoding handling.
        """
        valid, msg = cls.validate_file(file_source, filename)
        if not valid:
            raise ValueError(msg)

        if isinstance(file_source, str):
            ext = os.path.splitext(file_source)[1].lower()
            target = file_source
        else:
            ext = os.path.splitext(filename or "")[1].lower()
            target = io.BytesIO(file_source) if isinstance(file_source, bytes) else file_source

        if ext == ".csv":
            df = cls._read_csv_with_fallback(target)
        elif ext in [".xlsx", ".xls"]:
            try:
                df = pd.read_excel(target)
            except Exception as e:
                raise ValueError(f"Could not parse Excel spreadsheet: {str(e)}")
        else:
            raise ValueError(f"Unsupported format: {ext}")

        if df.empty:
            raise ValueError("The parsed dataset contains 0 rows.")

        # Clean column headers (strip extraneous whitespace)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    @classmethod
    def _read_csv_with_fallback(cls, target: Union[str, io.BytesIO]) -> pd.DataFrame:
        """Tries multiple encodings and separators to parse CSV reliably."""
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
        delimiters = [",", ";", "\t", "|"]

        last_error = None
        for enc in encodings:
            for sep in delimiters:
                try:
                    if isinstance(target, io.BytesIO):
                        target.seek(0)
                    df = pd.read_csv(target, encoding=enc, sep=sep, engine="python")
                    # Heuristic: if dataframe has only 1 column and contains delimiter, wrong separator
                    if len(df.columns) == 1 and any(d in str(df.columns[0]) for d in delimiters if d != sep):
                        continue
                    return df
                except Exception as e:
                    last_error = e
                    continue

        raise ValueError(f"Failed to read CSV file across supported encodings. Reason: {str(last_error)}")

    @classmethod
    def inspect_dataset(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Returns deep structural inspection and identified column classifications."""
        n_rows, n_cols = df.shape
        missing_count = int(df.isna().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        datetime_cols: List[str] = []
        text_cols: List[str] = []

        column_types: Dict[str, str] = {}
        sample_data: Dict[str, List[Any]] = {}

        for col in df.columns:
            series = df[col]
            inferred_type = cls._infer_column_type(series)
            column_types[col] = inferred_type

            if inferred_type == "numeric":
                numeric_cols.append(col)
            elif inferred_type == "datetime":
                datetime_cols.append(col)
            elif inferred_type == "categorical":
                categorical_cols.append(col)
            else:
                text_cols.append(col)

            # Preview sample top 5 values (JSON-serializable)
            sample_data[col] = [
                None if pd.isna(v) else (str(v) if isinstance(v, (pd.Timestamp, np.datetime64)) else v)
                for v in series.head(5).tolist()
            ]

        # Calculate memory footprint
        mem_bytes = df.memory_usage(deep=True).sum()
        mem_formatted = (
            f"{mem_bytes / 1024:.2f} KB" if mem_bytes < 1024 * 1024 else f"{mem_bytes / (1024 * 1024):.2f} MB"
        )

        return {
            "rows": n_rows,
            "columns": n_cols,
            "column_names": list(df.columns),
            "missing_values_total": missing_count,
            "missing_cells_pct": round((missing_count / (n_rows * n_cols)) * 100, 2) if n_rows * n_cols > 0 else 0.0,
            "duplicate_rows": duplicate_rows,
            "memory_usage": mem_formatted,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "text_columns": text_cols,
            "column_types": column_types,
            "preview_sample": sample_data,
        }

    @classmethod
    def _infer_column_type(cls, series: pd.Series) -> str:
        """Determines if a column is numeric, datetime, categorical, or freeform text."""
        # Check standard pandas numeric types
        if pd.api.types.is_numeric_dtype(series):
            # If binary (0/1) or very low unique values, check if categorical
            if series.nunique() <= 2 and not pd.api.types.is_float_dtype(series):
                return "categorical"
            return "numeric"

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Check if object column can be safely converted to datetime
        non_null = series.dropna().astype(str).str.strip()
        if len(non_null) > 0:
            sample = non_null.head(20)
            try:
                # Attempt parsing sample as dates
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().sum() >= max(3, int(0.8 * len(sample))):
                    return "datetime"
            except Exception:
                pass

            # Check if object column can be numeric (e.g. '$1,200', '14.5%')
            cleaned_sample = (
                sample.str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            converted = pd.to_numeric(cleaned_sample, errors="coerce")
            if converted.notna().sum() >= max(3, int(0.8 * len(sample))):
                return "numeric"

        # Low cardinality relative to row count -> categorical
        n_unique = series.nunique()
        total_valid = len(non_null)
        if total_valid > 0 and (n_unique <= 20 or (n_unique / total_valid < 0.25)):
            return "categorical"

        return "text"
