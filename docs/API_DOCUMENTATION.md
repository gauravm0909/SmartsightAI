# SmartSight AI - REST API Documentation

The SmartSight AI backend is built with FastAPI. When the backend is running, dynamic Swagger documentation is available at `http://localhost:8000/docs` and Redoc at `http://localhost:8000/redoc`.

---

## Base URL
```
http://localhost:8000/api
```

---

## 1. Datasets API (`/api/datasets`)

### `POST /api/datasets/upload`
Uploads a CSV or Excel spreadsheet into active memory and registers metadata in SQLite.
- **Content-Type:** `multipart/form-data`
- **Body:** `file` (Binary)
- **Response (200 OK):**
```json
{
  "success": true,
  "dataset_id": "a1b2c3d4",
  "filename": "sales.csv",
  "inspection": {
    "rows": 300,
    "columns": 10,
    "missing_values_total": 0,
    "numeric_columns": ["Quantity", "Unit_Price", "Total_Revenue"],
    "categorical_columns": ["Category", "Region"]
  }
}
```

### `POST /api/datasets/sample`
Loads one of the pre-bundled benchmark datasets (`sample-employees`, `sample-sales`, `sample-churn`).
- **Body:**
```json
{
  "sample_id": "sample-employees"
}
```

### `GET /api/datasets/{dataset_id}/preview`
Returns paginated tabular records.
- **Query Parameters:** `page` (int, default 1), `page_size` (int, default 25)
- **Response (200 OK):**
```json
{
  "dataset_id": "sample-employees",
  "total_rows": 250,
  "page": 1,
  "page_size": 25,
  "columns": ["Department", "Salary", "Experience_Years"],
  "records": [...]
}
```

---

## 2. Profiling API (`/api/profile`)

### `GET /api/profile/{dataset_id}`
Computes distributions, quality scores, and Pearson correlation matrices.
- **Response (200 OK):**
```json
{
  "success": true,
  "profile": {
    "overview": {
      "rows": 250,
      "columns": 9,
      "data_quality_score": 97.5,
      "missing_pct": 0.8
    },
    "numeric_stats": {
      "Salary": {
        "mean": 82341.25,
        "median": 79500.0,
        "std": 14210.5,
        "outlier_count": 4
      }
    }
  }
}
```

---

## 3. Data Cleaning API (`/api/clean`)

### `POST /api/clean/{dataset_id}`
Applies non-destructive cleaning transformations.
- **Body:**
```json
{
  "remove_duplicates": true,
  "handle_missing_num": "median",
  "handle_missing_cat": "mode",
  "strip_whitespace": true,
  "handle_outliers": "none"
}
```
- **Response (200 OK):**
```json
{
  "success": true,
  "summary": {
    "duplicates_removed": 0,
    "missing_filled_numerical": 2,
    "actions_taken": ["Trimmed whitespace across column names and string cells."]
  }
}
```

### `POST /api/clean/{dataset_id}/reset`
Reverts the dataset back to its raw original uncleaned state.

---

## 4. Visualizations API (`/api/visualize`)

### `POST /api/visualize/{dataset_id}`
Generates an interactive Plotly figure JSON specification.
- **Body:**
```json
{
  "chart_type": "Column",
  "x_col": "Department",
  "y_col": "Salary",
  "agg_func": "mean",
  "palette_name": "Corporate Indigo",
  "theme": "plotly_dark"
}
```
- **Response (200 OK):**
```json
{
  "success": true,
  "figure_json": {
    "data": [...],
    "layout": {...}
  }
}
```

---

## 5. AI Insights & Conversational NLP (`/api/insights` & `/api/ask`)

### `POST /api/insights/{dataset_id}`
Returns mathematical findings and optional Gemini executive narrative.
- **Body:**
```json
{
  "use_llm": false,
  "api_key": null
}
```

### `POST /api/ask/{dataset_id}`
Executes safe, allowlisted natural-language queries.
- **Body:**
```json
{
  "query": "Which department has the highest average salary?"
}
```
- **Response (200 OK):**
```json
{
  "answer": "The Engineering department has the highest average salary at 89,420.50.",
  "intent": "extreme_lookup",
  "confidence": "high",
  "chart_spec": {...}
}
```

---

## 6. Machine Learning API (`/api/ml`)

### `POST /api/ml/{dataset_id}/train`
Trains and evaluates ML pipelines with 80/20 train/test split.
- **Body:**
```json
{
  "target_column": "Salary",
  "feature_columns": ["Experience_Years", "Age", "Performance_Rating"],
  "algorithm": "Random Forest Regressor",
  "test_size": 0.2
}
```
- **Response (200 OK):**
```json
{
  "success": true,
  "task_type": "regression",
  "metrics": {
    "r2_score": 0.884,
    "mae": 3412.15,
    "rmse": 4510.32
  },
  "feature_importance": [
    {"feature": "Experience_Years", "importance": 0.742}
  ]
}
```

---

## 7. Reports Download API (`/api/reports`)

- `GET /api/reports/{dataset_id}/pdf`: Streams publication-ready PDF Executive Report.
- `GET /api/reports/{dataset_id}/excel`: Streams multi-sheet styled Excel workbook (`.xlsx`).
- `GET /api/reports/{dataset_id}/csv`: Streams cleaned UTF-8 CSV dataset.
